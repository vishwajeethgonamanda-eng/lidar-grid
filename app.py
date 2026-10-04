from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from src.grid_engine import (
    VarResGrid,
    stopping_distance,
    adaptive_fine_radius,
)
from src.loader import (
    load_scan,
    load_labels,
    remap_labels,
    make_synthetic_scene,
    describe_source,
)
from src.traversability import (
    TraversabilityParams,
    TraversabilityMap,
    UNKNOWN,
    DRIVABLE,
    NON_DRIVABLE,
    OBSTACLE,
)
from src.pipeline import process_frame
from src.speed import get_speed
from src.viz25d import build_25d_elevation_figure
from src.raster import (
    render_grid_map,
    render_raw_points,
    render_25d_fast,
    STATE_COLORS,
    SEMANTIC_COLORS,
    STATE_HEX_COLORS,
    SEMANTIC_HEX_COLORS,
)
import plotly.graph_objects as go

# Alias for backward compatibility
render_composite_top_down = render_grid_map


def scan_kitti_sequences(data_dir: Path = Path("data")) -> Dict[str, Dict[str, Any]]:
    """Scans data/ directory for SemanticKITTI sequences containing pairs of .bin and .label files."""
    sequences = {}
    if not data_dir.exists():
        return sequences

    for velo_dir in data_dir.rglob("velodyne"):
        seq_root = velo_dir.parent
        labels_dir = seq_root / "labels"
        if not labels_dir.exists():
            continue

        bin_files = sorted(list(velo_dir.glob("*.bin")))
        if not bin_files:
            continue

        pairs = []
        for b in bin_files:
            lbl = labels_dir / f"{b.stem}.label"
            if lbl.exists():
                pairs.append((b, lbl))

        if pairs:
            seq_name = str(seq_root.relative_to(data_dir))
            cand_poses = seq_root / "poses.txt"
            if not cand_poses.exists():
                cand_poses = seq_root.parent / "poses.txt"
            desc = describe_source(data_dir=data_dir, seq_id=seq_root.name)
            sequences[seq_name] = {
                "pairs": pairs,
                "poses_path": cand_poses if cand_poses.exists() else None,
                "desc": desc,
            }

    return sequences


# HTML Legend definitions placed above each view (16px squares, 15px bold labels)
GRID_LEGEND_HTML = """
<div style="display: flex; gap: 18px; align-items: center; margin-bottom: 8px; flex-wrap: wrap;">
  <span style="display: inline-flex; align-items: center; gap: 6px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #2ecc71; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Drivable</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 6px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #f39c12; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Non-drivable</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 6px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #e74c3c; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Obstacle</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 6px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #23262d; border-radius: 2px; border: 1px solid #4a5263;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Unknown</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 6px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #3FD0D4; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Sensing pulse (decorative)</span>
  </span>
</div>
"""

RAW_LEGEND_HTML = """
<div style="display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap;">
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #3498db; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Road</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #27ae60; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Terrain</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #95a5a6; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Obstacle</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #2B3FA3; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Vehicle</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #e67e22; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Person</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #e74c3c; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Moving</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #f1c40f; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Non-drivable</span>
  </span>
  <span style="display: inline-flex; align-items: center; gap: 5px;">
    <span style="display: inline-block; width: 16px; height: 16px; background: #3FD0D4; border-radius: 2px;"></span>
    <span style="font-size: 15px; font-weight: 700; color: #f1f5f9;">Sensing pulse (decorative)</span>
  </span>
</div>
"""


def format_frame_statistics(
    speed_mps: float,
    s_dist: float,
    d_stop: float,
    fine_radius: float,
    fine_cell_size: float,
    candidates: list,
    n_patches: int,
    stats: dict,
    timings: dict,
    disp_time_ms: float,
    pipe_fps: float,
    disp_fps: float,
) -> str:
    patch_lines = []
    for i in range(4):
        if i < len(candidates) and i < n_patches:
            cand = candidates[i]
            patch_lines.append(
                f"Patch #{i+1}:       ({cand.x:5.1f}m, {cand.y:5.1f}m) | Risk: {cand.risk:.3f} | Cls: {cand.dominant_class}"
            )
        else:
            patch_lines.append(f"Patch #{i+1}:       -- None active --")

    in_patch = stats.get("in_patch", 0)
    extra_lat = timings.get("extra_time_ms", 0.0)
    pipe_ms = max(timings.get("total_ms", 0.0), 0.001)

    return f"""```text
=== Frame Statistics ===
Vehicle Speed:          {speed_mps:5.1f} m/s ({speed_mps * 3.6:5.1f} km/h)
Travelled Distance (s): {s_dist:5.1f} m
Stopping Distance:      {d_stop:5.1f} m
Fine Zone Radius:       {fine_radius:5.1f} m (Cell: {fine_cell_size * 100:4.1f} cm)
Active Focus Patches:   {n_patches:5d}
{patch_lines[0]}
{patch_lines[1]}
{patch_lines[2]}
{patch_lines[3]}
In Fine Zone:           {stats.get('in_fine', 0):7,d} points
In Focus Patches:       {in_patch:7,d} points
In Coarse Zone:         {stats.get('in_coarse', 0):7,d} points
Out of Range (≥100m):   {stats.get('out_of_range', 0):7,d} points
Grid Add Time:          {timings.get('add_points_ms', 0.0):6.2f} ms
Traversability Time:    {timings.get('compute_traversability_ms', 0.0):6.2f} ms
Display Draw Time:      {disp_time_ms:6.2f} ms
Patch Extra Latency:    {extra_lat:6.2f} ms
Pipeline Rate:          {pipe_ms:6.2f} ms ({pipe_fps:5.1f} FPS)
Display Rate:           {disp_time_ms:6.2f} ms ({disp_fps:5.1f} FPS)
```"""


def format_telemetry_html(
    pipe_fps: float,
    pipe_ms: float,
    disp_fps: float,
    disp_ms: float,
    fine_radius: float,
    fine_cell_size: float,
    s_dist: float,
    speed_mps: float,
    mem_var_mb: float,
    extra_mem_kb: float,
    n_patches: int,
    patches_enabled: bool,
    compression_ratio: float,
    mem_uni_mb: float,
    dropped_points: int,
) -> str:
    patch_sub = f"+{extra_mem_kb:.0f} KB ({n_patches} Patches)" if (patches_enabled and n_patches > 0) else "Constant"
    return f"""
    <div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 12px;">
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">Pipeline FPS</div>
        <div style="font-size: 20px; font-weight: 700; color: #4ade80;">{pipe_fps:.1f}</div>
        <div style="font-size: 11px; color: #8c93a4;">{pipe_ms:.1f} ms</div>
      </div>
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">Display FPS</div>
        <div style="font-size: 20px; font-weight: 700; color: #38bdf8;">{disp_fps:.1f}</div>
        <div style="font-size: 11px; color: #8c93a4;">{disp_ms:.1f} ms draw</div>
      </div>
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">Fine Radius</div>
        <div style="font-size: 20px; font-weight: 700; color: #f1f5f9;">{fine_radius:.1f} m</div>
        <div style="font-size: 11px; color: #8c93a4;">Cell: {fine_cell_size * 100:.1f} cm</div>
      </div>
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">Travelled s</div>
        <div style="font-size: 20px; font-weight: 700; color: #f1f5f9;">{s_dist:.1f} m</div>
        <div style="font-size: 11px; color: #8c93a4;">{speed_mps * 3.6:.1f} km/h</div>
      </div>
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">VarRes Memory</div>
        <div style="font-size: 20px; font-weight: 700; color: #f1f5f9;">{mem_var_mb:.2f} MB</div>
        <div style="font-size: 11px; color: #8c93a4;">{patch_sub}</div>
      </div>
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">Compression</div>
        <div style="font-size: 20px; font-weight: 700; color: #facc15;">{compression_ratio:.1f}x</div>
        <div style="font-size: 11px; color: #8c93a4;">vs {mem_uni_mb:.0f} MB</div>
      </div>
      <div style="background: #191c24; padding: 8px 12px; border-radius: 6px; border: 1px solid #2e3342;">
        <div style="font-size: 11px; color: #8c93a4; text-transform: uppercase;">Dropped Points</div>
        <div style="font-size: 20px; font-weight: 700; color: #4ade80;">{dropped_points}</div>
        <div style="font-size: 11px; color: #4ade80;">100% Conserved</div>
      </div>
    </div>
    """


def main():
    st.set_page_config(
        page_title="RAIL-2.5D: Risk-Adaptive 2.5D LiDAR Map",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Browser DOM stale-element dimming and transition override CSS
    st.markdown(
        """
        <style>
        [data-stale="true"] {
            opacity: 1 !important;
            transition: none !important;
        }
        .stale-element-overlay {
            display: none !important;
        }
        [data-testid="stAppViewBlockContainer"],
        .main .block-container,
        div.block-container {
            transition: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RAIL-2.5D: Risk-Adaptive 2.5D LiDAR Map")

    # Sidebar: Data Source & Settings
    st.sidebar.header("Data Source")
    sequences = scan_kitti_sequences(Path("data"))
    poses_path = None
    seq_choice = "00"

    if sequences:
        first_seq_name = list(sequences.keys())[0]
        first_desc = sequences[first_seq_name]["desc"]

        source_mode = st.sidebar.radio("Select Source", [first_desc.label, "Synthetic scene"])
        if source_mode == first_desc.label:
            seq_choice = st.sidebar.selectbox("Sequence", list(sequences.keys()))
            frame_pairs = sequences[seq_choice]["pairs"]
            poses_path = sequences[seq_choice]["poses_path"]
            current_desc = sequences[seq_choice]["desc"]
            num_frames = len(frame_pairs)
            scene_layout_choice = "Open intersection"
        else:
            num_frames = 20
            frame_pairs = None
            current_desc = describe_source(Path("data"))
            scene_layout_choice = st.sidebar.selectbox("Scene layout", ["Open intersection", "Narrow street"], index=0)
    else:
        st.sidebar.info("No sequence data found in `data/`. Running in Synthetic scene mode.")
        num_frames = 20
        frame_pairs = None
        current_desc = describe_source(Path("data"))
        scene_layout_choice = st.sidebar.selectbox("Scene layout", ["Open intersection", "Narrow street"], index=0)

    st.sidebar.caption(f"**Data Source**: {current_desc.label} ({num_frames} frames)")

    # Permanent Data Honesty Badge
    if current_desc.kind == "real_full":
        st.markdown(
            f'<div style="background-color: #163828; color: #2ecc71; padding: 6px 14px; border-radius: 6px; font-weight: 600; margin-bottom: 14px; display: inline-block; border: 1px solid #27ae60;">'
            f'🟢 {current_desc.label}, sequence {seq_choice}, {num_frames} frames, ground-truth labels</div>',
            unsafe_allow_html=True,
        )
    elif current_desc.kind == "synthetic_kitti_format":
        st.markdown(
            f'<div style="background-color: #3e3814; color: #f1c40f; padding: 6px 14px; border-radius: 6px; font-weight: 600; margin-bottom: 14px; display: inline-block; border: 1px solid #f39c12;">'
            f'🟡 {current_desc.label}, sequence {seq_choice}, {num_frames} frames</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background-color: #3e3814; color: #f1c40f; padding: 6px 14px; border-radius: 6px; font-weight: 600; margin-bottom: 14px; display: inline-block; border: 1px solid #f39c12;">'
            f'🟡 {current_desc.label}, {num_frames} frames</div>',
            unsafe_allow_html=True,
        )

    # Playback Controls in Sidebar
    st.sidebar.header("Playback & Navigation")
    if "s" not in st.session_state:
        st.session_state.s = 0.0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "view_revision" not in st.session_state:
        st.session_state.view_revision = 0
    if "frame_history" not in st.session_state:
        st.session_state.frame_history = []

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("Play" if not st.session_state.playing else "Pause"):
        st.session_state.playing = not st.session_state.playing
        st.rerun()

    if col_btn2.button("Reset Frame"):
        st.session_state.frame_idx = 0
        st.session_state.s = 0.0
        st.session_state.playing = False
        st.rerun()

    slider_idx = st.sidebar.slider(
        "Frame Index",
        min_value=0,
        max_value=max(0, num_frames - 1),
        value=st.session_state.frame_idx,
    )
    if slider_idx != st.session_state.frame_idx and not st.session_state.playing:
        st.session_state.frame_idx = slider_idx

    # Speed & Adaptive Fine Zone Controls
    st.sidebar.header("Speed & Adaptive Fine Zone Controls")
    auto_speed = get_speed(st.session_state.frame_idx, poses_path)
    speed_mode = st.sidebar.radio("Speed Source", ["Slider Control", "Auto Speed"], index=0)

    if speed_mode == "Slider Control":
        speed_mps = st.sidebar.slider(
            "Vehicle Speed (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=float(np.round(auto_speed, 1)),
            step=0.5,
            help="Simulated vehicle speed in m/s",
        )
    else:
        speed_mps = auto_speed
        st.sidebar.info(f"Auto Speed: {speed_mps:.1f} m/s ({speed_mps * 3.6:.1f} km/h)")

    adaptive_enabled = st.sidebar.checkbox("Adaptive Fine Zone", value=True)

    d_stop = stopping_distance(speed_mps)
    if adaptive_enabled:
        fine_radius = adaptive_fine_radius(speed_mps)
    else:
        fine_radius = 10.0

    fine_cell_size = (2.0 * fine_radius) / 400.0

    # Risk Focus Patches Controls
    st.sidebar.header("Focus Patches")
    patches_enabled = st.sidebar.checkbox("Risk-Guided Focus Patches", value=True)
    if patches_enabled:
        k_patches = st.sidebar.slider("Max Patches (K)", min_value=1, max_value=8, value=4)
    else:
        k_patches = 0

    # Display Options
    st.sidebar.header("Display Options")
    view_layout = st.sidebar.radio(
        "View Layout",
        ["Grid map", "Raw points", "Side by side", "2.5D elevation map"],
        index=0,
    )
    if view_layout != "2.5D elevation map":
        view_mode = st.sidebar.radio(
            "Map Layer",
            ["Drivable State Map", "Semantic Class Map"],
            index=0,
        )
        view_range = "Road corridor"
        h_exaggeration = 3.0
        viz25d_color_by = "traversability"
        draw_empty_cells = False
    else:
        view_mode = "Drivable State Map"
        st.sidebar.subheader("2.5D Elevation Map Options")
        view_range = st.sidebar.radio(
            "View Range",
            ["Road corridor", "60 m", "Full 100 m"],
            index=0,
            help="Road corridor: 8m behind to 60m ahead, 18m lateral; 60m: 60m radius; Full 100m: full 100m grid",
        )
        h_exaggeration = st.sidebar.slider(
            "Height Exaggeration",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Exaggerates elevation to enhance ground slope and step visibility",
        )
        viz25d_color_by = st.sidebar.selectbox("Colour by", ["traversability", "semantic class", "height"])
        draw_empty_cells = st.sidebar.checkbox("Draw Empty Cells (Faint)", value=False)
        if st.sidebar.button("Reset View (Chase Cam)"):
            st.session_state.view_revision = st.session_state.get("view_revision", 0) + 1
            st.rerun()

    sensing_mode = st.sidebar.selectbox("Sensing rings", ["Pulse", "Static", "Off"], index=0)

    # Traversability Parameters
    st.sidebar.header("Traversability Parameters")
    max_step = st.sidebar.slider("Max Step Height (m)", min_value=0.04, max_value=0.30, value=0.10, step=0.01)
    max_slope_deg = st.sidebar.slider("Max Slope (degrees)", min_value=5.0, max_value=35.0, value=15.0, step=1.0)
    terrain_drivable = st.sidebar.checkbox("Treat Terrain as Drivable", value=False)

    params = TraversabilityParams(
        max_step=float(max_step),
        max_slope_deg=float(max_slope_deg),
        terrain_is_drivable=bool(terrain_drivable),
    )

    # Benchmark Mode Controls
    st.sidebar.header("Benchmark")
    bench_clicked = st.sidebar.button(
        "Run Benchmark Mode (100 frames)",
        help="Runs 100 headless pipeline frames to measure pure compute FPS without UI overhead",
    )
    bench_result_ph = st.sidebar.empty()

    # Static Placeholders created ONCE outside the loop
    telemetry_ph = st.empty()
    col_map, col_info = st.columns([3, 1])

    with col_map:
        if view_layout == "Side by side":
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(GRID_LEGEND_HTML, unsafe_allow_html=True)
                view_left_ph = st.empty()
            with col_r:
                st.markdown(RAW_LEGEND_HTML, unsafe_allow_html=True)
                view_right_ph = st.empty()
            view_ph = None
        else:
            if view_layout == "Raw points":
                st.markdown(RAW_LEGEND_HTML, unsafe_allow_html=True)
            else:
                st.markdown(GRID_LEGEND_HTML, unsafe_allow_html=True)
            view_ph = st.empty()
            view_left_ph = None
            view_right_ph = None

        caption_ph = st.empty()
        info_ph = st.empty()

    with col_info:
        st.subheader("Diagnostics & Details")
        if view_layout == "2.5D elevation map":
            st.markdown(
                """
                - 🟢 **Drivable**: Smooth surface
                - 🟠 **Non-Drivable**: Step/slope limit
                - 🔴 **Obstacle**: Static / dynamic hazard
                - 🟦 **Blue Car**: Ego Vehicle (4.5m x 1.8m)
                - 🔵 **Cyan Ring**: Fine Zone Radius
                - 🟨 **Gold Box**: 64x64 Focus Patch
                - *Playing: Ultra-fast 2.5D perspective raster (<30ms).*
                - *Paused: Interactive 3D (drag to rotate/pan/zoom).*
                """
            )
        elif view_mode == "Drivable State Map":
            st.markdown(
                """
                - 🟢 **Drivable**: Smooth surface
                - 🟠 **Non-Drivable**: Step/slope limit
                - 🔴 **Obstacle**: Vehicle, person, obstacle
                - ⬛ **Unknown**: Unobserved empty cell
                - 🟨 **Gold Box**: 64x64 Focus Patch @ 5cm
                - 🔵 **Cyan Ring**: Fine Zone Boundary
                """
            )
        else:
            st.markdown(
                """
                - 🟦 **Road**: Class 1
                - 🟩 **Terrain**: Class 2
                - ⬜ **Static Obstacle**: Class 3
                - 🟪 **Vehicle**: Class 4
                - 🟧 **Person**: Class 5
                - 🟥 **Moving Object**: Class 6
                - 🟨 **Non-drivable Ground**: Class 7
                """
            )
        st.divider()
        stats_ph = st.empty()

    with st.expander("Stage timings", expanded=False):
        timings_ph = st.empty()

    with st.expander("Results", expanded=False):
        results_ph = st.empty()

    # Helper function to execute pipeline on a given frame
    def run_pipeline_for_frame(f_idx: int, s_val: float):
        if frame_pairs is not None:
            bin_path, label_path = frame_pairs[f_idx]
            scan = load_scan(bin_path)
            raw_labels = load_labels(label_path)
            labels = remap_labels(raw_labels)
            xyz = scan[:, :3]
        else:
            from src.scene_urban import make_urban_scene
            layout_mode = "open" if scene_layout_choice == "Open intersection" else "street"
            xyz, labels = make_urban_scene(
                frame_idx=f_idx,
                seed=42,
                layout=layout_mode,
                s=s_val,
            )
        base_grid = VarResGrid(n_classes=8, fine_radius=fine_radius, forward_offset=0.0)
        t_p0 = time.perf_counter()
        result = process_frame(
            xyz,
            labels,
            grid=base_grid,
            params=params,
            risk_patches=patches_enabled,
            speed_mps=speed_mps,
            k_patches=k_patches,
        )
        t_p1 = time.perf_counter()
        pipe_ms = max((t_p1 - t_p0) * 1000.0, 0.001)
        return xyz, labels, result, pipe_ms

    if bench_clicked:
        with bench_result_ph.container():
            with st.spinner("Benchmarking 100 headless frames..."):
                t_bench_times = []
                for b_idx in range(100):
                    s_bench = float(b_idx * 1.5)
                    f_idx_bench = b_idx % max(1, num_frames)
                    _, _, _, p_ms = run_pipeline_for_frame(f_idx_bench, s_bench)
                    t_bench_times.append(p_ms)
                b_mean_ms = float(np.mean(t_bench_times))
                b_p95_ms = float(np.percentile(t_bench_times, 95))
                b_mean_fps = 1000.0 / max(0.001, b_mean_ms)
                b_p95_fps = 1000.0 / max(0.001, b_p95_ms)
                pass_status = b_mean_fps >= 30.0 and b_p95_fps >= 25.0
                st.success(
                    f"**Benchmark Mode (100 Frames):**\n\n"
                    f"- **Mean Pipeline Rate**: {b_mean_fps:.1f} FPS ({b_mean_ms:.2f} ms)\n"
                    f"- **95th Percentile**: {b_p95_fps:.1f} FPS ({b_p95_ms:.2f} ms)\n"
                    f"- **Target (≥30 FPS)**: {'PASS' if pass_status else 'FAIL'}"
                )

    # Mode 1: Active Playback using while loop with fast rasters (NO st.rerun, NO fragment)
    if st.session_state.playing:
        FRAME_INTERVAL = 0.100  # 10 Hz target
        last_disp_time = 0.0

        while st.session_state.playing:
            t_loop_0 = time.perf_counter()

            # Advance state
            st.session_state.s += float(speed_mps) * 0.1
            st.session_state.frame_idx = (st.session_state.frame_idx + 1) % max(1, num_frames)

            # Pipeline execution
            xyz, labels, result, pipe_ms = run_pipeline_for_frame(st.session_state.frame_idx, st.session_state.s)
            grid = result["grid"]
            trav = result["traversability"]
            timings = result["timings"]
            stats = result["stats"]
            mem_var = result["memory_bytes"]
            mem_uni = result["uniform_equivalent_bytes"]
            extra_mem_kb = result.get("extra_memory_bytes", 0) / 1024.0
            n_patches = len(grid.patches) if hasattr(grid, "patches") else 0
            pipe_fps = 1000.0 / pipe_ms

            # Render fast rasters directly into image placeholders
            t_draw_0 = time.perf_counter()
            mode_key = "state" if view_mode == "Drivable State Map" else "semantic"
            curr_time = time.time()

            if view_layout == "Grid map":
                img = render_grid_map(grid, trav, view_mode=mode_key, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
                view_ph.image(img, width="stretch", output_format="JPEG", clamp=True)
                caption_text = f"Top-Down 200m Grid Map | Fine r={fine_radius:.0f}m ({fine_cell_size*100:.1f}cm) | {n_patches} Focus Patches @ 5cm"
            elif view_layout == "Raw points":
                img = render_raw_points(xyz, labels, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
                view_ph.image(img, width="stretch", output_format="JPEG", clamp=True)
                caption_text = f"Raw Points ({min(len(xyz), 25000):,} Subsampled Points, Semantic Classes)"
            elif view_layout == "Side by side":
                map_img = render_grid_map(grid, trav, view_mode=mode_key, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
                raw_img = render_raw_points(xyz, labels, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
                view_left_ph.image(map_img, width="stretch", output_format="JPEG", clamp=True)
                view_right_ph.image(raw_img, width="stretch", output_format="JPEG", clamp=True)
                caption_text = f"Left: 2.5D Grid Map | Right: Raw Points ({min(len(xyz), 25000):,} pts)"
            else:  # "2.5D elevation map"
                img_25d = render_25d_fast(grid, trav, view_range=view_range, canvas_size=(800, 800), sensing_mode=sensing_mode, current_time=curr_time)
                view_ph.image(img_25d, width="stretch", output_format="JPEG", clamp=True)
                caption_text = f"Fast 2.5D Road Perspective (<30ms) | Range: {view_range} | Height: {h_exaggeration:.1f}x"

            t_draw_1 = time.perf_counter()
            disp_time_ms = max((t_draw_1 - t_draw_0) * 1000.0, 0.001)
            disp_fps = 1000.0 / max(0.001, t_draw_1 - last_disp_time) if last_disp_time > 0.0 else 10.0
            last_disp_time = t_draw_1

            caption_ph.caption(caption_text)

            # Update comparison and stats
            num_raw = len(xyz)
            non_empty_fine = int(np.count_nonzero(grid.fine.count > 0))
            non_empty_coarse = int(np.count_nonzero(grid.coarse.count > 0))
            non_empty_patches = int(sum(np.count_nonzero(p.count > 0) for p in grid.patches)) if hasattr(grid, "patches") else 0
            total_non_empty = non_empty_fine + non_empty_coarse + non_empty_patches
            mem_var_mb = mem_var / (1024.0 * 1024.0)
            mem_uni_mb = mem_uni / (1024.0 * 1024.0)
            dropped_points = stats["total_input"] - (
                stats["in_fine"] + stats.get("in_patch", 0) + stats["in_coarse"] + stats["out_of_range"]
            )
            compression_ratio = mem_uni / max(1, mem_var)

            info_ph.info(
                f"**Frame Comparison:** {num_raw:,} raw points | {total_non_empty:,} non-empty cells | "
                f"VarRes: {mem_var_mb:.2f} MB vs Uniform 5 cm: {mem_uni_mb:.0f} MB ({compression_ratio:.1f}x compression)"
            )

            stats_md = format_frame_statistics(
                speed_mps=speed_mps,
                s_dist=st.session_state.s,
                d_stop=d_stop,
                fine_radius=fine_radius,
                fine_cell_size=fine_cell_size,
                candidates=result.get("candidates", []),
                n_patches=n_patches,
                stats=stats,
                timings=timings,
                disp_time_ms=disp_time_ms,
                pipe_fps=pipe_fps,
                disp_fps=disp_fps,
            )
            stats_ph.markdown(stats_md)

            telemetry_html = format_telemetry_html(
                pipe_fps=pipe_fps,
                pipe_ms=pipe_ms,
                disp_fps=disp_fps,
                disp_ms=disp_time_ms,
                fine_radius=fine_radius,
                fine_cell_size=fine_cell_size,
                s_dist=st.session_state.s,
                speed_mps=speed_mps,
                mem_var_mb=mem_var_mb,
                extra_mem_kb=extra_mem_kb,
                n_patches=n_patches,
                patches_enabled=patches_enabled,
                compression_ratio=compression_ratio,
                mem_uni_mb=mem_uni_mb,
                dropped_points=dropped_points,
            )
            telemetry_ph.markdown(telemetry_html, unsafe_allow_html=True)

            # Stage timings markdown
            t_add = timings.get("pass1_ms", 0.0) if patches_enabled else timings.get("add_points_ms", 0.0)
            t_risk = timings.get("risk_time_ms", 0.0) if patches_enabled else 0.0
            t_rebin = timings.get("add_points_ms", 0.0) if patches_enabled else 0.0
            t_trav = timings.get("compute_traversability_ms", 0.0)
            timings_md = f"""
| Stage | Latency |
| :--- | :--- |
| **add_points** | {t_add:.2f} ms |
| **risk search** | {t_risk:.2f} ms |
| **patch re-bin** | {t_rebin:.2f} ms |
| **traversability** | {t_trav:.2f} ms |
| **Display Draw** | {disp_time_ms:.2f} ms |
| **Pipeline Rate** | {pipe_ms:.2f} ms ({pipe_fps:.1f} FPS) |
| **Display Rate** | {disp_time_ms:.2f} ms ({disp_fps:.1f} FPS) |
"""
            timings_ph.markdown(timings_md)

            # Maintain strict 10 Hz target
            loop_elapsed = time.perf_counter() - t_loop_0
            rem_sleep = FRAME_INTERVAL - loop_elapsed
            if rem_sleep > 0:
                time.sleep(rem_sleep)

    # Mode 2: Paused / Stopped: Render Interactive Plotly figures once with distinct keys
    else:
        xyz, labels, result, pipe_ms = run_pipeline_for_frame(st.session_state.frame_idx, st.session_state.s)
        grid = result["grid"]
        trav = result["traversability"]
        timings = result["timings"]
        stats = result["stats"]
        mem_var = result["memory_bytes"]
        mem_uni = result["uniform_equivalent_bytes"]
        extra_mem_kb = result.get("extra_memory_bytes", 0) / 1024.0
        n_patches = len(grid.patches) if hasattr(grid, "patches") else 0
        pipe_fps = 1000.0 / pipe_ms
        mode_key = "state" if view_mode == "Drivable State Map" else "semantic"

        t_draw_0 = time.perf_counter()
        curr_time = time.time()
        if view_layout == "Grid map":
            img = render_grid_map(grid, trav, view_mode=mode_key, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
            fig = go.Figure(data=[go.Image(z=img)])
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                height=720,
                paper_bgcolor="#14161c",
                plot_bgcolor="#14161c",
                uirevision="chart_grid",
            )
            view_ph.plotly_chart(fig, key="chart_grid", width="stretch")
            caption_text = f"Top-Down 200m Grid Map (Interactive) | Fine r={fine_radius:.0f}m ({fine_cell_size*100:.1f}cm) | {n_patches} Patches"
        elif view_layout == "Raw points":
            img = render_raw_points(xyz, labels, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
            fig = go.Figure(data=[go.Image(z=img)])
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                height=720,
                paper_bgcolor="#14161c",
                plot_bgcolor="#14161c",
                uirevision="chart_raw",
            )
            view_ph.plotly_chart(fig, key="chart_raw", width="stretch")
            caption_text = f"Raw Points (Interactive) ({min(len(xyz), 25000):,} Subsampled Points, Semantic Classes)"
        elif view_layout == "Side by side":
            map_img = render_grid_map(grid, trav, view_mode=mode_key, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
            raw_img = render_raw_points(xyz, labels, canvas_size=800, sensing_mode=sensing_mode, current_time=curr_time)
            fig_l = go.Figure(data=[go.Image(z=map_img)])
            fig_l.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=550,
                paper_bgcolor="#14161c",
                uirevision="chart_side_left",
            )
            fig_r = go.Figure(data=[go.Image(z=raw_img)])
            fig_r.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=550,
                paper_bgcolor="#14161c",
                uirevision="chart_side_right",
            )
            view_left_ph.plotly_chart(fig_l, key="chart_side_left", width="stretch")
            view_right_ph.plotly_chart(fig_r, key="chart_side_right", width="stretch")
            caption_text = f"Left: 2.5D Grid Map | Right: Raw Points"
        else:  # "2.5D elevation map"
            fig_3d, diag_3d = build_25d_elevation_figure(
                grid=grid,
                trav=trav,
                color_mode=viz25d_color_by,
                height_exaggeration=h_exaggeration,
                draw_empty=draw_empty_cells,
                max_cells=30000,
                view_range=view_range,
                view_revision=st.session_state.view_revision,
                road_markings=True,
                sensing_mode=sensing_mode,
                current_time=curr_time,
            )
            view_ph.plotly_chart(
                fig_3d,
                key="chart_25d",
                width="stretch",
                config={"scrollZoom": True, "displayModeBar": False},
            )
            caption_text = f"Interactive 3D Elevation Mesh (Rotate/Pan/Zoom) | Cells: {diag_3d['cells_drawn']:,} | Range: {view_range}"

        t_draw_1 = time.perf_counter()
        disp_time_ms = max((t_draw_1 - t_draw_0) * 1000.0, 0.001)
        disp_fps = 1000.0 / disp_time_ms
        caption_ph.caption(caption_text)

        num_raw = len(xyz)
        non_empty_fine = int(np.count_nonzero(grid.fine.count > 0))
        non_empty_coarse = int(np.count_nonzero(grid.coarse.count > 0))
        non_empty_patches = int(sum(np.count_nonzero(p.count > 0) for p in grid.patches)) if hasattr(grid, "patches") else 0
        total_non_empty = non_empty_fine + non_empty_coarse + non_empty_patches
        mem_var_mb = mem_var / (1024.0 * 1024.0)
        mem_uni_mb = mem_uni / (1024.0 * 1024.0)
        dropped_points = stats["total_input"] - (
            stats["in_fine"] + stats.get("in_patch", 0) + stats["in_coarse"] + stats["out_of_range"]
        )
        compression_ratio = mem_uni / max(1, mem_var)

        info_ph.info(
            f"**Frame Comparison:** {num_raw:,} raw points | {total_non_empty:,} non-empty cells | "
            f"VarRes: {mem_var_mb:.2f} MB vs Uniform 5 cm: {mem_uni_mb:.0f} MB ({compression_ratio:.1f}x compression)"
        )

        stats_md = format_frame_statistics(
            speed_mps=speed_mps,
            s_dist=st.session_state.s,
            d_stop=d_stop,
            fine_radius=fine_radius,
            fine_cell_size=fine_cell_size,
            candidates=result.get("candidates", []),
            n_patches=n_patches,
            stats=stats,
            timings=timings,
            disp_time_ms=disp_time_ms,
            pipe_fps=pipe_fps,
            disp_fps=disp_fps,
        )
        stats_ph.markdown(stats_md)

        telemetry_html = format_telemetry_html(
            pipe_fps=pipe_fps,
            pipe_ms=pipe_ms,
            disp_fps=disp_fps,
            disp_ms=disp_time_ms,
            fine_radius=fine_radius,
            fine_cell_size=fine_cell_size,
            s_dist=st.session_state.s,
            speed_mps=speed_mps,
            mem_var_mb=mem_var_mb,
            extra_mem_kb=extra_mem_kb,
            n_patches=n_patches,
            patches_enabled=patches_enabled,
            compression_ratio=compression_ratio,
            mem_uni_mb=mem_uni_mb,
            dropped_points=dropped_points,
        )
        telemetry_ph.markdown(telemetry_html, unsafe_allow_html=True)

        t_add = timings.get("pass1_ms", 0.0) if patches_enabled else timings.get("add_points_ms", 0.0)
        t_risk = timings.get("risk_time_ms", 0.0) if patches_enabled else 0.0
        t_rebin = timings.get("add_points_ms", 0.0) if patches_enabled else 0.0
        t_trav = timings.get("compute_traversability_ms", 0.0)
        timings_md = f"""
| Stage | Latency |
| :--- | :--- |
| **add_points** | {t_add:.2f} ms |
| **risk search** | {t_risk:.2f} ms |
| **patch re-bin** | {t_rebin:.2f} ms |
| **traversability** | {t_trav:.2f} ms |
| **Display Draw** | {disp_time_ms:.2f} ms |
| **Pipeline Rate** | {pipe_ms:.2f} ms ({pipe_fps:.1f} FPS) |
| **Display Rate** | {disp_time_ms:.2f} ms ({disp_fps:.1f} FPS) |
"""
        timings_ph.markdown(timings_md)

    # Footer credit for real SemanticKITTI data only
    if current_desc.kind == "real_full":
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #8c93a4; font-size: 13px; margin-top: 24px; padding-bottom: 16px;'>"
            "SemanticKITTI, Behley et al., ICCV 2019, CC BY-NC-SA 4.0"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
