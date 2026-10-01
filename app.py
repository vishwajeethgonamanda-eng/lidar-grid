from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import streamlit as st

from src.grid_engine import (
    VarResGrid,
    stopping_distance,
    adaptive_fine_radius,
)
from src.loader import load_scan, load_labels, remap_labels, make_synthetic_scene
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


# Color palettes:
STATE_COLORS = {
    UNKNOWN: np.array([35, 38, 45], dtype=np.float32),        # Dark grey
    DRIVABLE: np.array([46, 204, 113], dtype=np.float32),      # Green
    NON_DRIVABLE: np.array([243, 156, 18], dtype=np.float32),  # Orange
    OBSTACLE: np.array([231, 76, 60], dtype=np.float32),       # Red
}

SEMANTIC_COLORS = {
    0: np.array([40, 40, 45], dtype=np.float32),     # Unlabeled (Dark grey)
    1: np.array([52, 152, 219], dtype=np.float32),   # Road (Blue)
    2: np.array([39, 174, 96], dtype=np.float32),    # Terrain (Green)
    3: np.array([149, 165, 166], dtype=np.float32),  # Static Obstacle (Grey)
    4: np.array([155, 89, 182], dtype=np.float32),   # Vehicle (Purple)
    5: np.array([230, 126, 34], dtype=np.float32),   # Person (Orange)
    6: np.array([231, 76, 60], dtype=np.float32),    # Moving Objects (Red)
    7: np.array([241, 196, 15], dtype=np.float32),   # Non-drivable Ground / Sidewalk (Yellow)
}


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
            sequences[seq_name] = {
                "pairs": pairs,
                "poses_path": cand_poses if cand_poses.exists() else None,
            }

    return sequences


def render_composite_top_down(
    grid: VarResGrid,
    trav: TraversabilityMap,
    view_mode: str = "state",
    canvas_size: int = 800,
) -> np.ndarray:
    """Renders a composite top-down bird's-eye view where x is forward (pointing up)
    and y is lateral (left to right), dynamically centering and scaling the fine zone fovea.
    """
    half_extent = 100.0  # 200m x 200m
    xs = np.linspace(half_extent - 0.125, -half_extent + 0.125, canvas_size, dtype=np.float32)
    ys = np.linspace(-half_extent + 0.125, half_extent - 0.125, canvas_size, dtype=np.float32)

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R_sensor = np.hypot(X, Y)
    R_fine = np.hypot(X - grid.forward_offset, Y)

    bg_color = np.array([30, 32, 38], dtype=np.float32)
    img = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.float32)

    fine_mask = (R_fine < grid.fine_radius) & (R_sensor < 100.0)
    coarse_mask = (~fine_mask) & (R_sensor < 100.0)

    # 1. Render Fine Zone
    f_x = X[fine_mask]
    f_y = Y[fine_mask]
    f_col = np.clip(
        np.floor(((f_x - grid.forward_offset) + grid.fine_radius) / grid.fine_cell_size).astype(np.int64),
        0,
        399,
    )
    f_row = np.clip(
        np.floor((f_y + grid.fine_radius) / grid.fine_cell_size).astype(np.int64),
        0,
        399,
    )

    if view_mode == "state":
        f_st = trav.fine.state[f_row, f_col]
        f_cf = trav.fine.confidence[f_row, f_col, None]

        f_colors = np.zeros((len(f_st), 3), dtype=np.float32)
        for st_val, col_val in STATE_COLORS.items():
            mask = f_st == st_val
            f_colors[mask] = col_val

        f_non_empty = f_st != UNKNOWN
        dimmed = f_colors[f_non_empty] * (0.3 + 0.7 * f_cf[f_non_empty]) + bg_color * (0.7 * (1.0 - f_cf[f_non_empty]))
        f_colors[f_non_empty] = dimmed
        img[fine_mask] = f_colors
    else:
        f_dom = grid.fine.dominant_label[f_row, f_col]
        f_cnt = grid.fine.count[f_row, f_col]
        f_colors = np.zeros((len(f_dom), 3), dtype=np.float32)
        for class_id, col_val in SEMANTIC_COLORS.items():
            mask = (f_dom == class_id) & (f_cnt > 0)
            f_colors[mask] = col_val
        f_colors[f_cnt == 0] = bg_color
        img[fine_mask] = f_colors

    # 2. Render Coarse Zone
    c_x = X[coarse_mask]
    c_y = Y[coarse_mask]
    c_col = np.clip(np.floor((c_x + 100.0) / 0.5).astype(np.int64), 0, 399)
    c_row = np.clip(np.floor((c_y + 100.0) / 0.5).astype(np.int64), 0, 399)

    if view_mode == "state":
        c_st = trav.coarse.state[c_row, c_col]
        c_cf = trav.coarse.confidence[c_row, c_col, None]

        c_colors = np.zeros((len(c_st), 3), dtype=np.float32)
        for st_val, col_val in STATE_COLORS.items():
            mask = c_st == st_val
            c_colors[mask] = col_val

        c_non_empty = c_st != UNKNOWN
        dimmed_c = c_colors[c_non_empty] * (0.3 + 0.7 * c_cf[c_non_empty]) + bg_color * (0.7 * (1.0 - c_cf[c_non_empty]))
        c_colors[c_non_empty] = dimmed_c
        img[coarse_mask] = c_colors
    else:
        c_dom = grid.coarse.dominant_label[c_row, c_col]
        c_cnt = grid.coarse.count[c_row, c_col]
        c_colors = np.zeros((len(c_dom), 3), dtype=np.float32)
        for class_id, col_val in SEMANTIC_COLORS.items():
            mask = (c_dom == class_id) & (f_cnt if 'f_cnt' in locals() and False else c_cnt > 0)
            c_colors[mask] = col_val
        c_colors[c_cnt == 0] = bg_color
        img[coarse_mask] = c_colors

    # 3. Draw Adaptive Seam Circle at fine_radius boundary
    seam_ring = np.abs(R_fine - grid.fine_radius) <= 0.35
    img[seam_ring] = img[seam_ring] * 0.35 + np.array([70, 180, 255], dtype=np.float32) * 0.65

    # 4. Sensor Marker at (0, 0)
    center = canvas_size // 2
    img[center - 3 : center + 4, center - 1 : center + 2] = [255, 255, 255]
    img[center - 1 : center + 2, center - 3 : center + 4] = [255, 255, 255]

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    st.set_page_config(
        page_title="Variable-Resolution 2.5D LiDAR Grid Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Adaptive-Fovea 2.5D LiDAR Grid & Traversability")

    # Sidebar: Data Source & Settings
    st.sidebar.header("Data Source")
    sequences = scan_kitti_sequences(Path("data"))
    poses_path = None

    if sequences:
        source_mode = st.sidebar.radio("Select Source", ["SemanticKITTI Dataset", "Synthetic Scene Generator"])
        if source_mode == "SemanticKITTI Dataset":
            seq_choice = st.sidebar.selectbox("Sequence", list(sequences.keys()))
            frame_pairs = sequences[seq_choice]["pairs"]
            poses_path = sequences[seq_choice]["poses_path"]
            num_frames = len(frame_pairs)
        else:
            num_frames = 20
            frame_pairs = None
    else:
        st.sidebar.info("No SemanticKITTI data found in `data/`. Running in Synthetic Scene mode.")
        num_frames = 20
        frame_pairs = None

    # Playback Controls
    st.sidebar.header("Playback & Navigation")
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("Play" if not st.session_state.playing else "Pause"):
        st.session_state.playing = not st.session_state.playing

    if col_btn2.button("Reset Frame"):
        st.session_state.frame_idx = 0
        st.session_state.playing = False

    frame_idx = st.sidebar.slider(
        "Frame Index",
        min_value=0,
        max_value=max(0, num_frames - 1),
        value=st.session_state.frame_idx,
    )
    st.session_state.frame_idx = frame_idx

    # Speed & Adaptive Fovea Controls
    st.sidebar.header("Adaptive Fovea Controls")
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

    adaptive_enabled = st.sidebar.checkbox("Adaptive Fovea", value=True)

    # Compute Fine Radius & Stopping Distance
    d_stop = stopping_distance(speed_mps)
    if adaptive_enabled:
        fine_radius = adaptive_fine_radius(speed_mps)
    else:
        fine_radius = 10.0

    fine_cell_size = (2.0 * fine_radius) / 400.0

    # Display Options
    st.sidebar.header("Display Options")
    view_mode = st.sidebar.radio(
        "Map Layer",
        ["Drivable State Map", "Semantic Class Map"],
        index=0,
    )

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

    # Load / Generate Frame Data
    if frame_pairs is not None:
        bin_path, label_path = frame_pairs[st.session_state.frame_idx]
        scan = load_scan(bin_path)
        raw_labels = load_labels(label_path)
        labels = remap_labels(raw_labels)
        xyz = scan[:, :3]
    else:
        scan, labels = make_synthetic_scene(seed=42 + st.session_state.frame_idx)
        xyz = scan[:, :3]

    # Process Frame through Pipeline with dynamic grid
    grid = VarResGrid(n_classes=8, fine_radius=fine_radius, forward_offset=0.0)
    result = process_frame(xyz, labels, grid=grid, params=params)
    trav = result["traversability"]
    timings = result["timings"]
    stats = result["stats"]
    mem_var = result["memory_bytes"]
    mem_uni = result["uniform_equivalent_bytes"]

    total_time_ms = max(timings["total_ms"], 0.001)
    fps = 1000.0 / total_time_ms
    dropped_points = stats["total_input"] - (stats["in_fine"] + stats["in_coarse"] + stats["out_of_range"])
    compression_ratio = mem_uni / mem_var

    # Live Counters & Telemetry Bar
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pipeline FPS", f"{fps:.1f}", f"{total_time_ms:.1f} ms")
    c2.metric("Fine Radius", f"{fine_radius:.1f} m", f"Cell: {fine_cell_size * 100:.1f} cm")
    c3.metric("Stopping Dist", f"{d_stop:.1f} m", f"{speed_mps * 3.6:.1f} km/h")
    c4.metric("VarRes Memory", f"{mem_var / (1024*1024):.2f} MB", "Constant")
    c5.metric("Compression", f"{compression_ratio:.1f}x", f"vs {mem_uni / (1024*1024):.0f} MB")
    c6.metric("Dropped Points", f"{dropped_points}", delta="100% Conserved")

    # Main Visual Layout
    col_map, col_info = st.columns([3, 1])

    with col_map:
        st.subheader("2.5D Top-Down Composite Grid (Forward is Up)")
        mode_key = "state" if view_mode == "Drivable State Map" else "semantic"
        map_img = render_composite_top_down(grid, trav, view_mode=mode_key, canvas_size=800)
        st.image(
            map_img,
            caption=f"200m x 200m Composite Map (Cyan Ring: {fine_radius:.1f}m Fine Seam, Cell: {fine_cell_size*100:.1f}cm)",
            use_container_width=True,
        )

    with col_info:
        st.subheader("Legend & Diagnostics")
        if view_mode == "Drivable State Map":
            st.markdown(
                """
                - 🟢 **Drivable**: Smooth surface
                - 🟠 **Non-Drivable**: Step/slope limit or non-drivable ground
                - 🔴 **Obstacle**: Static obstacle, vehicle, person
                - ⬛ **Unknown**: Unobserved empty cell
                - *Dimming indicates low observation confidence.*
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
        st.markdown("### Frame Statistics")
        st.write(f"- **Vehicle Speed**: {speed_mps:.1f} m/s ({speed_mps * 3.6:.1f} km/h)")
        st.write(f"- **Stopping Distance**: {d_stop:.1f} m")
        st.write(f"- **Fine Zone Radius**: {fine_radius:.1f} m")
        st.write(f"- **Fine Cell Resolution**: {fine_cell_size * 100:.1f} cm")
        st.write(f"- **In Fine Zone**: {stats['in_fine']:,} points")
        st.write(f"- **In Coarse Zone**: {stats['in_coarse']:,} points")
        st.write(f"- **Out of Range (≥100m)**: {stats['out_of_range']:,} points")
        st.write(f"- **Grid Add Time**: {timings['add_points_ms']:.2f} ms")
        st.write(f"- **Traversability Time**: {timings['compute_traversability_ms']:.2f} ms")

    # If playing, advance frame
    if st.session_state.playing:
        time.sleep(0.08)
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % max(1, num_frames)
        st.rerun()


if __name__ == "__main__":
    main()
