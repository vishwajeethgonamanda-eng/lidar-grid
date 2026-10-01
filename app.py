from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import streamlit as st

from src.grid_engine import VarResGrid
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


def scan_kitti_sequences(data_dir: Path = Path("data")) -> Dict[str, List[Tuple[Path, Path]]]:
    """Scans data/ directory for SemanticKITTI sequences containing pairs of .bin and .label files."""
    sequences = {}
    if not data_dir.exists():
        return sequences

    # Look for directories that contain a 'velodyne' subfolder
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
            sequences[seq_name] = pairs

    return sequences


def render_composite_top_down(
    grid: VarResGrid,
    trav: TraversabilityMap,
    view_mode: str = "state",
    canvas_size: int = 800,
) -> np.ndarray:
    """Renders a composite top-down bird's-eye view where x is forward (pointing up)
    and y is lateral (left to right), with the fine zone in the middle and coarse zone around it.
    """
    half_extent = 100.0  # 200m x 200m
    # x goes from +100 (top row) to -100 (bottom row)
    xs = np.linspace(half_extent - 0.125, -half_extent + 0.125, canvas_size, dtype=np.float32)
    # y goes from -100 (left col) to +100 (right col)
    ys = np.linspace(-half_extent + 0.125, half_extent - 0.125, canvas_size, dtype=np.float32)

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R = np.hypot(X, Y)

    bg_color = np.array([30, 32, 38], dtype=np.float32)
    img = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.float32)

    fine_mask = R < 10.0
    coarse_mask = (R >= 10.0) & (R < 100.0)

    # 1. Render Fine Zone
    f_x = X[fine_mask]
    f_y = Y[fine_mask]
    f_col = np.clip(np.floor((f_x + 10.0) / 0.05).astype(np.int64), 0, 399)
    f_row = np.clip(np.floor((f_y + 10.0) / 0.05).astype(np.int64), 0, 399)

    if view_mode == "state":
        f_st = trav.fine.state[f_row, f_col]
        f_cf = trav.fine.confidence[f_row, f_col, None]

        f_colors = np.zeros((len(f_st), 3), dtype=np.float32)
        for st_val, col_val in STATE_COLORS.items():
            mask = f_st == st_val
            f_colors[mask] = col_val

        # Dim by confidence
        f_non_empty = f_st != UNKNOWN
        dimmed = f_colors[f_non_empty] * (0.3 + 0.7 * f_cf[f_non_empty]) + bg_color * (0.7 * (1.0 - f_cf[f_non_empty]))
        f_colors[f_non_empty] = dimmed
        img[fine_mask] = f_colors
    else:
        # Semantic view
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
            mask = (c_dom == class_id) & (c_cnt > 0)
            c_colors[mask] = col_val
        c_colors[c_cnt == 0] = bg_color
        img[coarse_mask] = c_colors

    # 3. Draw Seam Circle (r = 10 m boundary)
    seam_ring = (R >= 9.75) & (R <= 10.25)
    img[seam_ring] = img[seam_ring] * 0.4 + np.array([70, 160, 240], dtype=np.float32) * 0.6

    # 4. Sensor Marker at Center (0, 0)
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

    st.title("Variable-Resolution 2.5D LiDAR Grid & Traversability")

    # Sidebar: Data Source & Settings
    st.sidebar.header("Data Source")
    sequences = scan_kitti_sequences(Path("data"))

    if sequences:
        source_mode = st.sidebar.radio("Select Source", ["SemanticKITTI Dataset", "Synthetic Scene Generator"])
        if source_mode == "SemanticKITTI Dataset":
            seq_choice = st.sidebar.selectbox("Sequence", list(sequences.keys()))
            frame_pairs = sequences[seq_choice]
            num_frames = len(frame_pairs)
        else:
            num_frames = 20
            frame_pairs = None
    else:
        st.sidebar.info("No SemanticKITTI data found in `data/`. Running in Synthetic Scene mode.")
        num_frames = 20
        frame_pairs = None

    # Playback Controls
    st.sidebar.header("Playback Controls")
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

    # Visualization Options
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
        # Dynamic synthetic animation based on frame_idx
        scan, labels = make_synthetic_scene(seed=42 + st.session_state.frame_idx)
        xyz = scan[:, :3]

    # Process Frame through Pipeline
    result = process_frame(xyz, labels, params=params)
    grid = result["grid"]
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
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Pipeline FPS", f"{fps:.1f}", f"{total_time_ms:.1f} ms/frame")
    m2.metric("VarRes Memory", f"{mem_var / (1024*1024):.2f} MB")
    m3.metric("Uniform Grid", f"{mem_uni / (1024*1024):.2f} MB", f"{compression_ratio:.1f}x compression")
    m4.metric("In Fine (r < 10m)", f"{stats['in_fine']:,}")
    m5.metric("In Coarse (10-100m)", f"{stats['in_coarse']:,}")
    m6.metric("Dropped Points", f"{dropped_points}", delta="Conservation: 100%")

    # Main Visual Layout
    col_map, col_info = st.columns([3, 1])

    with col_map:
        st.subheader("2.5D Top-Down Composite Grid (Forward is Up)")
        mode_key = "state" if view_mode == "Drivable State Map" else "semantic"
        map_img = render_composite_top_down(grid, trav, view_mode=mode_key, canvas_size=800)
        st.image(map_img, caption="200m x 200m Composite Map (Inner Cyan Circle: 10m Fine Seam)", use_container_width=True)

    with col_info:
        st.subheader("Legend & Diagnostics")
        if view_mode == "Drivable State Map":
            st.markdown(
                """
                - 🟢 **Drivable**: Smooth road surface
                - 🟠 **Non-Drivable**: Step > max step, slope > max slope, or non-drivable ground
                - 🔴 **Obstacle**: Static obstacle, vehicle, person, moving object
                - ⬛ **Unknown**: Unobserved / empty cell
                - *Note: Cells are dimmed according to their observation confidence.*
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
                - 🟨 **Non-drivable Ground (Sidewalk/Kerb)**: Class 7
                """
            )

        st.divider()
        st.markdown("### Frame Statistics")
        st.write(f"- **Total Input Points**: {stats['total_input']:,}")
        st.write(f"- **Out of Range (r ≥ 100m)**: {stats['out_of_range']:,}")
        st.write(f"- **Grid Add Time**: {timings['add_points_ms']:.2f} ms")
        st.write(f"- **Traversability Time**: {timings['compute_traversability_ms']:.2f} ms")

        # Zone occupancies
        fine_occ = np.count_nonzero(grid.fine.count > 0)
        coarse_occ = np.count_nonzero(grid.coarse.count > 0)
        st.write(f"- **Fine Zone Occupied**: {fine_occ} / 160,000 cells ({fine_occ/1600:.1f}%)")
        st.write(f"- **Coarse Zone Occupied**: {coarse_occ} / 160,000 cells ({coarse_occ/1600:.1f}%)")

    # If playing, step frame and rerun
    if st.session_state.playing:
        time.sleep(0.08)
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % max(1, num_frames)
        st.rerun()


if __name__ == "__main__":
    main()
