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
    and y is lateral (left to right), dynamically centering and scaling the fine zone fovea
    and rendering high-resolution risk-guided focus patches.
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

    # 1. Render Coarse Zone
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

    # 2. Render Focus Patches (over coarse, outside fine)
    if hasattr(grid, "patches") and grid.patches:
        for idx, patch in enumerate(grid.patches):
            p_mask = (
                (np.abs(X - patch.center_x) <= patch.half_extent)
                & (np.abs(Y - patch.center_y) <= patch.half_extent)
                & (R_sensor < 100.0)
                & (~fine_mask)
            )
            if not np.any(p_mask):
                continue

            p_x = X[p_mask]
            p_y = Y[p_mask]
            p_col = np.clip(
                np.floor(((p_x - patch.center_x) + patch.half_extent) / patch.cell_size).astype(np.int64),
                0,
                patch.grid_size - 1,
            )
            p_row = np.clip(
                np.floor(((p_y - patch.center_y) + patch.half_extent) / patch.cell_size).astype(np.int64),
                0,
                patch.grid_size - 1,
            )

            if view_mode == "state" and hasattr(trav, "patches") and idx < len(trav.patches):
                p_trav = trav.patches[idx]
                p_st = p_trav.state[p_row, p_col]
                p_cf = p_trav.confidence[p_row, p_col, None]

                p_colors = np.zeros((len(p_st), 3), dtype=np.float32)
                for st_val, col_val in STATE_COLORS.items():
                    mask = p_st == st_val
                    p_colors[mask] = col_val

                p_non_empty = p_st != UNKNOWN
                dimmed_p = p_colors[p_non_empty] * (0.3 + 0.7 * p_cf[p_non_empty]) + bg_color * (0.7 * (1.0 - p_cf[p_non_empty]))
                p_colors[p_non_empty] = dimmed_p
                img[p_mask] = p_colors
            else:
                p_dom = patch.dominant_label[p_row, p_col]
                p_cnt = patch.count[p_row, p_col]
                p_colors = np.zeros((len(p_dom), 3), dtype=np.float32)
                for class_id, col_val in SEMANTIC_COLORS.items():
                    mask = (p_dom == class_id) & (p_cnt > 0)
                    p_colors[mask] = col_val
                p_colors[p_cnt == 0] = bg_color
                img[p_mask] = p_colors

    # 3. Render Fine Zone (highest priority)
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

    # 4. Draw Focus Patch Outlines
    if hasattr(grid, "patches") and grid.patches:
        gold_color = np.array([255, 215, 0], dtype=np.float32)
        for patch in grid.patches:
            dx = np.abs(X - patch.center_x)
            dy = np.abs(Y - patch.center_y)
            on_edge = (
                ((np.abs(dx - patch.half_extent) <= 0.35) & (dy <= patch.half_extent + 0.35))
                | ((np.abs(dy - patch.half_extent) <= 0.35) & (dx <= patch.half_extent + 0.35))
            )
            outline_mask = on_edge & (R_sensor < 100.0) & (~fine_mask)
            img[outline_mask] = img[outline_mask] * 0.25 + gold_color * 0.75

    # 5. Draw Adaptive Seam Circle at fine_radius boundary
    seam_ring = np.abs(R_fine - grid.fine_radius) <= 0.35
    img[seam_ring] = img[seam_ring] * 0.35 + np.array([70, 180, 255], dtype=np.float32) * 0.65

    # 6. Sensor Marker at (0, 0)
    center = canvas_size // 2
    img[center - 3 : center + 4, center - 1 : center + 2] = [255, 255, 255]
    img[center - 1 : center + 2, center - 3 : center + 4] = [255, 255, 255]

    uint_img = np.clip(img, 0, 255).astype(np.uint8)

    # 7. PIL Annotations: Range rings, fine-zone label, focus-patch labels, color legend
    pil_img = Image.fromarray(uint_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default(size=12)
    font_bold = ImageFont.load_default(size=13)

    # 7a. Thin, labelled range rings at 10, 30, 50, and 100 m
    for r in [10, 30, 50, 100]:
        r_px = int(r * 4.0)
        draw.ellipse([(400 - r_px, 400 - r_px), (400 + r_px, 400 + r_px)], outline=(85, 92, 105), width=1)
        draw.rectangle([(404, 400 - r_px - 14), (436, 400 - r_px - 2)], fill=(20, 22, 28, 200))
        draw.text((406, 400 - r_px - 14), f"{r}m", fill=(175, 182, 195), font=font)

    # 7b. Label on the blue fine-zone circle showing fine radius and fine cell size
    fine_cy = int((100.0 - grid.forward_offset) * 4.0)
    fine_r_px = int(grid.fine_radius * 4.0)
    fine_label = f"fine zone, r = {grid.fine_radius:.0f} m, {grid.fine_cell_size * 100:.1f} cm cells"
    fine_box_w = len(fine_label) * 7 + 12
    fine_lbl_x = max(10, min(canvas_size - fine_box_w - 10, 400 - fine_box_w // 2))
    fine_lbl_y = max(10, fine_cy - fine_r_px - 18)
    draw.rectangle([(fine_lbl_x, fine_lbl_y), (fine_lbl_x + fine_box_w, fine_lbl_y + 16)], fill=(12, 24, 42, 230), outline=(70, 180, 255))
    draw.text((fine_lbl_x + 6, fine_lbl_y + 1), fine_label, fill=(70, 210, 255), font=font)

    # 7c. "focus patch" label next to yellow patch squares
    if hasattr(grid, "patches") and grid.patches:
        for patch in grid.patches:
            px_max = int((patch.center_y + patch.half_extent + 100.0) * 4.0)
            py_min = int((100.0 - (patch.center_x + patch.half_extent)) * 4.0)
            tag_x = min(canvas_size - 90, px_max + 4)
            tag_y = max(10, py_min - 2)
            draw.rectangle([(tag_x, tag_y), (tag_x + 84, tag_y + 15)], fill=(25, 25, 12, 230), outline=(255, 215, 0))
            draw.text((tag_x + 4, tag_y + 1), "focus patch", fill=(255, 215, 0), font=font)

    # 7d. Small colour legend
    draw.rectangle([(15, 15), (145, 110)], fill=(20, 22, 28, 220), outline=(75, 82, 95))
    draw.text((22, 19), "Legend", fill=(230, 235, 240), font=font_bold)
    legend_items = [
        ("Drivable", (46, 204, 113)),
        ("Non-drivable", (243, 156, 18)),
        ("Obstacle", (231, 76, 60)),
        ("Unknown", (55, 60, 70)),
    ]
    for i, (name, col) in enumerate(legend_items):
        y_pos = 38 + i * 17
        draw.rectangle([(22, y_pos), (32, y_pos + 10)], fill=col)
        draw.text((38, y_pos - 1), name, fill=(210, 215, 225), font=font)

    return np.asarray(pil_img)


def render_raw_points(
    xyz: np.ndarray,
    labels: np.ndarray,
    canvas_size: int = 800,
) -> np.ndarray:
    """Renders a fast top-down scatter of raw LiDAR points (x forward / up, y lateral)
    with the exact same axes and range ([-100, 100] m) as the grid map.
    Subsampled to at most 20,000 points.
    """
    bg_color = np.array([30, 32, 38], dtype=np.uint8)
    img = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)

    n_pts = len(xyz)
    if n_pts == 0:
        return img

    if n_pts > 20000:
        step = max(1, n_pts // 20000)
        idx = np.arange(0, n_pts, step)[:20000]
        sub_xyz = xyz[idx]
        sub_lbls = labels[idx]
    else:
        sub_xyz = xyz
        sub_lbls = labels

    x = sub_xyz[:, 0]
    y = sub_xyz[:, 1]
    valid = (np.abs(x) <= 100.0) & (np.abs(y) <= 100.0)
    x = x[valid]
    y = y[valid]
    sub_lbls = sub_lbls[valid]

    px_row = np.clip(np.floor((100.0 - x) * 4.0).astype(np.int64), 0, canvas_size - 1)
    py_col = np.clip(np.floor((y + 100.0) * 4.0).astype(np.int64), 0, canvas_size - 1)

    palette = np.array([
        [40, 40, 45],     # 0 Unlabeled
        [52, 152, 219],   # 1 Road
        [39, 174, 96],    # 2 Terrain
        [149, 165, 166],  # 3 Static Obstacle
        [155, 89, 182],   # 4 Vehicle
        [230, 126, 34],   # 5 Person
        [231, 76, 60],    # 6 Moving Object
        [241, 196, 15],   # 7 Non-drivable Ground
    ], dtype=np.uint8)

    point_colors = palette[np.clip(sub_lbls, 0, 7)]

    # Draw points with 2x2 splat for visibility
    img[px_row, py_col] = point_colors
    px_p1 = np.clip(px_row + 1, 0, canvas_size - 1)
    py_p1 = np.clip(py_col + 1, 0, canvas_size - 1)
    img[px_p1, py_col] = point_colors
    img[px_row, py_p1] = point_colors
    img[px_p1, py_p1] = point_colors

    # Overlay range rings and annotations via PIL
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default(size=12)
    font_bold = ImageFont.load_default(size=13)

    # Range rings
    for r in [10, 30, 50, 100]:
        r_px = int(r * 4.0)
        draw.ellipse([(400 - r_px, 400 - r_px), (400 + r_px, 400 + r_px)], outline=(85, 92, 105), width=1)
        draw.rectangle([(404, 400 - r_px - 14), (436, 400 - r_px - 2)], fill=(20, 22, 28, 200))
        draw.text((406, 400 - r_px - 14), f"{r}m", fill=(175, 182, 195), font=font)

    # Sensor Crosshair at (400, 400)
    center = canvas_size // 2
    draw.line([(center - 4, center), (center + 4, center)], fill=(255, 255, 255), width=1)
    draw.line([(center, center - 4), (center, center + 4)], fill=(255, 255, 255), width=1)

    # Small legend for raw points
    draw.rectangle([(15, 15), (150, 185)], fill=(20, 22, 28, 220), outline=(75, 82, 95))
    draw.text((22, 19), "Raw Point Classes", fill=(230, 235, 240), font=font_bold)
    labels_names = [
        ("Road", (52, 152, 219)),
        ("Terrain", (39, 174, 96)),
        ("Obstacle", (149, 165, 166)),
        ("Vehicle", (155, 89, 182)),
        ("Person", (230, 126, 34)),
        ("Moving", (231, 76, 60)),
        ("Non-drivable", (241, 196, 15)),
    ]
    for i, (name, col) in enumerate(labels_names):
        y_pos = 38 + i * 20
        draw.rectangle([(22, y_pos), (32, y_pos + 11)], fill=col)
        draw.text((38, y_pos - 1), name, fill=(205, 210, 220), font=font)

    return np.asarray(pil_img)


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
        ["Grid map", "Raw points", "Side by side"],
        index=0,
    )
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
        from eval_patches import generate_patch_eval_scene
        xyz, labels = generate_patch_eval_scene(seed=42 + st.session_state.frame_idx)

    # Process Frame through Pipeline with dynamic grid and optional risk patches
    t_e2e_start = time.perf_counter()
    base_grid = VarResGrid(n_classes=8, fine_radius=fine_radius, forward_offset=0.0)
    result = process_frame(
        xyz,
        labels,
        grid=base_grid,
        params=params,
        risk_patches=patches_enabled,
        speed_mps=speed_mps,
        k_patches=k_patches,
    )
    grid = result["grid"]
    trav = result["traversability"]
    timings = result["timings"]
    stats = result["stats"]
    mem_var = result["memory_bytes"]
    mem_uni = result["uniform_equivalent_bytes"]
    extra_mem_kb = result.get("extra_memory_bytes", 0) / 1024.0
    n_patches = len(grid.patches) if hasattr(grid, "patches") else 0

    mode_key = "state" if view_mode == "Drivable State Map" else "semantic"
    map_img = render_composite_top_down(grid, trav, view_mode=mode_key, canvas_size=800)
    raw_img = render_raw_points(xyz, labels, canvas_size=800) if view_layout in ("Raw points", "Side by side") else None
    t_e2e_end = time.perf_counter()

    pipe_time_ms = max(timings["total_ms"], 0.001)
    pipe_fps = 1000.0 / pipe_time_ms
    e2e_time_ms = max((t_e2e_end - t_e2e_start) * 1000.0, 0.001)
    e2e_fps = 1000.0 / e2e_time_ms

    dropped_points = stats["total_input"] - (
        stats["in_fine"] + stats.get("in_patch", 0) + stats["in_coarse"] + stats["out_of_range"]
    )
    compression_ratio = mem_uni / mem_var

    # Live Counters & Telemetry Bar
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Pipeline FPS", f"{pipe_fps:.1f}", f"{pipe_time_ms:.1f} ms")
    c2.metric("End-to-End FPS", f"{e2e_fps:.1f}", f"{e2e_time_ms:.1f} ms (w/ draw)")
    c3.metric("Fine Radius", f"{fine_radius:.1f} m", f"Cell: {fine_cell_size * 100:.1f} cm")
    c4.metric("Stopping Dist", f"{d_stop:.1f} m", f"{speed_mps * 3.6:.1f} km/h")
    c5.metric(
        "VarRes Memory",
        f"{mem_var / (1024*1024):.2f} MB",
        f"+{extra_mem_kb:.0f} KB ({n_patches} Patches)" if (patches_enabled and n_patches > 0) else "Constant",
    )
    c6.metric("Compression", f"{compression_ratio:.1f}x", f"vs {mem_uni / (1024*1024):.0f} MB")
    c7.metric("Dropped Points", f"{dropped_points}", delta="100% Conserved")

    # Main Visual Layout
    col_map, col_info = st.columns([3, 1])

    with col_map:

        if view_layout == "Grid map":
            st.subheader("2.5D Top-Down Composite Grid (Forward is Up)")
            caption_text = f"200m x 200m Composite Map (Cyan Ring: {fine_radius:.1f}m Fine Seam, Cell: {fine_cell_size*100:.1f}cm"
            if patches_enabled and n_patches > 0:
                caption_text += f" | Gold Rects: {n_patches} Focus Patches @ 5cm)"
            else:
                caption_text += ")"
            st.image(map_img, caption=caption_text, use_container_width=True)

        elif view_layout == "Raw points":
            st.subheader("Raw LiDAR Point Cloud (Forward is Up)")
            st.image(raw_img, caption=f"200m x 200m Raw Points ({min(len(xyz), 20000):,} Subsampled Points, Semantic Classes)", use_container_width=True)

        else:  # "Side by side"
            st.subheader("Side-by-Side: Raw Points vs 2.5D Grid Map")
            c_left, c_right = st.columns(2)
            with c_left:
                st.image(raw_img, caption=f"Raw Points (≤20,000 pts)", use_container_width=True)
            with c_right:
                caption_sub = f"2.5D Grid (Fovea r={fine_radius:.0f}m)"
                if patches_enabled and n_patches > 0:
                    caption_sub += f" + {n_patches} Patches"
                st.image(map_img, caption=caption_sub, use_container_width=True)

        # One-line frame comparison computed from current frame
        num_raw = len(xyz)
        non_empty_fine = int(np.count_nonzero(grid.fine.count > 0))
        non_empty_coarse = int(np.count_nonzero(grid.coarse.count > 0))
        non_empty_patches = int(sum(np.count_nonzero(p.count > 0) for p in grid.patches)) if hasattr(grid, "patches") else 0
        total_non_empty = non_empty_fine + non_empty_coarse + non_empty_patches
        mem_var_mb = grid.memory_bytes() / (1024.0 * 1024.0)
        mem_uni_mb = grid.uniform_equivalent_bytes() / (1024.0 * 1024.0)
        ratio = mem_uni_mb / max(0.001, mem_var_mb)

        st.info(
            f"**Frame Comparison:** {num_raw:,} raw points | "
            f"{total_non_empty:,} non-empty grid cells | "
            f"Variable Grid: {mem_var_mb:.2f} MB vs Uniform 5 cm Grid: {mem_uni_mb:.0f} MB ({ratio:.1f}x compression)"
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
                - 🟨 **Gold Box**: 64x64 @ 5cm Risk Focus Patch
                - 🔵 **Cyan Ring**: Fine Zone Radius Seam
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
                - 🟨 **Gold Box**: 64x64 @ 5cm Risk Focus Patch
                - 🔵 **Cyan Ring**: Fine Zone Radius Seam
                """
            )

        st.divider()
        st.markdown("### Frame Statistics")
        st.write(f"- **Vehicle Speed**: {speed_mps:.1f} m/s ({speed_mps * 3.6:.1f} km/h)")
        st.write(f"- **Stopping Distance**: {d_stop:.1f} m")
        st.write(f"- **Fine Zone Radius**: {fine_radius:.1f} m (Cell: {fine_cell_size * 100:.1f} cm)")
        st.write(f"- **Active Focus Patches**: {n_patches}")
        if patches_enabled and result.get("candidates"):
            for idx, cand in enumerate(result["candidates"][:n_patches]):
                st.write(f"  * Patch #{idx+1}: ({cand.x:.1f}m, {cand.y:.1f}m) | Risk: {cand.risk:.3f} | Cls: {cand.dominant_class}")
        st.write(f"- **In Fine Zone**: {stats['in_fine']:,} points")
        if "in_patch" in stats:
            st.write(f"- **In Focus Patches**: {stats['in_patch']:,} points")
        st.write(f"- **In Coarse Zone**: {stats['in_coarse']:,} points")
        st.write(f"- **Out of Range (≥100m)**: {stats['out_of_range']:,} points")
        st.write(f"- **Grid Add Time**: {timings['add_points_ms']:.2f} ms")
        st.write(f"- **Traversability Time**: {timings['compute_traversability_ms']:.2f} ms")
        if patches_enabled:
            st.write(f"- **Extra Risk & Patch Latency**: {timings.get('extra_time_ms', 0.0):.2f} ms")

    # If playing, advance frame
    if st.session_state.playing:
        time.sleep(0.08)
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % max(1, num_frames)
        st.rerun()


if __name__ == "__main__":
    main()
