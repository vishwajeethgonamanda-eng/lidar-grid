from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.grid_engine import VarResGrid
from src.loader import load_scan, load_labels, remap_labels, make_synthetic_scene
from src.pipeline import process_frame
from src.speed import get_speed


def generate_synthetic_scene_with_targets(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic LiDAR scene based on make_synthetic_scene,
    augmented with road ground points and targets of classes 4 (vehicle),
    5 (person), and 6 (moving object) across distance bands 20-40 m and 40-60 m.
    """
    rng = np.random.default_rng(seed)

    # Base scene from loader (ground at z=0, kerb at r=10m, pole at 5m, box at 15m)
    base_scan, base_labels = make_synthetic_scene(seed=seed)

    # Extended road / terrain ground points extending from 15 m to 70 m
    n_ground_ext = 35000
    x_ground = rng.uniform(15.0, 70.0, size=n_ground_ext)
    y_ground = rng.uniform(-6.0, 6.0, size=n_ground_ext)
    z_ground = rng.normal(0.0, 0.02, size=n_ground_ext).astype(np.float32)
    # y within [-3, 3] is road (1), outside is terrain (2)
    labels_ground = np.where(np.abs(y_ground) <= 3.0, 1, 2).astype(np.int64)

    pts_list = [base_scan[:, :3], np.column_stack([x_ground, y_ground, z_ground])]
    lbls_list = [base_labels, labels_ground]

    # Target objects sized to fit within 3.2m x 3.2m focus patch with realistic point counts:
    # 20-40 m band
    # - Vehicle (class 4) at x~26 m (~120 points)
    # - Person (class 5) at x~32 m (~25 points)
    # - Moving object (class 6, e.g. bicyclist) at x~37 m (~35 points)
    # 40-60 m band
    # - Vehicle (class 4) at x~46 m (~80 points)
    # - Person (class 5) at x~52 m (~15 points)
    # - Moving object (class 6) at x~57 m (~25 points)
    targets = [
        # Band 20-40 m
        {"cls": 4, "cx": 26.0 + rng.uniform(-1.5, 1.5), "cy": 1.2 + rng.uniform(-0.5, 0.5), "dx": 2.8, "dy": 1.6, "dz": 1.4, "n": 120},
        {"cls": 5, "cx": 32.0 + rng.uniform(-1.5, 1.5), "cy": 1.5 + rng.uniform(-0.5, 0.5), "dx": 0.5, "dy": 0.5, "dz": 1.7, "n": 25},
        {"cls": 6, "cx": 37.0 + rng.uniform(-1.5, 1.5), "cy": -1.2 + rng.uniform(-0.5, 0.5), "dx": 1.8, "dy": 0.6, "dz": 1.5, "n": 35},
        # Band 40-60 m
        {"cls": 4, "cx": 46.0 + rng.uniform(-1.5, 1.5), "cy": -1.0 + rng.uniform(-0.5, 0.5), "dx": 2.8, "dy": 1.6, "dz": 1.5, "n": 80},
        {"cls": 5, "cx": 52.0 + rng.uniform(-1.5, 1.5), "cy": 1.8 + rng.uniform(-0.5, 0.5), "dx": 0.5, "dy": 0.5, "dz": 1.7, "n": 15},
        {"cls": 6, "cx": 57.0 + rng.uniform(-1.5, 1.5), "cy": 0.5 + rng.uniform(-0.5, 0.5), "dx": 1.8, "dy": 0.7, "dz": 1.5, "n": 25},
    ]

    for t in targets:
        ox = rng.uniform(t["cx"] - t["dx"] / 2.0, t["cx"] + t["dx"] / 2.0, size=t["n"])
        oy = rng.uniform(t["cy"] - t["dy"] / 2.0, t["cy"] + t["dy"] / 2.0, size=t["n"])
        oz = rng.uniform(0.04, t["dz"], size=t["n"]).astype(np.float32)
        pts_list.append(np.column_stack([ox, oy, oz]))
        lbls_list.append(np.full(t["n"], t["cls"], dtype=np.int64))

    xyz = np.vstack(pts_list).astype(np.float32)
    labels = np.concatenate(lbls_list).astype(np.int64)
    return xyz, labels


# Alias for app.py and other consumers
generate_patch_eval_scene = generate_synthetic_scene_with_targets


def compute_label_purity(
    xyz: np.ndarray,
    labels: np.ndarray,
    grid: VarResGrid,
    r_min: float,
    r_max: float,
    target_class: Optional[int] = None,
) -> Tuple[Optional[float], int]:
    """Computes label purity: fraction of points whose cell's dominant label
    equals the point's own label within the radial distance band [r_min, r_max).

    Args:
        xyz: (N, 3) LiDAR coordinates.
        labels: (N,) point labels.
        grid: Populated VarResGrid (with or without patches).
        r_min: Minimum distance in metres.
        r_max: Maximum distance in metres.
        target_class: If None, evaluates all classes in {4, 5, 6}; else specific class.

    Returns:
        (purity_ratio, point_count)
    """
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    if target_class is None:
        mask = (r >= r_min) & (r < r_max) & np.isin(labels, [4, 5, 6])
    else:
        mask = (r >= r_min) & (r < r_max) & (labels == target_class)

    n_pts = int(np.count_nonzero(mask))
    if n_pts == 0:
        return None, 0

    pts = xyz[mask]
    lbls = labels[mask]
    x = pts[:, 0]
    y = pts[:, 1]

    # 1. Coarse cell lookup for all points
    c_col = np.clip(np.floor((x + 100.0) / 0.5).astype(np.int64), 0, 399)
    c_row = np.clip(np.floor((y + 100.0) / 0.5).astype(np.int64), 0, 399)
    cell_dom = grid.coarse.dominant_label[c_row, c_col].copy()

    # 2. For points inside active focus patches, look up patch cell dominant label
    for p in grid.patches:
        in_p = (
            (x >= p.center_x - p.half_extent)
            & (x < p.center_x + p.half_extent)
            & (y >= p.center_y - p.half_extent)
            & (y < p.center_y + p.half_extent)
        )
        if np.any(in_p):
            px = x[in_p]
            py = y[in_p]
            p_col = np.clip(
                np.floor(((px - p.center_x) + p.half_extent) / p.cell_size).astype(np.int64),
                0,
                p.grid_size - 1,
            )
            p_row = np.clip(
                np.floor(((py - p.center_y) + p.half_extent) / p.cell_size).astype(np.int64),
                0,
                p.grid_size - 1,
            )
            cell_dom[in_p] = p.dominant_label[p_row, p_col]

    matches = int(np.count_nonzero(cell_dom == lbls))
    purity = float(matches) / float(n_pts)
    return purity, n_pts


def run_evaluation(n_frames: int = 15):
    print("=" * 96)
    print("EVALUATION: Label Purity With vs Without Risk-Guided Focus Patches")
    print("Definition: Fraction of points whose cell's dominant label equals the point's own label")
    print("=" * 96)

    # Check for real data in data/
    data_dir = Path("data")
    velo_dirs = list(data_dir.rglob("velodyne"))
    real_frames: List[Tuple[Path, Path]] = []
    poses_path: Optional[Path] = None

    if velo_dirs:
        seq_root = velo_dirs[0].parent
        labels_dir = seq_root / "labels"
        cand_poses = seq_root / "poses.txt"
        if not cand_poses.exists():
            cand_poses = seq_root.parent / "poses.txt"
        if cand_poses.exists():
            poses_path = cand_poses

        bin_files = sorted(list(velo_dirs[0].glob("*.bin")))[:n_frames]
        for b in bin_files:
            lbl = labels_dir / f"{b.stem}.label"
            if lbl.exists():
                real_frames.append((b, lbl))

    is_real = len(real_frames) > 0
    total_frames = len(real_frames) if is_real else n_frames
    print(f"Data Source: {'Real SemanticKITTI Frames' if is_real else 'Synthetic Dynamic Scene'}")
    print(f"Frames to Evaluate: {total_frames}\n")

    # Metrics collectors
    b1_no_patch = {4: [], 5: [], 6: [], "all": []}
    b1_with_patch = {4: [], 5: [], 6: [], "all": []}
    b2_no_patch = {4: [], 5: [], 6: [], "all": []}
    b2_with_patch = {4: [], 5: [], 6: [], "all": []}

    extra_memories_bytes = []
    extra_latencies_ms = []
    num_patches_list = []

    header = (
        f"{'Frame':^5} | {'Patches':^7} | "
        f"{'Band 20-40m (No / With)':^24} | "
        f"{'Band 40-60m (No / With)':^24} | "
        f"{'+Mem (KB)':^9} | {'+Time (ms)':^10}"
    )
    print(header)
    print("-" * len(header))

    for frame_idx in range(total_frames):
        if is_real:
            bin_path, label_path = real_frames[frame_idx]
            scan = load_scan(bin_path)
            raw_labels = load_labels(label_path)
            labels = remap_labels(raw_labels)
            xyz = scan[:, :3]
            speed = get_speed(frame_idx, poses_path)
        else:
            xyz, labels = generate_synthetic_scene_with_targets(seed=300 + frame_idx)
            # Speed varying between 15 m/s (~54 km/h) and 24 m/s (~86 km/h)
            speed = float(16.0 + 8.0 * np.sin(np.pi * frame_idx / (total_frames - 1)))

        # Baseline: without patches
        res_no = process_frame(xyz, labels, risk_patches=False)
        grid_no = res_no["grid"]

        # With risk focus patches
        res_with = process_frame(xyz, labels, risk_patches=True, speed_mps=speed, k_patches=4)
        grid_with = res_with["grid"]

        extra_mem_bytes = res_with["extra_memory_bytes"]
        extra_time_ms = res_with["timings"].get("extra_time_ms", 0.0)
        n_p = len(grid_with.patches)

        extra_memories_bytes.append(extra_mem_bytes)
        extra_latencies_ms.append(extra_time_ms)
        num_patches_list.append(n_p)

        # Distance Band 20-40 m
        p1_no_all, _ = compute_label_purity(xyz, labels, grid_no, 20.0, 40.0)
        p1_with_all, _ = compute_label_purity(xyz, labels, grid_with, 20.0, 40.0)
        if p1_no_all is not None:
            b1_no_patch["all"].append(p1_no_all)
            b1_with_patch["all"].append(p1_with_all)

        for c in (4, 5, 6):
            p_no, _ = compute_label_purity(xyz, labels, grid_no, 20.0, 40.0, target_class=c)
            p_with, _ = compute_label_purity(xyz, labels, grid_with, 20.0, 40.0, target_class=c)
            if p_no is not None:
                b1_no_patch[c].append(p_no)
                b1_with_patch[c].append(p_with)

        # Distance Band 40-60 m
        p2_no_all, _ = compute_label_purity(xyz, labels, grid_no, 40.0, 60.0)
        p2_with_all, _ = compute_label_purity(xyz, labels, grid_with, 40.0, 60.0)
        if p2_no_all is not None:
            b2_no_patch["all"].append(p2_no_all)
            b2_with_patch["all"].append(p2_with_all)

        for c in (4, 5, 6):
            p_no, _ = compute_label_purity(xyz, labels, grid_no, 40.0, 60.0, target_class=c)
            p_with, _ = compute_label_purity(xyz, labels, grid_with, 40.0, 60.0, target_class=c)
            if p_no is not None:
                b2_no_patch[c].append(p_no)
                b2_with_patch[c].append(p_with)

        b1_str = f"{p1_no_all * 100:.1f}% -> {p1_with_all * 100:.1f}%" if p1_no_all is not None else "N/A"
        b2_str = f"{p2_no_all * 100:.1f}% -> {p2_with_all * 100:.1f}%" if p2_no_all is not None else "N/A"

        print(
            f"{frame_idx:^5d} | {n_p:^7d} | "
            f"{b1_str:^24} | {b2_str:^24} | "
            f"{extra_mem_bytes / 1024.0:^9.1f} | {extra_time_ms:^10.2f}"
        )

    print("=" * 96)
    print("SUMMARY RESULTS (Averages Across All Evaluated Frames):")
    print("-" * 96)

    print("DISTANCE BAND 20-40 m:")
    for c, name in [(4, "Class 4 (Vehicle)      "), (5, "Class 5 (Person)       "), (6, "Class 6 (Moving Object)"), ("all", "Overall (Classes 4,5,6)")]:
        if b1_no_patch[c]:
            no_pct = float(np.mean(b1_no_patch[c])) * 100.0
            with_pct = float(np.mean(b1_with_patch[c])) * 100.0
            diff = with_pct - no_pct
            print(f"  * {name}: Without Patches = {no_pct:5.1f}%  |  With Patches = {with_pct:5.1f}%  |  Diff = {diff:+5.1f}%")

    print("\nDISTANCE BAND 40-60 m:")
    for c, name in [(4, "Class 4 (Vehicle)      "), (5, "Class 5 (Person)       "), (6, "Class 6 (Moving Object)"), ("all", "Overall (Classes 4,5,6)")]:
        if b2_no_patch[c]:
            no_pct = float(np.mean(b2_no_patch[c])) * 100.0
            with_pct = float(np.mean(b2_with_patch[c])) * 100.0
            diff = with_pct - no_pct
            print(f"  * {name}: Without Patches = {no_pct:5.1f}%  |  With Patches = {with_pct:5.1f}%  |  Diff = {diff:+5.1f}%")

    avg_mem_bytes = int(np.mean(extra_memories_bytes))
    avg_mem_kb = avg_mem_bytes / 1024.0
    avg_mem_mb = avg_mem_kb / 1024.0
    avg_time_ms = float(np.mean(extra_latencies_ms))
    avg_patches = float(np.mean(num_patches_list))

    print("\nRESOURCE & LATENCY COSTS:")
    print(f"  * Average Active Patches:     {avg_patches:.1f}")
    print(f"  * Extra Memory Cost:          {avg_mem_bytes:,} bytes ({avg_mem_kb:.1f} KB / {avg_mem_mb:.2f} MB)")
    print(f"  * Extra Frame Time Cost:      {avg_time_ms:.2f} ms")
    print("=" * 96)


if __name__ == "__main__":
    run_evaluation()
