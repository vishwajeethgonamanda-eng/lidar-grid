from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.grid_engine import VarResGrid
from src.loader import load_scan, load_labels, remap_labels, make_synthetic_scene
from src.pipeline import process_frame
from src.speed import get_speed


def generate_patch_eval_scene(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic scene specifically populated with dynamic / obstacle objects
    (classes 4, 5, 6) placed within distance bands 20-40m and 40-60m amidst ground/road points.
    """
    rng = np.random.default_rng(seed)

    # 1. Base scene (ground, kerb, near obstacles)
    base_scan, base_labels = make_synthetic_scene(seed=seed)

    # Ground road / terrain points extending to 70m
    n_road = 15000
    x_road = rng.uniform(15.0, 70.0, size=n_road)
    y_road = rng.uniform(-5.0, 5.0, size=n_road)
    z_road = rng.normal(0.0, 0.02, size=n_road).astype(np.float32)
    i_road = rng.uniform(0.1, 0.3, size=n_road).astype(np.float32)
    labels_road = np.full(n_road, 1, dtype=np.int64)

    pts_list = [base_scan[:, :3]]
    lbls_list = [base_labels]
    pts_list.append(np.column_stack([x_road, y_road, z_road]))
    lbls_list.append(labels_road)

    # Objects to place in 20-40m and 40-60m bands
    # Class 4: Vehicle, Class 5: Person, Class 6: Moving object
    objects = [
        # 20-40m band
        {"cls": 4, "cx": 25.0 + rng.uniform(-2, 2), "cy": 1.0 + rng.uniform(-0.5, 0.5), "dx": 3.8, "dy": 1.8, "dz": 1.4, "n": 600},
        {"cls": 5, "cx": 32.0 + rng.uniform(-2, 2), "cy": 1.5 + rng.uniform(-0.5, 0.5), "dx": 0.5, "dy": 0.5, "dz": 1.7, "n": 200},
        {"cls": 6, "cx": 38.0 + rng.uniform(-2, 2), "cy": -1.2 + rng.uniform(-0.5, 0.5), "dx": 1.8, "dy": 0.7, "dz": 1.5, "n": 300},
        # 40-60m band
        {"cls": 4, "cx": 46.0 + rng.uniform(-2, 2), "cy": -1.0 + rng.uniform(-0.5, 0.5), "dx": 4.0, "dy": 1.9, "dz": 1.5, "n": 400},
        {"cls": 5, "cx": 52.0 + rng.uniform(-2, 2), "cy": 1.8 + rng.uniform(-0.5, 0.5), "dx": 0.5, "dy": 0.5, "dz": 1.7, "n": 150},
        {"cls": 6, "cx": 57.0 + rng.uniform(-2, 2), "cy": 0.5 + rng.uniform(-0.5, 0.5), "dx": 1.9, "dy": 0.8, "dz": 1.5, "n": 250},
    ]

    for obj in objects:
        ox = rng.uniform(obj["cx"] - obj["dx"] / 2.0, obj["cx"] + obj["dx"] / 2.0, size=obj["n"])
        oy = rng.uniform(obj["cy"] - obj["dy"] / 2.0, obj["cy"] + obj["dy"] / 2.0, size=obj["n"])
        oz = rng.uniform(0.05, obj["dz"], size=obj["n"]).astype(np.float32)
        pts_list.append(np.column_stack([ox, oy, oz]))
        lbls_list.append(np.full(obj["n"], obj["cls"], dtype=np.int64))

    xyz = np.vstack(pts_list).astype(np.float32)
    labels = np.concatenate(lbls_list).astype(np.int64)
    return xyz, labels


def compute_point_purity(
    xyz: np.ndarray,
    labels: np.ndarray,
    grid: VarResGrid,
    target_classes: Tuple[int, ...] = (4, 5, 6),
    r_min: float = 20.0,
    r_max: float = 40.0,
) -> Tuple[float, float, int]:
    """Computes dominant label match rate and mean cell purity for points
    with labels in target_classes within [r_min, r_max).

    Returns:
        (dominant_match_pct, mean_cell_purity_pct, n_points)
    """
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    mask = (r >= r_min) & (r < r_max) & np.isin(labels, list(target_classes))
    target_pts = xyz[mask]
    target_lbls = labels[mask]

    n_pts = len(target_pts)
    if n_pts == 0:
        return 0.0, 0.0, 0

    x = target_pts[:, 0]
    y = target_pts[:, 1]

    # Initialize with coarse cell lookup for all points
    c_col = np.clip(np.floor((x + 100.0) / 0.5).astype(np.int64), 0, 399)
    c_row = np.clip(np.floor((y + 100.0) / 0.5).astype(np.int64), 0, 399)
    dom = grid.coarse.dominant_label[c_row, c_col].copy()
    cnt = grid.coarse.count[c_row, c_col].copy()
    class_cnt = grid.coarse.label_hist[c_row, c_col, target_lbls].copy()

    # Overwrite for points inside any active focus patch
    for p in grid.patches:
        p_mask = (np.abs(x - p.center_x) <= p.half_extent) & (np.abs(y - p.center_y) <= p.half_extent)
        if np.any(p_mask):
            px = x[p_mask]
            py = y[p_mask]
            plbl = target_lbls[p_mask]
            p_col = np.clip(np.floor(((px - p.center_x) + p.half_extent) / p.cell_size).astype(np.int64), 0, p.grid_size - 1)
            p_row = np.clip(np.floor(((py - p.center_y) + p.half_extent) / p.cell_size).astype(np.int64), 0, p.grid_size - 1)
            dom[p_mask] = p.dominant_label[p_row, p_col]
            cnt[p_mask] = p.count[p_row, p_col]
            class_cnt[p_mask] = p.label_hist[p_row, p_col, plbl]

    dom_match_pct = float(np.mean(dom == target_lbls)) * 100.0
    mean_purity_pct = float(np.mean(class_cnt / np.maximum(1, cnt))) * 100.0
    return dom_match_pct, mean_purity_pct, n_pts


def evaluate(n_frames: int = 15):
    print("=" * 88)
    print("EVALUATION: Risk-Guided Focus Patches vs Baseline Grid (Classes 4, 5, 6)")
    print("=" * 88)

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
    print(f"Data Mode: {'Real SemanticKITTI Sequence' if is_real else 'Synthetic Dynamic Scenes'}")
    print(f"Frames to evaluate: {total_frames}\n")

    # Metrics accumulators
    extra_mems_kb = []
    extra_times_ms = []

    b1_no_patch_dom, b1_with_patch_dom = [], []
    b1_no_patch_pur, b1_with_patch_pur = [], []

    b2_no_patch_dom, b2_with_patch_dom = [], []
    b2_no_patch_pur, b2_with_patch_pur = [], []

    header = (
        f"{'Frame':^5} | {'Patches':^7} | "
        f"{'Purity 20-40m (No / With)':^25} | "
        f"{'Purity 40-60m (No / With)':^25} | "
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
            xyz, labels = generate_patch_eval_scene(seed=200 + frame_idx)
            speed = 12.0

        # Baseline: No patches
        res_baseline = process_frame(xyz, labels, risk_patches=False)
        grid_base = res_baseline["grid"]

        # With risk focus patches
        res_patched = process_frame(xyz, labels, risk_patches=True, speed_mps=speed, k_patches=4)
        grid_patch = res_patched["grid"]

        # Band 20-40m
        b1_base_dom, b1_base_pur, _ = compute_point_purity(xyz, labels, grid_base, r_min=20.0, r_max=40.0)
        b1_pat_dom, b1_pat_pur, _ = compute_point_purity(xyz, labels, grid_patch, r_min=20.0, r_max=40.0)

        # Band 40-60m
        b2_base_dom, b2_base_pur, _ = compute_point_purity(xyz, labels, grid_base, r_min=40.0, r_max=60.0)
        b2_pat_dom, b2_pat_pur, _ = compute_point_purity(xyz, labels, grid_patch, r_min=40.0, r_max=60.0)

        n_patches = len(grid_patch.patches)
        extra_mem_kb = res_patched["extra_memory_bytes"] / 1024.0
        extra_time_ms = res_patched["timings"].get("extra_time_ms", 0.0)

        b1_no_patch_dom.append(b1_base_dom)
        b1_with_patch_dom.append(b1_pat_dom)
        b1_no_patch_pur.append(b1_base_pur)
        b1_with_patch_pur.append(b1_pat_pur)

        b2_no_patch_dom.append(b2_base_dom)
        b2_with_patch_dom.append(b2_pat_dom)
        b2_no_patch_pur.append(b2_base_pur)
        b2_with_patch_pur.append(b2_pat_pur)

        extra_mems_kb.append(extra_mem_kb)
        extra_times_ms.append(extra_time_ms)

        b1_str = f"{b1_base_pur:4.1f}% -> {b1_pat_pur:4.1f}%"
        b2_str = f"{b2_base_pur:4.1f}% -> {b2_pat_pur:4.1f}%"

        print(
            f"{frame_idx:^5d} | {n_patches:^7d} | "
            f"{b1_str:^25} | {b2_str:^25} | "
            f"{extra_mem_kb:^9.1f} | {extra_time_ms:^10.2f}"
        )

    print("=" * 88)
    print("SUMMARY RESULTS (Averages across all evaluated frames):")
    print(f"- Distance Band 20-40 m:")
    print(f"    * Dominant Label Match:   {np.mean(b1_no_patch_dom):.1f}%  -->  {np.mean(b1_with_patch_dom):.1f}% (+{np.mean(b1_with_patch_dom) - np.mean(b1_no_patch_dom):.1f}%)")
    print(f"    * Mean Cell Label Purity: {np.mean(b1_no_patch_pur):.1f}%  -->  {np.mean(b1_with_patch_pur):.1f}% (+{np.mean(b1_with_patch_pur) - np.mean(b1_no_patch_pur):.1f}%)")
    print(f"- Distance Band 40-60 m:")
    print(f"    * Dominant Label Match:   {np.mean(b2_no_patch_dom):.1f}%  -->  {np.mean(b2_with_patch_dom):.1f}% (+{np.mean(b2_with_patch_dom) - np.mean(b2_no_patch_dom):.1f}%)")
    print(f"    * Mean Cell Label Purity: {np.mean(b2_no_patch_pur):.1f}%  -->  {np.mean(b2_with_patch_pur):.1f}% (+{np.mean(b2_with_patch_pur) - np.mean(b2_no_patch_pur):.1f}%)")
    print(f"- Average Extra Memory Allocated:   {np.mean(extra_mems_kb):.1f} KB ({np.mean(extra_mems_kb)/1024.0:.2f} MB)")
    print(f"- Average Extra Frame Latency:      {np.mean(extra_times_ms):.2f} ms")
    print("=" * 88)


if __name__ == "__main__":
    evaluate()
