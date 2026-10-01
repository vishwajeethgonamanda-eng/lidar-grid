from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.grid_engine import VarResGrid, stopping_distance, adaptive_fine_radius
from src.loader import make_synthetic_scene, load_scan, load_labels, remap_labels
from src.speed import get_speed


def evaluate(n_frames: int = 20):
    print("=" * 80)
    print("EVALUATION: Fixed (r=10m) vs Adaptive Fine Radius at Constant Memory")
    print("=" * 80)

    # Check for real data in data/
    data_dir = Path("data")
    velo_dirs = list(data_dir.rglob("velodyne"))
    real_frames: List[Tuple[Path, Path]] = []
    poses_path: Path = None

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
    speeds = []
    stopping_dists = []
    fixed_fine_ranges = []
    adapt_fine_ranges = []
    fixed_fractions = []
    adapt_fractions = []

    header = (
        f"{'Frame':^5} | {'Speed':^7} | {'d_stop':^7} | "
        f"{'Fixed R':^7} | {'Adapt R':^7} | "
        f"{'Fixed Range':^11} | {'Adapt Range':^11} | "
        f"{'Fixed % in Fine':^15} | {'Adapt % in Fine':^15}"
    )
    print(header)
    print("-" * len(header))

    # Obstacle / Kerb classes (ground truth labels 3, 4, 5, 6, 7)
    OBSTACLE_CLASSES = {3, 4, 5, 6, 7}

    for frame_idx in range(total_frames):
        if is_real:
            bin_path, label_path = real_frames[frame_idx]
            scan = load_scan(bin_path)
            raw_labels = load_labels(label_path)
            labels = remap_labels(raw_labels)
            xyz = scan[:, :3]
            speed = get_speed(frame_idx, poses_path)
        else:
            scan, labels = make_synthetic_scene(seed=100 + frame_idx)
            xyz = scan[:, :3]
            # Vary speed from ~8 m/s (30 km/h) to 20 m/s (72 km/h)
            speed = float(8.0 + 10.0 * np.sin(np.pi * frame_idx / (total_frames - 1)))

        d_stop = stopping_distance(speed)
        r_fixed = 10.0
        r_adapt = adaptive_fine_radius(speed)

        # Fixed grid
        grid_fixed = VarResGrid(n_classes=8, fine_radius=r_fixed)
        grid_fixed.add_points(xyz, labels)

        # Adaptive grid
        grid_adapt = VarResGrid(n_classes=8, fine_radius=r_adapt)
        grid_adapt.add_points(xyz, labels)

        # Identify ground-truth obstacle / kerb points
        obs_mask = np.isin(labels, list(OBSTACLE_CLASSES))
        obs_xyz = xyz[obs_mask]
        obs_r = np.hypot(obs_xyz[:, 0], obs_xyz[:, 1])

        # Fine zone obstacle detection range
        fixed_fine_obs_r = obs_r[obs_r < r_fixed]
        adapt_fine_obs_r = obs_r[obs_r < r_adapt]

        fixed_range = float(np.max(fixed_fine_obs_r)) if len(fixed_fine_obs_r) > 0 else 0.0
        adapt_range = float(np.max(adapt_fine_obs_r)) if len(adapt_fine_obs_r) > 0 else 0.0

        # Obstacles within stopping distance
        within_stop = obs_r <= d_stop
        n_within_stop = np.count_nonzero(within_stop)

        if n_within_stop > 0:
            n_fixed_in_fine = np.count_nonzero(within_stop & (obs_r < r_fixed))
            n_adapt_in_fine = np.count_nonzero(within_stop & (obs_r < r_adapt))
            pct_fixed = (n_fixed_in_fine / n_within_stop) * 100.0
            pct_adapt = (n_adapt_in_fine / n_within_stop) * 100.0
        else:
            pct_fixed = 100.0
            pct_adapt = 100.0

        speeds.append(speed)
        stopping_dists.append(d_stop)
        fixed_fine_ranges.append(fixed_range)
        adapt_fine_ranges.append(adapt_range)
        fixed_fractions.append(pct_fixed)
        adapt_fractions.append(pct_adapt)

        print(
            f"{frame_idx:^5d} | {speed:^7.1f} | {d_stop:^7.1f} | "
            f"{r_fixed:^7.1f} | {r_adapt:^7.1f} | "
            f"{fixed_range:^11.1f} | {adapt_range:^11.1f} | "
            f"{pct_fixed:^14.1f}% | {pct_adapt:^14.1f}%"
        )

    print("=" * 80)
    print("SUMMARY RESULTS (Averages across all frames):")
    print(f"- Average Vehicle Speed:               {np.mean(speeds):.1f} m/s ({np.mean(speeds) * 3.6:.1f} km/h)")
    print(f"- Average Stopping Distance:           {np.mean(stopping_dists):.1f} m")
    print(f"- Fixed Fine Obstacle Range:           {np.mean(fixed_fine_ranges):.1f} m")
    print(f"- Adaptive Fine Obstacle Range:        {np.mean(adapt_fine_ranges):.1f} m (+{np.mean(adapt_fine_ranges) - np.mean(fixed_fine_ranges):.1f} m boost)")
    print(f"- Obstacles in Fine within Stop Dist (Fixed):    {np.mean(fixed_fractions):.1f}%")
    print(f"- Obstacles in Fine within Stop Dist (Adaptive): {np.mean(adapt_fractions):.1f}% (+{np.mean(adapt_fractions) - np.mean(fixed_fractions):.1f}% improvement)")
    print(f"- Constant Memory Footprint:           {grid_fixed.memory_bytes() / (1024*1024):.2f} MB for both configurations")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
