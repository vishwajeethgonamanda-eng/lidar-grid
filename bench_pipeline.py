from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.pipeline import process_frame
from src.loader import load_scan, load_labels, remap_labels, make_synthetic_scene
from src.speed import get_speed
from eval_patches import generate_synthetic_scene_with_targets


def load_benchmark_frames(n_frames: int = 100) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], str]:
    """Loads frames for benchmarking: real SemanticKITTI if in data/, else synthetic."""
    data_dir = Path("data")
    velo_dirs = list(data_dir.rglob("velodyne"))
    real_frames = []
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

    frames = []
    if real_frames:
        mode_str = f"Real SemanticKITTI ({len(real_frames)} frames)"
        for idx, (b_path, l_path) in enumerate(real_frames):
            scan = load_scan(b_path)
            raw_labels = load_labels(l_path)
            labels = remap_labels(raw_labels)
            xyz = scan[:, :3]
            speed = get_speed(idx, poses_path)
            frames.append((xyz, labels, speed))
    else:
        mode_str = f"Synthetic Scene ({n_frames} frames)"
        for idx in range(n_frames):
            xyz, labels = generate_synthetic_scene_with_targets(seed=1000 + idx)
            speed = float(16.0 + 8.0 * np.sin(np.pi * idx / (n_frames - 1)))
            frames.append((xyz, labels, speed))

    return frames, mode_str


def run_benchmark(n_frames: int = 100):
    print("=" * 80)
    print(f"HEADLESS PIPELINE BENCHMARK (100 Frames)")
    print("=" * 80)

    frames, mode_str = load_benchmark_frames(n_frames)
    print(f"Data Source: {mode_str}")
    print(f"Number of frames: {len(frames)}\n")

    for patches_on in (False, True):
        label = "WITH RISK PATCHES (Patches ON)" if patches_on else "BASELINE (Patches OFF)"
        print(f"--- Running: {label} ---")

        total_times = []
        add_times = []
        risk_times = []
        rebin_times = []
        trav_times = []

        # Warmup frame
        w_xyz, w_lbls, w_spd = frames[0]
        process_frame(w_xyz, w_lbls, risk_patches=patches_on, speed_mps=w_spd)

        for xyz, labels, speed in frames:
            t0 = time.perf_counter()
            res = process_frame(xyz, labels, risk_patches=patches_on, speed_mps=speed, k_patches=4)
            t1 = time.perf_counter()

            tot_ms = (t1 - t0) * 1000.0
            total_times.append(tot_ms)

            timings = res["timings"]
            if not patches_on:
                add_times.append(timings.get("add_points_ms", 0.0))
                risk_times.append(0.0)
                rebin_times.append(0.0)
                trav_times.append(timings.get("compute_traversability_ms", 0.0))
            else:
                # With patches
                # pass 1 add_points + risk_time + pass 2 re-binning + traversability
                pass1_add = timings.get("pass1_ms", timings.get("extra_time_ms", 0.0) - timings.get("risk_time_ms", 0.0))
                r_time = timings.get("risk_time_ms", 0.0)
                rebin_add = timings.get("add_points_ms", 0.0)
                trav_ms = timings.get("compute_traversability_ms", 0.0)

                add_times.append(pass1_add)
                risk_times.append(r_time)
                rebin_times.append(rebin_add)
                trav_times.append(trav_ms)

        mean_ms = float(np.mean(total_times))
        median_ms = float(np.median(total_times))
        p95_ms = float(np.percentile(total_times, 95))
        worst_ms = float(np.max(total_times))

        mean_fps = 1000.0 / mean_ms
        median_fps = 1000.0 / median_ms

        print(f"  * Mean Frame Time:   {mean_ms:6.2f} ms  ({mean_fps:5.1f} FPS)")
        print(f"  * Median Frame Time: {median_ms:6.2f} ms  ({median_fps:5.1f} FPS)")
        print(f"  * 95th Percentile:   {p95_ms:6.2f} ms")
        print(f"  * Worst Frame Time:  {worst_ms:6.2f} ms")

        print("  * Per-Stage Breakdown (Mean ms):")
        if not patches_on:
            print(f"      - add_points:       {np.mean(add_times):6.2f} ms ({np.mean(add_times)/mean_ms*100:4.1f}%)")
            print(f"      - traversability:   {np.mean(trav_times):6.2f} ms ({np.mean(trav_times)/mean_ms*100:4.1f}%)")
        else:
            print(f"      - add_points (P1):  {np.mean(add_times):6.2f} ms ({np.mean(add_times)/mean_ms*100:4.1f}%)")
            print(f"      - risk_search:      {np.mean(risk_times):6.2f} ms ({np.mean(risk_times)/mean_ms*100:4.1f}%)")
            print(f"      - re-binning (P2):  {np.mean(rebin_times):6.2f} ms ({np.mean(rebin_times)/mean_ms*100:4.1f}%)")
            print(f"      - traversability:   {np.mean(trav_times):6.2f} ms ({np.mean(trav_times)/mean_ms*100:4.1f}%)")
        print()

    print("=" * 80)


if __name__ == "__main__":
    run_benchmark(100)
