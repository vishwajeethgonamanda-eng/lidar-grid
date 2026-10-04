"""Standalone CLI pipeline runner for RAIL-2.5D.

Processes scans from SemanticKITTI sequence (or synthetic scene fallback),
computes traversability and risk-guided focus patches, logs per-frame metrics
to CSV, and prints a final performance summary.

Usage:
    python run_pipeline.py --seq 00 --frames 300 --out results/
"""
import argparse
import csv
from pathlib import Path
import time
from typing import Optional, Tuple, List
import numpy as np

from src.pipeline import process_frame
from src.loader import load_scan, load_labels, remap_labels
from src.speed import get_speed
from src.scene_urban import make_urban_scene
from src.traversability import OBSTACLE, TraversabilityParams


def parse_args():
    parser = argparse.ArgumentParser(description="RAIL-2.5D Standalone CLI Pipeline Runner")
    parser.add_argument("--seq", type=str, default="00", help="SemanticKITTI sequence ID (default: 00)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process (default: 300)")
    parser.add_argument("--out", type=str, default="results/", help="Output directory for metrics (default: results/)")
    parser.add_argument("--k_patches", type=int, default=4, help="Max focus patches (default: 4)")
    parser.add_argument("--speed", type=float, default=None, help="Fixed vehicle speed in m/s (default: auto from poses)")
    return parser.parse_args()


def load_dataset(seq_id: str, n_frames: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], str]:
    """Loads frames from data/ if available for sequence, else generates synthetic scenes."""
    data_dir = Path("data")
    target_velo = None

    if data_dir.exists():
        # Look for sequences matching seq_id
        for velo_dir in data_dir.rglob("velodyne"):
            seq_root = velo_dir.parent
            if seq_id in str(seq_root):
                target_velo = velo_dir
                break
        if target_velo is None:
            # Fall back to any velodyne dir
            velo_dirs = list(data_dir.rglob("velodyne"))
            if velo_dirs:
                target_velo = velo_dirs[0]

    frames = []
    if target_velo is not None:
        seq_root = target_velo.parent
        labels_dir = seq_root / "labels"
        cand_poses = seq_root / "poses.txt"
        if not cand_poses.exists():
            cand_poses = seq_root.parent / "poses.txt"
        poses_path = cand_poses if cand_poses.exists() else None

        bin_files = sorted(list(target_velo.glob("*.bin")))
        real_pairs = []
        for b in bin_files:
            lbl = labels_dir / f"{b.stem}.label"
            if lbl.exists():
                real_pairs.append((b, lbl))

        if real_pairs:
            for idx in range(n_frames):
                b, lbl = real_pairs[idx % len(real_pairs)]
                scan = load_scan(b)
                raw_labels = load_labels(lbl)
                labels = remap_labels(raw_labels)
                xyz = scan[:, :3]
                speed = get_speed(idx % len(real_pairs), poses_path)
                frames.append((xyz, labels, speed))

    if frames:
        mode_str = f"SemanticKITTI sequence {seq_id} ({len(real_pairs)} unique frames, cycled to {len(frames)} frames from {target_velo.parent})"
    else:
        mode_str = (
            f"[NOTICE: Real SemanticKITTI sequence '{seq_id}' not found in data/]\n"
            f"Using synthetic 64-beam urban street LiDAR generator ({n_frames} frames)."
        )
        for idx in range(n_frames):
            s_val = float(idx * 1.5)
            xyz, labels = make_urban_scene(frame_idx=idx, seed=42 + idx, layout="street", s=s_val)
            speed = float(16.0 + 6.0 * np.sin(np.pi * idx / max(1, n_frames - 1)))
            frames.append((xyz, labels, speed))

    return frames, mode_str


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    print("=" * 80)
    print("RAIL-2.5D STANDALONE PIPELINE RUNNER")
    print("=" * 80)

    frames, mode_info = load_dataset(args.seq, args.frames)
    print(f"Data source: {mode_info}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Frames to process: {len(frames)}\n")

    # Warmup
    w_xyz, w_lbls, w_spd = frames[0]
    process_frame(w_xyz, w_lbls, risk_patches=True, speed_mps=w_spd, k_patches=args.k_patches)

    records = []
    total_start_time = time.perf_counter()

    for idx, (xyz, labels, speed) in enumerate(frames):
        spd = args.speed if args.speed is not None else speed
        t0 = time.perf_counter()
        res = process_frame(xyz, labels, risk_patches=True, speed_mps=spd, k_patches=args.k_patches)
        t1 = time.perf_counter()

        frame_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / max(0.001, frame_ms)

        timings = res["timings"]
        add_ms = timings.get("pass1_ms", timings.get("add_points_ms", 0.0))
        risk_ms = timings.get("risk_time_ms", 0.0)
        trav_ms = timings.get("compute_traversability_ms", 0.0)

        grid = res["grid"]
        trav = res["traversability"]
        n_patches = len(grid.patches) if hasattr(grid, "patches") else 0

        # Count obstacle cells in fine and coarse grids
        n_obs = int(np.count_nonzero(trav.fine.state == OBSTACLE) + np.count_nonzero(trav.coarse.state == OBSTACLE))
        if hasattr(trav, "patches"):
            n_obs += int(sum(np.count_nonzero(p.state == OBSTACLE) for p in trav.patches))

        records.append({
            "frame": idx,
            "n_points": len(xyz),
            "add_points_ms": f"{add_ms:.3f}",
            "trav_ms": f"{trav_ms:.3f}",
            "risk_ms": f"{risk_ms:.3f}",
            "total_ms": f"{frame_ms:.3f}",
            "fps": f"{fps:.2f}",
            "n_patches": n_patches,
            "n_obstacles": n_obs,
        })

        if (idx + 1) % 50 == 0 or idx == len(frames) - 1:
            print(f"Processed frame {idx + 1:4d} / {len(frames):4d} | Latency: {frame_ms:5.2f} ms ({fps:5.1f} FPS) | Patches: {n_patches}")

    total_time_s = time.perf_counter() - total_start_time

    # Write metrics to CSV
    fieldnames = ["frame", "n_points", "add_points_ms", "trav_ms", "risk_ms", "total_ms", "fps", "n_patches", "n_obstacles"]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Summary statistics
    all_ms = [float(r["total_ms"]) for r in records]
    all_fps = [float(r["fps"]) for r in records]
    mean_ms = np.mean(all_ms)
    p95_ms = np.percentile(all_ms, 95)
    mean_fps = np.mean(all_fps)
    p95_fps = 1000.0 / max(0.001, p95_ms)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY:")
    print(f"  * Total Frames Processed: {len(records)}")
    print(f"  * Total Execution Time:    {total_time_s:.2f} s")
    print(f"  * Mean Frame Latency:      {mean_ms:.2f} ms")
    print(f"  * 95th Percentile Latency: {p95_ms:.2f} ms")
    print(f"  * Mean Pipeline Rate:      {mean_fps:.1f} FPS")
    print(f"  * 95th Percentile Rate:    {p95_fps:.1f} FPS")
    print(f"  * Metrics saved to:        {metrics_csv.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
