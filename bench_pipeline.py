"""Comprehensive performance benchmark for RAIL-2.5D.

Evaluates pipeline processing and fast rasterisation across:
  - 50,000 and 100,000 points per frame
  - Risk-guided focus patches OFF vs ON
  - Pipeline only vs Pipeline + 2.5D Fast Raster
  - Repeated 3 times per configuration to report median of means and spread

Usage:
    python bench_pipeline.py --frames 300
"""
import argparse
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import numba
    NUMBA_VERSION = numba.__version__
    _HAS_NUMBA = True
except ImportError:
    NUMBA_VERSION = "None (numba not installed)"
    _HAS_NUMBA = False

from src.pipeline import process_frame
from src.loader import load_scan, load_labels, remap_labels, describe_source, SourceDescription
from src.speed import get_speed
from src.scene_urban import make_urban_scene
from src.raster import render_25d_fast, render_grid_map
from src.bench_config import get_bench_config


def get_cpu_model() -> str:
    """Retrieves CPU model name."""
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        if name:
            return name.strip()
    except Exception:
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "Unknown CPU"


def load_base_scans(n_frames: int = 300) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], SourceDescription]:
    """Loads or generates full base scans once using describe_source()."""
    desc = describe_source("data")

    if desc.kind != "synthetic_scene" and desc.path is not None:
        seq_root = desc.path
        labels_dir = seq_root / "labels"
        velo_dir = seq_root / "velodyne"
        cand_poses = seq_root / "poses.txt"
        if not cand_poses.exists():
            cand_poses = seq_root.parent / "poses.txt"
        poses_path = cand_poses if cand_poses.exists() else None

        bin_files = sorted(list(velo_dir.glob("*.bin")))
        real_pairs = []
        for b in bin_files:
            lbl = labels_dir / f"{b.stem}.label"
            if lbl.exists():
                real_pairs.append((b, lbl))

        if real_pairs:
            frames = []
            for idx in range(n_frames):
                b, lbl = real_pairs[idx % len(real_pairs)]
                scan = load_scan(b)
                raw_labels = load_labels(lbl)
                labels = remap_labels(raw_labels)
                xyz = scan[:, :3]
                speed = get_speed(idx % len(real_pairs), poses_path)
                frames.append((xyz, labels, speed))
            return frames, desc

    frames = []
    for idx in range(n_frames):
        s_val = float(idx * 1.5)
        xyz, labels = make_urban_scene(frame_idx=idx, seed=42 + idx, layout="street", s=s_val)
        speed = float(16.0 + 6.0 * np.sin(np.pi * idx / max(1, n_frames - 1)))
        frames.append((xyz, labels, speed))
    return frames, desc


def prepare_point_dataset(base_frames: List[Tuple[np.ndarray, np.ndarray, float]], target_pts: int):
    """Subsamples each scan to exact target point count."""
    sub_frames = []
    for xyz, labels, speed in base_frames:
        n = len(xyz)
        if n >= target_pts:
            sub_idx = np.linspace(0, n - 1, target_pts, dtype=np.int32)
            sub_frames.append((xyz[sub_idx], labels[sub_idx], speed))
        else:
            tile_n = (target_pts // n) + 1
            t_xyz = np.tile(xyz, (tile_n, 1))[:target_pts]
            t_lbl = np.tile(labels, tile_n)[:target_pts]
            sub_frames.append((t_xyz, t_lbl, speed))
    return sub_frames


def benchmark_single_pass(
    frames: List[Tuple[np.ndarray, np.ndarray, float]],
    patches_on: bool,
    with_raster: bool = False,
    raster_view: str = "25d",
    k_patches: int = 4,
) -> Dict[str, Any]:
    """Executes a single timed pass across all frames."""
    latencies = []
    pipe_latencies = []
    raster_latencies = []

    for xyz, labels, speed in frames:
        t0 = time.perf_counter()
        res = process_frame(xyz, labels, risk_patches=patches_on, speed_mps=speed, k_patches=k_patches)
        t_pipe_end = time.perf_counter()

        pipe_ms = (t_pipe_end - t0) * 1000.0
        pipe_latencies.append(pipe_ms)

        if with_raster:
            t_r0 = time.perf_counter()
            if raster_view == "25d":
                _ = render_25d_fast(res["grid"], res["traversability"])
            else:
                _ = render_grid_map(res["grid"], res["traversability"])
            t_r1 = time.perf_counter()
            raster_ms = (t_r1 - t_r0) * 1000.0
            raster_latencies.append(raster_ms)
            total_ms = (t_r1 - t0) * 1000.0
        else:
            total_ms = pipe_ms

        latencies.append(total_ms)

    lat_arr = np.array(latencies)
    mean_ms = float(np.mean(lat_arr))
    p50_ms = float(np.median(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))

    return {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "mean_fps": 1000.0 / max(0.001, mean_ms),
        "p50_fps": 1000.0 / max(0.001, p50_ms),
        "p95_fps": 1000.0 / max(0.001, p95_ms),
        "p99_fps": 1000.0 / max(0.001, p99_ms),
        "pipe_mean_ms": float(np.mean(pipe_latencies)),
        "raster_mean_ms": float(np.mean(raster_latencies)) if raster_latencies else 0.0,
        "latencies": latencies,
    }


def benchmark_configuration(
    frames: List[Tuple[np.ndarray, np.ndarray, float]],
    patches_on: bool,
    with_raster: bool = False,
    raster_view: str = "25d",
    n_repeats: int = 3,
    k_patches: int = 4,
) -> Dict[str, Any]:
    """Runs benchmark 3 times and reports median of means plus spread."""
    # Exclude 10-frame warm-up from timing
    warmup_count = min(10, len(frames))
    warmup_frames = frames[:warmup_count]
    for w_xyz, w_lbls, w_spd in warmup_frames:
        res = process_frame(w_xyz, w_lbls, risk_patches=patches_on, speed_mps=w_spd, k_patches=k_patches)
        if with_raster:
            if raster_view == "25d":
                _ = render_25d_fast(res["grid"], res["traversability"])
            else:
                _ = render_grid_map(res["grid"], res["traversability"])

    # Run n_repeats passes
    run_stats = []
    run_means = []
    for rep in range(n_repeats):
        stats = benchmark_single_pass(
            frames,
            patches_on=patches_on,
            with_raster=with_raster,
            raster_view=raster_view,
            k_patches=k_patches,
        )
        run_stats.append(stats)
        run_means.append(stats["mean_ms"])

    # Identify median run
    sorted_indices = np.argsort(run_means)
    median_idx = sorted_indices[len(sorted_indices) // 2]
    median_stats = dict(run_stats[median_idx])

    min_mean = min(run_means)
    max_mean = max(run_means)
    spread_ms = max_mean - min_mean
    spread_pct = (spread_ms / max(0.001, median_stats["mean_ms"])) * 100.0

    median_stats["spread_pct"] = spread_pct
    median_stats["spread_ms"] = spread_ms
    median_stats["run_means"] = run_means

    return median_stats


def run_full_benchmark(n_frames: int = 300):
    bcfg = get_bench_config()

    print("=" * 102, flush=True)
    print(f"RAIL-2.5D COMPREHENSIVE PERFORMANCE BENCHMARK ({n_frames} FRAMES, 3 REPEATS PER CONFIG)", flush=True)
    print("=" * 102, flush=True)

    base_frames, desc = load_base_scans(n_frames=n_frames)

    print("Environment & System Telemetry:", flush=True)
    print(f"  * CPU Model:       {get_cpu_model()}", flush=True)
    print(f"  * Python Version:  {platform.python_version()}", flush=True)
    print(f"  * Numba JIT:       {'ACTIVE (Compiled njit)' if _HAS_NUMBA else 'DISABLED (Pure NumPy)'} (v{NUMBA_VERSION})", flush=True)
    print(f"  * Frames Tested:   {n_frames} (10-frame warm-up excluded from timing)", flush=True)
    print(f"  * Data Source:     {desc.label} (sequence '{desc.name}', {desc.n_frames} unique scans, cycled to {n_frames} frames)", flush=True)
    print("\nBenchmark Grid Configuration (from src/bench_config.py):", flush=True)
    print(f"  * Fine-Zone:       radius {bcfg.fine_radius:.1f} m, cell size {bcfg.fine_cell_size:.2f} m ({bcfg.fine_grid_size}x{bcfg.fine_grid_size})", flush=True)
    print(f"  * Coarse-Zone:     cell size {bcfg.coarse_cell_size:.2f} m ({bcfg.coarse_grid_size}x{bcfg.coarse_grid_size})", flush=True)
    print(f"  * Focus Patches:   {bcfg.patch_size:.1f} m x {bcfg.patch_size:.1f} m ({bcfg.patch_cell_size:.2f} m cells, limit: {bcfg.k_patches})", flush=True)
    print(f"  * Spatial Range:   {bcfg.range_m:.1f} m radius ({bcfg.range_m*2:.0f} m x {bcfg.range_m*2:.0f} m area)", flush=True)
    print("=" * 102, flush=True)

    final_pass_100k = True

    for n_pts in [50000, 100000]:
        print(f"\n==================== BENCHMARK DATASET: {n_pts:,} POINTS / FRAME ====================", flush=True)
        frames = prepare_point_dataset(base_frames, n_pts)
        print(f"Prepared {len(frames)} frames with exact {n_pts:,} points per frame (3 runs per config).\n", flush=True)

        header = f"{'Configuration':<38} | {'Median ms':>9} | {'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8} | {'Spread':>7} | {'Mean FPS':>9} | {'Target':>7}"
        print(header, flush=True)
        print("-" * 108, flush=True)

        configs = [
            ("Pipeline Only (Patches OFF)", False, False),
            ("Pipeline Only (Patches ON)", True, False),
            ("Pipeline + 2.5D Raster (Patches OFF)", False, True),
            ("Pipeline + 2.5D Raster (Patches ON)", True, True),
        ]

        for name, patches_on, with_raster in configs:
            stats = benchmark_configuration(
                frames,
                patches_on=patches_on,
                with_raster=with_raster,
                n_repeats=3,
                k_patches=bcfg.k_patches,
            )
            is_pass = (stats["mean_fps"] >= 30.0) and (stats["p95_fps"] >= 25.0)
            status_str = "PASS" if is_pass else "FAIL"

            if n_pts == 100000 and patches_on and not with_raster and not is_pass:
                final_pass_100k = False

            print(
                f"{name:<38} | {stats['mean_ms']:9.2f} | {stats['p50_ms']:8.2f} | "
                f"{stats['p95_ms']:8.2f} | {stats['p99_ms']:8.2f} | {stats['spread_pct']:6.1f}% | "
                f"{stats['mean_fps']:8.1f} | {status_str:>7}",
                flush=True,
            )
            if stats["spread_pct"] > 20.0:
                print(f"    -> WARNING: Spread across 3 runs is {stats['spread_pct']:.1f}% (> 20.0%)", flush=True)
            if with_raster:
                print(f"    -> breakdown: pipeline {stats['pipe_mean_ms']:.2f} ms + raster {stats['raster_mean_ms']:.2f} ms", flush=True)

        print("-" * 108, flush=True)

    print("\n" + "=" * 102, flush=True)
    print("FINAL VERDICT FOR 100k POINTS (PATCHES ON, >=30 FPS TARGET):", flush=True)
    print(f"Data Source: {desc.label} (sequence '{desc.name}')", flush=True)
    if final_pass_100k:
        print(">>> SUCCESS: PIPELINE MEETS >= 30 FPS TARGET (MEAN >= 30 FPS, P95 >= 25 FPS) <<<", flush=True)
    else:
        print(">>> FAIL: PIPELINE DID NOT MEET TARGET <<<", flush=True)
    print("=" * 102, flush=True)

    if desc.kind == "real_full":
        print("\nCredit: SemanticKITTI, Behley et al., ICCV 2019, CC BY-NC-SA 4.0\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAIL-2.5D Benchmark Runner")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to benchmark (default: 300)")
    args = parser.parse_args()
    run_full_benchmark(n_frames=args.frames)
