"""Standalone CLI pipeline runner for RAIL-2.5D.

Processes scans from SemanticKITTI sequence (or synthetic scene fallback),
computes traversability and risk-guided focus patches, logs per-frame metrics
to CSV, and prints a final performance summary.

Usage:
    python run_pipeline.py --frames 300 --points 100000 --out results
"""
import argparse
import csv
from pathlib import Path
import time
from typing import Optional, Tuple, List, Set, Dict, Any
import numpy as np

try:
    import numba
    _HAS_NUMBA = True
    NUMBA_VERSION = numba.__version__
except ImportError:
    _HAS_NUMBA = False
    NUMBA_VERSION = "None (not installed)"

from src.pipeline import process_frame
from src.loader import load_scan, load_labels, remap_labels, describe_source
from src.speed import get_speed
from src.scene_urban import make_urban_scene
from src.traversability import OBSTACLE, DRIVABLE, NON_DRIVABLE, TraversabilityParams, compute_too_steep
from eval_patches import compute_label_purity
from src.bench_config import get_bench_config


def parse_args():
    parser = argparse.ArgumentParser(description="RAIL-2.5D Standalone CLI Pipeline Runner")
    parser.add_argument("--seq", type=str, default="00", help="SemanticKITTI sequence ID (default: 00)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process (default: 300)")
    parser.add_argument("--points", type=int, default=100000, help="Exact point count per frame (default: 100000)")
    parser.add_argument("--out", type=str, default="results", help="Output directory for metrics (default: results)")
    parser.add_argument("--k_patches", type=int, default=4, help="Max focus patches (default: 4)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repeats (default: 3)")
    parser.add_argument("--speed", type=float, default=None, help="Fixed vehicle speed in m/s (default: auto from poses)")
    return parser.parse_args()


def load_dataset(seq_id: str, n_frames: int):
    """Loads frames from data/ using describe_source(), or falls back to synthetic scene."""
    desc = describe_source("data", seq_id=seq_id)
    frames = []

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
            for idx in range(n_frames):
                b, lbl = real_pairs[idx % len(real_pairs)]
                scan = load_scan(b)
                raw_labels = load_labels(lbl)
                labels = remap_labels(raw_labels)
                xyz = scan[:, :3]
                speed = get_speed(idx % len(real_pairs), poses_path)
                frames.append((xyz, labels, speed))

            mode_str = f"{desc.label} (sequence '{desc.name}', {len(real_pairs)} unique scans, cycled to {len(frames)} frames from {desc.path})"
            return frames, mode_str, desc

    mode_str = f"{desc.label} ({n_frames} frames from built-in 64-beam urban generator)"
    for idx in range(n_frames):
        s_val = float(idx * 1.5)
        xyz, labels = make_urban_scene(frame_idx=idx, seed=42 + idx, layout="street", s=s_val)
        speed = float(16.0 + 6.0 * np.sin(np.pi * idx / max(1, n_frames - 1)))
        frames.append((xyz, labels, speed))

    return frames, mode_str, desc


def prepare_point_dataset(base_frames: List[Tuple[np.ndarray, np.ndarray, float]], target_pts: int):
    """Subsamples or tiles each scan to exact target point count."""
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


def compute_occupied_cells_agreement(grid, trav) -> Tuple[float, int, int]:
    """Computes traversability agreement on occupied cells only.

    Denominator: count of occupied cells (count > 0) with a valid semantic class (1..7).
    Numerator: count of such occupied cells where actual traversability state matches
               the expected state:
                 - Class 1 (Road, Parking): DRIVABLE (1)
                 - Class 2 (Terrain) & 7 (Sidewalk): NON_DRIVABLE (2)
                 - Class 3, 4, 5, 6 (Static/Dynamic Obstacles): OBSTACLE (3)
    """
    total_occupied = 0
    total_matching = 0

    def tally(dom, state, occ):
        nonlocal total_occupied, total_matching
        if not np.any(occ):
            return
        d = dom[occ]
        s = state[occ]
        valid = (d >= 1) & (d <= 7)
        if np.any(valid):
            vd = d[valid]
            vs = s[valid]
            exp = np.zeros(len(vd), dtype=np.int8)
            exp[vd == 1] = 1
            exp[(vd == 2) | (vd == 7)] = 2
            exp[(vd >= 3) & (vd <= 6)] = 3
            total_occupied += len(vd)
            total_matching += int(np.count_nonzero(vs == exp))

    tally(grid.fine.dominant_label, trav.fine.state, grid.fine.count > 0)
    tally(grid.coarse.dominant_label, trav.coarse.state, grid.coarse.count > 0)
    if hasattr(grid, "patches") and hasattr(trav, "patches"):
        for p, pt in zip(grid.patches, trav.patches):
            tally(p.dominant_label, pt.state, p.count > 0)

    ratio = float(total_matching) / max(1, total_occupied)
    return ratio, total_matching, total_occupied


def tally_per_class_agreement(grid, trav, accumulator: Dict[int, List[int]], params: TraversabilityParams):
    """Tally occupied cells, semantic matching cells, and geometry-only matching cells across classes (1..7)."""
    def tally(zone, ztrav):
        occ = zone.count > 0
        if not np.any(occ):
            return
        dom = zone.dominant_label
        state = ztrav.state

        thresh_dz = np.float32(zone.cell_size * np.tan(np.deg2rad(params.max_slope_deg)))
        inv_count = np.zeros(zone.count.shape, dtype=np.float32)
        inv_count[occ] = 1.0 / zone.count[occ]
        z_m = (zone.z_sum * inv_count).astype(np.float32)

        diff_v = np.abs(z_m[1:, :] - z_m[:-1, :])
        both_v = (diff_v > thresh_dz) & occ[1:, :] & occ[:-1, :]
        diff_h = np.abs(z_m[:, 1:] - z_m[:, :-1])
        both_h = (diff_h > thresh_dz) & occ[:, 1:] & occ[:, :-1]
        too_steep = np.zeros(zone.count.shape, dtype=bool)
        too_steep[1:, :] |= both_v
        too_steep[:-1, :] |= both_v
        too_steep[:, 1:] |= both_h
        too_steep[:, :-1] |= both_h

        step_height = zone.z_max - zone.z_min
        too_high_step = step_height > params.max_step
        non_flat = too_high_step | too_steep

        for c in range(1, 8):
            mask_c = occ & (dom == c)
            if not np.any(mask_c):
                continue
            n_c = int(np.count_nonzero(mask_c))
            accumulator[c][0] += n_c

            if c == 1:
                exp_sem = 1  # DRIVABLE
                n_match_geom = int(np.count_nonzero((~non_flat) & mask_c))
            elif c in (2, 7):
                exp_sem = 2  # NON_DRIVABLE
                n_match_geom = int(np.count_nonzero(non_flat & mask_c))
            else:
                exp_sem = 3  # OBSTACLE
                n_match_geom = int(np.count_nonzero(non_flat & mask_c))

            n_match_sem = int(np.count_nonzero(state[mask_c] == exp_sem))
            accumulator[c][1] += n_match_sem
            accumulator[c][2] += n_match_geom

    tally(grid.fine, trav.fine)
    tally(grid.coarse, trav.coarse)
    if hasattr(grid, "patches") and hasattr(trav, "patches"):
        for p, pt in zip(grid.patches, trav.patches):
            tally(p, pt)


def tally_road_diagnostics(grid, trav, diag_acc: Dict[str, Dict[str, int]], params: TraversabilityParams):
    """Diagnose ground-truth Road cells (class 1) that are not marked DRIVABLE."""
    def inspect_zone(zone, ztrav, zname):
        occ = zone.count > 0
        road = occ & (zone.dominant_label == 1)
        nondriv = road & (ztrav.state != 1)
        if not np.any(nondriv):
            return

        diag_acc["zone"][zname] += int(np.count_nonzero(nondriv))

        too_steep = compute_too_steep(zone, params)

        step_height = zone.z_max - zone.z_min
        too_high_step = step_height > params.max_step

        rows, cols = np.where(nondriv)
        if zname == "fine":
            xs = (cols + 0.5) * zone.cell_size - 10.0 + grid.forward_offset
            ys = (rows + 0.5) * zone.cell_size - 10.0
        elif zname == "coarse":
            xs = (cols + 0.5) * zone.cell_size - 100.0
            ys = (rows + 0.5) * zone.cell_size - 100.0
        else:
            xs = zone.center_x + (cols + 0.5) * zone.cell_size - zone.half_extent
            ys = zone.center_y + (rows + 0.5) * zone.cell_size - zone.half_extent

        dists = np.hypot(xs, ys)
        diag_acc["range"]["0-10"] += int(np.count_nonzero(dists < 10.0))
        diag_acc["range"]["10-30"] += int(np.count_nonzero((dists >= 10.0) & (dists < 30.0)))
        diag_acc["range"]["30-60"] += int(np.count_nonzero((dists >= 30.0) & (dists < 60.0)))
        diag_acc["range"]["60-100"] += int(np.count_nonzero(dists >= 60.0))

        steep_nd = too_steep[rows, cols]
        step_nd = too_high_step[rows, cols]
        min_pts = params.min_points_fine if zname == "fine" else params.min_points_coarse
        few_pts = zone.count[rows, cols] < min_pts

        both = steep_nd & step_nd
        slope_only = steep_nd & (~step_nd)
        step_only = step_nd & (~steep_nd)
        few_only = few_pts & (~steep_nd) & (~step_nd)
        other = (~steep_nd) & (~step_nd) & (~few_pts)

        diag_acc["cause"]["slope"] += int(np.count_nonzero(slope_only))
        diag_acc["cause"]["step"] += int(np.count_nonzero(step_only))
        diag_acc["cause"]["both"] += int(np.count_nonzero(both))
        diag_acc["cause"]["few_pts"] += int(np.count_nonzero(few_only))
        diag_acc["cause"]["other"] += int(np.count_nonzero(other))

    inspect_zone(grid.fine, trav.fine, "fine")
    inspect_zone(grid.coarse, trav.coarse, "coarse")
    if hasattr(grid, "patches") and hasattr(trav, "patches"):
        for p, pt in zip(grid.patches, trav.patches):
            inspect_zone(p, pt, "patch")


def count_sparse_5cm_cells(xyz: np.ndarray) -> int:
    """Counts distinct occupied 5 cm cells over [-100, 100) m."""
    r_sensor = np.hypot(xyz[:, 0], xyz[:, 1])
    in_range = r_sensor < 100.0
    if not np.any(in_range):
        return 0
    pts = xyz[in_range]
    col = np.clip(np.floor((pts[:, 0] + 100.0) / 0.05).astype(np.int64), 0, 3999)
    row = np.clip(np.floor((pts[:, 1] + 100.0) / 0.05).astype(np.int64), 0, 3999)
    flat_idx = row * 4000 + col
    return int(len(np.unique(flat_idx)))


def main():
    args = parse_args()
    bcfg = get_bench_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    class_csv = out_dir / "class_agreement.csv"

    print("=" * 80)
    print("RAIL-2.5D STANDALONE PIPELINE RUNNER")
    print("=" * 80)

    base_frames, mode_info, desc = load_dataset(args.seq, args.frames)
    if args.points is not None:
        frames = prepare_point_dataset(base_frames, args.points)
        pts_str = f"{args.points:,} points / frame"
    else:
        frames = base_frames
        pts_str = "original point counts"

    print(f"Data source:       {mode_info}")
    print(f"Points per frame:  {pts_str}")
    print(f"Output directory:  {out_dir.resolve()}")
    print(f"Frames to process: {len(frames)}")
    print(f"Numba JIT status:  {'ACTIVE (njit compiled)' if _HAS_NUMBA else 'DISABLED (Pure NumPy)'} (v{NUMBA_VERSION})")
    print("\nBenchmark Grid Configuration (from src/bench_config.py):")
    print(f"  * Fine-Zone:       radius {bcfg.fine_radius:.1f} m, cell size {bcfg.fine_cell_size:.2f} m ({bcfg.fine_grid_size}x{bcfg.fine_grid_size})")
    print(f"  * Coarse-Zone:     cell size {bcfg.coarse_cell_size:.2f} m ({bcfg.coarse_grid_size}x{bcfg.coarse_grid_size})")
    print(f"  * Focus Patches:   {bcfg.patch_size:.1f} m x {bcfg.patch_size:.1f} m ({bcfg.patch_cell_size:.2f} m cells, limit: {bcfg.k_patches})")
    print(f"  * Spatial Range:   {bcfg.range_m:.1f} m radius ({bcfg.range_m*2:.0f} m x {bcfg.range_m*2:.0f} m area)")
    print("=" * 80 + "\n")

    # Discover classes present across dataset
    classes_present: Set[int] = set()
    for _, lbls, _ in frames:
        classes_present.update(np.unique(lbls).tolist())
    classes_present_str = ",".join(str(c) for c in sorted(classes_present) if c > 0)

    # 1. Warm-up: 10 frames excluded from timing (matching bench_pipeline.py)
    warmup_count = min(10, len(frames))
    warmup_frames = frames[:warmup_count]
    print(f"Warming up pipeline with {warmup_count} frames (excluded from timing)...", flush=True)
    for w_xyz, w_lbls, w_spd in warmup_frames:
        process_frame(w_xyz, w_lbls, risk_patches=False, speed_mps=w_spd)
        process_frame(w_xyz, w_lbls, risk_patches=True, speed_mps=w_spd, k_patches=bcfg.k_patches)

    # 2. Run Patches OFF configuration 3 times
    print(f"\nRunning {args.repeats} passes for Pipeline Only (Patches OFF)...", flush=True)
    runs_off_lats = []
    runs_off_means = []
    for rep in range(args.repeats):
        lats = []
        for xyz, labels, speed in frames:
            spd = args.speed if args.speed is not None else speed
            t0 = time.perf_counter()
            _ = process_frame(xyz, labels, risk_patches=False, speed_mps=spd)
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1000.0)
        m_val = float(np.mean(lats))
        runs_off_means.append(m_val)
        runs_off_lats.append(lats)
        print(f"  -> Pass {rep + 1}/{args.repeats} (Patches OFF): mean {m_val:.2f} ms ({1000.0/m_val:.1f} FPS)")

    # Select median run for Patches OFF
    med_off_idx = int(np.argsort(runs_off_means)[len(runs_off_means) // 2])
    best_off_lats = runs_off_lats[med_off_idx]
    spread_off_ms = max(runs_off_means) - min(runs_off_means)
    spread_off_pct = (spread_off_ms / max(0.001, runs_off_means[med_off_idx])) * 100.0
    print(f"Patches OFF: Median mean = {runs_off_means[med_off_idx]:.2f} ms, spread = {spread_off_pct:.1f}%")
    if spread_off_pct > 20.0:
        print(f"WARNING: Patches OFF spread across {args.repeats} runs is {spread_off_pct:.1f}% (> 20.0%)!")

    # 3. Run Patches ON configuration 3 times
    print(f"\nRunning {args.repeats} passes for Pipeline Only (Patches ON)...", flush=True)
    runs_on_data = []
    runs_on_means = []

    trav_params = TraversabilityParams()
    road_diag_csv = out_dir / "road_diagnostics.csv"

    for rep in range(args.repeats):
        rep_lats = []
        rep_records = []
        rep_class_tallies: Dict[int, List[int]] = {c: [0, 0, 0] for c in range(1, 8)}
        rep_road_diag = {
            "zone": {"fine": 0, "coarse": 0, "patch": 0},
            "range": {"0-10": 0, "10-30": 0, "30-60": 0, "60-100": 0},
            "cause": {"slope": 0, "step": 0, "both": 0, "few_pts": 0, "other": 0},
        }

        for idx, (xyz, labels, speed) in enumerate(frames):
            spd = args.speed if args.speed is not None else speed
            t0 = time.perf_counter()
            res_on = process_frame(xyz, labels, risk_patches=True, speed_mps=spd, k_patches=bcfg.k_patches)
            t1 = time.perf_counter()
            pipe_ms = (t1 - t0) * 1000.0
            rep_lats.append(pipe_ms)

            timings = res_on["timings"]
            add_points_ms = timings.get("pass1_ms", timings.get("add_points_ms", 0.0))
            rebinning_ms = timings.get("add_points_ms", 0.0)
            risk_search_ms = timings.get("risk_time_ms", 0.0)
            traversability_ms = timings.get("compute_traversability_ms", 0.0)

            overhead_ms = max(0.0, pipe_ms - (add_points_ms + rebinning_ms + risk_search_ms + traversability_ms))

            grid = res_on["grid"]
            trav = res_on["traversability"]
            patch_count = len(grid.patches) if hasattr(grid, "patches") else 0

            fine_cells = int(np.count_nonzero(grid.fine.count > 0))
            coarse_cells = int(np.count_nonzero(grid.coarse.count > 0))
            patch_cells = int(sum(np.count_nonzero(p.count > 0) for p in grid.patches)) if hasattr(grid, "patches") else 0

            memory_var_mb = float(grid.memory_bytes()) / (1024.0 * 1024.0)
            memory_uniform_mb = float(grid.uniform_equivalent_bytes()) / (1024.0 * 1024.0)

            # Sparse uniform occupied cells
            sparse_cells = count_sparse_5cm_cells(xyz)
            sparse_uniform_mb = float(sparse_cells * 60) / (1024.0 * 1024.0)

            # Traversability agreement & diagnostics
            trav_agreement, n_match, n_occ = compute_occupied_cells_agreement(grid, trav)
            tally_per_class_agreement(grid, trav, rep_class_tallies, trav_params)
            tally_road_diagnostics(grid, trav, rep_road_diag, trav_params)

            purity_20_40, _ = compute_label_purity(xyz, labels, grid, 20.0, 40.0)
            purity_40_60, _ = compute_label_purity(xyz, labels, grid, 40.0, 60.0)

            rep_records.append({
                "frame": idx,
                "patches_off_ms": f"{best_off_lats[idx]:.3f}",
                "patches_on_ms": f"{pipe_ms:.3f}",
                "add_points_ms": f"{add_points_ms:.3f}",
                "rebinning_ms": f"{rebinning_ms:.3f}",
                "re_binning_ms": f"{rebinning_ms:.3f}",
                "traversability_ms": f"{traversability_ms:.3f}",
                "risk_search_ms": f"{risk_search_ms:.3f}",
                "overhead_ms": f"{overhead_ms:.3f}",
                "total_ms": f"{pipe_ms:.3f}",
                "fine_cells": fine_cells,
                "coarse_cells": coarse_cells,
                "patch_cells": patch_cells,
                "sparse_uniform_cells": sparse_cells,
                "sparse_uniform_mb": f"{sparse_uniform_mb:.2f}",
                "memory_var_mb": f"{memory_var_mb:.2f}",
                "memory_uniform_mb": f"{memory_uniform_mb:.2f}",
                "patch_count": patch_count,
                "source_label": desc.label,
                "source_kind": desc.kind,
                "classes_present": classes_present_str,
                "traversability_agreement": f"{trav_agreement:.4f}",
                "purity_20_40m": f"{(purity_20_40 if purity_20_40 is not None else 0.0):.4f}",
                "purity_40_60m": f"{(purity_40_60 if purity_40_60 is not None else 0.0):.4f}",
                "fine_cell_size": f"{grid.fine.cell_size:.3f}",
                "coarse_cell_size": f"{grid.coarse.cell_size:.3f}",
                "patch_cell_size": f"{(grid.patches[0].cell_size if grid.patches else 0.05):.3f}",
                "fine_radius": f"{grid.fine_radius:.1f}",
            })

        m_on = float(np.mean(rep_lats))
        runs_on_means.append(m_on)
        runs_on_data.append((rep_lats, rep_records, rep_class_tallies, rep_road_diag))
        print(f"  -> Pass {rep + 1}/{args.repeats} (Patches ON):  mean {m_on:.2f} ms ({1000.0/m_on:.1f} FPS)")

    # Select median run for Patches ON
    med_on_idx = int(np.argsort(runs_on_means)[len(runs_on_means) // 2])
    best_on_lats, best_records, best_class_tallies, best_road_diag = runs_on_data[med_on_idx]
    spread_on_ms = max(runs_on_means) - min(runs_on_means)
    spread_on_pct = (spread_on_ms / max(0.001, runs_on_means[med_on_idx])) * 100.0
    print(f"Patches ON:  Median mean = {runs_on_means[med_on_idx]:.2f} ms, spread = {spread_on_pct:.1f}%")
    if spread_on_pct > 20.0:
        print(f"WARNING: Patches ON spread across {args.repeats} runs is {spread_on_pct:.1f}% (> 20.0%)!")

    # Write metrics to CSV from the median run
    fieldnames = list(best_records[0].keys())
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_records)

    # Write per-class agreement CSV
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        cwriter = csv.writer(f)
        cwriter.writerow([
            "class_id",
            "occupied_cells",
            "matching_cells",
            "agreement_pct",
            "matching_cells_geom",
            "agreement_pct_geom",
            "is_by_construction",
        ])
        for cid in sorted(best_class_tallies.keys()):
            occ, match_sem, match_geom = best_class_tallies[cid]
            pct_sem = (float(match_sem) / max(1, occ) * 100.0) if occ > 0 else 0.0
            pct_geom = (float(match_geom) / max(1, occ) * 100.0) if occ > 0 else 0.0
            by_constr = "True" if cid in (2, 3, 4, 5, 6, 7) else "False"
            cwriter.writerow([cid, occ, match_sem, f"{pct_sem:.2f}", match_geom, f"{pct_geom:.2f}", by_constr])

    # Write road diagnostics CSV
    with open(road_diag_csv, "w", newline="", encoding="utf-8") as f:
        rwriter = csv.writer(f)
        rwriter.writerow(["category", "item", "count", "share_pct"])
        total_nd = max(1, sum(best_road_diag["zone"].values()))
        for cat in ["zone", "range", "cause"]:
            for item, count in best_road_diag[cat].items():
                pct = (float(count) / total_nd) * 100.0
                rwriter.writerow([cat, item, count, f"{pct:.2f}"])

    # Summary statistics
    mean_on_ms = float(np.mean(best_on_lats))
    p50_on_ms = float(np.percentile(best_on_lats, 50))
    p95_on_ms = float(np.percentile(best_on_lats, 95))
    p99_on_ms = float(np.percentile(best_on_lats, 99))
    mean_on_fps = 1000.0 / max(0.001, mean_on_ms)
    p95_on_fps = 1000.0 / max(0.001, p95_on_ms)
    p99_on_fps = 1000.0 / max(0.001, p99_on_ms)

    mean_off_ms = float(np.mean(best_off_lats))
    mean_off_fps = 1000.0 / max(0.001, mean_off_ms)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY (MEDIAN OF 3 RUNS):")
    print(f"  * Total Frames:            {len(best_records)}")
    print(f"  * Patches OFF Median Mean: {mean_off_ms:.2f} ms ({mean_off_fps:.1f} FPS) [spread: {spread_off_pct:.1f}%]")
    print(f"  * Patches ON Median Mean:  {mean_on_ms:.2f} ms ({mean_on_fps:.1f} FPS) [spread: {spread_on_pct:.1f}%]")
    print(f"  * Patches ON 95th %ile:    {p95_on_ms:.2f} ms ({p95_on_fps:.1f} FPS)")
    print(f"  * Patches ON 99th %ile:    {p99_on_ms:.2f} ms ({p99_on_fps:.1f} FPS)")
    print(f"  * Metrics saved to:        {metrics_csv.resolve()}")
    print(f"  * Class agreement:         {class_csv.resolve()}")
    print(f"  * Road diagnostics:        {road_diag_csv.resolve()}")
    print("=" * 80)

    if desc.kind == "real_full":
        print("\nCredit: SemanticKITTI, Behley et al., ICCV 2019, CC BY-NC-SA 4.0\n")


if __name__ == "__main__":
    main()
