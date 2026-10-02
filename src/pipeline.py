import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.grid_engine import VarResGrid
from src.traversability import compute_traversability, TraversabilityParams, TraversabilityMap
from src.risk import compute_candidate_risks, CandidateObject


def process_frame(
    xyz: np.ndarray,
    labels: np.ndarray,
    grid: Optional[VarResGrid] = None,
    params: Optional[Union[TraversabilityParams, Dict[str, Any]]] = None,
    risk_patches: bool = False,
    speed_mps: float = 10.0,
    k_patches: int = 4,
) -> Dict[str, Any]:
    """Processes a single LiDAR frame through the variable-resolution grid and traversability engine.

    Resets the grid at the start of each frame.

    Args:
        xyz: (N, 3) float array of LiDAR point coordinates (x, y, z in metres).
        labels: (N,) int array of point semantic labels.
        grid: Optional pre-allocated VarResGrid instance. If None, a new grid with n_classes=8 is created.
        params: Optional TraversabilityParams or parameter dict.
        risk_patches: If True, identifies high-risk candidate objects outside fine zone,
                      allocates 64x64 focus patches, and re-bins points with focus patches.
        speed_mps: Current vehicle speed in m/s used for risk horizon evaluation.
        k_patches: Maximum number of focus patches to allocate.

    Returns:
        dict containing:
            - "grid": VarResGrid instance
            - "traversability": TraversabilityMap containing fine, coarse, and patch results
            - "timings": dict with "add_points_ms", "compute_traversability_ms", "risk_time_ms", "total_ms"
            - "stats": dict with "in_fine", "in_patch", "in_coarse", "out_of_range", "total_input"
            - "memory_bytes": int
            - "uniform_equivalent_bytes": int
            - "extra_memory_bytes": int
            - "candidates": List[CandidateObject]
    """
    if not risk_patches:
        if grid is None:
            grid = VarResGrid(n_classes=8)
        else:
            grid.reset()

        # 1. Point insertion with timing
        t0 = time.perf_counter()
        grid.add_points(xyz, labels)
        t1 = time.perf_counter()
        add_points_ms = (t1 - t0) * 1000.0

        # 2. Traversability analysis with timing
        t2 = time.perf_counter()
        trav = compute_traversability(grid, params=params)
        t3 = time.perf_counter()
        compute_traversability_ms = (t3 - t2) * 1000.0

        total_ms = (t3 - t0) * 1000.0
        stats = grid.stats()

        return {
            "grid": grid,
            "traversability": trav,
            "timings": {
                "add_points_ms": float(add_points_ms),
                "compute_traversability_ms": float(compute_traversability_ms),
                "risk_time_ms": 0.0,
                "total_ms": float(total_ms),
            },
            "stats": stats,
            "memory_bytes": int(grid.memory_bytes()),
            "uniform_equivalent_bytes": int(grid.uniform_equivalent_bytes()),
            "extra_memory_bytes": 0,
            "candidates": [],
        }

    # risk_patches is True:
    # 1. Pass 1: Build initial grid without patches to evaluate coarse risk
    fine_radius = grid.fine_radius if grid is not None else 10.0
    forward_offset = grid.forward_offset if grid is not None else 0.0
    n_classes = grid.n_classes if grid is not None else 8

    init_grid = VarResGrid(n_classes=n_classes, fine_radius=fine_radius, forward_offset=forward_offset)
    t_pass1_0 = time.perf_counter()
    init_grid.add_points(xyz, labels)
    t_pass1_1 = time.perf_counter()
    pass1_ms = (t_pass1_1 - t_pass1_0) * 1000.0

    # 2. Compute risk and select top K focus patch candidates
    t_risk_0 = time.perf_counter()
    candidates = compute_candidate_risks(init_grid, speed_mps=speed_mps, k=k_patches)
    patch_centers = [(c.x, c.y) for c in candidates]
    t_risk_1 = time.perf_counter()
    risk_time_ms = (t_risk_1 - t_risk_0) * 1000.0

    # 3. Pass 2: Re-bin points with focus patches
    patched_grid = VarResGrid(
        n_classes=n_classes,
        fine_radius=fine_radius,
        forward_offset=forward_offset,
        patches=patch_centers,
    )
    t_add_0 = time.perf_counter()
    patched_grid.add_points(xyz, labels)
    t_add_1 = time.perf_counter()
    add_points_ms = (t_add_1 - t_add_0) * 1000.0

    # 4. Traversability analysis across fine, coarse, and patch zones
    t_trav_0 = time.perf_counter()
    trav = compute_traversability(patched_grid, params=params)
    t_trav_1 = time.perf_counter()
    compute_traversability_ms = (t_trav_1 - t_trav_0) * 1000.0

    extra_time_ms = pass1_ms + risk_time_ms
    total_ms = pass1_ms + risk_time_ms + add_points_ms + compute_traversability_ms
    extra_memory_bytes = sum(p.nbytes for p in patched_grid.patches)
    stats = patched_grid.stats()

    return {
        "grid": patched_grid,
        "traversability": trav,
        "timings": {
            "add_points_ms": float(add_points_ms),
            "compute_traversability_ms": float(compute_traversability_ms),
            "risk_time_ms": float(risk_time_ms),
            "extra_time_ms": float(extra_time_ms),
            "total_ms": float(total_ms),
        },
        "stats": stats,
        "memory_bytes": int(patched_grid.memory_bytes()),
        "uniform_equivalent_bytes": int(patched_grid.uniform_equivalent_bytes()),
        "extra_memory_bytes": int(extra_memory_bytes),
        "candidates": candidates,
    }
