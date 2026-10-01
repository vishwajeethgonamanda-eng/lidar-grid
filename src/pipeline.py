import time
from typing import Any, Dict, Optional, Union
import numpy as np

from src.grid_engine import VarResGrid
from src.traversability import compute_traversability, TraversabilityParams, TraversabilityMap


def process_frame(
    xyz: np.ndarray,
    labels: np.ndarray,
    grid: Optional[VarResGrid] = None,
    params: Optional[Union[TraversabilityParams, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Processes a single LiDAR frame through the variable-resolution grid and traversability engine.

    Resets the grid at the start of each frame.

    Args:
        xyz: (N, 3) float array of LiDAR point coordinates (x, y, z in metres).
        labels: (N,) int array of point semantic labels.
        grid: Optional pre-allocated VarResGrid instance. If None, a new grid with n_classes=8 is created.
        params: Optional TraversabilityParams or parameter dict.

    Returns:
        dict containing:
            - "grid": VarResGrid instance
            - "traversability": TraversabilityMap containing fine and coarse results
            - "timings": dict with "add_points_ms", "compute_traversability_ms", "total_ms"
            - "stats": dict with "in_fine", "in_coarse", "out_of_range", "total_input"
            - "memory_bytes": int
            - "uniform_equivalent_bytes": int
    """
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
            "total_ms": float(total_ms),
        },
        "stats": stats,
        "memory_bytes": int(grid.memory_bytes()),
        "uniform_equivalent_bytes": int(grid.uniform_equivalent_bytes()),
    }
