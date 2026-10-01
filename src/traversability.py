from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np

# Traversability state constants:
UNKNOWN = np.int8(0)       # empty cell (count == 0)
DRIVABLE = np.int8(1)      # safe ground/road
NON_DRIVABLE = np.int8(2)  # geometrically unsafe or prohibited ground
OBSTACLE = np.int8(3)      # obstacles / objects


@dataclass
class TraversabilityParams:
    """Parameters for traversability computation.

    Attributes:
        max_step: Maximum allowed vertical step height (z_max - z_min) in metres.
        max_slope_deg: Maximum allowed slope in degrees relative to occupied 4-neighbours.
        terrain_is_drivable: If True, terrain (label 2) is considered drivable when flat.
        min_points_fine: Number of points required in fine zone for full point-count confidence.
        min_points_coarse: Number of points required in coarse zone for full point-count confidence.
        var_scale: Reference vertical variance scale (m^2) for variance penalty calculation.
    """
    max_step: float = 0.10
    max_slope_deg: float = 15.0
    terrain_is_drivable: bool = False
    min_points_fine: int = 3
    min_points_coarse: int = 5
    var_scale: float = 0.05


class TraversabilityResult:
    """Traversability outputs (state and confidence) for a single grid zone."""

    def __init__(self, state: np.ndarray, confidence: np.ndarray):
        self.state = np.asarray(state, dtype=np.int8)
        self.confidence = np.asarray(confidence, dtype=np.float32)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.state
        yield self.confidence

    def __getitem__(self, item: Union[str, int]) -> np.ndarray:
        if item in ("state", 0):
            return self.state
        if item in ("confidence", 1):
            return self.confidence
        raise KeyError(f"Invalid key for TraversabilityResult: {item}")


class TraversabilityMap:
    """Traversability outputs across fine and coarse zones."""

    def __init__(self, fine: TraversabilityResult, coarse: TraversabilityResult):
        self.fine = fine
        self.coarse = coarse

    def __iter__(self) -> Iterator[TraversabilityResult]:
        yield self.fine
        yield self.coarse

    def __getitem__(self, item: str) -> TraversabilityResult:
        if item == "fine":
            return self.fine
        if item == "coarse":
            return self.coarse
        raise KeyError(f"Invalid key for TraversabilityMap: {item}")


def _compute_zone_traversability(
    zone: Any,
    params: TraversabilityParams,
    min_points: int,
) -> TraversabilityResult:
    """Vectorized traversability evaluation for a single grid zone.

    Confidence formula:
        confidence = clip(count / min_points, 0.0, 1.0) * (1.0 / (1.0 + max(0, z_var) / var_scale))
        Empty cells (count == 0) receive confidence 0.0.
    """
    count = zone.count
    shape = count.shape
    occupied = count > 0

    state = np.full(shape, UNKNOWN, dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)

    if not np.any(occupied):
        return TraversabilityResult(state, confidence)

    # 1. Compute Confidence
    c_count = np.clip(count.astype(np.float32) / float(min_points), 0.0, 1.0)
    z_var_safe = np.maximum(0.0, zone.z_var.astype(np.float32))
    p_var = 1.0 / (1.0 + z_var_safe / float(params.var_scale))
    confidence[occupied] = (c_count * p_var)[occupied]

    # 2. Rule b: Dominant label in {3, 4, 5, 6} -> obstacle (3)
    dom_label = zone.dominant_label
    is_obstacle = np.isin(dom_label, [3, 4, 5, 6]) & occupied
    state[is_obstacle] = OBSTACLE

    # Remaining candidate cells to evaluate for drivability
    candidates = occupied & (~is_obstacle)

    # 3. Rule c: Step height = z_max - z_min > max_step -> non-drivable (2)
    step_height = zone.z_max - zone.z_min
    too_high_step = (step_height > params.max_step) & candidates

    # 4. Rule d: Slope against 4 non-empty neighbours -> non-drivable (2)
    z_m = zone.z_mean
    dz_up = np.zeros(shape, dtype=np.float64)
    dz_down = np.zeros(shape, dtype=np.float64)
    dz_left = np.zeros(shape, dtype=np.float64)
    dz_right = np.zeros(shape, dtype=np.float64)

    valid_up = np.zeros(shape, dtype=bool)
    valid_down = np.zeros(shape, dtype=bool)
    valid_left = np.zeros(shape, dtype=bool)
    valid_right = np.zeros(shape, dtype=bool)

    # Up: (r, c) vs (r-1, c)
    dz_up[1:, :] = np.abs(z_m[1:, :] - z_m[:-1, :])
    valid_up[1:, :] = occupied[1:, :] & occupied[:-1, :]

    # Down: (r, c) vs (r+1, c)
    dz_down[:-1, :] = np.abs(z_m[:-1, :] - z_m[1:, :])
    valid_down[:-1, :] = occupied[:-1, :] & occupied[1:, :]

    # Left: (r, c) vs (r, c-1)
    dz_left[:, 1:] = np.abs(z_m[:, 1:] - z_m[:, :-1])
    valid_left[:, 1:] = occupied[:, 1:] & occupied[:, :-1]

    # Right: (r, c) vs (r, c+1)
    dz_right[:, :-1] = np.abs(z_m[:, :-1] - z_m[:, 1:])
    valid_right[:, :-1] = occupied[:, :-1] & occupied[:, 1:]

    dz_up = np.where(valid_up, dz_up, 0.0)
    dz_down = np.where(valid_down, dz_down, 0.0)
    dz_left = np.where(valid_left, dz_left, 0.0)
    dz_right = np.where(valid_right, dz_right, 0.0)

    max_dz = np.maximum(np.maximum(dz_up, dz_down), np.maximum(dz_left, dz_right))
    steepest_slope_deg = np.rad2deg(np.arctan(max_dz / zone.cell_size))
    too_steep_slope = (steepest_slope_deg > params.max_slope_deg) & candidates

    # 5. Rule e: Dominant label 7 (non-drivable ground) or 2 (terrain unless terrain_is_drivable)
    if params.terrain_is_drivable:
        prohibited_label = (dom_label == 7) & candidates
    else:
        prohibited_label = ((dom_label == 7) | (dom_label == 2)) & candidates

    # Mark non-drivable
    is_non_drivable = too_high_step | too_steep_slope | prohibited_label
    state[is_non_drivable] = NON_DRIVABLE

    # 6. Rule f: Otherwise dominant label 1 (road) -> drivable (1). Anything else -> non-drivable (2)
    remaining = candidates & (~is_non_drivable)
    if params.terrain_is_drivable:
        is_drivable = ((dom_label == 1) | (dom_label == 2)) & remaining
    else:
        is_drivable = (dom_label == 1) & remaining

    state[is_drivable] = DRIVABLE
    state[remaining & (~is_drivable)] = NON_DRIVABLE

    return TraversabilityResult(state, confidence)


def compute_traversability(
    grid: Any,
    params: Optional[Union[TraversabilityParams, Dict[str, Any]]] = None,
) -> TraversabilityMap:
    """Computes 2.5D traversability states and confidences for VarResGrid zones.

    Args:
        grid: Instance of VarResGrid containing fine and coarse zones.
        params: Optional TraversabilityParams or dict of parameters.

    Returns:
        TraversabilityMap containing `fine` and `coarse` TraversabilityResult objects,
        each with `state` (int8) and `confidence` (float32).
    """
    if params is None:
        p = TraversabilityParams()
    elif isinstance(params, dict):
        p = TraversabilityParams(**params)
    else:
        p = params

    # Resolve min_points per zone
    min_fine = getattr(p, "min_points_fine", getattr(p, "min_points", 3))
    min_coarse = getattr(p, "min_points_coarse", getattr(p, "min_points", 5))

    fine_res = _compute_zone_traversability(grid.fine, p, min_fine)
    coarse_res = _compute_zone_traversability(grid.coarse, p, min_coarse)

    return TraversabilityMap(fine=fine_res, coarse=coarse_res)
