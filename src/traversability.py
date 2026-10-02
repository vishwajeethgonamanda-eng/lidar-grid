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
    """Traversability outputs across fine, coarse, and optional focus patch zones."""

    def __init__(
        self,
        fine: TraversabilityResult,
        coarse: TraversabilityResult,
        patches: Optional[List[TraversabilityResult]] = None,
    ):
        self.fine = fine
        self.coarse = coarse
        self.patches = patches if patches is not None else []

    def __iter__(self) -> Iterator[TraversabilityResult]:
        yield self.fine
        yield self.coarse

    def __getitem__(self, item: str) -> Union[TraversabilityResult, List[TraversabilityResult]]:
        if item == "fine":
            return self.fine
        if item == "coarse":
            return self.coarse
        if item == "patches":
            return self.patches
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
    c_occ = count[occupied].astype(np.float32)
    inv_c = 1.0 / count[occupied]
    mean = zone.z_sum[occupied] * inv_c
    mean_sq = zone.z_sq_sum[occupied] * inv_c
    var_safe = np.maximum(0.0, mean_sq - mean * mean).astype(np.float32)
    p_var = 1.0 / (1.0 + var_safe / np.float32(params.var_scale))
    confidence[occupied] = np.clip(c_occ / float(min_points), 0.0, 1.0) * p_var

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
    thresh_dz = np.float32(zone.cell_size * np.tan(np.deg2rad(params.max_slope_deg)))
    inv_count = np.zeros(shape, dtype=np.float32)
    inv_count[occupied] = 1.0 / count[occupied]
    z_m = (zone.z_sum * inv_count).astype(np.float32)

    diff_v = np.abs(z_m[1:, :] - z_m[:-1, :])
    both_v = (diff_v > thresh_dz) & occupied[1:, :] & occupied[:-1, :]

    diff_h = np.abs(z_m[:, 1:] - z_m[:, :-1])
    both_h = (diff_h > thresh_dz) & occupied[:, 1:] & occupied[:, :-1]

    too_steep = np.zeros(shape, dtype=bool)
    too_steep[1:, :] |= both_v
    too_steep[:-1, :] |= both_v
    too_steep[:, 1:] |= both_h
    too_steep[:, :-1] |= both_h
    too_steep_slope = too_steep & candidates

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
    cached_fine: Optional[TraversabilityResult] = None,
) -> TraversabilityMap:
    """Computes 2.5D traversability states and confidences for VarResGrid zones.

    Args:
        grid: Instance of VarResGrid containing fine and coarse zones.
        params: Optional TraversabilityParams or dict of parameters.
        cached_fine: Optional precomputed TraversabilityResult for the fine zone to avoid recomputation.

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

    fine_res = cached_fine if cached_fine is not None else _compute_zone_traversability(grid.fine, p, min_fine)
    coarse_res = _compute_zone_traversability(grid.coarse, p, min_coarse)

    patch_results = []
    if hasattr(grid, "patches") and grid.patches:
        for p_zone in grid.patches:
            patch_results.append(_compute_zone_traversability(p_zone, p, min_fine))

    return TraversabilityMap(fine=fine_res, coarse=coarse_res, patches=patch_results)
