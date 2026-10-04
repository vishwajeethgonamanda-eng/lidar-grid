from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

# Traversability state constants:
UNKNOWN = np.int8(0)       # empty cell (count == 0)
DRIVABLE = np.int8(1)      # safe ground/road
NON_DRIVABLE = np.int8(2)  # geometrically unsafe or prohibited ground
OBSTACLE = np.int8(3)      # obstacles / objects

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _traversability_kernel(
        count, z_min, z_max, z_sum, z_sq_sum, label_hist,
        cell_size, max_step, max_slope_deg, terrain_is_drivable, min_points, var_scale,
        state_out, conf_out,
    ):
        rows, cols = count.shape
        thresh_dz = np.float32(cell_size * np.tan(np.deg2rad(max_slope_deg)))

        z_m = np.zeros((rows, cols), dtype=np.float32)
        dom_label = np.zeros((rows, cols), dtype=np.int32)

        for r in range(rows):
            for c in range(cols):
                cnt = count[r, c]
                if cnt > 0:
                    inv_c = 1.0 / cnt
                    m = z_sum[r, c] * inv_c
                    z_m[r, c] = np.float32(m)
                    msq = z_sq_sum[r, c] * inv_c
                    v = np.float32(max(0.0, msq - m * m))
                    p_var = np.float32(1.0) / (np.float32(1.0) + v / np.float32(var_scale))
                    c_occ = np.float32(cnt)
                    c_ratio = min(np.float32(1.0), c_occ / np.float32(min_points))
                    conf_out[r, c] = c_ratio * p_var

                    best_lbl = 0
                    max_lbl_cnt = -1
                    for l in range(label_hist.shape[2]):
                        lh = label_hist[r, c, l]
                        if lh > max_lbl_cnt:
                            max_lbl_cnt = lh
                            best_lbl = l
                    dom_label[r, c] = best_lbl

        # Slope check: 4-neighbours
        too_steep = np.zeros((rows, cols), dtype=np.bool_)
        for r in range(rows):
            for c in range(cols):
                if count[r, c] > 0:
                    if r + 1 < rows and count[r + 1, c] > 0:
                        if abs(z_m[r + 1, c] - z_m[r, c]) > thresh_dz:
                            too_steep[r, c] = True
                            too_steep[r + 1, c] = True
                    if c + 1 < cols and count[r, c + 1] > 0:
                        if abs(z_m[r, c + 1] - z_m[r, c]) > thresh_dz:
                            too_steep[r, c] = True
                            too_steep[r, c + 1] = True

        for r in range(rows):
            for c in range(cols):
                cnt = count[r, c]
                if cnt == 0:
                    continue

                dl = dom_label[r, c]
                if dl == 3 or dl == 4 or dl == 5 or dl == 6:
                    state_out[r, c] = OBSTACLE
                    continue

                step_height = z_max[r, c] - z_min[r, c]
                too_high_step = step_height > max_step
                is_steep = too_steep[r, c]

                if terrain_is_drivable:
                    prohibited = (dl == 7)
                else:
                    prohibited = (dl == 7 or dl == 2)

                if too_high_step or is_steep or prohibited:
                    state_out[r, c] = NON_DRIVABLE
                    continue

                if terrain_is_drivable:
                    drivable = (dl == 1 or dl == 2)
                else:
                    drivable = (dl == 1)

                if drivable:
                    state_out[r, c] = DRIVABLE
                else:
                    state_out[r, c] = NON_DRIVABLE

    # Warmup
    _w_c = np.zeros((1, 1), dtype=np.int32)
    _w_f32 = np.zeros((1, 1), dtype=np.float32)
    _w_f64 = np.zeros((1, 1), dtype=np.float64)
    _w_h = np.zeros((1, 1, 1), dtype=np.int32)
    _w_s = np.zeros((1, 1), dtype=np.int8)
    _traversability_kernel(
        _w_c, _w_f32, _w_f32, _w_f64, _w_f64, _w_h,
        0.05, 0.1, 15.0, False, 3, 0.05, _w_s, _w_f32
    )


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

    if _HAS_NUMBA:
        _traversability_kernel(
            zone.count,
            zone.z_min,
            zone.z_max,
            zone.z_sum,
            zone.z_sq_sum,
            zone.label_hist,
            np.float32(zone.cell_size),
            np.float32(params.max_step),
            np.float32(params.max_slope_deg),
            bool(params.terrain_is_drivable),
            int(min_points),
            np.float32(params.var_scale),
            state,
            confidence,
        )
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
