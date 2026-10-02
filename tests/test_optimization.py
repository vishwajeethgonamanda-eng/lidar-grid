import numpy as np
import pytest
from eval_patches import generate_synthetic_scene_with_targets
from src.grid_engine import VarResGrid
from src.loader import make_synthetic_scene
from src.pipeline import process_frame
from src.traversability import (
    TraversabilityParams,
    TraversabilityResult,
    UNKNOWN,
    OBSTACLE,
    NON_DRIVABLE,
    DRIVABLE,
)


def unoptimized_zone_traversability(
    zone,
    params: TraversabilityParams,
    min_points: int,
) -> TraversabilityResult:
    """Original unoptimized reference implementation using float64 differences and rad2deg(arctan)."""
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

    dz_up[1:, :] = np.abs(z_m[1:, :] - z_m[:-1, :])
    valid_up[1:, :] = occupied[1:, :] & occupied[:-1, :]

    dz_down[:-1, :] = np.abs(z_m[:-1, :] - z_m[1:, :])
    valid_down[:-1, :] = occupied[:-1, :] & occupied[1:, :]

    dz_left[:, 1:] = np.abs(z_m[:, 1:] - z_m[:, :-1])
    valid_left[:, 1:] = occupied[:, 1:] & occupied[:, :-1]

    dz_right[:, :-1] = np.abs(z_m[:, :-1] - z_m[:, 1:])
    valid_right[:, :-1] = occupied[:, :-1] & occupied[:, 1:]

    dz_up = np.where(valid_up, dz_up, 0.0)
    dz_down = np.where(valid_down, dz_down, 0.0)
    dz_left = np.where(valid_left, dz_left, 0.0)
    dz_right = np.where(valid_right, dz_right, 0.0)

    max_dz = np.maximum(np.maximum(dz_up, dz_down), np.maximum(dz_left, dz_right))
    steepest_slope_deg = np.rad2deg(np.arctan(max_dz / zone.cell_size))
    too_steep_slope = (steepest_slope_deg > params.max_slope_deg) & candidates

    # 5. Rule e: Dominant label 7 or 2
    if params.terrain_is_drivable:
        prohibited_label = (dom_label == 7) & candidates
    else:
        prohibited_label = ((dom_label == 7) | (dom_label == 2)) & candidates

    is_non_drivable = too_high_step | too_steep_slope | prohibited_label
    state[is_non_drivable] = NON_DRIVABLE

    remaining = candidates & (~is_non_drivable)
    if params.terrain_is_drivable:
        is_drivable = ((dom_label == 1) | (dom_label == 2)) & remaining
    else:
        is_drivable = (dom_label == 1) & remaining

    state[is_drivable] = DRIVABLE
    state[remaining & (~is_drivable)] = NON_DRIVABLE

    return TraversabilityResult(state, confidence)


def unoptimized_process_frame_patches(xyz, labels, speed_mps=15.0, k_patches=4, params=None):
    """Reference implementation of process_frame with risk patches (full re-binning)."""
    from src.risk import compute_candidate_risks
    p = TraversabilityParams() if params is None else params
    init_grid = VarResGrid(n_classes=8, fine_radius=10.0, forward_offset=0.0)
    init_grid.add_points(xyz, labels)

    cands = compute_candidate_risks(init_grid, speed_mps=speed_mps, k=k_patches)
    patch_centers = [(c.x, c.y) for c in cands]

    patched_grid = VarResGrid(n_classes=8, fine_radius=10.0, forward_offset=0.0, patches=patch_centers)
    patched_grid.add_points(xyz, labels)

    fine_res = unoptimized_zone_traversability(patched_grid.fine, p, 3)
    coarse_res = unoptimized_zone_traversability(patched_grid.coarse, p, 5)
    patch_res = [unoptimized_zone_traversability(pz, p, 3) for pz in patched_grid.patches]

    return patched_grid, fine_res, coarse_res, patch_res


def test_optimized_vs_original_synthetic_scenes():
    """Verifies that optimized process_frame produces strictly identical outputs across frames."""
    params = TraversabilityParams(max_step=0.10, max_slope_deg=15.0)

    for seed in [42, 101, 777]:
        xyz, labels = generate_synthetic_scene_with_targets(seed=seed)

        # 1. Run reference implementation
        ref_grid, ref_fine_res, ref_coarse_res, ref_patch_res = unoptimized_process_frame_patches(
            xyz, labels, speed_mps=15.0, k_patches=4, params=params
        )

        # 2. Run optimized pipeline
        opt_result = process_frame(
            xyz, labels, risk_patches=True, speed_mps=15.0, k_patches=4, params=params
        )
        opt_grid = opt_result["grid"]
        opt_trav = opt_result["traversability"]

        # Assert stats equality
        assert opt_grid.stats() == ref_grid.stats(), f"Stats mismatch for seed {seed}"

        # Assert fine zone equality
        np.testing.assert_array_equal(opt_grid.fine.count, ref_grid.fine.count)
        np.testing.assert_array_equal(opt_grid.fine.z_min, ref_grid.fine.z_min)
        np.testing.assert_array_equal(opt_grid.fine.z_max, ref_grid.fine.z_max)
        np.testing.assert_array_equal(opt_grid.fine.label_hist, ref_grid.fine.label_hist)
        np.testing.assert_array_equal(opt_trav.fine.state, ref_fine_res.state)
        np.testing.assert_allclose(opt_trav.fine.confidence, ref_fine_res.confidence, atol=1e-5)

        # Assert coarse zone equality
        np.testing.assert_array_equal(opt_grid.coarse.count, ref_grid.coarse.count)
        np.testing.assert_array_equal(opt_grid.coarse.z_min, ref_grid.coarse.z_min)
        np.testing.assert_array_equal(opt_grid.coarse.z_max, ref_grid.coarse.z_max)
        np.testing.assert_array_equal(opt_grid.coarse.label_hist, ref_grid.coarse.label_hist)
        np.testing.assert_array_equal(opt_trav.coarse.state, ref_coarse_res.state)
        np.testing.assert_allclose(opt_trav.coarse.confidence, ref_coarse_res.confidence, atol=1e-5)

        # Assert patch zones equality
        assert len(opt_grid.patches) == len(ref_grid.patches)
        for idx in range(len(opt_grid.patches)):
            np.testing.assert_array_equal(opt_grid.patches[idx].count, ref_grid.patches[idx].count)
            np.testing.assert_array_equal(opt_grid.patches[idx].z_min, ref_grid.patches[idx].z_min)
            np.testing.assert_array_equal(opt_grid.patches[idx].z_max, ref_grid.patches[idx].z_max)
            np.testing.assert_array_equal(opt_grid.patches[idx].label_hist, ref_grid.patches[idx].label_hist)
            np.testing.assert_array_equal(opt_trav.patches[idx].state, ref_patch_res[idx].state)
            np.testing.assert_allclose(opt_trav.patches[idx].confidence, ref_patch_res[idx].confidence, atol=1e-5)


def test_optimized_vs_original_patches_off():
    """Verifies that with patches off, results are strictly identical to the reference."""
    scan, labels = make_synthetic_scene(seed=999)
    xyz = scan[:, :3]
    params = TraversabilityParams(max_step=0.10, max_slope_deg=15.0)

    # Reference
    ref_grid = VarResGrid(n_classes=8)
    ref_grid.add_points(xyz, labels)
    ref_fine = unoptimized_zone_traversability(ref_grid.fine, params, 3)
    ref_coarse = unoptimized_zone_traversability(ref_grid.coarse, params, 5)

    # Optimized process_frame
    opt_result = process_frame(xyz, labels, risk_patches=False, params=params)
    opt_trav = opt_result["traversability"]

    np.testing.assert_array_equal(opt_trav.fine.state, ref_fine.state)
    np.testing.assert_allclose(opt_trav.fine.confidence, ref_fine.confidence, atol=1e-5)
    np.testing.assert_array_equal(opt_trav.coarse.state, ref_coarse.state)
    np.testing.assert_allclose(opt_trav.coarse.confidence, ref_coarse.confidence, atol=1e-5)
