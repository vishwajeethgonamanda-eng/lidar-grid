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
    _compute_zone_traversability,
)
import src.traversability as trav_mod


def unoptimized_zone_traversability(
    zone,
    params: TraversabilityParams,
    min_points: int,
) -> TraversabilityResult:
    """NumPy reference implementation using the NumPy traversability path."""
    orig_numba = trav_mod._HAS_NUMBA
    try:
        trav_mod._HAS_NUMBA = False
        return _compute_zone_traversability(zone, params, min_points)
    finally:
        trav_mod._HAS_NUMBA = orig_numba


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
