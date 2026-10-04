import numpy as np
import pytest
from src.grid_engine import VarResGrid
from src.loader import make_synthetic_scene
from src.traversability import (
    compute_traversability,
    TraversabilityParams,
    UNKNOWN,
    DRIVABLE,
    NON_DRIVABLE,
    OBSTACLE,
)


def test_empty_cell_is_unknown():
    grid = VarResGrid(n_classes=8)
    res = compute_traversability(grid)

    # All cells empty -> state 0 (UNKNOWN) and confidence 0.0
    assert np.all(res.fine.state == UNKNOWN)
    assert np.all(res.fine.confidence == 0.0)
    assert np.all(res.coarse.state == UNKNOWN)
    assert np.all(res.coarse.confidence == 0.0)


def test_output_shapes_match():
    grid = VarResGrid(n_classes=8)
    res = compute_traversability(grid)

    assert res.fine.state.shape == grid.fine.count.shape
    assert res.fine.confidence.shape == grid.fine.count.shape
    assert res.coarse.state.shape == grid.coarse.count.shape
    assert res.coarse.confidence.shape == grid.coarse.count.shape

    assert res.fine.state.dtype == np.int8
    assert res.fine.confidence.dtype == np.float32
    assert res.coarse.state.dtype == np.int8
    assert res.coarse.confidence.dtype == np.float32


def test_flat_road_patch_is_drivable():
    grid = VarResGrid(n_classes=8)
    # Add flat road points (z=0, label=1) in fine grid (e.g. at x=2, y=2)
    # 20 points in the same cell
    n_pts = 20
    x = np.full(n_pts, 2.02, dtype=np.float32)
    y = np.full(n_pts, 2.02, dtype=np.float32)
    z = np.zeros(n_pts, dtype=np.float32)
    labels = np.full(n_pts, 1, dtype=np.int64)
    grid.add_points(np.column_stack([x, y, z]), labels)

    res = compute_traversability(grid)

    # col = floor((2.02 + 10) / 0.05) = 240, row = 240
    col = int(np.floor((2.02 + 10.0) / 0.05))
    row = int(np.floor((2.02 + 10.0) / 0.05))
    assert res.fine.state[row, col] == DRIVABLE
    assert res.fine.confidence[row, col] > 0.8


def test_step_height_overrides_road_label():
    grid = VarResGrid(n_classes=8)
    # Cell with label=1 (road), but points at z=0.0 and z=0.15 (15 cm step)
    pts = np.array([
        [1.02, 1.02, 0.00],
        [1.02, 1.02, 0.00],
        [1.02, 1.02, 0.15],
        [1.02, 1.02, 0.15],
    ], dtype=np.float32)
    labels = np.array([1, 1, 1, 1], dtype=np.int64)
    grid.add_points(pts, labels)

    res = compute_traversability(grid, TraversabilityParams(max_step=0.10))

    col = int(np.floor((1.02 + 10.0) / 0.05))
    row = int(np.floor((1.02 + 10.0) / 0.05))
    assert res.fine.state[row, col] == NON_DRIVABLE


def test_slope_steep_vs_gentle():
    params = TraversabilityParams(max_slope_deg=15.0, slope_baseline_m=0.05)

    # 1. Gentle ramp (e.g., 5 degrees):
    # dx = 0.05 m, dz for 5 deg is 0.05 * tan(5 deg) ~ 0.0044 m
    grid_gentle = VarResGrid(n_classes=8)
    pts_g1 = np.array([[2.02, 2.02, 0.00]] * 10, dtype=np.float32)  # cell A
    pts_g2 = np.array([[2.07, 2.02, 0.0044]] * 10, dtype=np.float32) # neighbor cell B (dx = 0.05 m)
    grid_gentle.add_points(np.vstack([pts_g1, pts_g2]), np.full(20, 1, dtype=np.int64))
    res_gentle = compute_traversability(grid_gentle, params)

    col_a = int(np.floor((2.02 + 10.0) / 0.05))
    row_a = int(np.floor((2.02 + 10.0) / 0.05))
    assert res_gentle.fine.state[row_a, col_a] == DRIVABLE

    # 2. Steep ramp exceeding both 15 degrees and min_slope_dz_m (3 cm):
    # dx = 0.05 m, dz = 0.035 m (3.5 cm >= 3 cm), atan(0.035 / 0.05) ~ 35.0 deg > 15 deg
    grid_steep = VarResGrid(n_classes=8)
    pts_s1 = np.array([[2.02, 2.02, 0.00]] * 10, dtype=np.float32)
    pts_s2 = np.array([[2.07, 2.02, 0.035]] * 10, dtype=np.float32)
    grid_steep.add_points(np.vstack([pts_s1, pts_s2]), np.full(20, 1, dtype=np.int64))
    res_steep = compute_traversability(grid_steep, params)

    assert res_steep.fine.state[row_a, col_a] == NON_DRIVABLE

    # 3. Steep slope angle (>15 deg) but dz < 3 cm is NOT rejected under default min_slope_dz_m = 0.03:
    grid_sub3cm = VarResGrid(n_classes=8)
    pts_sub1 = np.array([[2.02, 2.02, 0.00]] * 10, dtype=np.float32)
    pts_sub2 = np.array([[2.07, 2.02, 0.025]] * 10, dtype=np.float32)  # dz = 2.5 cm < 3 cm, angle ~ 26.5 deg > 15 deg
    grid_sub3cm.add_points(np.vstack([pts_sub1, pts_sub2]), np.full(20, 1, dtype=np.int64))
    res_sub3cm = compute_traversability(grid_sub3cm, params)
    assert res_sub3cm.fine.state[row_a, col_a] == DRIVABLE

    # If min_slope_dz_m = 0.02 is configured, that same 2.5 cm ramp is rejected:
    params_low_dz = TraversabilityParams(max_slope_deg=15.0, slope_baseline_m=0.05, min_slope_dz_m=0.02)
    res_low_dz = compute_traversability(grid_sub3cm, params_low_dz)
    assert res_low_dz.fine.state[row_a, col_a] == NON_DRIVABLE


def test_person_dominated_cell_is_obstacle():
    grid = VarResGrid(n_classes=8)
    # Cell with label=5 (person)
    pts = np.array([[3.02, 1.02, 1.0]] * 5, dtype=np.float32)
    labels = np.full(5, 5, dtype=np.int64)
    grid.add_points(pts, labels)

    res = compute_traversability(grid)
    col = int(np.floor((3.02 + 10.0) / 0.05))
    row = int(np.floor((1.02 + 10.0) / 0.05))
    assert res.fine.state[row, col] == OBSTACLE


def test_confidence_point_count_comparison():
    grid = VarResGrid(n_classes=8)
    # Cell A with 1 point
    grid.add_points(np.array([[2.02, 2.02, 0.0]], dtype=np.float32), np.array([1], dtype=np.int64))
    # Cell B with 50 points
    grid.add_points(np.array([[4.02, 4.02, 0.0]] * 50, dtype=np.float32), np.full(50, 1, dtype=np.int64))

    res = compute_traversability(grid)

    col_a = int(np.floor((2.02 + 10.0) / 0.05))
    row_a = int(np.floor((2.02 + 10.0) / 0.05))
    col_b = int(np.floor((4.02 + 10.0) / 0.05))
    row_b = int(np.floor((4.02 + 10.0) / 0.05))

    conf_1pt = res.fine.confidence[row_a, col_a]
    conf_50pt = res.fine.confidence[row_b, col_b]
    assert conf_1pt < conf_50pt


def test_synthetic_kerb_seam_both_sides():
    scan, labels = make_synthetic_scene(seed=42)
    grid = VarResGrid(n_classes=8)
    grid.add_points(scan[:, :3], labels)

    res = compute_traversability(grid, TraversabilityParams(max_step=0.10))

    # Fine side kerb cells: x in [9.7, 10.0), y around 0.0
    # Fine grid col ~ floor((9.7+10)/0.05)=394..399, row ~ 200
    fine_kerb_states = res.fine.state[196:204, 394:400]
    assert np.any(fine_kerb_states == NON_DRIVABLE), "Fine side kerb must have non-drivable cells"

    # Coarse side kerb cells: x in [10.0, 10.3], y around 0.0
    # Coarse grid col ~ floor((10.0+100)/0.5)=220, row ~ 200
    coarse_kerb_states = res.coarse.state[198:202, 220:222]
    assert np.any(coarse_kerb_states == NON_DRIVABLE), "Coarse side kerb must have non-drivable cells"


def test_flat_plane_with_2cm_noise_drivable():
    """Flat ground plane with 2 cm range noise must achieve >= 98% DRIVABLE cells."""
    rng = np.random.default_rng(42)
    n_pts = 100000
    x = rng.uniform(-8.0, 8.0, size=n_pts).astype(np.float32)
    y = rng.uniform(-8.0, 8.0, size=n_pts).astype(np.float32)
    z = rng.normal(0.0, 0.02, size=n_pts).astype(np.float32)
    labels = np.ones(n_pts, dtype=np.int64)

    grid = VarResGrid(n_classes=8)
    grid.add_points(np.column_stack([x, y, z]), labels)
    res = compute_traversability(grid, TraversabilityParams())

    fine_occ = grid.fine.count > 0
    fine_road = fine_occ & (grid.fine.dominant_label == 1)
    fine_driv = fine_road & (res.fine.state == DRIVABLE)
    driv_pct = np.count_nonzero(fine_driv) / np.count_nonzero(fine_road) * 100.0
    assert driv_pct >= 98.0, f"Expected >= 98% drivable on flat plane + 2 cm noise, got {driv_pct:.2f}%"


def test_ramp_20deg_nondrivable():
    """A 20 degree ramp >= 8 m long must have >= 90% cells marked NON_DRIVABLE."""
    rng = np.random.default_rng(42)
    x_ramp = rng.uniform(1.0, 9.0, size=80000).astype(np.float32)
    y_ramp = rng.uniform(-1.0, 1.0, size=80000).astype(np.float32)
    z_ramp = (x_ramp * np.tan(np.deg2rad(20.0))).astype(np.float32)
    labels_ramp = np.ones(len(x_ramp), dtype=np.int64)

    grid_ramp = VarResGrid(n_classes=8)
    grid_ramp.add_points(np.column_stack([x_ramp, y_ramp, z_ramp]), labels_ramp)
    res_ramp = compute_traversability(grid_ramp, TraversabilityParams())

    ramp_occ = grid_ramp.fine.count > 0
    ramp_road = ramp_occ & (grid_ramp.fine.dominant_label == 1)
    ramp_nd = ramp_road & (res_ramp.fine.state == NON_DRIVABLE)
    nd_pct = np.count_nonzero(ramp_nd) / np.count_nonzero(ramp_road) * 100.0
    assert nd_pct >= 90.0, f"Expected >= 90% non-drivable on 20 deg ramp, got {nd_pct:.2f}%"


def test_kerb_12cm_flagged_by_step_rule():
    """A 12 cm kerb step must be flagged NON_DRIVABLE by the step height rule."""
    grid_kerb = VarResGrid(n_classes=8)
    pts_kerb = np.array([
        [2.02, 2.02, 0.00],
        [2.02, 2.02, 0.00],
        [2.02, 2.02, 0.12],
        [2.02, 2.02, 0.12],
    ], dtype=np.float32)
    grid_kerb.add_points(pts_kerb, np.array([1, 1, 1, 1], dtype=np.int64))
    res_kerb = compute_traversability(grid_kerb, TraversabilityParams(max_step=0.10))
    col = int(np.floor((2.02 + 10.0) / 0.05))
    row = int(np.floor((2.02 + 10.0) / 0.05))
    assert res_kerb.fine.state[row, col] == NON_DRIVABLE


def test_numba_vs_numpy_parity():
    """Verifies complete parity between Numba kernel and NumPy fallback."""
    import src.traversability as trav_mod
    scan, labels = make_synthetic_scene(seed=42)
    grid = VarResGrid(n_classes=8)
    grid.add_points(scan[:, :3], labels)

    params = TraversabilityParams()
    res_nb = compute_traversability(grid, params)

    orig_nb = trav_mod._HAS_NUMBA
    try:
        trav_mod._HAS_NUMBA = False
        res_np = compute_traversability(grid, params)
    finally:
        trav_mod._HAS_NUMBA = orig_nb

    np.testing.assert_array_equal(res_nb.fine.state, res_np.fine.state)
    np.testing.assert_allclose(res_nb.fine.confidence, res_np.fine.confidence, atol=1e-5)
    np.testing.assert_array_equal(res_nb.coarse.state, res_np.coarse.state)
    np.testing.assert_allclose(res_nb.coarse.confidence, res_np.coarse.confidence, atol=1e-5)
    assert len(res_nb.patches) == len(res_np.patches)
    for p_nb, p_np in zip(res_nb.patches, res_np.patches):
        np.testing.assert_array_equal(p_nb.state, p_np.state)
        np.testing.assert_allclose(p_nb.confidence, p_np.confidence, atol=1e-5)
