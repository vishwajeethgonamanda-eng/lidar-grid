"""Unit tests for the rasterization engine in src/raster.py."""
import time
import numpy as np
import pytest

from src.grid_engine import VarResGrid
from src.traversability import (
    TraversabilityParams,
    compute_traversability,
    DRIVABLE,
    OBSTACLE,
)
from src.pipeline import process_frame
from src.raster import (
    render_grid_map,
    render_raw_points,
    render_25d_fast,
    DRIVABLE_COLOR,
    OBSTACLE_COLOR,
    CAR_BODY_COLOR,
    BG_COLOR,
)


def _make_dummy_grid_and_trav():
    grid = VarResGrid(n_classes=8, fine_radius=20.0, forward_offset=0.0)
    params = TraversabilityParams()

    # Add a flat ground plane of 10,000 points
    xs = np.random.uniform(-15.0, 15.0, 10000)
    ys = np.random.uniform(-15.0, 15.0, 10000)
    zs = np.full(10000, -0.05)
    xyz = np.column_stack([xs, ys, zs])
    labels = np.ones(10000, dtype=np.uint32)  # road class 1
    grid.add_points(xyz, labels)
    trav = compute_traversability(grid, params)
    return grid, trav, xyz, labels


def test_raster_output_shape_and_dtype():
    grid, trav, xyz, labels = _make_dummy_grid_and_trav()

    img_grid = render_grid_map(grid, trav)
    assert img_grid.shape == (800, 800, 3)
    assert img_grid.dtype == np.uint8

    img_raw = render_raw_points(xyz, labels)
    assert img_raw.shape == (800, 800, 3)
    assert img_raw.dtype == np.uint8

    img_25d = render_25d_fast(grid, trav)
    assert img_25d.shape == (800, 800, 3)
    assert img_25d.dtype == np.uint8


def test_raster_deterministic():
    grid, trav, xyz, labels = _make_dummy_grid_and_trav()

    img1 = render_grid_map(grid, trav)
    img2 = render_grid_map(grid, trav)
    np.testing.assert_array_equal(img1, img2)

    img_raw1 = render_raw_points(xyz, labels)
    img_raw2 = render_raw_points(xyz, labels)
    np.testing.assert_array_equal(img_raw1, img_raw2)

    img_25d1 = render_25d_fast(grid, trav)
    img_25d2 = render_25d_fast(grid, trav)
    np.testing.assert_array_equal(img_25d1, img_25d2)


def test_raster_car_icon_not_background():
    grid, trav, xyz, labels = _make_dummy_grid_and_trav()

    img_grid = render_grid_map(grid, trav)
    # Center pixel (400, 400) is inside the ego car icon
    center_color = img_grid[400, 400]
    assert not np.array_equal(center_color, BG_COLOR)

    img_raw = render_raw_points(xyz, labels)
    center_raw = img_raw[400, 400]
    assert not np.array_equal(center_raw, BG_COLOR)

    img_25d = render_25d_fast(grid, trav)
    # Car box is drawn in the lower-middle foreground
    car_region = img_25d[450:550, 380:420]
    assert not np.all(car_region == BG_COLOR)


def test_raster_drivable_cell_color():
    # Construct a grid with a single known drivable cell at (+10m, 0m)
    grid = VarResGrid(n_classes=8, fine_radius=20.0, forward_offset=0.0)
    params = TraversabilityParams()

    # Place points strictly at (x=10.0, y=0.0, z=-0.05)
    xyz = np.array([[10.0, 0.0, -0.05], [10.05, 0.0, -0.05], [10.0, 0.05, -0.05]])
    labels = np.array([1, 1, 1], dtype=np.uint32)
    grid.add_points(xyz, labels)
    trav = compute_traversability(grid, params)

    img = render_grid_map(grid, trav)
    # Physical x=10m forward, y=0m lateral -> canvas row = (100 - 10) * 4 = 360, col = (0 + 100) * 4 = 400
    expected_row = 360
    expected_col = 400

    # The pixel or its immediate neighbor should have DRIVABLE_COLOR
    patch_area = img[expected_row - 2 : expected_row + 3, expected_col - 2 : expected_col + 3]
    matches = np.all(patch_area == DRIVABLE_COLOR, axis=-1)
    assert np.any(matches), f"Expected drivable color {DRIVABLE_COLOR} near ({expected_row}, {expected_col})"


def test_raster_performance_under_50ms_on_100k_points():
    # 100k points benchmark
    N = 100000
    xs = np.random.uniform(-40.0, 40.0, N)
    ys = np.random.uniform(-40.0, 40.0, N)
    zs = np.random.uniform(-0.3, 1.5, N)
    xyz = np.column_stack([xs, ys, zs])
    labels = np.random.randint(1, 8, N, dtype=np.uint32)

    grid = VarResGrid(n_classes=8, fine_radius=20.0, forward_offset=0.0)
    params = TraversabilityParams()
    grid.add_points(xyz, labels)
    trav = compute_traversability(grid, params)

    # 1. render_grid_map
    t0 = time.perf_counter()
    _ = render_grid_map(grid, trav)
    t_grid = (time.perf_counter() - t0) * 1000.0

    # 2. render_raw_points on 100k points
    t0 = time.perf_counter()
    _ = render_raw_points(xyz, labels)
    t_raw = (time.perf_counter() - t0) * 1000.0

    # 3. render_25d_fast
    t0 = time.perf_counter()
    _ = render_25d_fast(grid, trav)
    t_25d = (time.perf_counter() - t0) * 1000.0

    print(f"Timing on 100k pts: grid={t_grid:.2f}ms, raw={t_raw:.2f}ms, 25d={t_25d:.2f}ms")
    assert t_grid < 50.0, f"render_grid_map took {t_grid:.2f}ms (must be < 50ms)"
    assert t_raw < 50.0, f"render_raw_points took {t_raw:.2f}ms (must be < 50ms)"
    assert t_25d < 50.0, f"render_25d_fast took {t_25d:.2f}ms (must be < 50ms)"
