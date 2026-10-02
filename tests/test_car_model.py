"""Unit tests for the ego car model, styling, and sensing pulse rings."""
import time
import numpy as np
import pytest

from src.car_model import (
    build_car_mesh,
    build_other_car_mesh,
    top_down_polygons,
    compute_sensing_rings,
    CAR_BODY_RED,
    CAR_BODY_RED_RGB,
    OTHER_CAR_BLUE,
    RING_TEAL,
)
from src.grid_engine import VarResGrid
from src.traversability import TraversabilityParams, compute_traversability
from src.raster import render_grid_map, render_raw_points, render_25d_fast


def test_car_mesh_triangle_count_under_200():
    mesh = build_car_mesh()
    tri_count = mesh["triangle_count"]
    assert tri_count < 200, f"Car mesh has {tri_count} triangles (must be < 200)"
    assert tri_count > 30, f"Car mesh has too few triangles ({tri_count})"


def test_faces_and_colors_equal_length():
    mesh = build_car_mesh()
    assert len(mesh["faces"]) == len(mesh["colors"])

    other_mesh = build_other_car_mesh(OTHER_CAR_BLUE)
    assert len(other_mesh["faces"]) == len(other_mesh["colors"])


def test_car_bounds_dimensions():
    mesh = build_car_mesh()
    bounds = mesh["bounds"]
    dx = bounds["x"][1] - bounds["x"][0]
    dy = bounds["y"][1] - bounds["y"][0]
    dz = bounds["z"][1] - bounds["z"][0]

    assert np.isclose(dx, 4.6, atol=0.15), f"Expected length ~4.6m, got {dx:.2f}m"
    assert np.isclose(dy, 1.9, atol=0.15), f"Expected width ~1.9m, got {dy:.2f}m"
    assert np.isclose(dz, 1.6, atol=0.15), f"Expected height ~1.6m, got {dz:.2f}m"


def test_build_car_mesh_cached():
    mesh1 = build_car_mesh()
    mesh2 = build_car_mesh()
    # Verifies caching returns the exact same object reference
    assert mesh1 is mesh2


def test_top_down_polygons_non_empty():
    polys = top_down_polygons()
    assert len(polys) >= 5, "Expected at least 5 component polygons for top-down car"
    for p in polys:
        assert "polygon" in p and len(p["polygon"]) >= 3
        assert "fill" in p and len(p["fill"]) == 3


def test_ring_alphas_decrease_inner_to_outer():
    # In static mode
    rings_static = compute_sensing_rings("Static")
    assert len(rings_static) == 3
    # Check radii increase: 3.0 < 5.5 < 8.0
    assert rings_static[0][0] < rings_static[1][0] < rings_static[2][0]
    # Check alphas decrease: 0.28 > 0.20 > 0.12
    assert rings_static[0][1] > rings_static[1][1] > rings_static[2][1]

    # In pulse mode
    rings_pulse = compute_sensing_rings("Pulse", current_time=0.4)
    assert len(rings_pulse) == 3
    assert rings_pulse[0][0] < rings_pulse[1][0] < rings_pulse[2][0]
    assert rings_pulse[0][1] > rings_pulse[1][1] > rings_pulse[2][1]


def test_red_body_color_in_rendered_raster():
    grid = VarResGrid(n_classes=8, fine_radius=20.0, forward_offset=0.0)
    params = TraversabilityParams()
    trav = compute_traversability(grid, params)

    img = render_grid_map(grid, trav)
    # The car is at origin: x in [-0.9, 3.7], y in [-0.95, 0.95].
    # Canvas center is (400, 400). Let's check a patch around (400, 400).
    car_patch = img[385:415, 395:405]
    # Check for presence of red body color
    red_match = np.all(car_patch == np.array(CAR_BODY_RED_RGB, dtype=np.uint8), axis=-1)
    assert np.any(red_match), f"Expected red body color {CAR_BODY_RED_RGB} in rendered car pixels"


def test_renderer_performance_under_50ms_on_100k_points():
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

    # 2. render_raw_points
    t0 = time.perf_counter()
    _ = render_raw_points(xyz, labels)
    t_raw = (time.perf_counter() - t0) * 1000.0

    # 3. render_25d_fast
    t0 = time.perf_counter()
    _ = render_25d_fast(grid, trav)
    t_25d = (time.perf_counter() - t0) * 1000.0

    assert t_grid < 50.0, f"render_grid_map took {t_grid:.2f}ms"
    assert t_raw < 50.0, f"render_raw_points took {t_raw:.2f}ms"
    assert t_25d < 50.0, f"render_25d_fast took {t_25d:.2f}ms"
