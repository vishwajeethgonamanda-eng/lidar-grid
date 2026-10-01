import numpy as np
import pytest
from src.grid_engine import VarResGrid


def generate_synthetic_scene(seed=42):
    """Generates synthetic LiDAR scene:
    - flat ground at z=0
    - a 15 cm kerb across r=10 m
    - a few random points beyond 100 m
    """
    rng = np.random.default_rng(seed)

    # 1. Flat ground at z=0 distributed inside r < 95 m
    n_ground = 10000
    r_ground = rng.uniform(0.1, 95.0, size=n_ground)
    theta_ground = rng.uniform(0, 2 * np.pi, size=n_ground)
    x_ground = r_ground * np.cos(theta_ground)
    y_ground = r_ground * np.sin(theta_ground)
    z_ground = np.zeros(n_ground, dtype=np.float32)
    labels_ground = rng.integers(0, 3, size=n_ground)

    # 2. 15 cm kerb running across r = 10 m line (e.g. from r=9.5 to r=10.5 m)
    # The kerb has a lower surface (z=0.0) and upper surface (z=0.15)
    n_kerb = 2000
    r_kerb = rng.uniform(9.5, 10.5, size=n_kerb)
    theta_kerb = rng.uniform(-0.3, 0.3, size=n_kerb)  # along positive x-axis
    x_kerb = r_kerb * np.cos(theta_kerb)
    y_kerb = r_kerb * np.sin(theta_kerb)
    # Half the points at z=0.0, half at z=0.15 to simulate the vertical kerb step
    z_kerb = rng.choice([0.0, 0.15], size=n_kerb).astype(np.float32)
    labels_kerb = np.full(n_kerb, 1, dtype=np.int64)

    # 3. Random points beyond 100 m
    n_out = 50
    r_out = rng.uniform(100.5, 150.0, size=n_out)
    theta_out = rng.uniform(0, 2 * np.pi, size=n_out)
    x_out = r_out * np.cos(theta_out)
    y_out = r_out * np.sin(theta_out)
    z_out = rng.uniform(-1.0, 5.0, size=n_out).astype(np.float32)
    labels_out = rng.integers(0, 3, size=n_out)

    xyz = np.vstack([
        np.column_stack([x_ground, y_ground, z_ground]),
        np.column_stack([x_kerb, y_kerb, z_kerb]),
        np.column_stack([x_out, y_out, z_out]),
    ]).astype(np.float32)

    labels = np.concatenate([labels_ground, labels_kerb, labels_out]).astype(np.int64)
    return xyz, labels


def test_conservation_basic():
    xyz, labels = generate_synthetic_scene(seed=123)
    grid = VarResGrid(n_classes=4)
    grid.add_points(xyz, labels)

    r = np.hypot(xyz[:, 0], xyz[:, 1])
    expected_fine = np.count_nonzero(r < 10.0)
    expected_coarse = np.count_nonzero((r >= 10.0) & (r < 100.0))
    expected_out = np.count_nonzero(r >= 100.0)
    expected_in_range = expected_fine + expected_coarse

    stats = grid.stats()

    # Rule 1: sum(fine.count) + sum(coarse.count) == number of points with r < 100
    total_grid_count = int(np.sum(grid.fine.count) + np.sum(grid.coarse.count))
    assert total_grid_count == expected_in_range

    # Rule 1: in_fine + in_coarse + out_of_range == total_input
    assert stats["in_fine"] == expected_fine
    assert stats["in_coarse"] == expected_coarse
    assert stats["out_of_range"] == expected_out
    assert stats["total_input"] == len(xyz)
    assert stats["in_fine"] + stats["in_coarse"] + stats["out_of_range"] == stats["total_input"]


def test_boundary_points():
    grid = VarResGrid(n_classes=3)
    # Point at r = 9.999 goes to fine; point at r = 10.0 goes to coarse
    pts = np.array([
        [9.999, 0.0, 1.0],
        [10.0, 0.0, 2.0],
        [0.0, 9.999, 1.5],
        [0.0, 10.0, 2.5],
    ], dtype=np.float32)
    labels = np.array([0, 1, 2, 0], dtype=np.int64)

    grid.add_points(pts, labels)
    stats = grid.stats()

    assert stats["in_fine"] == 2
    assert stats["in_coarse"] == 2
    assert stats["out_of_range"] == 0
    assert np.sum(grid.fine.count) == 2
    assert np.sum(grid.coarse.count) == 2


def test_exact_boundaries_counted_once():
    """Points at both zone edges: 10 m (fine/coarse) and 100 m (coarse/out)."""
    grid = VarResGrid(n_classes=3)
    pts = np.array([
        [9.999, 0.0, 0.0],    # just inside fine
        [10.0, 0.0, 0.0],     # exactly on the seam -> coarse, NOT fine
        [99.999, 0.0, 0.0],   # just inside coarse
        [100.0, 0.0, 0.0],    # exactly at the limit -> out of range
    ], dtype=np.float32)
    labels = np.array([0, 1, 2, 0], dtype=np.int64)

    grid.add_points(pts, labels)
    stats = grid.stats()

    # Every point lands in exactly one place
    assert stats["in_fine"] == 1
    assert stats["in_coarse"] == 2
    assert stats["out_of_range"] == 1
    assert stats["total_input"] == 4
    assert stats["in_fine"] + stats["in_coarse"] + stats["out_of_range"] == 4

    # The cell arrays agree with the stats (nothing dropped or doubled)
    assert int(np.sum(grid.fine.count)) == 1
    assert int(np.sum(grid.coarse.count)) == 2


def test_edge_cases():
    grid = VarResGrid(n_classes=3)

    # 1. Empty input
    empty_xyz = np.zeros((0, 3), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int64)
    grid.add_points(empty_xyz, empty_labels)
    stats = grid.stats()
    assert stats["total_input"] == 0
    assert stats["in_fine"] == 0
    assert stats["in_coarse"] == 0
    assert stats["out_of_range"] == 0
    assert np.sum(grid.fine.count) == 0
    assert np.sum(grid.coarse.count) == 0

    # 2. Single point in fine
    grid.add_points(np.array([[1.0, 2.0, 0.5]], dtype=np.float32), np.array([1], dtype=np.int64))
    assert grid.stats()["in_fine"] == 1
    assert grid.stats()["total_input"] == 1
    assert np.sum(grid.fine.count) == 1

    # 3. All points out of range
    grid.reset()
    far_pts = np.array([
        [105.0, 0.0, 0.0],
        [0.0, 150.0, 1.0],
        [-200.0, 0.0, 2.0],
    ], dtype=np.float32)
    grid.add_points(far_pts, np.array([0, 1, 2], dtype=np.int64))
    stats = grid.stats()
    assert stats["total_input"] == 3
    assert stats["in_fine"] == 0
    assert stats["in_coarse"] == 0
    assert stats["out_of_range"] == 3
    assert np.sum(grid.fine.count) == 0
    assert np.sum(grid.coarse.count) == 0


@pytest.mark.parametrize("radius", [15.0, 30.0])
def test_conservation_variable_fine_radius(radius):
    xyz, labels = generate_synthetic_scene(seed=2026)
    grid = VarResGrid(n_classes=4, fine_radius=radius)
    grid.add_points(xyz, labels)

    r_sensor = np.hypot(xyz[:, 0], xyz[:, 1])
    expected_fine = np.count_nonzero((r_sensor < radius) & (r_sensor < 100.0))
    expected_coarse = np.count_nonzero((r_sensor >= radius) & (r_sensor < 100.0))
    expected_out = np.count_nonzero(r_sensor >= 100.0)

    stats = grid.stats()
    assert stats["in_fine"] == expected_fine
    assert stats["in_coarse"] == expected_coarse
    assert stats["out_of_range"] == expected_out
    assert stats["total_input"] == len(xyz)
    assert int(np.sum(grid.fine.count) + np.sum(grid.coarse.count)) == (expected_fine + expected_coarse)

    # Constant memory verification
    default_grid = VarResGrid(n_classes=4, fine_radius=10.0)
    assert grid.memory_bytes() == default_grid.memory_bytes()


@pytest.mark.parametrize("radius", [15.0, 30.0])
def test_exact_boundary_variable_fine_radius(radius):
    grid = VarResGrid(n_classes=3, fine_radius=radius)
    pts = np.array([
        [radius - 0.001, 0.0, 1.0],  # just inside fine
        [radius, 0.0, 2.0],          # exactly on boundary -> coarse
    ], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    grid.add_points(pts, labels)
    stats = grid.stats()

    assert stats["in_fine"] == 1
    assert stats["in_coarse"] == 1
    assert stats["out_of_range"] == 0


def test_helper_functions():
    from src.grid_engine import stopping_distance, adaptive_fine_radius

    # At v = 0, stopping distance is 0, radius clamped to r_min (10.0)
    assert stopping_distance(0.0) == 0.0
    assert adaptive_fine_radius(0.0) == 10.0

    # At v = 10 m/s: reaction = 10*1.0 = 10m, braking = 100 / (2*6) = 8.33m -> d = 18.33m
    d_10 = stopping_distance(10.0, reaction_s=1.0, decel=6.0)
    assert np.isclose(d_10, 18.333333, atol=1e-3)
    r_10 = adaptive_fine_radius(10.0, r_min=10.0, r_max=30.0)
    assert np.isclose(r_10, 18.333333, atol=1e-3)

    # At high speed (v = 20 m/s): d = 20 + 400/12 = 53.33m -> clamped to r_max (30.0)
    d_20 = stopping_distance(20.0)
    assert d_20 > 30.0
    r_20 = adaptive_fine_radius(20.0, r_min=10.0, r_max=30.0)
    assert r_20 == 30.0