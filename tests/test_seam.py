import numpy as np
import pytest
from src.grid_engine import VarResGrid


def generate_synthetic_scene(seed=42):
    """Generates a synthetic LiDAR scene:
    - flat ground at z=0
    - a 15 cm kerb running across r=10 m line
    - a few random points beyond 100 m
    """
    rng = np.random.default_rng(seed)

    # 1. Flat ground at z=0 (inside r < 90 m)
    n_ground = 5000
    r_ground = rng.uniform(0.5, 90.0, size=n_ground)
    theta_ground = rng.uniform(0, 2 * np.pi, size=n_ground)
    x_ground = r_ground * np.cos(theta_ground)
    y_ground = r_ground * np.sin(theta_ground)
    z_ground = np.zeros(n_ground, dtype=np.float32)
    labels_ground = np.zeros(n_ground, dtype=np.int64)

    # 2. 15 cm kerb running across r = 10 m (e.g. spanning r in [9.7, 10.3] m)
    # Dense sampling so cells on both fine (r < 10) and coarse (r >= 10) sides
    # have both low (z=0.0) and high (z=0.15) points.
    n_kerb = 6000
    r_kerb = rng.uniform(9.7, 10.3, size=n_kerb)
    theta_kerb = rng.uniform(-0.2, 0.2, size=n_kerb)
    x_kerb = r_kerb * np.cos(theta_kerb)
    y_kerb = r_kerb * np.sin(theta_kerb)
    z_kerb = rng.choice([0.0, 0.15], size=n_kerb).astype(np.float32)
    labels_kerb = np.ones(n_kerb, dtype=np.int64)

    # 3. Random points beyond 100 m
    n_out = 40
    r_out = rng.uniform(101.0, 140.0, size=n_out)
    theta_out = rng.uniform(0, 2 * np.pi, size=n_out)
    x_out = r_out * np.cos(theta_out)
    y_out = r_out * np.sin(theta_out)
    z_out = rng.uniform(-0.5, 2.0, size=n_out).astype(np.float32)
    labels_out = np.full(n_out, 2, dtype=np.int64)

    xyz = np.vstack([
        np.column_stack([x_ground, y_ground, z_ground]),
        np.column_stack([x_kerb, y_kerb, z_kerb]),
        np.column_stack([x_out, y_out, z_out]),
    ]).astype(np.float32)

    labels = np.concatenate([labels_ground, labels_kerb, labels_out]).astype(np.int64)
    return xyz, labels


def test_seam_kerb_height():
    xyz, labels = generate_synthetic_scene(seed=999)
    grid = VarResGrid(n_classes=3)
    grid.add_points(xyz, labels)

    # Fine grid cells along the kerb (r < 10 m)
    # Check that cells containing both levels of the kerb have z_max - z_min >= 0.14
    fine_dz = grid.fine.z_max - grid.fine.z_min
    fine_occupied = grid.fine.count > 0
    # Fine side kerb cells:
    # Origin is at (-10, -10), so x in [9.7, 10.0) corresponds to col around floor((9.7 + 10)/0.05) ~ 394..399
    # y around 0 corresponds to row floor((0 + 10)/0.05) ~ 200
    fine_kerb_region_dz = fine_dz[195:205, 390:400]
    fine_kerb_detected = np.any(fine_kerb_region_dz >= 0.14)
    assert fine_kerb_detected, "Fine grid failed to preserve kerb height (z_max - z_min >= 0.14) on fine side of seam"

    # Coarse grid cells along the kerb (r >= 10 m)
    coarse_dz = grid.coarse.z_max - grid.coarse.z_min
    # Coarse grid: half_extent = 100, cell_size = 0.5.
    # x in [10.0, 10.3] -> col = floor((10.0 + 100)/0.5) = 220
    # y around 0 -> row = floor((0 + 100)/0.5) = 200
    coarse_kerb_region_dz = coarse_dz[198:202, 220:222]
    coarse_kerb_detected = np.any(coarse_kerb_region_dz >= 0.14)
    assert coarse_kerb_detected, "Coarse grid failed to preserve kerb height (z_max - z_min >= 0.14) on coarse side of seam"


def test_exact_seam_routing():
    grid = VarResGrid(n_classes=2)
    # Specific points right at the seam:
    # 9.999 m from origin -> fine
    # 10.000 m from origin -> coarse
    xyz = np.array([
        [9.999, 0.0, 0.5],
        [10.000, 0.0, 0.8],
    ], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    grid.add_points(xyz, labels)
    stats = grid.stats()

    assert stats["in_fine"] == 1
    assert stats["in_coarse"] == 1
    assert stats["out_of_range"] == 0

    # In fine: point at (9.999, 0)
    # col = floor((9.999 + 10) / 0.05) = floor(19.999 / 0.05) = 399
    # row = floor((0.0 + 10) / 0.05) = 200
    assert grid.fine.count[200, 399] == 1
    assert np.isclose(grid.fine.z_min[200, 399], 0.5)

    # In coarse: point at (10.0, 0)
    # col = floor((10.0 + 100) / 0.5) = floor(110.0 / 0.5) = 220
    # row = floor((0.0 + 100) / 0.5) = 200
    assert grid.coarse.count[200, 220] == 1
    assert np.isclose(grid.coarse.z_min[200, 220], 0.8)
