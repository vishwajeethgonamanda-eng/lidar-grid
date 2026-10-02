import numpy as np
import pytest
from src.grid_engine import VarResGrid
from src.loader import make_synthetic_scene
from src.traversability import compute_traversability, DRIVABLE, NON_DRIVABLE, OBSTACLE


def test_conservation_with_patches():
    scan, labels = make_synthetic_scene(seed=777)
    xyz = scan[:, :3]

    # Two non-overlapping patches at (25.0, 0.0) and (40.0, 5.0)
    patches = [(25.0, 0.0), (40.0, 5.0)]
    grid = VarResGrid(n_classes=8, fine_radius=10.0, patches=patches)
    grid.add_points(xyz, labels)

    stats = grid.stats()
    # Conservation rule
    total_grid = int(
        np.sum(grid.fine.count)
        + np.sum(grid.coarse.count)
        + sum(np.sum(p.count) for p in grid.patches)
    )

    r_sensor = np.hypot(xyz[:, 0], xyz[:, 1])
    expected_in_range = int(np.count_nonzero(r_sensor < 100.0))

    assert total_grid == expected_in_range
    assert stats["total_input"] == len(xyz)
    assert (
        stats["in_fine"]
        + stats["in_patch"]
        + stats["in_coarse"]
        + stats["out_of_range"]
        == stats["total_input"]
    )


def test_exact_boundary_patch_edge():
    # Patch centered at (20.0, 0.0), snapped 5 cm lattice
    # Patch size is 64 x 64 cells at 5 cm -> 3.2 m square -> half extent = 1.6 m
    # Extent in x: [20.0 - 1.6, 20.0 + 1.6) = [18.4, 21.6)
    # Extent in y: [-1.6, 1.6)
    patches = [(20.0, 0.0)]
    grid = VarResGrid(n_classes=8, fine_radius=10.0, patches=patches)

    pts = np.array([
        [18.400, 0.0, 0.0],  # exactly on lower x boundary of patch -> inside patch
        [21.599, 0.0, 0.0],  # just inside upper x boundary of patch -> inside patch
        [21.600, 0.0, 0.0],  # on/beyond upper boundary -> coarse
        [18.399, 0.0, 0.0],  # outside lower boundary -> coarse
    ], dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)

    grid.add_points(pts, labels)
    stats = grid.stats()

    assert stats["in_patch"] == 2
    assert stats["in_coarse"] == 2
    assert stats["in_fine"] == 0
    assert stats["out_of_range"] == 0


def test_fine_zone_wins_over_patch():
    # If a patch center is at (8.0, 0.0), overlapping with fine circle (fine_radius = 10.0)
    # Point at (7.0, 0.0) has r = 7.0 < 10.0 (inside fine zone) AND inside patch
    # Fine zone must win!
    patches = [(8.0, 0.0)]
    grid = VarResGrid(n_classes=8, fine_radius=10.0, patches=patches)

    pts = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)

    grid.add_points(pts, labels)
    stats = grid.stats()

    assert stats["in_fine"] == 1
    assert stats["in_patch"] == 0
    assert stats["in_coarse"] == 0


def test_no_patch_equals_current_results():
    scan, labels = make_synthetic_scene(seed=42)
    xyz = scan[:, :3]

    grid_default = VarResGrid(n_classes=8, fine_radius=10.0)
    grid_default.add_points(xyz, labels)

    grid_none = VarResGrid(n_classes=8, fine_radius=10.0, patches=None)
    grid_none.add_points(xyz, labels)

    grid_empty = VarResGrid(n_classes=8, fine_radius=10.0, patches=[])
    grid_empty.add_points(xyz, labels)

    assert grid_default.stats() == grid_none.stats()
    assert grid_default.stats() == grid_empty.stats()
    assert np.array_equal(grid_default.fine.count, grid_none.fine.count)
    assert np.array_equal(grid_default.coarse.count, grid_none.coarse.count)
    assert grid_default.memory_bytes() == grid_none.memory_bytes()


def test_patch_non_overlap_pruning():
    # Patch 1 at (20.0, 0.0). Patch 2 at (21.0, 0.0) overlaps patch 1 (< 3.2 m distance)
    # Patch 3 at (30.0, 0.0) is far away and non-overlapping.
    # Second patch should be dropped.
    patches = [(20.0, 0.0), (21.0, 0.0), (30.0, 0.0)]
    grid = VarResGrid(n_classes=8, fine_radius=10.0, patches=patches)

    assert len(grid.patches) == 2
    assert np.isclose(grid.patches[0].center_x, 20.0)
    assert np.isclose(grid.patches[1].center_x, 30.0)


def test_patch_traversability_computation():
    patches = [(20.0, 0.0)]
    grid = VarResGrid(n_classes=8, fine_radius=10.0, patches=patches)

    # Road in patch at (20.0, 0.0)
    pts = np.array([[20.0, 0.0, 0.0]] * 10, dtype=np.float32)
    labels = np.full(10, 1, dtype=np.int64)
    grid.add_points(pts, labels)

    trav = compute_traversability(grid)
    assert hasattr(trav, "patches")
    assert len(trav.patches) == 1
    patch_res = trav.patches[0]
    assert patch_res.state.shape == (64, 64)
    assert patch_res.confidence.shape == (64, 64)

    # Center cell of patch (col 32, row 32)
    assert patch_res.state[32, 32] == DRIVABLE
    assert patch_res.confidence[32, 32] > 0.8
