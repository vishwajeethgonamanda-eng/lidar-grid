import numpy as np
import pytest
from src.loader import make_synthetic_scene
from src.grid_engine import VarResGrid
from src.traversability import TraversabilityParams
from src.pipeline import process_frame


def test_process_frame_synthetic():
    scan, labels = make_synthetic_scene(seed=123)
    xyz = scan[:, :3]

    result = process_frame(xyz, labels)

    # Required keys in dictionary
    assert "grid" in result
    assert "traversability" in result
    assert "timings" in result
    assert "stats" in result
    assert "memory_bytes" in result
    assert "uniform_equivalent_bytes" in result

    # Validate conservation
    stats = result["stats"]
    assert stats["total_input"] == len(xyz)
    assert stats["in_fine"] + stats["in_coarse"] + stats["out_of_range"] == stats["total_input"]

    # Validate memory savings
    mem_var = result["memory_bytes"]
    mem_uniform = result["uniform_equivalent_bytes"]
    assert mem_var < mem_uniform
    assert mem_uniform / mem_var == pytest.approx(50.0)

    # Validate timings
    timings = result["timings"]
    assert timings["add_points_ms"] > 0
    assert timings["compute_traversability_ms"] > 0
    assert timings["total_ms"] >= timings["add_points_ms"] + timings["compute_traversability_ms"]

    # Validate traversability output present
    trav = result["traversability"]
    assert hasattr(trav, "fine")
    assert hasattr(trav, "coarse")
    assert trav.fine.state.shape == (400, 400)
    assert trav.coarse.state.shape == (400, 400)


def test_process_frame_grid_reset_between_frames():
    scan, labels = make_synthetic_scene(seed=42)
    xyz = scan[:, :3]

    grid = VarResGrid(n_classes=8)

    # Frame 1
    res1 = process_frame(xyz, labels, grid=grid)
    count1 = res1["stats"]["total_input"]

    # Frame 2 with half the points
    half_xyz = xyz[: len(xyz) // 2]
    half_labels = labels[: len(labels) // 2]
    res2 = process_frame(half_xyz, half_labels, grid=grid)
    count2 = res2["stats"]["total_input"]

    # Because grid is reset at start of frame, count2 should match len(half_xyz), not accumulate
    assert count2 == len(half_xyz)
    assert count2 < count1
    assert int(grid.fine.count.sum() + grid.coarse.count.sum()) == (
        res2["stats"]["in_fine"] + res2["stats"]["in_coarse"]
    )
