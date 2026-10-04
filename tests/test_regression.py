"""Regression test verifying that pipeline outputs match the golden reference.

Compares grid statistics, cell counts, traversability states and confidences,
focus patches, and candidate risks against the pre-optimization golden reference
generated with fixed random seeds.
Exact match is enforced for discrete quantities (labels, states, counts),
and a tolerance of 1e-5 is enforced for floating point values.
"""
from pathlib import Path
import numpy as np
import pytest

from eval_patches import generate_synthetic_scene_with_targets
from src.scene_urban import make_urban_scene
from src.pipeline import process_frame


GOLDEN_PATH = Path(__file__).parent / "golden_reference.npz"


def _verify_scene(prefix: str, xyz: np.ndarray, labels: np.ndarray, golden_data: dict, speed_mps: float = 15.0):
    res = process_frame(xyz, labels, risk_patches=True, speed_mps=speed_mps, k_patches=4)
    grid = res["grid"]
    trav = res["traversability"]

    # 1. Global stats
    assert res["stats"]["in_fine"] == int(golden_data[f"{prefix}_stats_in_fine"])
    assert res["stats"]["in_patch"] == int(golden_data[f"{prefix}_stats_in_patch"])
    assert res["stats"]["in_coarse"] == int(golden_data[f"{prefix}_stats_in_coarse"])
    assert res["stats"]["out_of_range"] == int(golden_data[f"{prefix}_stats_out_of_range"])
    assert res["stats"]["total_input"] == int(golden_data[f"{prefix}_stats_total_input"])

    # 2. Fine zone grid stats
    np.testing.assert_array_equal(grid.fine.count, golden_data[f"{prefix}_fine_count"])
    np.testing.assert_array_equal(grid.fine.label_hist, golden_data[f"{prefix}_fine_label_hist"])
    np.testing.assert_allclose(grid.fine.z_min, golden_data[f"{prefix}_fine_z_min"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grid.fine.z_max, golden_data[f"{prefix}_fine_z_max"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grid.fine.z_sum, golden_data[f"{prefix}_fine_z_sum"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grid.fine.z_sq_sum, golden_data[f"{prefix}_fine_z_sq_sum"], atol=1e-5, rtol=1e-5)

    # 3. Coarse zone grid stats
    np.testing.assert_array_equal(grid.coarse.count, golden_data[f"{prefix}_coarse_count"])
    np.testing.assert_array_equal(grid.coarse.label_hist, golden_data[f"{prefix}_coarse_label_hist"])
    np.testing.assert_allclose(grid.coarse.z_min, golden_data[f"{prefix}_coarse_z_min"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grid.coarse.z_max, golden_data[f"{prefix}_coarse_z_max"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grid.coarse.z_sum, golden_data[f"{prefix}_coarse_z_sum"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grid.coarse.z_sq_sum, golden_data[f"{prefix}_coarse_z_sq_sum"], atol=1e-5, rtol=1e-5)

    # 4. Traversability outputs
    np.testing.assert_array_equal(trav.fine.state, golden_data[f"{prefix}_trav_fine_state"])
    np.testing.assert_allclose(trav.fine.confidence, golden_data[f"{prefix}_trav_fine_conf"], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(trav.coarse.state, golden_data[f"{prefix}_trav_coarse_state"])
    np.testing.assert_allclose(trav.coarse.confidence, golden_data[f"{prefix}_trav_coarse_conf"], atol=1e-5, rtol=1e-5)

    # 5. Focus patches
    n_patches = int(golden_data[f"{prefix}_n_patches"])
    assert len(grid.patches) == n_patches
    assert len(trav.patches) == n_patches

    for i in range(n_patches):
        p = grid.patches[i]
        assert np.isclose(p.center_x, golden_data[f"{prefix}_patch_{i}_cx"], atol=1e-5)
        assert np.isclose(p.center_y, golden_data[f"{prefix}_patch_{i}_cy"], atol=1e-5)
        np.testing.assert_array_equal(p.count, golden_data[f"{prefix}_patch_{i}_count"])
        np.testing.assert_array_equal(p.label_hist, golden_data[f"{prefix}_patch_{i}_label_hist"])
        np.testing.assert_allclose(p.z_min, golden_data[f"{prefix}_patch_{i}_z_min"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(p.z_max, golden_data[f"{prefix}_patch_{i}_z_max"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(p.z_sum, golden_data[f"{prefix}_patch_{i}_z_sum"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(p.z_sq_sum, golden_data[f"{prefix}_patch_{i}_z_sq_sum"], atol=1e-5, rtol=1e-5)

        np.testing.assert_array_equal(trav.patches[i].state, golden_data[f"{prefix}_patch_{i}_trav_state"])
        np.testing.assert_allclose(trav.patches[i].confidence, golden_data[f"{prefix}_patch_{i}_trav_conf"], atol=1e-5, rtol=1e-5)

    # 6. Candidate objects
    cands = res["candidates"]
    golden_cands = golden_data[f"{prefix}_candidates"]
    assert len(cands) == len(golden_cands)
    for c, gc in zip(cands, golden_cands):
        assert np.isclose(c.x, gc[0], atol=1e-5)
        assert np.isclose(c.y, gc[1], atol=1e-5)
        assert np.isclose(c.risk, gc[2], atol=1e-5)
        assert c.dominant_class == int(gc[3])
        assert bool(c.is_mixed) == bool(gc[4])


def test_regression_synthetic_scene_matches_golden():
    assert GOLDEN_PATH.exists(), f"Golden reference file {GOLDEN_PATH} missing"
    golden_data = np.load(GOLDEN_PATH)
    xyz, labels = generate_synthetic_scene_with_targets(seed=42)
    _verify_scene("scene1", xyz, labels, golden_data, speed_mps=15.0)


def test_regression_urban_scene_matches_golden():
    assert GOLDEN_PATH.exists(), f"Golden reference file {GOLDEN_PATH} missing"
    golden_data = np.load(GOLDEN_PATH)
    xyz, labels = make_urban_scene(frame_idx=0, seed=42)
    _verify_scene("scene2", xyz, labels, golden_data, speed_mps=15.0)
