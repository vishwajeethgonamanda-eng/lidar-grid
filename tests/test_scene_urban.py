"""Unit tests for realistic urban street LiDAR scene generator."""
import numpy as np
import pytest
from src.scene_urban import make_urban_scene


def test_scene_urban_shapes_and_types():
    xyz, labels = make_urban_scene(frame_idx=0, seed=42)
    assert isinstance(xyz, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert xyz.ndim == 2
    assert xyz.shape[1] == 3
    assert labels.ndim == 1
    assert xyz.shape[0] == labels.shape[0]
    # Aim for roughly 60,000 to 120,000 points
    assert 60000 <= len(xyz) <= 120000


def test_scene_urban_labels_valid_and_required_classes_present():
    xyz, labels = make_urban_scene(frame_idx=0, seed=42)
    # Labels must be in 0..7
    assert np.all(labels >= 0)
    assert np.all(labels <= 7)

    unique_classes = set(np.unique(labels))
    # All classes 1 (road), 3 (buildings/obstacles), 4 (vehicles), 5 (pedestrians), 7 (sidewalks)
    required = {1, 3, 4, 5, 7}
    for req_cls in required:
        assert req_cls in unique_classes, f"Class {req_cls} missing from generated urban scene"


def test_scene_urban_majority_within_100m():
    xyz, labels = make_urban_scene(frame_idx=0, seed=42)
    dists = np.linalg.norm(xyz[:, :2], axis=1)
    within_100m = np.count_nonzero(dists < 100.0)
    fraction_within = within_100m / len(dists)
    # The vast majority of points must be within 100m
    assert fraction_within > 0.85


def test_scene_urban_determinism():
    xyz1, labels1 = make_urban_scene(frame_idx=5, seed=123)
    xyz2, labels2 = make_urban_scene(frame_idx=5, seed=123)
    np.testing.assert_array_equal(xyz1, xyz2)
    np.testing.assert_array_equal(labels1, labels2)


def test_scene_urban_pedestrian_movement():
    xyz_f0, labels_f0 = make_urban_scene(frame_idx=0, seed=42)
    xyz_f1, labels_f1 = make_urban_scene(frame_idx=10, seed=42)
    # Pedestrian points (class 5) should have moved
    peds_f0 = xyz_f0[labels_f0 == 5]
    peds_f1 = xyz_f1[labels_f1 == 5]
    assert len(peds_f0) > 0
    assert len(peds_f1) > 0
    # Mean position of pedestrians changes between frames
    mean_x_f0 = np.mean(peds_f0[:, 0])
    mean_x_f1 = np.mean(peds_f1[:, 0])
    assert not np.isclose(mean_x_f0, mean_x_f1, atol=0.05)


def test_scene_urban_open_quadrants():
    xyz, labels = make_urban_scene(frame_idx=0, seed=42, layout="open")
    dists = np.linalg.norm(xyz[:, :2], axis=1)
    in_100 = dists < 100.0
    xyz_100 = xyz[in_100]
    total_100 = len(xyz_100)
    assert total_100 > 0

    x = xyz_100[:, 0]
    y = xyz_100[:, 1]

    # Four quadrants around the car
    q1 = (x > 0) & (y > 0)
    q2 = (x < 0) & (y > 0)
    q3 = (x < 0) & (y < 0)
    q4 = (x > 0) & (y < 0)

    for q_idx, q_mask in enumerate([q1, q2, q3, q4], start=1):
        frac = np.count_nonzero(q_mask) / total_100
        assert frac >= 0.10, f"Quadrant {q_idx} has only {frac*100:.1f}% of points within 100m (expected >= 10%)"


def test_scene_urban_ego_motion_differs():
    xyz_s0, _ = make_urban_scene(frame_idx=0, seed=42, s=0.0)
    xyz_s15, _ = make_urban_scene(frame_idx=0, seed=42, s=15.0)
    # Scenes at s=0 and s=15 must differ
    assert not np.array_equal(xyz_s0, xyz_s15)


def test_scene_urban_pole_shift():
    # In street layout, reference pole is at world x = 20.0, y = 4.2
    xyz_s15, labels_s15 = make_urban_scene(frame_idx=0, seed=42, layout="street", s=15.0)
    # Extract pole points near y=4.2 and above ground
    pole_mask = (labels_s15 == 3) & (np.abs(xyz_s15[:, 1] - 4.2) < 0.4) & (xyz_s15[:, 2] > 0.5) & (xyz_s15[:, 0] > 0.0) & (xyz_s15[:, 0] < 10.0)
    assert np.any(pole_mask), "Pole points not found at s=15"
    pole_x = xyz_s15[pole_mask, 0]
    mean_x = float(np.mean(pole_x))
    # Must appear at about x = 5 (within 0.5 m)
    assert np.isclose(mean_x, 5.0, atol=0.5), f"Pole expected at ~5.0m, found at {mean_x:.2f}m"


def test_scene_urban_periodicity():
    # City block pattern repeats periodically every 120 m
    xyz_0, lbl_0 = make_urban_scene(frame_idx=0, seed=42, s=0.0)
    xyz_120, lbl_120 = make_urban_scene(frame_idx=0, seed=42, s=120.0)
    np.testing.assert_allclose(xyz_0, xyz_120, atol=1e-3)
    np.testing.assert_array_equal(lbl_0, lbl_120)


def test_scene_urban_determinism_with_s():
    xyz_a, lbl_a = make_urban_scene(frame_idx=3, seed=99, s=23.4)
    xyz_b, lbl_b = make_urban_scene(frame_idx=3, seed=99, s=23.4)
    np.testing.assert_array_equal(xyz_a, xyz_b)
    np.testing.assert_array_equal(lbl_a, lbl_b)
