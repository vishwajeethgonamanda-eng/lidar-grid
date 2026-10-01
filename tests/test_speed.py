from pathlib import Path
import numpy as np
import pytest
from src.speed import (
    load_kitti_poses,
    compute_speeds_from_poses,
    get_synthetic_speed,
    get_speed,
    stopping_distance,
    adaptive_fine_radius,
)


def test_speed_from_poses(tmp_path):
    poses_file = tmp_path / "poses.txt"
    # Write 3 poses with constant 1 m forward movement per frame (at 10 Hz = 10 m/s)
    # 12 values: r11..r33, tx, ty, tz
    p0 = [1, 0, 0, 0.0, 0, 1, 0, 0.0, 0, 0, 1, 0.0]
    p1 = [1, 0, 0, 1.0, 0, 1, 0, 0.0, 0, 0, 1, 0.0]
    p2 = [1, 0, 0, 2.0, 0, 1, 0, 0.0, 0, 0, 1, 0.0]
    np.savetxt(poses_file, [p0, p1, p2])

    translations = load_kitti_poses(poses_file)
    assert translations.shape == (3, 3)
    assert np.allclose(translations[:, 0], [0.0, 1.0, 2.0])

    speeds = compute_speeds_from_poses(translations, fps=10.0)
    assert len(speeds) == 3
    assert np.allclose(speeds, [10.0, 10.0, 10.0])

    assert np.isclose(get_speed(0, poses_file), 10.0)
    assert np.isclose(get_speed(1, poses_file), 10.0)


def test_synthetic_speed_fallback():
    # Calling get_speed without poses file uses synthetic speed
    spd0 = get_speed(0, poses_path=None)
    spd10 = get_speed(10, poses_path=None)
    assert spd0 > 0.0
    assert spd10 > 0.0

    synth = get_synthetic_speed(0, base_speed_mps=15.0)
    assert synth == 15.0


def test_speed_stopping_distance_integration():
    spd = 12.0
    d_stop = stopping_distance(spd)
    assert d_stop > 12.0

    r = adaptive_fine_radius(spd)
    assert 10.0 <= r <= 30.0
