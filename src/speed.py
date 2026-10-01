from pathlib import Path
from typing import Optional, Union
import numpy as np

from src.grid_engine import stopping_distance, adaptive_fine_radius


def load_kitti_poses(poses_path: Union[str, Path]) -> np.ndarray:
    """Loads SemanticKITTI poses file where each line has 12 floats (3x4 transformation matrix).

    Returns:
        (N, 3) float64 array of translation vectors [tx, ty, tz] in metres.
    """
    path = Path(poses_path)
    if not path.exists():
        raise FileNotFoundError(f"Poses file not found: {path}")

    raw = np.loadtxt(path)
    if raw.ndim == 1 and raw.size == 12:
        raw = raw.reshape(1, 12)
    elif raw.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Translation components are at indices 3, 7, 11
    tx = raw[:, 3]
    ty = raw[:, 7]
    tz = raw[:, 11]
    return np.column_stack([tx, ty, tz])


def compute_speeds_from_poses(translations: np.ndarray, fps: float = 10.0) -> np.ndarray:
    """Computes speeds in metres/sec from consecutive translation positions at given sensor rate (default 10 Hz).

    Returns:
        (N,) float array of speeds in m/s.
    """
    n_frames = len(translations)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    if n_frames == 1:
        return np.zeros(1, dtype=np.float32)

    diffs = np.diff(translations, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    step_speeds = step_distances * float(fps)

    # Prepend first speed to frame 0
    speeds = np.empty(n_frames, dtype=np.float32)
    speeds[0] = step_speeds[0]
    speeds[1:] = step_speeds
    return speeds


def get_synthetic_speed(
    frame_idx: int,
    base_speed_mps: float = 12.0,
    amplitude_mps: float = 6.0,
    period_frames: int = 40,
) -> float:
    """Generates a smooth, realistic varying synthetic vehicle speed in m/s."""
    idx = max(0, int(frame_idx))
    oscillation = float(amplitude_mps) * np.sin(2.0 * np.pi * idx / float(period_frames))
    return float(max(0.0, float(base_speed_mps) + oscillation))


def get_speed(frame_idx: int, poses_path: Optional[Union[str, Path]] = None) -> float:
    """Retrieves speed in m/s for a given frame.

    Falls back to synthetic speed if poses_path is None or file does not exist.
    """
    if poses_path is not None and Path(poses_path).exists():
        try:
            translations = load_kitti_poses(poses_path)
            speeds = compute_speeds_from_poses(translations, fps=10.0)
            if 0 <= frame_idx < len(speeds):
                return float(speeds[frame_idx])
        except Exception:
            pass

    return get_synthetic_speed(frame_idx)
