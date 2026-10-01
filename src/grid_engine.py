from typing import Any, Dict
import numpy as np


def stopping_distance(speed_mps: float, reaction_s: float = 1.0, decel: float = 6.0) -> float:
    """Calculates stopping distance: v * reaction_s + v**2 / (2 * decel)."""
    v = max(0.0, float(speed_mps))
    d_reaction = v * float(reaction_s)
    d_braking = (v ** 2) / (2.0 * float(decel))
    return float(d_reaction + d_braking)


def adaptive_fine_radius(speed_mps: float, r_min: float = 10.0, r_max: float = 30.0) -> float:
    """Returns stopping distance clamped between r_min and r_max."""
    d_stop = stopping_distance(speed_mps)
    return float(np.clip(d_stop, float(r_min), float(r_max)))


class GridZone:
    """Represents a dense 2.5D grid zone with per-cell statistics."""

    def __init__(
        self,
        half_extent: float,
        cell_size: float,
        grid_size: int,
        n_classes: int,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ):
        self.half_extent = float(half_extent)
        self.cell_size = float(cell_size)
        self.grid_size = int(grid_size)
        self.n_classes = int(n_classes)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.reset()

    def reset(self) -> None:
        shape = (self.grid_size, self.grid_size)
        self.count = np.zeros(shape, dtype=np.int32)
        self.z_min = np.full(shape, np.inf, dtype=np.float32)
        self.z_max = np.full(shape, -np.inf, dtype=np.float32)
        self.z_sum = np.zeros(shape, dtype=np.float64)
        self.z_sq_sum = np.zeros(shape, dtype=np.float64)
        self.label_hist = np.zeros((self.grid_size, self.grid_size, self.n_classes), dtype=np.int32)

    def add_points(self, xyz: np.ndarray, labels: np.ndarray) -> None:
        if len(xyz) == 0:
            return

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        col = np.clip(
            np.floor(((x - self.center_x) + self.half_extent) / self.cell_size).astype(np.int64),
            0,
            self.grid_size - 1,
        )
        row = np.clip(
            np.floor(((y - self.center_y) + self.half_extent) / self.cell_size).astype(np.int64),
            0,
            self.grid_size - 1,
        )
        flat_idx = row * self.grid_size + col

        # Unbuffered in-place vectorized accumulations
        np.add.at(self.count.ravel(), flat_idx, 1)
        np.minimum.at(self.z_min.ravel(), flat_idx, z)
        np.maximum.at(self.z_max.ravel(), flat_idx, z)
        np.add.at(self.z_sum.ravel(), flat_idx, z)
        np.add.at(self.z_sq_sum.ravel(), flat_idx, z * z)

        # Label histogram accumulation
        if self.n_classes > 0:
            valid_labels = np.clip(labels, 0, self.n_classes - 1)
            hist_flat_idx = flat_idx * self.n_classes + valid_labels
            np.add.at(self.label_hist.ravel(), hist_flat_idx, 1)

    @property
    def z_mean(self) -> np.ndarray:
        return np.where(self.count > 0, self.z_sum / np.maximum(self.count, 1), np.nan)

    @property
    def z_var(self) -> np.ndarray:
        mean = np.where(self.count > 0, self.z_sum / np.maximum(self.count, 1), 0.0)
        mean_sq = np.where(self.count > 0, self.z_sq_sum / np.maximum(self.count, 1), 0.0)
        return np.where(self.count > 0, np.maximum(0.0, mean_sq - mean**2), 0.0)

    @property
    def dominant_label(self) -> np.ndarray:
        return np.argmax(self.label_hist, axis=-1)

    @property
    def nbytes(self) -> int:
        return int(
            self.count.nbytes
            + self.z_min.nbytes
            + self.z_max.nbytes
            + self.z_sum.nbytes
            + self.z_sq_sum.nbytes
            + self.label_hist.nbytes
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class VarResGrid:
    """Variable-Resolution 2.5D LiDAR Grid Engine with adjustable fine radius at constant memory.

    - Fine zone: Circle of fine_radius around (forward_offset, 0).
                 Array remains 400 x 400, fine cell size = 2 * fine_radius / 400.
    - Coarse zone: 50 cm cells, array 400 x 400 covering [-100, 100) m for all other points within 100m.
    - Out of range: r_sensor >= 100 m
    """

    def __init__(self, n_classes: int, fine_radius: float = 10.0, forward_offset: float = 0.0):
        self.n_classes = int(n_classes)
        self.fine_radius = float(fine_radius)
        self.forward_offset = float(forward_offset)
        self.fine_cell_size = (2.0 * self.fine_radius) / 400.0

        self.fine = GridZone(
            half_extent=self.fine_radius,
            cell_size=self.fine_cell_size,
            grid_size=400,
            n_classes=self.n_classes,
            center_x=self.forward_offset,
            center_y=0.0,
        )
        self.coarse = GridZone(
            half_extent=100.0,
            cell_size=0.5,
            grid_size=400,
            n_classes=self.n_classes,
            center_x=0.0,
            center_y=0.0,
        )
        self._stats = {
            "in_fine": 0,
            "in_coarse": 0,
            "out_of_range": 0,
            "total_input": 0,
        }

    def reset(self) -> None:
        self.fine.reset()
        self.coarse.reset()
        self._stats = {
            "in_fine": 0,
            "in_coarse": 0,
            "out_of_range": 0,
            "total_input": 0,
        }

    def add_points(self, xyz: np.ndarray, labels: np.ndarray) -> None:
        """Add batch of (x, y, z) points and labels. Fully vectorized."""
        xyz = np.asarray(xyz, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if xyz.ndim == 1 and xyz.size >= 3:
            xyz = xyz.reshape(1, -1)
        if labels.ndim == 0:
            labels = labels.reshape(1)

        n_pts = len(xyz)
        if n_pts == 0:
            return

        x = xyz[:, 0]
        y = xyz[:, 1]

        r_sensor = np.hypot(x, y)
        r_fine = np.hypot(x - self.forward_offset, y)

        mask_fine = (r_fine < self.fine_radius) & (r_sensor < 100.0)
        mask_coarse = (~mask_fine) & (r_sensor < 100.0)
        mask_out = r_sensor >= 100.0

        n_fine = int(np.count_nonzero(mask_fine))
        n_coarse = int(np.count_nonzero(mask_coarse))
        n_out = int(np.count_nonzero(mask_out))

        self._stats["total_input"] += n_pts
        self._stats["in_fine"] += n_fine
        self._stats["in_coarse"] += n_coarse
        self._stats["out_of_range"] += n_out

        if n_fine > 0:
            self.fine.add_points(xyz[mask_fine], labels[mask_fine])

        if n_coarse > 0:
            self.coarse.add_points(xyz[mask_coarse], labels[mask_coarse])

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def memory_bytes(self) -> int:
        return self.fine.nbytes + self.coarse.nbytes

    def uniform_equivalent_bytes(self) -> int:
        """Memory if same stats were stored in a uniform 5 cm grid over the 100 m radius."""
        bytes_per_cell = (
            self.fine.count.itemsize
            + self.fine.z_min.itemsize
            + self.fine.z_max.itemsize
            + self.fine.z_sum.itemsize
            + self.fine.z_sq_sum.itemsize
            + self.fine.label_hist.itemsize * self.n_classes
        )
        uniform_cells = 4000 * 4000
        return int(uniform_cells * bytes_per_cell)
