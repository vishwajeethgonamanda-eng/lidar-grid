from typing import Dict, Any
import numpy as np


class GridZone:
    """Represents a dense 2.5D grid zone with per-cell statistics."""

    def __init__(self, half_extent: float, cell_size: float, grid_size: int, n_classes: int):
        self.half_extent = float(half_extent)
        self.cell_size = float(cell_size)
        self.grid_size = int(grid_size)
        self.n_classes = int(n_classes)
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
            np.floor((x + self.half_extent) / self.cell_size).astype(np.int64),
            0,
            self.grid_size - 1,
        )
        row = np.clip(
            np.floor((y + self.half_extent) / self.cell_size).astype(np.int64),
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
    """Variable-Resolution 2.5D LiDAR Grid Engine.

    - Fine zone:   r < 10 m,        5 cm cells, array 400 x 400 covering [-10, 10) m
    - Coarse zone: 10 <= r < 100 m, 50 cm cells, array 400 x 400 covering [-100, 100) m
    - Out of range: r >= 100 m
    """

    def __init__(self, n_classes: int):
        self.n_classes = int(n_classes)
        self.fine = GridZone(half_extent=10.0, cell_size=0.05, grid_size=400, n_classes=self.n_classes)
        self.coarse = GridZone(half_extent=100.0, cell_size=0.5, grid_size=400, n_classes=self.n_classes)
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

        if xyz.ndim == 1 and xyz.size == 3:
            xyz = xyz.reshape(1, 3)
        if labels.ndim == 0:
            labels = labels.reshape(1)

        n_pts = len(xyz)
        if n_pts == 0:
            return

        # Continuous point radius determines the zone
        r = np.hypot(xyz[:, 0], xyz[:, 1])

        mask_fine = r < 10.0
        mask_coarse = (r >= 10.0) & (r < 100.0)
        mask_out = r >= 100.0

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
        """Memory if same stats were stored in a uniform 5 cm grid over the 100 m radius.
        Covering [-100, 100) m at 5 cm resolution yields 4000 x 4000 cells.
        """
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
