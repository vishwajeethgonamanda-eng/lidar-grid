from typing import Any, Dict, List, Optional
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _accumulate_points_numba(
        x, y, z, labels,
        center_x, center_y, half_extent, cell_size,
        grid_size, n_classes,
        count_flat, z_min_flat, z_max_flat, z_sum_flat, z_sq_sum_flat, label_hist_flat,
    ):
        cx = np.float32(center_x)
        cy = np.float32(center_y)
        he = np.float32(half_extent)
        cs = np.float32(cell_size)
        n = len(x)
        for i in range(n):
            xi = x[i]
            yi = y[i]
            zi = z[i]

            c = int(np.floor(((xi - cx) + he) / cs))
            if c < 0:
                c = 0
            elif c >= grid_size:
                c = grid_size - 1

            r = int(np.floor(((yi - cy) + he) / cs))
            if r < 0:
                r = 0
            elif r >= grid_size:
                r = grid_size - 1

            flat_idx = r * grid_size + c

            count_flat[flat_idx] += 1
            if zi < z_min_flat[flat_idx]:
                z_min_flat[flat_idx] = zi
            if zi > z_max_flat[flat_idx]:
                z_max_flat[flat_idx] = zi
            z_sum_flat[flat_idx] += zi
            z_sq_sum_flat[flat_idx] += np.float64(np.float32(zi * zi))

            if n_classes > 0:
                lbl = labels[i]
                if lbl < 0:
                    lbl = 0
                elif lbl >= n_classes:
                    lbl = n_classes - 1
                label_hist_flat[flat_idx * n_classes + lbl] += 1

    @njit(fastmath=True, cache=True)
    def _bin_var_res_no_patches_numba(
        x, y, z, labels,
        forward_offset, fine_radius,
        fine_half_extent, fine_cell_size, fine_grid_size, fine_n_classes,
        fine_count, fine_z_min, fine_z_max, fine_z_sum, fine_z_sq_sum, fine_label_hist,
        coarse_half_extent, coarse_cell_size, coarse_grid_size, coarse_n_classes,
        coarse_count, coarse_z_min, coarse_z_max, coarse_z_sum, coarse_z_sq_sum, coarse_label_hist,
        stats_out,
    ):
        n = len(x)
        fine_cx = np.float32(forward_offset)
        fine_cy = np.float32(0.0)
        fine_he = np.float32(fine_half_extent)
        fine_cs = np.float32(fine_cell_size)
        fine_r = np.float32(fine_radius)

        coarse_cx = np.float32(0.0)
        coarse_cy = np.float32(0.0)
        coarse_he = np.float32(coarse_half_extent)
        coarse_cs = np.float32(coarse_cell_size)

        n_fine = 0
        n_coarse = 0
        n_out = 0

        for i in range(n):
            xi = x[i]
            yi = y[i]
            zi = z[i]
            lbl = labels[i]

            if np.hypot(xi, yi) >= 100.0:
                n_out += 1
                continue

            if np.hypot(xi - fine_cx, yi) < fine_r:
                n_fine += 1
                c = int(np.floor(((xi - fine_cx) + fine_he) / fine_cs))
                if c < 0: c = 0
                elif c >= fine_grid_size: c = fine_grid_size - 1
                r = int(np.floor(((yi - fine_cy) + fine_he) / fine_cs))
                if r < 0: r = 0
                elif r >= fine_grid_size: r = fine_grid_size - 1
                idx = r * fine_grid_size + c
                fine_count[idx] += 1
                if zi < fine_z_min[idx]: fine_z_min[idx] = zi
                if zi > fine_z_max[idx]: fine_z_max[idx] = zi
                fine_z_sum[idx] += zi
                fine_z_sq_sum[idx] += np.float64(np.float32(zi * zi))
                if fine_n_classes > 0:
                    l_v = lbl
                    if l_v < 0: l_v = 0
                    elif l_v >= fine_n_classes: l_v = fine_n_classes - 1
                    fine_label_hist[idx * fine_n_classes + l_v] += 1
            else:
                n_coarse += 1
                c = int(np.floor(((xi - coarse_cx) + coarse_he) / coarse_cs))
                if c < 0: c = 0
                elif c >= coarse_grid_size: c = coarse_grid_size - 1
                r = int(np.floor(((yi - coarse_cy) + coarse_he) / coarse_cs))
                if r < 0: r = 0
                elif r >= coarse_grid_size: r = coarse_grid_size - 1
                idx = r * coarse_grid_size + c
                coarse_count[idx] += 1
                if zi < coarse_z_min[idx]: coarse_z_min[idx] = zi
                if zi > coarse_z_max[idx]: coarse_z_max[idx] = zi
                coarse_z_sum[idx] += zi
                coarse_z_sq_sum[idx] += np.float64(np.float32(zi * zi))
                if coarse_n_classes > 0:
                    l_v = lbl
                    if l_v < 0: l_v = 0
                    elif l_v >= coarse_n_classes: l_v = coarse_n_classes - 1
                    coarse_label_hist[idx * coarse_n_classes + l_v] += 1

        stats_out[0] = n_fine
        stats_out[1] = 0
        stats_out[2] = n_coarse
        stats_out[3] = n_out
        stats_out[4] = n

    @njit(fastmath=True, cache=True)
    def _bin_var_res_with_patches_numba(
        x, y, z, labels,
        forward_offset, fine_radius,
        fine_half_extent, fine_cell_size, fine_grid_size, fine_n_classes,
        fine_count, fine_z_min, fine_z_max, fine_z_sum, fine_z_sq_sum, fine_label_hist,
        coarse_half_extent, coarse_cell_size, coarse_grid_size, coarse_n_classes,
        coarse_count, coarse_z_min, coarse_z_max, coarse_z_sum, coarse_z_sq_sum, coarse_label_hist,
        patch_cx, patch_cy, patch_he,
        patch_counts, patch_z_mins, patch_z_maxs, patch_z_sums, patch_z_sq_sums, patch_label_hists,
        stats_out,
    ):
        n = len(x)
        n_patches = len(patch_cx)
        fine_cx = np.float32(forward_offset)
        fine_cy = np.float32(0.0)
        fine_he = np.float32(fine_half_extent)
        fine_cs = np.float32(fine_cell_size)
        fine_r = np.float32(fine_radius)

        coarse_cx = np.float32(0.0)
        coarse_cy = np.float32(0.0)
        coarse_he = np.float32(coarse_half_extent)
        coarse_cs = np.float32(coarse_cell_size)

        n_fine = 0
        n_patch = 0
        n_coarse = 0
        n_out = 0

        for i in range(n):
            xi = x[i]
            yi = y[i]
            zi = z[i]
            lbl = labels[i]

            if np.hypot(xi, yi) >= 100.0:
                n_out += 1
                continue

            if np.hypot(xi - fine_cx, yi) < fine_r:
                n_fine += 1
                c = int(np.floor(((xi - fine_cx) + fine_he) / fine_cs))
                if c < 0: c = 0
                elif c >= fine_grid_size: c = fine_grid_size - 1
                r = int(np.floor(((yi - fine_cy) + fine_he) / fine_cs))
                if r < 0: r = 0
                elif r >= fine_grid_size: r = fine_grid_size - 1
                idx = r * fine_grid_size + c
                fine_count[idx] += 1
                if zi < fine_z_min[idx]: fine_z_min[idx] = zi
                if zi > fine_z_max[idx]: fine_z_max[idx] = zi
                fine_z_sum[idx] += zi
                fine_z_sq_sum[idx] += np.float64(np.float32(zi * zi))
                if fine_n_classes > 0:
                    l_v = lbl
                    if l_v < 0: l_v = 0
                    elif l_v >= fine_n_classes: l_v = fine_n_classes - 1
                    fine_label_hist[idx * fine_n_classes + l_v] += 1
            else:
                in_p = False
                for p_idx in range(n_patches):
                    pcx = patch_cx[p_idx]
                    pcy = patch_cy[p_idx]
                    phe = patch_he[p_idx]
                    if xi >= pcx - phe and xi < pcx + phe and yi >= pcy - phe and yi < pcy + phe:
                        n_patch += 1
                        in_p = True
                        c = int(np.floor(((xi - pcx) + phe) / np.float32(0.05)))
                        if c < 0: c = 0
                        elif c >= 64: c = 63
                        r = int(np.floor(((yi - pcy) + phe) / np.float32(0.05)))
                        if r < 0: r = 0
                        elif r >= 64: r = 63
                        idx = p_idx * 4096 + r * 64 + c
                        patch_counts[idx] += 1
                        if zi < patch_z_mins[idx]: patch_z_mins[idx] = zi
                        if zi > patch_z_maxs[idx]: patch_z_maxs[idx] = zi
                        patch_z_sums[idx] += zi
                        patch_z_sq_sums[idx] += np.float64(np.float32(zi * zi))
                        if coarse_n_classes > 0:
                            l_v = lbl
                            if l_v < 0: l_v = 0
                            elif l_v >= coarse_n_classes: l_v = coarse_n_classes - 1
                            patch_label_hists[idx * coarse_n_classes + l_v] += 1
                        break
                if not in_p:
                    n_coarse += 1
                    c = int(np.floor(((xi - coarse_cx) + coarse_he) / coarse_cs))
                    if c < 0: c = 0
                    elif c >= coarse_grid_size: c = coarse_grid_size - 1
                    r = int(np.floor(((yi - coarse_cy) + coarse_he) / coarse_cs))
                    if r < 0: r = 0
                    elif r >= coarse_grid_size: r = coarse_grid_size - 1
                    idx = r * coarse_grid_size + c
                    coarse_count[idx] += 1
                    if zi < coarse_z_min[idx]: coarse_z_min[idx] = zi
                    if zi > coarse_z_max[idx]: coarse_z_max[idx] = zi
                    coarse_z_sum[idx] += zi
                    coarse_z_sq_sum[idx] += np.float64(np.float32(zi * zi))
                    if coarse_n_classes > 0:
                        l_v = lbl
                        if l_v < 0: l_v = 0
                        elif l_v >= coarse_n_classes: l_v = coarse_n_classes - 1
                        coarse_label_hist[idx * coarse_n_classes + l_v] += 1

        stats_out[0] = n_fine
        stats_out[1] = n_patch
        stats_out[2] = n_coarse
        stats_out[3] = n_out
        stats_out[4] = n

    @njit(fastmath=True, cache=True)
    def _bin_pass2_numba(
        x, y, z, labels,
        forward_offset, fine_radius,
        patch_cx, patch_cy, patch_he,
        patch_counts, patch_z_mins, patch_z_maxs, patch_z_sums, patch_z_sq_sums, patch_label_hists,
        coarse_half_extent, coarse_cell_size, coarse_grid_size, coarse_n_classes,
        coarse_count, coarse_z_min, coarse_z_max, coarse_z_sum, coarse_z_sq_sum, coarse_label_hist,
        stats_out,
    ):
        n = len(x)
        n_patches = len(patch_cx)
        fine_cx = np.float32(forward_offset)
        fine_r = np.float32(fine_radius)

        coarse_cx = np.float32(0.0)
        coarse_cy = np.float32(0.0)
        coarse_he = np.float32(coarse_half_extent)
        coarse_cs = np.float32(coarse_cell_size)

        n_patch = 0
        n_coarse = 0

        for i in range(n):
            xi = x[i]
            yi = y[i]
            zi = z[i]
            lbl = labels[i]

            if np.hypot(xi, yi) >= 100.0:
                continue

            if np.hypot(xi - fine_cx, yi) < fine_r:
                continue

            in_p = False
            for p_idx in range(n_patches):
                pcx = patch_cx[p_idx]
                pcy = patch_cy[p_idx]
                phe = patch_he[p_idx]
                if xi >= pcx - phe and xi < pcx + phe and yi >= pcy - phe and yi < pcy + phe:
                    n_patch += 1
                    in_p = True
                    c = int(np.floor(((xi - pcx) + phe) / np.float32(0.05)))
                    if c < 0: c = 0
                    elif c >= 64: c = 63
                    r = int(np.floor(((yi - pcy) + phe) / np.float32(0.05)))
                    if r < 0: r = 0
                    elif r >= 64: r = 63
                    idx = p_idx * 4096 + r * 64 + c
                    patch_counts[idx] += 1
                    if zi < patch_z_mins[idx]: patch_z_mins[idx] = zi
                    if zi > patch_z_maxs[idx]: patch_z_maxs[idx] = zi
                    patch_z_sums[idx] += zi
                    patch_z_sq_sums[idx] += np.float64(np.float32(zi * zi))
                    if coarse_n_classes > 0:
                        l_v = lbl
                        if l_v < 0: l_v = 0
                        elif l_v >= coarse_n_classes: l_v = coarse_n_classes - 1
                        patch_label_hists[idx * coarse_n_classes + l_v] += 1
                    break
            if not in_p:
                n_coarse += 1
                c = int(np.floor(((xi - coarse_cx) + coarse_he) / coarse_cs))
                if c < 0: c = 0
                elif c >= coarse_grid_size: c = coarse_grid_size - 1
                r = int(np.floor(((yi - coarse_cy) + coarse_he) / coarse_cs))
                if r < 0: r = 0
                elif r >= coarse_grid_size: r = coarse_grid_size - 1
                idx = r * coarse_grid_size + c
                coarse_count[idx] += 1
                if zi < coarse_z_min[idx]: coarse_z_min[idx] = zi
                if zi > coarse_z_max[idx]: coarse_z_max[idx] = zi
                coarse_z_sum[idx] += zi
                coarse_z_sq_sum[idx] += np.float64(np.float32(zi * zi))
                if coarse_n_classes > 0:
                    l_v = lbl
                    if l_v < 0: l_v = 0
                    elif l_v >= coarse_n_classes: l_v = coarse_n_classes - 1
                    coarse_label_hist[idx * coarse_n_classes + l_v] += 1

        stats_out[0] = n_patch
        stats_out[1] = n_coarse

    # Warmup compilation on minimal arrays
    _w_f = np.zeros(1, dtype=np.float32)
    _w_i = np.zeros(1, dtype=np.int64)
    _w_st = np.zeros(5, dtype=np.int64)
    _w_st2 = np.zeros(2, dtype=np.int64)
    _accumulate_points_numba(
        _w_f, _w_f, _w_f, _w_i, 0.0, 0.0, 10.0, 0.05, 1, 1,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32)
    )
    _bin_var_res_no_patches_numba(
        _w_f, _w_f, _w_f, _w_i, 0.0, 10.0, 10.0, 0.05, 1, 1,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        100.0, 0.5, 1, 1,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        _w_st
    )
    _bin_var_res_with_patches_numba(
        _w_f, _w_f, _w_f, _w_i, 0.0, 10.0, 10.0, 0.05, 1, 1,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        100.0, 0.5, 1, 1,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        _w_f, _w_f, _w_f,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        _w_st
    )
    _bin_pass2_numba(
        _w_f, _w_f, _w_f, _w_i, 0.0, 10.0,
        _w_f, _w_f, _w_f,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        100.0, 0.5, 1, 1,
        np.zeros(1, np.int32), np.full(1, np.inf, np.float32), np.full(1, -np.inf, np.float32),
        np.zeros(1, np.float64), np.zeros(1, np.float64), np.zeros(1, np.int32),
        _w_st2
    )


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
        if not hasattr(self, "count"):
            self.count = np.zeros(shape, dtype=np.int32)
            self.z_min = np.full(shape, np.inf, dtype=np.float32)
            self.z_max = np.full(shape, -np.inf, dtype=np.float32)
            self.z_sum = np.zeros(shape, dtype=np.float64)
            self.z_sq_sum = np.zeros(shape, dtype=np.float64)
            self.label_hist = np.zeros((self.grid_size, self.grid_size, self.n_classes), dtype=np.int32)
        else:
            self.count.fill(0)
            self.z_min.fill(np.inf)
            self.z_max.fill(-np.inf)
            self.z_sum.fill(0.0)
            self.z_sq_sum.fill(0.0)
            self.label_hist.fill(0)

    def add_points(self, xyz: np.ndarray, labels: np.ndarray) -> None:
        if len(xyz) == 0:
            return

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        if _HAS_NUMBA:
            _accumulate_points_numba(
                np.ascontiguousarray(x, dtype=np.float32),
                np.ascontiguousarray(y, dtype=np.float32),
                np.ascontiguousarray(z, dtype=np.float32),
                np.ascontiguousarray(labels, dtype=np.int64),
                self.center_x,
                self.center_y,
                self.half_extent,
                self.cell_size,
                self.grid_size,
                self.n_classes,
                self.count.ravel(),
                self.z_min.ravel(),
                self.z_max.ravel(),
                self.z_sum.ravel(),
                self.z_sq_sum.ravel(),
                self.label_hist.ravel(),
            )
        else:
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

    def __init__(
        self,
        n_classes: int,
        fine_radius: float = 10.0,
        forward_offset: float = 0.0,
        patches: Optional[List[Any]] = None,
    ):
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

        # Build non-overlapping 64 x 64 focus patches (3.2m square at 5cm)
        self.patches: List[GridZone] = []
        if patches:
            for p in patches:
                if hasattr(p, "x") and hasattr(p, "y"):
                    cx, cy = float(p.x), float(p.y)
                elif isinstance(p, (tuple, list)):
                    cx, cy = float(p[0]), float(p[1])
                else:
                    continue

                # Snap center to 5 cm lattice
                cx_snap = round(cx / 0.05) * 0.05
                cy_snap = round(cy / 0.05) * 0.05

                # Enforce non-overlapping patches (drop lower-priority / later patch)
                # Patch half-extent is 1.6 m, so two patches overlap if |dx| < 3.2 and |dy| < 3.2
                overlaps = False
                for existing in self.patches:
                    if abs(cx_snap - existing.center_x) < 3.2 and abs(cy_snap - existing.center_y) < 3.2:
                        overlaps = True
                        break

                if not overlaps:
                    patch_zone = GridZone(
                        half_extent=1.6,
                        cell_size=0.05,
                        grid_size=64,
                        n_classes=self.n_classes,
                        center_x=cx_snap,
                        center_y=cy_snap,
                    )
                    self.patches.append(patch_zone)

        self._stats = {
            "in_fine": 0,
            "in_patch": 0,
            "in_coarse": 0,
            "out_of_range": 0,
            "total_input": 0,
        }

    def reset(self) -> None:
        self.fine.reset()
        self.coarse.reset()
        for p in self.patches:
            p.reset()
        self._stats = {
            "in_fine": 0,
            "in_patch": 0,
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
        z = xyz[:, 2]

        if _HAS_NUMBA:
            x_c = np.ascontiguousarray(x, dtype=np.float32)
            y_c = np.ascontiguousarray(y, dtype=np.float32)
            z_c = np.ascontiguousarray(z, dtype=np.float32)
            l_c = np.ascontiguousarray(labels, dtype=np.int64)
            st = np.zeros(5, dtype=np.int64)

            if not self.patches:
                _bin_var_res_no_patches_numba(
                    x_c, y_c, z_c, l_c,
                    self.forward_offset, self.fine_radius,
                    self.fine.half_extent, self.fine.cell_size, self.fine.grid_size, self.fine.n_classes,
                    self.fine.count.ravel(), self.fine.z_min.ravel(), self.fine.z_max.ravel(),
                    self.fine.z_sum.ravel(), self.fine.z_sq_sum.ravel(), self.fine.label_hist.ravel(),
                    self.coarse.half_extent, self.coarse.cell_size, self.coarse.grid_size, self.coarse.n_classes,
                    self.coarse.count.ravel(), self.coarse.z_min.ravel(), self.coarse.z_max.ravel(),
                    self.coarse.z_sum.ravel(), self.coarse.z_sq_sum.ravel(), self.coarse.label_hist.ravel(),
                    st,
                )
            else:
                n_p = len(self.patches)
                p_cx = np.array([p.center_x for p in self.patches], dtype=np.float32)
                p_cy = np.array([p.center_y for p in self.patches], dtype=np.float32)
                p_he = np.array([p.half_extent for p in self.patches], dtype=np.float32)
                p_c = np.empty(n_p * 4096, dtype=np.int32)
                p_min = np.empty(n_p * 4096, dtype=np.float32)
                p_max = np.empty(n_p * 4096, dtype=np.float32)
                p_sum = np.empty(n_p * 4096, dtype=np.float64)
                p_sq = np.empty(n_p * 4096, dtype=np.float64)
                p_h = np.empty(n_p * 4096 * self.n_classes, dtype=np.int32)
                for i, p in enumerate(self.patches):
                    p_c[i * 4096 : (i + 1) * 4096] = p.count.ravel()
                    p_min[i * 4096 : (i + 1) * 4096] = p.z_min.ravel()
                    p_max[i * 4096 : (i + 1) * 4096] = p.z_max.ravel()
                    p_sum[i * 4096 : (i + 1) * 4096] = p.z_sum.ravel()
                    p_sq[i * 4096 : (i + 1) * 4096] = p.z_sq_sum.ravel()
                    p_h[i * 4096 * self.n_classes : (i + 1) * 4096 * self.n_classes] = p.label_hist.ravel()

                _bin_var_res_with_patches_numba(
                    x_c, y_c, z_c, l_c,
                    self.forward_offset, self.fine_radius,
                    self.fine.half_extent, self.fine.cell_size, self.fine.grid_size, self.fine.n_classes,
                    self.fine.count.ravel(), self.fine.z_min.ravel(), self.fine.z_max.ravel(),
                    self.fine.z_sum.ravel(), self.fine.z_sq_sum.ravel(), self.fine.label_hist.ravel(),
                    self.coarse.half_extent, self.coarse.cell_size, self.coarse.grid_size, self.coarse.n_classes,
                    self.coarse.count.ravel(), self.coarse.z_min.ravel(), self.coarse.z_max.ravel(),
                    self.coarse.z_sum.ravel(), self.coarse.z_sq_sum.ravel(), self.coarse.label_hist.ravel(),
                    p_cx, p_cy, p_he,
                    p_c, p_min, p_max, p_sum, p_sq, p_h,
                    st,
                )

                for i, p in enumerate(self.patches):
                    p.count[:] = p_c[i * 4096 : (i + 1) * 4096].reshape(64, 64)
                    p.z_min[:] = p_min[i * 4096 : (i + 1) * 4096].reshape(64, 64)
                    p.z_max[:] = p_max[i * 4096 : (i + 1) * 4096].reshape(64, 64)
                    p.z_sum[:] = p_sum[i * 4096 : (i + 1) * 4096].reshape(64, 64)
                    p.z_sq_sum[:] = p_sq[i * 4096 : (i + 1) * 4096].reshape(64, 64)
                    p.label_hist[:] = p_h[i * 4096 * self.n_classes : (i + 1) * 4096 * self.n_classes].reshape(64, 64, self.n_classes)

            self._stats["total_input"] += int(st[4])
            self._stats["in_fine"] += int(st[0])
            self._stats["in_patch"] += int(st[1])
            self._stats["in_coarse"] += int(st[2])
            self._stats["out_of_range"] += int(st[3])
            return

        r_sensor = np.hypot(x, y)
        r_fine = np.hypot(x - self.forward_offset, y)

        mask_out = r_sensor >= 100.0
        mask_fine = (r_fine < self.fine_radius) & (~mask_out)
        mask_remaining = (~mask_fine) & (~mask_out)

        n_fine = int(np.count_nonzero(mask_fine))
        n_out = int(np.count_nonzero(mask_out))

        if n_fine > 0:
            self.fine.add_points(xyz[mask_fine], labels[mask_fine])

        n_in_patch = 0
        if self.patches:
            for p in self.patches:
                mask_p = (
                    mask_remaining
                    & (x >= p.center_x - p.half_extent)
                    & (x < p.center_x + p.half_extent)
                    & (y >= p.center_y - p.half_extent)
                    & (y < p.center_y + p.half_extent)
                )
                cnt_p = int(np.count_nonzero(mask_p))
                if cnt_p > 0:
                    p.add_points(xyz[mask_p], labels[mask_p])
                    n_in_patch += cnt_p
                    mask_remaining = mask_remaining & (~mask_p)

        n_coarse = int(np.count_nonzero(mask_remaining))
        if n_coarse > 0:
            self.coarse.add_points(xyz[mask_remaining], labels[mask_remaining])

        self._stats["total_input"] += n_pts
        self._stats["in_fine"] += n_fine
        self._stats["in_patch"] += n_in_patch
        self._stats["in_coarse"] += n_coarse
        self._stats["out_of_range"] += n_out

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def memory_bytes(self) -> int:
        patch_bytes = sum(p.nbytes for p in self.patches)
        return self.fine.nbytes + self.coarse.nbytes + patch_bytes

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
