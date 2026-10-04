from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import ndimage

from src.grid_engine import VarResGrid, stopping_distance


@dataclass
class RiskParams:
    """Parameters for risk scoring.

    Attributes:
        corridor_half_width: Half-width of path corridor (metres either side of forward axis).
        stopping_dist_multiplier: Forward horizon multiplier (1.5 x stopping distance).
        min_risk: Minimum risk score threshold to consider as a focus candidate.
        k: Maximum number of focus patch candidates to return.
        mixed_cell_weight_factor: Multiplier applied to mixed-cell secondary candidates.
    """
    corridor_half_width: float = 2.0
    stopping_dist_multiplier: float = 1.5
    min_risk: float = 0.01
    k: int = 4
    mixed_cell_weight_factor: float = 0.5


@dataclass
class CandidateObject:
    """Represents a focus patch candidate object."""
    x: float
    y: float
    risk: float
    dominant_class: int
    is_mixed: bool = False


# Class priority weights:
CLASS_WEIGHTS = {
    5: 1.0,  # person
    6: 0.9,  # moving object
    4: 0.6,  # vehicle
}


try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Precomputed coarse grid cell center coordinates (metres)
_COARSE_X = (-100.0 + (np.arange(400, dtype=np.float32) + 0.5) * 0.5)
_COARSE_Y = (-100.0 + (np.arange(400, dtype=np.float32) + 0.5) * 0.5)[:, None]
_COARSE_X_1D = (-100.0 + (np.arange(400, dtype=np.float32) + 0.5) * 0.5)
_COARSE_Y_1D = (-100.0 + (np.arange(400, dtype=np.float32) + 0.5) * 0.5)
_FULL_X = np.broadcast_to(_COARSE_X, (400, 400)).astype(np.float32)
_FULL_Y = np.broadcast_to(_COARSE_Y, (400, 400)).astype(np.float32)
_STRUCTURE = ndimage.generate_binary_structure(2, 2)

if _HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _find_all_candidates_numba(
        count, label_hist, coarse_x, coarse_y,
        forward_offset, fine_radius,
        target_mask_out, dom_label_out,
        sec_r_out, sec_c_out, sec_cls_out,
    ):
        fine_r_sq = np.float32(fine_radius * fine_radius)
        out_sq = np.float32(100.0 * 100.0)
        rows, cols = count.shape
        sec_count = 0

        for r in range(rows):
            cy = coarse_y[r]
            for c in range(cols):
                cnt = count[r, c]
                if cnt == 0:
                    continue
                cx = coarse_x[c]
                if cx * cx + cy * cy >= out_sq:
                    continue
                dx = cx - forward_offset
                if dx * dx + cy * cy < fine_r_sq:
                    continue

                best_lbl = 0
                max_cnt = -1
                for l in range(8):
                    lh = label_hist[r, c, l]
                    if lh > max_cnt:
                        max_cnt = lh
                        best_lbl = l
                dom_label_out[r, c] = best_lbl

                if best_lbl == 4 or best_lbl == 5 or best_lbl == 6:
                    target_mask_out[r, c] = True
                elif cnt >= 2:
                    if label_hist[r, c, 4] > 0 or label_hist[r, c, 5] > 0 or label_hist[r, c, 6] > 0:
                        counts_copy = [label_hist[r, c, l] for l in range(8)]
                        lbls = [0, 1, 2, 3, 4, 5, 6, 7]
                        for i in range(7):
                            for j in range(7 - i):
                                if counts_copy[j] > counts_copy[j + 1]:
                                    tmp_c = counts_copy[j]
                                    counts_copy[j] = counts_copy[j + 1]
                                    counts_copy[j + 1] = tmp_c
                                    tmp_l = lbls[j]
                                    lbls[j] = lbls[j + 1]
                                    lbls[j + 1] = tmp_l

                        second_count = counts_copy[6]
                        if (float(second_count) / float(cnt)) >= 0.30:
                            sec_label = lbls[6]
                            dom_l = lbls[7]
                            best_cls = -1
                            best_w = 0.0
                            for l in (sec_label, dom_l):
                                w = 0.0
                                if l == 5:
                                    w = 1.0
                                elif l == 6:
                                    w = 0.9
                                elif l == 4:
                                    w = 0.6
                                if w > best_w:
                                    best_w = w
                                    best_cls = l
                            if best_cls >= 0:
                                sec_r_out[sec_count] = r
                                sec_c_out[sec_count] = c
                                sec_cls_out[sec_count] = best_cls
                                sec_count += 1
        return sec_count

    # Warmup JIT
    _w_cnt = np.zeros((1, 1), dtype=np.int32)
    _w_lh = np.zeros((1, 1, 8), dtype=np.int32)
    _w_cx = np.zeros(1, dtype=np.float32)
    _w_cy = np.zeros(1, dtype=np.float32)
    _w_tm = np.zeros((1, 1), dtype=np.bool_)
    _w_dl = np.zeros((1, 1), dtype=np.int32)
    _w_sr = np.zeros(1, dtype=np.int32)
    _w_sc = np.zeros(1, dtype=np.int32)
    _w_scls = np.zeros(1, dtype=np.int32)
    _find_all_candidates_numba(
        _w_cnt, _w_lh, _w_cx, _w_cy, 0.0, 10.0,
        _w_tm, _w_dl, _w_sr, _w_sc, _w_scls
    )


def compute_candidate_risks(
    grid: VarResGrid,
    speed_mps: float = 10.0,
    k: int = 4,
    min_risk: float = 0.01,
    params: Optional[RiskParams] = None,
) -> List[CandidateObject]:
    """Finds high-risk candidate objects outside the fine zone from the coarse grid.

    Risk formula:
        risk = class_weight * path_factor * proximity_factor
    """
    if params is None:
        p = RiskParams(k=k, min_risk=min_risk)
    else:
        p = params

    coarse = grid.coarse
    d_stop = stopping_distance(speed_mps)
    forward_horizon = p.stopping_dist_multiplier * max(d_stop, 5.0)

    candidates: List[CandidateObject] = []

    if _HAS_NUMBA:
        target_mask = np.zeros((400, 400), dtype=np.bool_)
        dom_label = np.zeros((400, 400), dtype=np.int32)
        sec_r = np.zeros(1000, dtype=np.int32)
        sec_c = np.zeros(1000, dtype=np.int32)
        sec_cls = np.zeros(1000, dtype=np.int32)

        sec_cnt = _find_all_candidates_numba(
            coarse.count, coarse.label_hist, _COARSE_X_1D, _COARSE_Y_1D,
            np.float32(grid.forward_offset), np.float32(grid.fine_radius),
            target_mask, dom_label, sec_r, sec_c, sec_cls
        )

        if np.any(target_mask):
            labeled_arr, num_features = ndimage.label(target_mask, structure=_STRUCTURE)
            counts = np.bincount(labeled_arr.ravel())
            sum_x = np.bincount(labeled_arr.ravel(), weights=_FULL_X.ravel())
            sum_y = np.bincount(labeled_arr.ravel(), weights=_FULL_Y.ravel())
            cx_all = sum_x[1:] / counts[1:]
            cy_all = sum_y[1:] / counts[1:]

            feat_labels = [[] for _ in range(num_features + 1)]
            t_rows, t_cols = np.where(target_mask)
            f_ids = labeled_arr[t_rows, t_cols]
            l_vals = dom_label[t_rows, t_cols]
            for fid, lval in zip(f_ids, l_vals):
                feat_labels[fid].append(lval)

            for feat_id in range(1, num_features + 1):
                cx = float(cx_all[feat_id - 1])
                cy = float(cy_all[feat_id - 1])

                labels_in_feat = feat_labels[feat_id]
                cls_present = [c for c in [5, 6, 4] if c in labels_in_feat]
                best_cls = cls_present[0] if cls_present else int(labels_in_feat[0])
                w_class = CLASS_WEIGHTS.get(best_cls, 0.5)

                dy = max(0.0, abs(cy) - p.corridor_half_width)
                f_y = float(np.exp(-(dy ** 2) / (2.0 * (6.0 ** 2))))

                if cx < 0.0:
                    dx = abs(cx)
                    f_x = float(np.exp(-(dx ** 2) / (2.0 * (5.0 ** 2))))
                elif cx > forward_horizon:
                    dx = cx - forward_horizon
                    f_x = float(np.exp(-(dx ** 2) / (2.0 * (10.0 ** 2))))
                else:
                    f_x = 1.0

                path_factor = f_y * f_x
                dist = float(np.hypot(cx, cy))
                proximity_factor = float(np.clip(1.0 - dist / 100.0, 0.0, 1.0))

                risk = w_class * path_factor * proximity_factor
                if risk >= p.min_risk:
                    candidates.append(
                        CandidateObject(
                            x=cx,
                            y=cy,
                            risk=risk,
                            dominant_class=best_cls,
                            is_mixed=False,
                        )
                    )

        for i in range(sec_cnt):
            r_idx = sec_r[i]
            c_idx = sec_c[i]
            best_c = int(sec_cls[i])
            w_class = CLASS_WEIGHTS[best_c] * p.mixed_cell_weight_factor
            cx = float(_FULL_X[r_idx, c_idx])
            cy = float(_FULL_Y[r_idx, c_idx])
            dy = max(0.0, abs(cy) - p.corridor_half_width)
            f_y = float(np.exp(-(dy ** 2) / (2.0 * (4.0 ** 2))))
            if cx < 0.0:
                dx = abs(cx)
                f_x = float(np.exp(-(dx ** 2) / (2.0 * (5.0 ** 2))))
            elif cx > forward_horizon:
                dx = cx - forward_horizon
                f_x = float(np.exp(-(dx ** 2) / (2.0 * (10.0 ** 2))))
            else:
                f_x = 1.0
            dist = float(np.hypot(cx, cy))
            proximity_factor = float(np.clip(1.0 - dist / 100.0, 0.0, 1.0))
            risk = w_class * (f_y * f_x) * proximity_factor
            if risk >= p.min_risk:
                candidates.append(
                    CandidateObject(
                        x=cx,
                        y=cy,
                        risk=risk,
                        dominant_class=best_c,
                        is_mixed=True,
                    )
                )

        candidates.sort(key=lambda c: c.risk, reverse=True)
        return candidates[: p.k]

    # NumPy Fallback
    occupied = coarse.count > 0
    grid_x = _COARSE_X
    grid_y = _COARSE_Y

    r_fine = np.hypot(grid_x - grid.forward_offset, grid_y)
    r_sensor = np.hypot(grid_x, grid_y)
    outside_fine = (r_fine >= grid.fine_radius) & (r_sensor < 100.0)

    dom_label = coarse.dominant_label

    # 1. Primary candidates
    target_mask = occupied & outside_fine & np.isin(dom_label, [4, 5, 6])

    if np.any(target_mask):
        structure = ndimage.generate_binary_structure(2, 2)
        labeled_arr, num_features = ndimage.label(target_mask, structure=structure)
        full_x = np.broadcast_to(grid_x, (400, 400))
        full_y = np.broadcast_to(grid_y, (400, 400))

        for feat_id in range(1, num_features + 1):
            feat_mask = labeled_arr == feat_id
            cx = float(np.mean(full_x[feat_mask]))
            cy = float(np.mean(full_y[feat_mask]))

            labels_in_feat = dom_label[feat_mask]
            cls_present = [c for c in [5, 6, 4] if np.any(labels_in_feat == c)]
            best_cls = cls_present[0] if cls_present else int(labels_in_feat[0])
            w_class = CLASS_WEIGHTS.get(best_cls, 0.5)

            dy = max(0.0, abs(cy) - p.corridor_half_width)
            f_y = float(np.exp(-(dy ** 2) / (2.0 * (6.0 ** 2))))

            if cx < 0.0:
                dx = abs(cx)
                f_x = float(np.exp(-(dx ** 2) / (2.0 * (5.0 ** 2))))
            elif cx > forward_horizon:
                dx = cx - forward_horizon
                f_x = float(np.exp(-(dx ** 2) / (2.0 * (10.0 ** 2))))
            else:
                f_x = 1.0

            path_factor = f_y * f_x
            dist = float(np.hypot(cx, cy))
            proximity_factor = float(np.clip(1.0 - dist / 100.0, 0.0, 1.0))

            risk = w_class * path_factor * proximity_factor
            if risk >= p.min_risk:
                candidates.append(
                    CandidateObject(
                        x=cx,
                        y=cy,
                        risk=risk,
                        dominant_class=best_cls,
                        is_mixed=False,
                    )
                )

    # 2. Secondary candidates
    has_relevant = (coarse.label_hist[:, :, 4] > 0) | (coarse.label_hist[:, :, 5] > 0) | (coarse.label_hist[:, :, 6] > 0)
    cand_cells_mask = occupied & outside_fine & (~target_mask) & (coarse.count >= 2) & has_relevant

    if np.any(cand_cells_mask):
        rows, cols = np.where(cand_cells_mask)
        coarse_hist = coarse.label_hist
        full_x = np.broadcast_to(grid_x, (400, 400))
        full_y = np.broadcast_to(grid_y, (400, 400))
        for r_idx, c_idx in zip(rows, cols):
            h = coarse_hist[r_idx, c_idx]
            total_pts = coarse.count[r_idx, c_idx]

            sorted_h = np.sort(h)
            second_count = sorted_h[-2]
            if (second_count / total_pts) >= 0.30:
                top_labels = np.argsort(h)
                sec_label = int(top_labels[-2])
                dom_l = int(top_labels[-1])

                relevant_cls = [l for l in [sec_label, dom_l] if l in CLASS_WEIGHTS]
                if relevant_cls:
                    best_cls = max(relevant_cls, key=lambda l: CLASS_WEIGHTS[l])
                    w_class = CLASS_WEIGHTS[best_cls] * p.mixed_cell_weight_factor

                    cx = float(full_x[r_idx, c_idx])
                    cy = float(full_y[r_idx, c_idx])

                    dy = max(0.0, abs(cy) - p.corridor_half_width)
                    f_y = float(np.exp(-(dy ** 2) / (2.0 * (4.0 ** 2))))

                    if cx < 0.0:
                        dx = abs(cx)
                        f_x = float(np.exp(-(dx ** 2) / (2.0 * (5.0 ** 2))))
                    elif cx > forward_horizon:
                        dx = cx - forward_horizon
                        f_x = float(np.exp(-(dx ** 2) / (2.0 * (10.0 ** 2))))
                    else:
                        f_x = 1.0

                    dist = float(np.hypot(cx, cy))
                    proximity_factor = float(np.clip(1.0 - dist / 100.0, 0.0, 1.0))

                    risk = w_class * (f_y * f_x) * proximity_factor
                    if risk >= p.min_risk:
                        candidates.append(
                            CandidateObject(
                                x=cx,
                                y=cy,
                                risk=risk,
                                dominant_class=best_cls,
                                is_mixed=True,
                            )
                        )

    candidates.sort(key=lambda c: c.risk, reverse=True)
    return candidates[: p.k]
