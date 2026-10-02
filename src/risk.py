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

    Where:
        - class_weight: person (5) = 1.0, moving object (6) = 0.9, vehicle (4) = 0.6.
          Secondary mixed histogram cells receive 0.5 * class_weight.
        - path_factor: 1.0 if object center is inside the driving corridor
          (|y| <= 2.0 m, 0 <= x <= 1.5 * stopping_distance), falling off smoothly
          outside via a Gaussian decay:
              path_factor = exp(-delta_y^2 / (2 * sigma_y^2)) * exp(-delta_x^2 / (2 * sigma_x^2))
        - proximity_factor: clip(1.0 - distance / 100.0, 0.0, 1.0) where distance = sqrt(x^2 + y^2).

    Returns:
        Top K candidates sorted in descending order of risk score.
    """
    if params is None:
        p = RiskParams(k=k, min_risk=min_risk)
    else:
        p = params

    coarse = grid.coarse
    occupied = coarse.count > 0

    # Grid cell coordinates in metres (400 x 400 covering [-100, 100) at 0.5 m)
    grid_x = -100.0 + (np.tile(np.arange(400), (400, 1)) + 0.5) * 0.5
    grid_y = -100.0 + (np.tile(np.arange(400)[:, None], (1, 400)) + 0.5) * 0.5

    r_fine = np.hypot(grid_x - grid.forward_offset, grid_y)
    r_sensor = np.hypot(grid_x, grid_y)

    outside_fine = (r_fine >= grid.fine_radius) & (r_sensor < 100.0)

    d_stop = stopping_distance(speed_mps)
    forward_horizon = p.stopping_dist_multiplier * max(d_stop, 5.0)

    dom_label = coarse.dominant_label
    candidates: List[CandidateObject] = []

    # 1. Primary candidates: connected components with dominant label in {4, 5, 6}
    target_mask = occupied & outside_fine & np.isin(dom_label, [4, 5, 6])

    if np.any(target_mask):
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        labeled_arr, num_features = ndimage.label(target_mask, structure=structure)

        for feat_id in range(1, num_features + 1):
            feat_mask = labeled_arr == feat_id
            cx = float(np.mean(grid_x[feat_mask]))
            cy = float(np.mean(grid_y[feat_mask]))

            # Determine dominant class in component
            labels_in_feat = dom_label[feat_mask]
            # Pick highest priority class
            cls_present = [c for c in [5, 6, 4] if np.any(labels_in_feat == c)]
            best_cls = cls_present[0] if cls_present else int(labels_in_feat[0])
            w_class = CLASS_WEIGHTS.get(best_cls, 0.5)

            # Path factor
            # Lateral falloff
            dy = max(0.0, abs(cy) - p.corridor_half_width)
            f_y = float(np.exp(-(dy ** 2) / (2.0 * (6.0 ** 2))))

            # Longitudinal falloff
            if cx < 0.0:
                dx = abs(cx)
                f_x = float(np.exp(-(dx ** 2) / (2.0 * (5.0 ** 2))))
            elif cx > forward_horizon:
                dx = cx - forward_horizon
                f_x = float(np.exp(-(dx ** 2) / (2.0 * (10.0 ** 2))))
            else:
                f_x = 1.0

            path_factor = f_y * f_x

            # Proximity factor
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

    # 2. Secondary candidates: coarse cells with mixed label histograms
    # Second most frequent label has at least 30% of cell points
    coarse_hist = coarse.label_hist  # (400, 400, n_classes)
    cand_cells_mask = occupied & outside_fine & (~target_mask)

    if np.any(cand_cells_mask):
        rows, cols = np.where(cand_cells_mask)
        for r_idx, c_idx in zip(rows, cols):
            h = coarse_hist[r_idx, c_idx]
            total_pts = coarse.count[r_idx, c_idx]
            if total_pts < 2:
                continue

            sorted_h = np.sort(h)
            second_count = sorted_h[-2]
            if (second_count / total_pts) >= 0.30:
                # Find which label is the second most common
                top_labels = np.argsort(h)
                sec_label = int(top_labels[-2])
                dom_l = int(top_labels[-1])

                # Check if either top or second label is an obstacle/person/vehicle
                relevant_cls = [l for l in [sec_label, dom_l] if l in CLASS_WEIGHTS]
                if relevant_cls:
                    best_cls = max(relevant_cls, key=lambda l: CLASS_WEIGHTS[l])
                    w_class = CLASS_WEIGHTS[best_cls] * p.mixed_cell_weight_factor

                    cx = float(grid_x[r_idx, c_idx])
                    cy = float(grid_y[r_idx, c_idx])

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

    # Sort candidates in descending order of risk and take top K
    candidates.sort(key=lambda c: c.risk, reverse=True)
    return candidates[: p.k]
