import numpy as np
import pytest
from src.grid_engine import VarResGrid, stopping_distance
from src.risk import compute_candidate_risks, RiskParams, CandidateObject


def test_risk_ranking_person_in_path_vs_side_vs_building():
    # Coarse grid spans [-100, 100) m with 0.5 m cells (400 x 400).
    # Sensor at (0, 0).
    grid = VarResGrid(n_classes=8, fine_radius=10.0)

    # 1. Person in direct driving path ahead at x = 25m, y = 0.5m (inside corridor)
    pts_person_path = np.array([[25.0, 0.5, 0.5]] * 20, dtype=np.float32)
    labels_person_path = np.full(20, 5, dtype=np.int64)  # class 5: person

    # 2. Person off to the side at x = 25m, y = 12.0m (outside corridor)
    pts_person_side = np.array([[25.0, 12.0, 0.5]] * 20, dtype=np.float32)
    labels_person_side = np.full(20, 5, dtype=np.int64)  # class 5: person

    # 3. Building at x = 30m, y = 0.0m (in path, but class 3: static obstacle, not candidate)
    pts_building = np.array([[30.0, 0.0, 1.5]] * 20, dtype=np.float32)
    labels_building = np.full(20, 3, dtype=np.int64)  # class 3: building/obstacle

    all_pts = np.vstack([pts_person_path, pts_person_side, pts_building])
    all_labels = np.concatenate([labels_person_path, labels_person_side, labels_building])
    grid.add_points(all_pts, all_labels)

    speed_mps = 15.0  # stopping distance ~ 33.75 m
    candidates = compute_candidate_risks(grid, speed_mps=speed_mps, k=4)

    # Building must not be a candidate
    classes_found = [c.dominant_class for c in candidates]
    assert 3 not in classes_found, "Static building must not be a focus candidate"

    # Must find candidates for person
    assert len(candidates) >= 2
    in_path_cand = next(c for c in candidates if abs(c.y) < 2.0)
    side_cand = next(c for c in candidates if abs(c.y) > 5.0)

    # Person in path must score strictly higher than person off to the side
    assert in_path_cand.risk > side_cand.risk


def test_top_k_candidates_cap():
    grid = VarResGrid(n_classes=8, fine_radius=10.0)

    # Place 8 vehicles at different positions outside the fine zone
    pts_list = []
    labels_list = []
    for i in range(8):
        x = 20.0 + i * 4.0
        y = (i - 4) * 2.5
        pts_list.append(np.array([[x, y, 0.5]] * 10, dtype=np.float32))
        labels_list.append(np.full(10, 4, dtype=np.int64))  # vehicle (4)

    grid.add_points(np.vstack(pts_list), np.concatenate(labels_list))

    candidates = compute_candidate_risks(grid, speed_mps=12.0, k=4)
    assert len(candidates) <= 4, "Must cap candidates at top K"
    # Ensure sorted by descending risk
    risks = [c.risk for c in candidates]
    assert risks == sorted(risks, reverse=True)


def test_mixed_histogram_cell_candidate():
    grid = VarResGrid(n_classes=8, fine_radius=10.0)

    # Coarse cell at x = 20.25, y = 0.25 with 60% road (1) and 40% person (5)
    # Dominant is road, but second most common is person with >= 30% points
    pts_road = np.array([[20.25, 0.25, 0.0]] * 6, dtype=np.float32)
    labels_road = np.full(6, 1, dtype=np.int64)

    pts_person = np.array([[20.25, 0.25, 0.5]] * 4, dtype=np.float32)
    labels_person = np.full(4, 5, dtype=np.int64)

    grid.add_points(np.vstack([pts_road, pts_person]), np.concatenate([labels_road, labels_person]))

    candidates = compute_candidate_risks(grid, speed_mps=15.0, k=4, min_risk=0.01)
    # Should detect the mixed cell candidate
    assert len(candidates) >= 1
    assert any(c.is_mixed for c in candidates)
