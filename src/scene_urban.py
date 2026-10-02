"""Urban street LiDAR scene generator simulating a 64-beam spinning LiDAR with ego-motion.

Generates realistic street and open-intersection scans with ground rings,
roads, sidewalks with kerbs, buildings, poles, parked cars, and pedestrians.
World geometry is positioned in world coordinates and shifted by -s (metres travelled)
with spatial periodicity so objects stream past the moving vehicle.
"""
from typing import Tuple
import numpy as np


def _ray_aabb_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Ray-AABB intersection for parallel ray directions."""
    eps = 1e-7
    inv_dir = 1.0 / np.where(np.abs(ray_dir) < eps, np.sign(ray_dir) * eps + (ray_dir == 0) * eps, ray_dir)

    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir

    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    t_enter = np.max(t_near, axis=-1)
    t_exit = np.min(t_far, axis=-1)

    hit_mask = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    return hit_mask, t_enter


def make_urban_scene(
    frame_idx: int = 0,
    seed: int = 42,
    layout: str = "open",
    s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates a 64-beam spinning LiDAR scan in an urban scene with ego-motion.

    Args:
        frame_idx: frame counter for time-dependent pedestrian walking.
        seed: random seed for range noise.
        layout: "open" (wide 4-way intersection / plaza) or "street" (narrow corridor).
        s: vehicle longitudinal distance travelled in metres (ego-motion).
           World objects shift by -s along x relative to the car.
           Pattern repeats periodically every L = 120 m (city blocks).

    Returns:
        xyz: (N, 3) float32 coordinates relative to the vehicle at (0, 0, 1.7).
        labels: (N,) uint32 semantic class labels (0..7).
    """
    rng = np.random.default_rng(seed + frame_idx * 1007)

    # 1. 64-beam elevation angles and 1800 azimuth steps
    elev_deg = np.linspace(-24.8, 2.0, 64, dtype=np.float32)
    azim_deg = np.linspace(0.0, 360.0, 1800, endpoint=False, dtype=np.float32)

    elev_rad = np.radians(elev_deg)
    azim_rad = np.radians(azim_deg)

    e_grid, a_grid = np.meshgrid(elev_rad, azim_rad, indexing="ij")
    cos_e = np.cos(e_grid).ravel()
    sin_e = np.sin(e_grid).ravel()
    cos_a = np.cos(a_grid).ravel()
    sin_a = np.sin(a_grid).ravel()

    ray_dir = np.empty((len(cos_e), 3), dtype=np.float32)
    ray_dir[:, 0] = cos_e * cos_a
    ray_dir[:, 1] = cos_e * sin_a
    ray_dir[:, 2] = sin_e

    sensor_origin = np.array([0.0, 0.0, 1.7], dtype=np.float32)
    max_range = 105.0

    t_min = np.full(len(ray_dir), np.inf, dtype=np.float32)
    labels = np.zeros(len(ray_dir), dtype=np.uint32)

    # 2. Ego-motion parameters: periodicity L = 120 m
    period_L = 120.0
    s_mod = float(s) % period_L

    # Downward rays hitting ground
    down_mask = ray_dir[:, 2] < -1e-4
    idx_down = np.where(down_mask)[0]
    dz = ray_dir[down_mask, 2]
    dx = ray_dir[down_mask, 0]
    dy = ray_dir[down_mask, 1]

    # Time for walking pedestrians: 10 Hz LiDAR -> 0.1s per frame
    # Pedestrian walks at ~1.4 m/s in world coordinates
    ped_time = float(frame_idx) * 0.1

    if layout == "open":
        # Ground hit at z = 0.0 (Road):
        t0 = -1.7 / dz
        x0_rel = t0 * dx
        y0_rel = t0 * dy

        # Convert vehicle-relative x to periodic world coordinate
        x0_world = (x0_rel + s_mod + period_L / 2.0) % period_L - period_L / 2.0

        # Road along x (|y| <= 4.5m) OR intersection crossing road along y (|x_world| <= 4.5m)
        road_hit = (t0 > 0.0) & (t0 < max_range) & ((np.abs(y0_rel) <= 4.5) | (np.abs(x0_world) <= 4.5))
        idx_road = idx_down[road_hit]
        t_min[idx_road] = t0[road_hit]
        labels[idx_road] = 1  # Road

        # Ground hit at z = 0.15 (Sidewalk):
        t15 = -1.55 / dz
        x15_rel = t15 * dx
        y15_rel = t15 * dy
        x15_world = (x15_rel + s_mod + period_L / 2.0) % period_L - period_L / 2.0

        walk_geom = (
            ((np.abs(y15_rel) > 4.5) & (np.abs(y15_rel) <= 8.0))
            | ((np.abs(x15_world) > 4.5) & (np.abs(x15_world) <= 8.0))
        )
        walk_hit = (t15 > 0.0) & (t15 < max_range) & walk_geom & (~road_hit)
        idx_walk = idx_down[walk_hit]
        t_min[idx_walk] = t15[walk_hit]
        labels[idx_walk] = 7  # Sidewalk

        # Plaza / Terrain beyond sidewalks
        terr_hit = (t15 > 0.0) & (t15 < max_range) & (~road_hit) & (~walk_hit)
        idx_terr = idx_down[terr_hit]
        t_min[idx_terr] = t15[terr_hit]
        labels[idx_terr] = 2  # Terrain / Plaza

        # Template building blocks in world coordinates [-60, 60]
        # Shifted by -s_mod and replicated in adjacent cycles to cover [-105, 105] m
        base_bldgs = [
            # Quadrant 1 (+x, +y)
            (np.array([ 25.0,  25.0, 0.0], dtype=np.float32), np.array([ 42.0,  42.0, 16.0], dtype=np.float32)),
            (np.array([ 48.0,  25.0, 0.0], dtype=np.float32), np.array([ 70.0,  42.0, 18.0], dtype=np.float32)),
            (np.array([ 25.0,  48.0, 0.0], dtype=np.float32), np.array([ 42.0,  70.0, 15.0], dtype=np.float32)),
            # Quadrant 2 (-x, +y)
            (np.array([-42.0,  25.0, 0.0], dtype=np.float32), np.array([-25.0,  42.0, 16.0], dtype=np.float32)),
            (np.array([-70.0,  25.0, 0.0], dtype=np.float32), np.array([-48.0,  42.0, 18.0], dtype=np.float32)),
            (np.array([-42.0,  48.0, 0.0], dtype=np.float32), np.array([-25.0,  70.0, 15.0], dtype=np.float32)),
            # Quadrant 3 (-x, -y)
            (np.array([-42.0, -42.0, 0.0], dtype=np.float32), np.array([-25.0, -25.0, 16.0], dtype=np.float32)),
            (np.array([-70.0, -42.0, 0.0], dtype=np.float32), np.array([-48.0, -25.0, 18.0], dtype=np.float32)),
            (np.array([-42.0, -70.0, 0.0], dtype=np.float32), np.array([-25.0, -48.0, 15.0], dtype=np.float32)),
            # Quadrant 4 (+x, -y)
            (np.array([ 25.0, -42.0, 0.0], dtype=np.float32), np.array([ 42.0, -25.0, 16.0], dtype=np.float32)),
            (np.array([ 48.0, -42.0, 0.0], dtype=np.float32), np.array([ 70.0, -25.0, 18.0], dtype=np.float32)),
            (np.array([ 25.0, -70.0, 0.0], dtype=np.float32), np.array([ 42.0, -48.0, 15.0], dtype=np.float32)),
        ]

        base_cars = [
            (np.array([ 12.0, -4.2, 0.0], dtype=np.float32), np.array([ 16.5, -2.4, 1.5], dtype=np.float32)),
            (np.array([-18.0,  2.4, 0.0], dtype=np.float32), np.array([-13.5,  4.2, 1.5], dtype=np.float32)),
            (np.array([-4.2,  14.0, 0.0], dtype=np.float32), np.array([-2.4,  18.5, 1.5], dtype=np.float32)),
            (np.array([ 2.4, -20.0, 0.0], dtype=np.float32), np.array([ 4.2, -15.5, 1.5], dtype=np.float32)),
            (np.array([ 24.0,  2.4, 0.0], dtype=np.float32), np.array([ 28.5,  4.2, 1.6], dtype=np.float32)),
        ]

        base_poles = [
            (20.0, 5.5),  # Reference pole at x = 20 m
            (6.5, 6.5), (-6.5, 6.5), (-6.5, -6.5), (6.5, -6.5),
            (-20.0, 5.5), (20.0, -5.5), (-20.0, -5.5),
            (5.5, 20.0), (-5.5, 20.0), (5.5, -20.0), (-5.5, -20.0),
        ]

        # Pedestrians in world coordinates walking at 1.4 m/s
        base_peds = [
            (10.0 + 1.4 * ped_time, 5.8),
            (-12.0 - 1.2 * ped_time, -6.0),
            (7.5, -10.0 + 1.3 * ped_time),
        ]

    else:
        # Narrow street layout
        t_road = -1.7 / dz
        y_road = sensor_origin[1] + t_road * dy
        road_hit = (t_road > 0.0) & (t_road < max_range) & (np.abs(y_road) <= 3.5)
        idx_road = idx_down[road_hit]
        t_min[idx_road] = t_road[road_hit]
        labels[idx_road] = 1

        t_walk = -1.55 / dz
        y_walk = sensor_origin[1] + t_walk * dy
        walk_hit = (t_walk > 0.0) & (t_walk < max_range) & (np.abs(y_walk) > 3.5) & (np.abs(y_walk) <= 6.5)
        idx_walk = idx_down[walk_hit]
        t_min[idx_walk] = t_walk[walk_hit]
        labels[idx_walk] = 7

        t_terr = -1.50 / dz
        y_terr = sensor_origin[1] + t_terr * dy
        terr_hit = (t_terr > 0.0) & (t_terr < max_range) & (np.abs(y_terr) > 6.5) & (np.abs(y_terr) <= 12.0)
        idx_terr = idx_down[terr_hit]
        t_min[idx_terr] = t_terr[terr_hit]
        labels[idx_terr] = 2

        base_bldgs = [
            (np.array([-50.0,  12.0, 0.0], dtype=np.float32), np.array([-10.0,  25.0, 16.0], dtype=np.float32)),
            (np.array([ -8.0,  12.5, 0.0], dtype=np.float32), np.array([ 30.0,  26.0, 18.0], dtype=np.float32)),
            (np.array([ 32.0,  12.0, 0.0], dtype=np.float32), np.array([ 55.0,  25.0, 15.0], dtype=np.float32)),
            (np.array([-50.0, -25.0, 0.0], dtype=np.float32), np.array([-15.0, -12.0, 17.0], dtype=np.float32)),
            (np.array([-13.0, -26.0, 0.0], dtype=np.float32), np.array([ 25.0, -12.5, 19.0], dtype=np.float32)),
            (np.array([ 27.0, -25.0, 0.0], dtype=np.float32), np.array([ 55.0, -12.0, 16.0], dtype=np.float32)),
        ]

        base_cars = [
            (np.array([  8.0, -3.2, 0.0], dtype=np.float32), np.array([ 12.5, -1.5, 1.5], dtype=np.float32)),
            (np.array([ 18.0, -3.2, 0.0], dtype=np.float32), np.array([ 22.6, -1.5, 1.6], dtype=np.float32)),
            (np.array([-16.0, -3.2, 0.0], dtype=np.float32), np.array([-11.5, -1.5, 1.5], dtype=np.float32)),
            (np.array([ 14.0,  1.5, 0.0], dtype=np.float32), np.array([ 18.5,  3.2, 1.5], dtype=np.float32)),
            (np.array([ 30.0,  1.5, 0.0], dtype=np.float32), np.array([ 34.7,  3.2, 1.6], dtype=np.float32)),
        ]

        base_poles = [
            (20.0, 4.2),  # Reference pole at x = 20 m
            (-30.0, 4.2), (-15.0, 4.2), (5.0, 4.2), (38.0, 4.2), (55.0, 4.2),
            (-30.0, -4.2), (-15.0, -4.2), (5.0, -4.2), (20.0, -4.2), (38.0, -4.2), (55.0, -4.2),
        ]

        base_peds = [
            (10.0 + 1.4 * ped_time, 4.6),
            (28.0 - 1.3 * ped_time, -4.5),
        ]

    # Shift world objects by -s_mod with periodic repetition across adjacent cycles
    # Cycle offsets: k in [-1, 0, 1] ensures 360° coverage in [-105, 105] m relative to car
    cycle_offsets = [-period_L, 0.0, period_L]

    # 3. Raycast Buildings (Class 3)
    for k_off in cycle_offsets:
        shift_x = k_off - s_mod
        for b_min, b_max in base_bldgs:
            box_min_shifted = np.array([b_min[0] + shift_x, b_min[1], b_min[2]], dtype=np.float32)
            box_max_shifted = np.array([b_max[0] + shift_x, b_max[1], b_max[2]], dtype=np.float32)
            # Skip if box is completely outside visible sensor range
            if box_max_shifted[0] < -105.0 or box_min_shifted[0] > 105.0:
                continue
            hit, t_hit = _ray_aabb_intersect(sensor_origin, ray_dir, box_min_shifted, box_max_shifted)
            closer = hit & (t_hit < t_min) & (t_hit > 0.0) & (t_hit < max_range)
            t_min[closer] = t_hit[closer]
            labels[closer] = 3

    # 4. Raycast Poles (Class 3)
    for k_off in cycle_offsets:
        shift_x = k_off - s_mod
        for px, py in base_poles:
            p_rel_x = px + shift_x
            if p_rel_x < -105.0 or p_rel_x > 105.0:
                continue
            p_min = np.array([p_rel_x - 0.12, py - 0.12, 0.15], dtype=np.float32)
            p_max = np.array([p_rel_x + 0.12, py + 0.12, 5.50], dtype=np.float32)
            hit, t_hit = _ray_aabb_intersect(sensor_origin, ray_dir, p_min, p_max)
            closer = hit & (t_hit < t_min) & (t_hit > 0.0)
            t_min[closer] = t_hit[closer]
            labels[closer] = 3

    # 5. Raycast Parked Cars (Class 4)
    for k_off in cycle_offsets:
        shift_x = k_off - s_mod
        for c_min, c_max in base_cars:
            c_min_shifted = np.array([c_min[0] + shift_x, c_min[1], c_min[2]], dtype=np.float32)
            c_max_shifted = np.array([c_max[0] + shift_x, c_max[1], c_max[2]], dtype=np.float32)
            if c_max_shifted[0] < -105.0 or c_min_shifted[0] > 105.0:
                continue
            hit, t_hit = _ray_aabb_intersect(sensor_origin, ray_dir, c_min_shifted, c_max_shifted)
            closer = hit & (t_hit < t_min) & (t_hit > 0.0)
            t_min[closer] = t_hit[closer]
            labels[closer] = 4

    # 6. Raycast Pedestrians (Class 5)
    for k_off in cycle_offsets:
        shift_x = k_off - s_mod
        for ped_x, ped_y in base_peds:
            # Wrap pedestrian position within cycle
            ped_x_mod = (ped_x + period_L / 2.0) % period_L - period_L / 2.0
            p_rel_x = ped_x_mod + shift_x
            if p_rel_x < -105.0 or p_rel_x > 105.0:
                continue
            p_min = np.array([p_rel_x - 0.25, ped_y - 0.25, 0.15], dtype=np.float32)
            p_max = np.array([p_rel_x + 0.25, ped_y + 0.25, 1.90], dtype=np.float32)
            hit, t_hit = _ray_aabb_intersect(sensor_origin, ray_dir, p_min, p_max)
            closer = hit & (t_hit < t_min) & (t_hit > 0.0)
            t_min[closer] = t_hit[closer]
            labels[closer] = 5

    # 7. Filter valid physical hits and apply Gaussian range noise
    valid = np.isfinite(t_min) & (t_min > 0.2) & (t_min <= max_range)
    t_valid = t_min[valid]
    dir_valid = ray_dir[valid]
    lbl_valid = labels[valid]

    noise = rng.normal(0.0, 0.015, size=len(t_valid)).astype(np.float32)
    t_noisy = np.maximum(0.1, t_valid + noise)

    xyz = sensor_origin + dir_valid * t_noisy[:, None]

    return xyz.astype(np.float32), lbl_valid.astype(np.uint32)
