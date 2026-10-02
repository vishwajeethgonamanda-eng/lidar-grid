"""Ego vehicle model and styling engine for RAIL-2.5D.

Defines the geometry, colors, 3D meshes, and 2D top-down representations
for the ego crossover vehicle, other road vehicles, and decorative sensing rings.
"""
import functools
from typing import Any, Dict, List, Tuple
import numpy as np


# Color constants
CAR_BODY_RED = "#B3202F"      # Primary crossover red
CAR_BODY_DARK = "#7E1420"     # Shadow / lower trim red
CAR_GLASS = "#141B26"         # Dark glass cabin & windshield
CAR_TRIM = "#1A1A1A"          # Dark roof trim & wheels
HEADLIGHT = "#F4F7FA"         # Bright white headlights
TAILLIGHT = "#D7263D"         # Rear tail lights
RING_TEAL = "#3FD0D4"         # Sensing pulse teal
OTHER_CAR_BLUE = "#2B3FA3"    # Secondary vehicle blue
OTHER_CAR_DARK = "#1C2430"    # Secondary vehicle dark grey


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Converts '#RRGGBB' to an (R, G, B) integer tuple."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


CAR_BODY_RED_RGB = hex_to_rgb(CAR_BODY_RED)
CAR_BODY_DARK_RGB = hex_to_rgb(CAR_BODY_DARK)
CAR_GLASS_RGB = hex_to_rgb(CAR_GLASS)
CAR_TRIM_RGB = hex_to_rgb(CAR_TRIM)
HEADLIGHT_RGB = hex_to_rgb(HEADLIGHT)
TAILLIGHT_RGB = hex_to_rgb(TAILLIGHT)
RING_TEAL_RGB = hex_to_rgb(RING_TEAL)
OTHER_CAR_BLUE_RGB = hex_to_rgb(OTHER_CAR_BLUE)
OTHER_CAR_DARK_RGB = hex_to_rgb(OTHER_CAR_DARK)

# Base concentric sensing rings: (radius_m, base_alpha)
BASE_SENSING_RINGS = [
    (3.0, 0.28),
    (5.5, 0.20),
    (8.0, 0.12),
]


def _make_box_mesh_data(
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
    color: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Constructs vertices (8, 3), triangles (12, 3), and colors (12,) for a box."""
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 5, 1], [0, 4, 5],  # front
        [2, 7, 3], [2, 6, 7],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int32)

    colors = [color] * 12
    return corners, triangles, colors


def _make_wedge_mesh_data(
    x_back: float, x_front: float,
    y0: float, y1: float,
    z_base: float, z_top_back: float, z_top_front: float,
    color: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Constructs a sloped hood wedge (6 vertices, 8 triangles)."""
    verts = np.array([
        [x_back,  y0, z_base],        # 0
        [x_front, y0, z_base],        # 1
        [x_front, y1, z_base],        # 2
        [x_back,  y1, z_base],        # 3
        [x_back,  y0, z_top_back],    # 4
        [x_front, y0, z_top_front],   # 5
        [x_front, y1, z_top_front],   # 6
        [x_back,  y1, z_top_back],    # 7
    ], dtype=np.float32)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top sloped
        [0, 5, 1], [0, 4, 5],  # front
        [2, 7, 3], [2, 6, 7],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int32)

    colors = [color] * 12
    return verts, triangles, colors


@functools.lru_cache(maxsize=16)
def _build_car_geometry(body_color: str) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Builds the 3D car mesh geometry for a crossover vehicle with dimensions
    4.6m long, 1.9m wide, 1.6m tall, with origin at the rear axle.
    
    Returns (vertices, faces, colors_tuple). Total triangles < 200.
    """
    all_verts = []
    all_faces = []
    all_colors = []
    v_offset = 0

    def add_submesh(v, f, c):
        nonlocal v_offset
        all_verts.append(v)
        all_faces.append(f + v_offset)
        all_colors.extend(c)
        v_offset += len(v)

    # 1. Lower body / chassis (4.6m long: x in [-0.9, 3.7], y in [-0.95, 0.95], z in [0.25, 0.78])
    v, f, c = _make_box_mesh_data(-0.90, 3.70, -0.95, 0.95, 0.25, 0.78, body_color)
    add_submesh(v, f, c)

    # 2. Sloped hood / bonnet (x in [1.90, 3.65], y in [-0.88, 0.88], sloped from z=0.98 down to z=0.82)
    v, f, c = _make_wedge_mesh_data(1.90, 3.65, -0.88, 0.88, 0.78, 0.98, 0.82, CAR_BODY_DARK)
    add_submesh(v, f, c)

    # 3. Cabin / greenhouse with raked glass (x in [-0.55, 1.90], y in [-0.82, 0.82], z in [0.78, 1.56])
    v, f, c = _make_box_mesh_data(-0.55, 1.90, -0.82, 0.82, 0.78, 1.56, CAR_GLASS)
    add_submesh(v, f, c)

    # 4. Sunroof panel on roof (x in [0.05, 1.15], y in [-0.55, 0.55], z in [1.56, 1.60])
    v, f, c = _make_box_mesh_data(0.05, 1.15, -0.55, 0.55, 1.56, 1.60, CAR_TRIM)
    add_submesh(v, f, c)

    # 5. Four dark wheels (z in [0.0, 0.55], radius 0.38m)
    # Rear wheels centered at x = 0.0, front wheels at x = 2.70
    wheels = [
        # Front Left
        (2.35, 3.05, 0.74, 0.95, 0.0, 0.55),
        # Front Right
        (2.35, 3.05, -0.95, -0.74, 0.0, 0.55),
        # Rear Left
        (-0.35, 0.35, 0.74, 0.95, 0.0, 0.55),
        # Rear Right
        (-0.35, 0.35, -0.95, -0.74, 0.0, 0.55),
    ]
    for w in wheels:
        v, f, c = _make_box_mesh_data(w[0], w[1], w[2], w[3], w[4], w[5], CAR_TRIM)
        add_submesh(v, f, c)

    # 6. Two headlights at the front (x in [3.65, 3.70], y = +-0.65, z in [0.65, 0.80])
    v, f, c = _make_box_mesh_data(3.65, 3.70, 0.50, 0.85, 0.65, 0.80, HEADLIGHT)
    add_submesh(v, f, c)
    v, f, c = _make_box_mesh_data(3.65, 3.70, -0.85, -0.50, 0.65, 0.80, HEADLIGHT)
    add_submesh(v, f, c)

    # 7. Two taillights at the rear (x in [-0.90, -0.85], y = +-0.65, z in [0.65, 0.80])
    v, f, c = _make_box_mesh_data(-0.90, -0.85, 0.50, 0.85, 0.65, 0.80, TAILLIGHT)
    add_submesh(v, f, c)
    v, f, c = _make_box_mesh_data(-0.90, -0.85, -0.85, -0.50, 0.65, 0.80, TAILLIGHT)
    add_submesh(v, f, c)

    concat_verts = np.vstack(all_verts)
    concat_faces = np.vstack(all_faces)
    colors_tuple = tuple(all_colors)

    return concat_verts, concat_faces, colors_tuple


@functools.lru_cache(maxsize=4)
def build_car_mesh(body_color: str = CAR_BODY_RED) -> Dict[str, Any]:
    """Returns cached mesh data dictionary for the ego crossover vehicle."""
    verts, faces, colors = _build_car_geometry(body_color)
    return {
        "vertices": verts,
        "faces": faces,
        "colors": list(colors),
        "triangle_count": len(faces),
        "bounds": {
            "x": (float(verts[:, 0].min()), float(verts[:, 0].max())),
            "y": (float(verts[:, 1].min()), float(verts[:, 1].max())),
            "z": (float(verts[:, 2].min()), float(verts[:, 2].max())),
        },
    }


@functools.lru_cache(maxsize=8)
def build_other_car_mesh(colour: str = OTHER_CAR_BLUE) -> Dict[str, Any]:
    """Returns cached mesh data dictionary for secondary scene vehicles in custom color."""
    return build_car_mesh(body_color=colour)


def top_down_polygons() -> List[Dict[str, Any]]:
    """Returns 2D polygon layers for the crossover vehicle seen from above (+x is forward).
    
    Coordinates are in meters relative to the rear axle:
    Length: [-0.9m, 3.7m], Width: [-0.95m, 0.95m].
    """
    return [
        # 1. Wheels (4 dark rectangles)
        {
            "name": "wheel_fl",
            "polygon": [(2.35, 0.74), (3.05, 0.74), (3.05, 0.95), (2.35, 0.95)],
            "fill": CAR_TRIM_RGB,
            "outline": (10, 10, 10),
        },
        {
            "name": "wheel_fr",
            "polygon": [(2.35, -0.95), (3.05, -0.95), (3.05, -0.74), (2.35, -0.74)],
            "fill": CAR_TRIM_RGB,
            "outline": (10, 10, 10),
        },
        {
            "name": "wheel_rl",
            "polygon": [(-0.35, 0.74), (0.35, 0.74), (0.35, 0.95), (-0.35, 0.95)],
            "fill": CAR_TRIM_RGB,
            "outline": (10, 10, 10),
        },
        {
            "name": "wheel_rr",
            "polygon": [(-0.35, -0.95), (0.35, -0.95), (0.35, -0.74), (-0.35, -0.74)],
            "fill": CAR_TRIM_RGB,
            "outline": (10, 10, 10),
        },
        # 2. Main Red Body (tapered front bumper & aerodynamic contour)
        {
            "name": "body",
            "polygon": [
                (-0.90, -0.92), (3.40, -0.92), (3.70, -0.70), (3.70, 0.70),
                (3.40, 0.92), (-0.90, 0.92), (-0.90, -0.92),
            ],
            "fill": CAR_BODY_RED_RGB,
            "outline": None,
        },
        # 3. Sloped Hood Crease Lines (Bonnet)
        {
            "name": "hood",
            "polygon": [
                (1.90, -0.65), (3.45, -0.55), (3.45, 0.55), (1.90, 0.65),
            ],
            "fill": CAR_BODY_RED_RGB,
            "outline": CAR_BODY_DARK_RGB,
        },
        # 4. Dark Glass Cabin & Windshield
        {
            "name": "glass",
            "polygon": [
                (-0.55, -0.65), (1.90, -0.65), (1.90, 0.65), (-0.55, 0.65),
            ],
            "fill": CAR_GLASS_RGB,
            "outline": CAR_TRIM_RGB,
        },
        # 5. Sunroof Panel
        {
            "name": "sunroof",
            "polygon": [
                (0.05, -0.45), (1.15, -0.45), (1.15, 0.45), (0.05, 0.45),
            ],
            "fill": (15, 18, 24),
            "outline": (35, 42, 54),
        },
        # 6. Front Headlights
        {
            "name": "headlight_l",
            "polygon": [(3.62, 0.50), (3.70, 0.50), (3.70, 0.85), (3.62, 0.85)],
            "fill": HEADLIGHT_RGB,
            "outline": (220, 230, 240),
        },
        {
            "name": "headlight_r",
            "polygon": [(3.62, -0.85), (3.70, -0.85), (3.70, -0.50), (3.62, -0.50)],
            "fill": HEADLIGHT_RGB,
            "outline": (220, 230, 240),
        },
        # 7. Rear Taillights
        {
            "name": "taillight_l",
            "polygon": [(-0.90, 0.50), (-0.84, 0.50), (-0.84, 0.85), (-0.90, 0.85)],
            "fill": TAILLIGHT_RGB,
            "outline": (160, 20, 35),
        },
        {
            "name": "taillight_r",
            "polygon": [(-0.90, -0.85), (-0.84, -0.85), (-0.84, -0.50), (-0.90, -0.50)],
            "fill": TAILLIGHT_RGB,
            "outline": (160, 20, 35),
        },
    ]


def compute_sensing_rings(
    mode: str = "Pulse",
    current_time: float = 0.0,
) -> List[Tuple[float, float]]:
    """Calculates effective radius and alpha for the 3 concentric sensing rings.
    
    Mode:
    - 'Off': returns empty list
    - 'Static': returns fixed radii (3.0, 5.5, 8.0 m) and alphas (0.28, 0.20, 0.12)
    - 'Pulse': pulse cycle of 1.2s expanding radius by up to 1.5m and fading alpha
    """
    if mode == "Off":
        return []

    if mode == "Static":
        return [(r, a) for r, a in BASE_SENSING_RINGS]

    # Pulse mode (1.2 second period)
    cycle = 1.2
    phase = (current_time % cycle) / cycle
    pulse_dr = 1.5 * phase
    pulse_fade = 1.0 - 0.55 * phase

    rings = []
    for base_r, base_a in BASE_SENSING_RINGS:
        r = base_r + pulse_dr
        a = float(np.clip(base_a * pulse_fade, 0.02, 0.35))
        rings.append((r, a))

    return rings
