"""High-performance NumPy/PIL rasterization engine for RAIL-2.5D.

Provides sub-30ms vectorized rendering for top-down grid maps, raw LiDAR points,
and pseudo-3D perspective road corridor views.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.grid_engine import VarResGrid
from src.traversability import (
    TraversabilityMap,
    UNKNOWN,
    DRIVABLE,
    NON_DRIVABLE,
    OBSTACLE,
)
from src.car_model import (
    top_down_polygons,
    compute_sensing_rings,
    CAR_BODY_RED_RGB,
    OTHER_CAR_BLUE_RGB,
    OTHER_CAR_DARK_RGB,
    RING_TEAL_RGB,
    build_car_mesh,
    hex_to_rgb,
)


# Shared RGB uint8 color constants
BG_COLOR = np.array([20, 22, 28], dtype=np.uint8)           # Dark background (#14161c)
UNKNOWN_COLOR = np.array([35, 38, 45], dtype=np.uint8)       # Unknown cell (#23262d)
DRIVABLE_COLOR = np.array([46, 204, 113], dtype=np.uint8)     # Navigable (#2ecc71)
NON_DRIVABLE_COLOR = np.array([243, 156, 18], dtype=np.uint8) # Warning / slope limit (#f39c12)
OBSTACLE_COLOR = np.array([231, 76, 60], dtype=np.uint8)      # Obstacle / hazard (#e74c3c)
CAR_BODY_COLOR = np.array(CAR_BODY_RED_RGB, dtype=np.uint8)   # Red crossover (#B3202F)

STATE_COLORS = {
    UNKNOWN: UNKNOWN_COLOR,
    DRIVABLE: DRIVABLE_COLOR,
    NON_DRIVABLE: NON_DRIVABLE_COLOR,
    OBSTACLE: OBSTACLE_COLOR,
}

STATE_PALETTE = np.array([
    UNKNOWN_COLOR,
    DRIVABLE_COLOR,
    NON_DRIVABLE_COLOR,
    OBSTACLE_COLOR,
], dtype=np.uint8)

SEMANTIC_PALETTE = np.array([
    [40, 40, 45],     # 0 Unlabeled
    [52, 152, 219],   # 1 Road (Blue)
    [39, 174, 96],    # 2 Terrain (Green)
    [149, 165, 166],  # 3 Static Obstacle (Grey)
    [155, 89, 182],   # 4 Vehicle (Purple default)
    [230, 126, 34],   # 5 Person (Orange)
    [231, 76, 60],    # 6 Moving Object (Red)
    [241, 196, 15],   # 7 Non-drivable Ground / Sidewalk (Yellow)
], dtype=np.uint8)

SEMANTIC_COLORS = {i: SEMANTIC_PALETTE[i] for i in range(8)}

STATE_HEX_COLORS = {
    UNKNOWN: "#23262d",
    DRIVABLE: "#2ecc71",
    NON_DRIVABLE: "#f39c12",
    OBSTACLE: "#e74c3c",
}

SEMANTIC_HEX_COLORS = {
    0: "#28282d",
    1: "#3498db",
    2: "#27ae60",
    3: "#95a5a6",
    4: "#9b59b6",
    5: "#e67e22",
    6: "#e74c3c",
    7: "#f1c40f",
}

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Preloaded fonts
_FONT_RING = ImageFont.load_default(size=14)
_FONT_LABEL = ImageFont.load_default(size=13)

# Precomputed 3D Car Mesh
_CAR_MESH_DATA = build_car_mesh()
_CAR_MESH_VERTICES = np.ascontiguousarray(_CAR_MESH_DATA["vertices"], dtype=np.float32)
_CAR_MESH_FACES = _CAR_MESH_DATA["faces"]
_CAR_MESH_COLORS = [hex_to_rgb(c) for c in _CAR_MESH_DATA["colors"]]

# Precomputed 800x800 range mask: distance >= 100m outside circular bounds
_CANVAS_SIZE = 800
_GRID_Y, _GRID_X = np.ogrid[:_CANVAS_SIZE, :_CANVAS_SIZE]
_DIST_FROM_CENTER = np.hypot(_GRID_X - 400.0, _GRID_Y - 400.0)
_MASK_OUTSIDE_100M = _DIST_FROM_CENTER >= 400.0

if _HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _mask_outside_circle_numba(img, bg_r, bg_g, bg_b):
        rows, cols, _ = img.shape
        cr = rows // 2
        cc = cols // 2
        r_sq = (rows // 2) ** 2
        for r in range(rows):
            dr = r - cr
            dr2 = dr * dr
            for c in range(cols):
                dc = c - cc
                if dr2 + dc * dc >= r_sq:
                    img[r, c, 0] = bg_r
                    img[r, c, 1] = bg_g
                    img[r, c, 2] = bg_b

    @njit(fastmath=True, cache=True)
    def _paint_columns_numba(img, y0, y1, x0, x1, colors):
        n = len(y0)
        for i in range(n):
            r0 = y0[i]
            r1 = y1[i] + 1
            c0 = x0[i]
            c1 = x1[i]
            cr = colors[i, 0]
            cg = colors[i, 1]
            cb = colors[i, 2]
            for r in range(r0, r1):
                for c in range(c0, c1):
                    img[r, c, 0] = cr
                    img[r, c, 1] = cg
                    img[r, c, 2] = cb

    # Warmup
    _w_img = np.zeros((2, 2, 3), dtype=np.uint8)
    _mask_outside_circle_numba(_w_img, 20, 22, 28)
    _paint_columns_numba(_w_img, np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32),
                         np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32),
                         np.zeros((1, 3), dtype=np.uint8))


def _draw_topdown_car_icon(draw: ImageDraw.ImageDraw, center: int = 400, scale: float = 4.0) -> None:
    """Draws the crossover ego vehicle using top_down_polygons() (+x forward is UP)."""
    layers = top_down_polygons()
    for layer in layers:
        poly_px = [(center + y_m * scale, center - x_m * scale) for x_m, y_m in layer["polygon"]]
        fill = layer.get("fill")
        outline = layer.get("outline")
        draw.polygon(poly_px, fill=fill, outline=outline)


def render_grid_map(
    grid: VarResGrid,
    trav: TraversabilityMap,
    view_mode: str = "state",
    canvas_size: int = 800,
    sensing_mode: str = "Pulse",
    current_time: float = 0.0,
) -> np.ndarray:
    """Renders a composite bird's-eye view raster map where +x is forward (pointing UP)
    and +y is lateral (pointing RIGHT).
    
    Fast vectorized mapping: coarse grid expanded by 2x, non-empty fine-zone and focus patch
    cells stamped directly into the buffer, range rings and annotations overlaid via PIL.
    """
    scale = canvas_size / 200.0  # 4.0 px/m for 800x800 canvas covering [-100, 100] m
    center = canvas_size // 2

    # 1. Base Coarse Zone (400x400 cells at 0.5m = 2x2 px per cell)
    if view_mode == "state":
        # trav.coarse.state shape: (400, 400) where row is y, col is x
        c_state = trav.coarse.state
        c_colors = STATE_PALETTE[np.clip(c_state, 0, 3)]
    else:
        c_dom = grid.coarse.dominant_label
        c_cnt = grid.coarse.count
        c_colors = SEMANTIC_PALETTE[np.clip(c_dom, 0, 7)]
        c_colors[c_cnt == 0] = UNKNOWN_COLOR

    # Map to screen: col is x (forward is UP -> flip vertically), row is y (lateral is RIGHT)
    # coarse array has indexing coarse[row_y, col_x].
    # Screen has row_px (along -x), col_px (along +y).
    # transpose(1, 0, 2) turns [y, x] into [x, y].
    # flip along axis 0 turns +x (col 399) to row_px 0 (top of image).
    screen_coarse = np.flip(c_colors.transpose(1, 0, 2), axis=0)
    img = np.repeat(np.repeat(screen_coarse, 2, axis=0), 2, axis=1)

    # 2. Render Focus Patches (over coarse, 64x64 at 5cm)
    if hasattr(grid, "patches") and grid.patches:
        for idx, patch in enumerate(grid.patches):
            p_cnt = patch.count
            p_nz = p_cnt > 0
            if not np.any(p_nz):
                continue
            r_idx, c_idx = np.where(p_nz)

            # Physical coordinates of patch cells
            # patch col is x, row is y
            x_p = patch.center_x - patch.half_extent + (c_idx + 0.5) * patch.cell_size
            y_p = patch.center_y - patch.half_extent + (r_idx + 0.5) * patch.cell_size

            px_r = np.clip(np.floor((100.0 - x_p) * scale).astype(np.int64), 0, canvas_size - 1)
            px_c = np.clip(np.floor((y_p + 100.0) * scale).astype(np.int64), 0, canvas_size - 1)

            if view_mode == "state" and hasattr(trav, "patches") and idx < len(trav.patches):
                p_st = trav.patches[idx].state[r_idx, c_idx]
                p_col = STATE_PALETTE[np.clip(p_st, 0, 3)]
            else:
                p_dom = patch.dominant_label[r_idx, c_idx]
                p_col = SEMANTIC_PALETTE[np.clip(p_dom, 0, 7)]

            img[px_r, px_c] = p_col

    # 3. Render Fine Zone (highest priority: 400x400 cells inside fine_radius)
    f_cnt = grid.fine.count
    f_nz = f_cnt > 0
    if np.any(f_nz):
        r_f, c_f = np.where(f_nz)
        # col_f is x offset, row_f is y offset
        x_f = grid.forward_offset - grid.fine_radius + (c_f + 0.5) * grid.fine_cell_size
        y_f = -grid.fine_radius + (r_f + 0.5) * grid.fine_cell_size

        # Mask strictly within fine_radius
        dist_f = np.hypot(x_f - grid.forward_offset, y_f)
        in_fine = dist_f <= grid.fine_radius
        if np.any(in_fine):
            x_f = x_f[in_fine]
            y_f = y_f[in_fine]
            r_f = r_f[in_fine]
            c_f = c_f[in_fine]

            px_r = np.clip(np.floor((100.0 - x_f) * scale).astype(np.int64), 0, canvas_size - 1)
            px_c = np.clip(np.floor((y_f + 100.0) * scale).astype(np.int64), 0, canvas_size - 1)

            if view_mode == "state":
                f_st = trav.fine.state[r_f, c_f]
                f_col = STATE_PALETTE[np.clip(f_st, 0, 3)]
            else:
                f_dom = grid.fine.dominant_label[r_f, c_f]
                f_col = SEMANTIC_PALETTE[np.clip(f_dom, 0, 7)]

            img[px_r, px_c] = f_col

    # 4. Mask circular 100m range
    if _HAS_NUMBA:
        _mask_outside_circle_numba(img, 20, 22, 28)
    else:
        img[_MASK_OUTSIDE_100M] = BG_COLOR

    # 5. PIL Vector Overlay: Range Rings, Fine-Zone Ring, Focus Patches, Car Icon, Labels
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font_ring = _FONT_RING
    font_label = _FONT_LABEL

    # Range Rings at 10, 30, 50, 100 m
    for r in [10, 30, 50, 100]:
        r_px = int(r * scale)
        draw.ellipse([(center - r_px, center - r_px), (center + r_px, center + r_px)], outline=(85, 92, 105), width=1)
        # Ring labels: 14 px with dark label background, positioned laterally to prevent collisions
        lbl_x = min(canvas_size - 42, center + r_px - 14)
        lbl_y = center - 8
        draw.rectangle([(lbl_x - 3, lbl_y - 2), (lbl_x + 36, lbl_y + 16)], fill=(16, 18, 24, 230), outline=(50, 55, 68))
        draw.text((lbl_x + 2, lbl_y - 1), f"{r}m", fill=(200, 208, 222), font=font_ring)

    # Cyan Fine Zone Boundary Ring
    fine_cx = center
    fine_cy = int((100.0 - grid.forward_offset) * scale)
    fine_r_px = int(grid.fine_radius * scale)
    draw.ellipse(
        [(fine_cx - fine_r_px, fine_cy - fine_r_px), (fine_cx + fine_r_px, fine_cy + fine_r_px)],
        outline=(70, 180, 255),
        width=2,
    )

    # Fine-zone label (13 px, positioned so it does not overlap range ring tags)
    fine_label = f"fine zone: r={grid.fine_radius:.0f}m, {grid.fine_cell_size * 100:.1f}cm"
    f_lbl_w = len(fine_label) * 7 + 12
    f_lbl_x = max(10, min(canvas_size - f_lbl_w - 10, fine_cx - f_lbl_w // 2))
    f_lbl_y = min(canvas_size - 25, fine_cy + fine_r_px + 4)
    draw.rectangle([(f_lbl_x, f_lbl_y), (f_lbl_x + f_lbl_w, f_lbl_y + 17)], fill=(12, 24, 42, 230), outline=(70, 180, 255))
    draw.text((f_lbl_x + 6, f_lbl_y + 1), fine_label, fill=(70, 210, 255), font=font_label)

    # Focus-patch boxes and labels (13 px, gold box)
    if hasattr(grid, "patches") and grid.patches:
        for idx, patch in enumerate(grid.patches):
            px_min = int((patch.center_y - patch.half_extent + 100.0) * scale)
            px_max = int((patch.center_y + patch.half_extent + 100.0) * scale)
            py_min = int((100.0 - (patch.center_x + patch.half_extent)) * scale)
            py_max = int((100.0 - (patch.center_x - patch.half_extent)) * scale)

            draw.rectangle([(px_min, py_min), (px_max, py_max)], outline=(255, 215, 0), width=2)
            # Patch label tag placed outside the box
            tag_x = min(canvas_size - 95, px_max + 4)
            tag_y = max(10, py_min - 2)
            draw.rectangle([(tag_x, tag_y), (tag_x + 88, tag_y + 17)], fill=(25, 25, 12, 230), outline=(255, 215, 0))
            draw.text((tag_x + 4, tag_y + 1), "focus patch", fill=(255, 215, 0), font=font_label)

    # 6. Sensing rings on ground under car (localized bbox alpha-composite)
    if sensing_mode != "Off":
        rings = compute_sensing_rings(mode=sensing_mode, current_time=current_time)
        if rings:
            max_r_m = max(r_m for r_m, _ in rings)
            max_r_px = int(max_r_m * scale) + 4
            u_min = max(0, center - max_r_px)
            u_max = min(canvas_size, center + max_r_px)
            v_min = max(0, center - max_r_px)
            v_max = min(canvas_size, center + max_r_px)
            box_w = u_max - u_min
            box_h = v_max - v_min
            if box_w > 0 and box_h > 0:
                overlay_sub = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
                ov_draw = ImageDraw.Draw(overlay_sub)
                sub_center_x = center - u_min
                sub_center_y = center - v_min
                for r_m, alpha in sorted(rings, key=lambda x: -x[0]):
                    r_px = int(r_m * scale)
                    fill_rgba = (RING_TEAL_RGB[0], RING_TEAL_RGB[1], RING_TEAL_RGB[2], int(alpha * 255))
                    edge_rgba = (RING_TEAL_RGB[0], RING_TEAL_RGB[1], RING_TEAL_RGB[2], int(min(1.0, alpha * 2.2) * 255))
                    ov_draw.ellipse(
                        [(sub_center_x - r_px, sub_center_y - r_px), (sub_center_x + r_px, sub_center_y + r_px)],
                        fill=fill_rgba,
                        outline=edge_rgba,
                        width=1,
                    )
                sub_crop = pil_img.crop((u_min, v_min, u_max, v_max)).convert("RGBA")
                comp_sub = Image.alpha_composite(sub_crop, overlay_sub).convert("RGB")
                pil_img.paste(comp_sub, (u_min, v_min))
                draw = ImageDraw.Draw(pil_img)

    # 7. Top-down Car Icon at origin
    _draw_topdown_car_icon(draw, center=center, scale=scale)

    return np.asarray(pil_img)


def render_raw_points(
    xyz: np.ndarray,
    labels: np.ndarray,
    canvas_size: int = 800,
    sensing_mode: str = "Pulse",
    current_time: float = 0.0,
) -> np.ndarray:
    """Renders a fast top-down scatter of raw LiDAR points (x forward / up, y lateral)
    with circular 100m range rings, alternating vehicle tints, sensing pulse, and ego car icon.
    """
    scale = canvas_size / 200.0
    center = canvas_size // 2

    img = np.full((canvas_size, canvas_size, 3), BG_COLOR, dtype=np.uint8)

    n_pts = len(xyz)
    if n_pts > 0:
        if n_pts > 25000:
            step = max(1, n_pts // 25000)
            idx = np.arange(0, n_pts, step)[:25000]
            sub_xyz = xyz[idx]
            sub_lbls = labels[idx]
        else:
            sub_xyz = xyz
            sub_lbls = labels

        x = sub_xyz[:, 0]
        y = sub_xyz[:, 1]
        valid = (np.abs(x) <= 100.0) & (np.abs(y) <= 100.0)
        x = x[valid]
        y = y[valid]
        sub_lbls = sub_lbls[valid]

        point_colors = SEMANTIC_PALETTE[np.clip(sub_lbls, 0, 7)].copy()
        # Tint other vehicles with alternating OTHER_CAR_BLUE and OTHER_CAR_DARK
        veh_mask = (sub_lbls == 4)
        if np.any(veh_mask):
            veh_coords = np.column_stack([x[veh_mask], y[veh_mask]])
            keys = np.round(veh_coords / 4.5).astype(int)
            _, inv = np.unique(keys, axis=0, return_inverse=True)
            car_colors = np.where(
                (inv % 2 == 0)[:, None],
                np.array(OTHER_CAR_BLUE_RGB, dtype=np.uint8),
                np.array(OTHER_CAR_DARK_RGB, dtype=np.uint8),
            )
            point_colors[veh_mask] = car_colors

        px_row = np.clip(np.floor((100.0 - x) * scale).astype(np.int64), 0, canvas_size - 1)
        px_col = np.clip(np.floor((y + 100.0) * scale).astype(np.int64), 0, canvas_size - 1)

        # 2x2 splat for crisp visibility
        img[px_row, px_col] = point_colors
        px_r1 = np.clip(px_row + 1, 0, canvas_size - 1)
        px_c1 = np.clip(px_col + 1, 0, canvas_size - 1)
        img[px_r1, px_col] = point_colors
        img[px_row, px_c1] = point_colors
        img[px_r1, px_c1] = point_colors

    # Mask outside 100m circle
    if _HAS_NUMBA:
        _mask_outside_circle_numba(img, 20, 22, 28)
    else:
        img[_MASK_OUTSIDE_100M] = BG_COLOR

    # PIL Annotations: Range rings, Sensing Rings & car icon
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font_ring = _FONT_RING

    for r in [10, 30, 50, 100]:
        r_px = int(r * scale)
        draw.ellipse([(center - r_px, center - r_px), (center + r_px, center + r_px)], outline=(85, 92, 105), width=1)
        lbl_x = min(canvas_size - 42, center + r_px - 14)
        lbl_y = center - 8
        draw.rectangle([(lbl_x - 3, lbl_y - 2), (lbl_x + 36, lbl_y + 16)], fill=(16, 18, 24, 230), outline=(50, 55, 68))
        draw.text((lbl_x + 2, lbl_y - 1), f"{r}m", fill=(200, 208, 222), font=font_ring)

    # Sensing rings on ground under car (localized bbox alpha-composite)
    if sensing_mode != "Off":
        rings = compute_sensing_rings(mode=sensing_mode, current_time=current_time)
        if rings:
            max_r_m = max(r_m for r_m, _ in rings)
            max_r_px = int(max_r_m * scale) + 4
            u_min = max(0, center - max_r_px)
            u_max = min(canvas_size, center + max_r_px)
            v_min = max(0, center - max_r_px)
            v_max = min(canvas_size, center + max_r_px)
            box_w = u_max - u_min
            box_h = v_max - v_min
            if box_w > 0 and box_h > 0:
                overlay_sub = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
                ov_draw = ImageDraw.Draw(overlay_sub)
                sub_center_x = center - u_min
                sub_center_y = center - v_min
                for r_m, alpha in sorted(rings, key=lambda x: -x[0]):
                    r_px = int(r_m * scale)
                    fill_rgba = (RING_TEAL_RGB[0], RING_TEAL_RGB[1], RING_TEAL_RGB[2], int(alpha * 255))
                    edge_rgba = (RING_TEAL_RGB[0], RING_TEAL_RGB[1], RING_TEAL_RGB[2], int(min(1.0, alpha * 2.2) * 255))
                    ov_draw.ellipse(
                        [(sub_center_x - r_px, sub_center_y - r_px), (sub_center_x + r_px, sub_center_y + r_px)],
                        fill=fill_rgba,
                        outline=edge_rgba,
                        width=1,
                    )
                sub_crop = pil_img.crop((u_min, v_min, u_max, v_max)).convert("RGBA")
                comp_sub = Image.alpha_composite(sub_crop, overlay_sub).convert("RGB")
                pil_img.paste(comp_sub, (u_min, v_min))
                draw = ImageDraw.Draw(pil_img)

    _draw_topdown_car_icon(draw, center=center, scale=scale)
    return np.asarray(pil_img)


def render_25d_fast(
    grid: VarResGrid,
    trav: TraversabilityMap,
    view_range: str = "Road corridor",
    canvas_size: Tuple[int, int] = (800, 800),
    max_cells: int = 25000,
    sensing_mode: str = "Pulse",
    current_time: float = 0.0,
) -> np.ndarray:
    """Renders a fast pseudo-3D perspective road corridor view using NumPy and painter's algorithm.
    
    Fixed perspective camera behind and above the car looking down the road (+x forward).
    Cells rendered as elevated columns colored by traversability state (green drivable,
    orange non-drivable, red obstacle). 3D ego crossover model and sensing rings rendered at origin.
    Runs in under 25 ms.
    """
    canvas_h, canvas_w = canvas_size
    img = np.full((canvas_h, canvas_w, 3), (16, 18, 24), dtype=np.uint8)

    # 1. Gather all non-empty cells across fine, patches, and coarse zones
    cx_list, cy_list, z_base_list, z_top_list, state_list, prio_list = [], [], [], [], [], []

    # Fine zone
    f_cnt = grid.fine.count
    f_nz = f_cnt > 0
    if np.any(f_nz):
        r_f, c_f = np.where(f_nz)
        x_f = grid.forward_offset - grid.fine_radius + (c_f + 0.5) * grid.fine_cell_size
        y_f = -grid.fine_radius + (r_f + 0.5) * grid.fine_cell_size
        st_f = trav.fine.state[r_f, c_f]
        z_min_f = grid.fine.z_min[r_f, c_f]
        z_max_f = grid.fine.z_max[r_f, c_f]

        cx_list.append(x_f)
        cy_list.append(y_f)
        z_base_list.append(z_min_f)
        z_top_list.append(z_max_f)
        state_list.append(st_f)
        # Priority: Obstacle (3) > Non-drivable (2) > Drivable (1) > Unknown (0)
        prio_f = np.where(st_f == OBSTACLE, 3, np.where(st_f == NON_DRIVABLE, 2, np.where(st_f == DRIVABLE, 1, 0)))
        prio_list.append(prio_f)

    # Focus Patches
    if hasattr(grid, "patches") and grid.patches:
        for idx, patch in enumerate(grid.patches):
            p_nz = patch.count > 0
            if not np.any(p_nz):
                continue
            r_p, c_p = np.where(p_nz)
            x_p = patch.center_x - patch.half_extent + (c_p + 0.5) * patch.cell_size
            y_p = patch.center_y - patch.half_extent + (r_p + 0.5) * patch.cell_size
            st_p = trav.patches[idx].state[r_p, c_p] if (hasattr(trav, "patches") and idx < len(trav.patches)) else UNKNOWN
            z_min_p = patch.z_min[r_p, c_p]
            z_max_p = patch.z_max[r_p, c_p]

            cx_list.append(x_p)
            cy_list.append(y_p)
            z_base_list.append(z_min_p)
            z_top_list.append(z_max_p)
            state_list.append(st_p)
            prio_list.append(np.full_like(st_p, 3))  # Focus patches high priority

    # Coarse zone
    c_nz = grid.coarse.count > 0
    if np.any(c_nz):
        r_c, c_c = np.where(c_nz)
        x_c = -100.0 + (c_c + 0.5) * grid.coarse.cell_size
        y_c = -100.0 + (r_c + 0.5) * grid.coarse.cell_size
        st_c = trav.coarse.state[r_c, c_c]
        z_min_c = grid.coarse.z_min[r_c, c_c]
        z_max_c = grid.coarse.z_max[r_c, c_c]

        cx_list.append(x_c)
        cy_list.append(y_c)
        z_base_list.append(z_min_c)
        z_top_list.append(z_max_c)
        state_list.append(st_c)
        prio_c = np.where(st_c == OBSTACLE, 3, np.where(st_c == NON_DRIVABLE, 2, np.where(st_c == DRIVABLE, 1, 0)))
        prio_list.append(prio_c)

    if not cx_list:
        return img

    all_x = np.concatenate(cx_list)
    all_y = np.concatenate(cy_list)
    all_zb = np.concatenate(z_base_list)
    all_zt = np.concatenate(z_top_list)
    all_st = np.concatenate(state_list)
    all_prio = np.concatenate(prio_list)

    # 2. Filter by View Range
    if view_range == "Road corridor":
        valid_range = (all_x >= -8.0) & (all_x <= 60.0) & (np.abs(all_y) <= 18.0)
    elif view_range == "60 m":
        valid_range = np.hypot(all_x, all_y) <= 60.0
    else:  # Full 100 m
        valid_range = np.hypot(all_x, all_y) <= 100.0

    all_x = all_x[valid_range]
    all_y = all_y[valid_range]
    all_zb = all_zb[valid_range]
    all_zt = all_zt[valid_range]
    all_st = all_st[valid_range]
    all_prio = all_prio[valid_range]

    # Priority-based capping
    n_tot = len(all_x)
    if n_tot > max_cells:
        # Sort by priority descending
        prio_order = np.argsort(-all_prio)
        keep = prio_order[:max_cells]
        all_x = all_x[keep]
        all_y = all_y[keep]
        all_zb = all_zb[keep]
        all_zt = all_zt[keep]
        all_st = all_st[keep]

    # Clip column heights to prevent sky-high spikes
    dz = np.clip(all_zt - all_zb, 0.05, 3.5)
    all_zt = all_zb + dz

    # 3. Perspective Camera Projection
    # Camera placed behind (-10m) and above (+5.0m) the vehicle looking down the road
    xc, yc, zc = -10.0, 0.0, 5.0
    look_dist = 30.0
    pitch = np.arctan2(zc, look_dist - xc)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)

    x_rel = all_x - xc
    y_rel = all_y - yc
    zb_rel = all_zb - zc
    zt_rel = all_zt - zc

    # Camera coordinate rotation (pitch down)
    x_cam = x_rel * cos_p - zb_rel * sin_p
    valid_cam = x_cam > 1.0

    all_x = all_x[valid_cam]
    x_cam = x_cam[valid_cam]
    y_cam = y_rel[valid_cam]
    zb_cam = x_rel[valid_cam] * sin_p + zb_rel[valid_cam] * cos_p
    zt_cam = x_rel[valid_cam] * sin_p + zt_rel[valid_cam] * cos_p
    all_st = all_st[valid_cam]

    # Painter's algorithm: sort from far to near (decreasing x_cam)
    order = np.argsort(-x_cam)
    x_cam = x_cam[order]
    y_cam = y_cam[order]
    zb_cam = zb_cam[order]
    zt_cam = zt_cam[order]
    all_st = all_st[order]

    # Screen projection
    fu = canvas_w * 0.95
    fv = canvas_h * 0.95
    cu = canvas_w / 2.0
    cv = canvas_h * 0.58

    u = np.clip(np.round(cu + fu * (y_cam / x_cam)).astype(np.int32), 0, canvas_w - 1)
    v_b = np.clip(np.round(cv - fv * (zb_cam / x_cam)).astype(np.int32), 0, canvas_h - 1)
    v_t = np.clip(np.round(cv - fv * (zt_cam / x_cam)).astype(np.int32), 0, canvas_h - 1)
    colors = STATE_PALETTE[np.clip(all_st, 0, 3)]

    # Draw columns with width scaled by distance
    half_w = np.clip(np.round(fu * 0.25 / x_cam).astype(np.int32), 1, 6)
    x0_arr = np.clip(u - half_w, 0, canvas_w)
    x1_arr = np.clip(u + half_w + 1, 0, canvas_w)
    y0_arr = np.minimum(v_t, v_b)
    y1_arr = np.maximum(v_t, v_b)

    # Vectorized / looped span painting
    m_pts = len(u)
    if _HAS_NUMBA:
        _paint_columns_numba(
            img,
            np.ascontiguousarray(y0_arr, dtype=np.int32),
            np.ascontiguousarray(y1_arr, dtype=np.int32),
            np.ascontiguousarray(x0_arr, dtype=np.int32),
            np.ascontiguousarray(x1_arr, dtype=np.int32),
            np.ascontiguousarray(colors, dtype=np.uint8),
        )
    else:
        for i in range(m_pts):
            img[y0_arr[i]:y1_arr[i] + 1, x0_arr[i]:x1_arr[i]] = colors[i]

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # 4. Sensing rings projected on ground under car (localized bbox alpha-composite)
    if sensing_mode != "Off":
        rings = compute_sensing_rings(mode=sensing_mode, current_time=current_time)
        if rings:
            th_ring = np.linspace(0, 2 * np.pi, 36)
            all_ru, all_rv, ring_polys, ring_props = [], [], [], []
            for r_m, alpha in sorted(rings, key=lambda x: -x[0]):
                rx_m = r_m * np.cos(th_ring)
                ry_m = r_m * np.sin(th_ring)
                rx_rel = rx_m - xc
                ry_rel = ry_m - yc
                rz_rel = -zc
                r_xc = rx_rel * cos_p - rz_rel * sin_p
                r_zc = rx_rel * sin_p + rz_rel * cos_p
                if np.all(r_xc > 0.5):
                    ru = cu + fu * (ry_rel / r_xc)
                    rv = cv - fv * (r_zc / r_xc)
                    all_ru.append(ru)
                    all_rv.append(rv)
                    ring_polys.append(list(zip(ru, rv)))
                    fill_rgba = (RING_TEAL_RGB[0], RING_TEAL_RGB[1], RING_TEAL_RGB[2], int(alpha * 255))
                    edge_rgba = (RING_TEAL_RGB[0], RING_TEAL_RGB[1], RING_TEAL_RGB[2], int(min(1.0, alpha * 2.2) * 255))
                    ring_props.append((fill_rgba, edge_rgba))

            if all_ru:
                cat_u = np.concatenate(all_ru)
                cat_v = np.concatenate(all_rv)
                u_min = max(0, int(np.floor(np.min(cat_u))) - 2)
                u_max = min(canvas_w, int(np.ceil(np.max(cat_u))) + 2)
                v_min = max(0, int(np.floor(np.min(cat_v))) - 2)
                v_max = min(canvas_h, int(np.ceil(np.max(cat_v))) + 2)
                box_w = u_max - u_min
                box_h = v_max - v_min
                if box_w > 0 and box_h > 0:
                    overlay_sub = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
                    ov_draw = ImageDraw.Draw(overlay_sub)
                    for poly, (fill_rgba, edge_rgba) in zip(ring_polys, ring_props):
                        poly_offset = [(x - u_min, y - v_min) for x, y in poly]
                        ov_draw.polygon(poly_offset, fill=fill_rgba, outline=edge_rgba)
                    sub_crop = pil_img.crop((u_min, v_min, u_max, v_max)).convert("RGBA")
                    comp_sub = Image.alpha_composite(sub_crop, overlay_sub).convert("RGB")
                    pil_img.paste(comp_sub, (u_min, v_min))
                    draw = ImageDraw.Draw(pil_img)

    # 5. Project and Draw 3D Ego Crossover Model at Origin using precached car mesh
    v_car = _CAR_MESH_VERTICES
    f_car = _CAR_MESH_FACES
    c_car = _CAR_MESH_COLORS

    # Project vertices
    v_rel = v_car - np.array([xc, yc, zc], dtype=np.float32)
    car_v_xc = v_rel[:, 0] * cos_p - v_rel[:, 2] * sin_p
    car_v_yc = v_rel[:, 1]
    car_v_zc = v_rel[:, 0] * sin_p + v_rel[:, 2] * cos_p

    car_v_u = cu + fu * (car_v_yc / car_v_xc)
    car_v_v = cv - fv * (car_v_zc / car_v_xc)

    # Painter's algorithm for car faces: sort far to near by mean camera x depth
    face_depths = np.mean(car_v_xc[f_car], axis=1)
    car_face_order = np.argsort(-face_depths)

    for f_idx in car_face_order:
        idx0, idx1, idx2 = f_car[f_idx]
        tri_pts = [
            (float(car_v_u[idx0]), float(car_v_v[idx0])),
            (float(car_v_u[idx1]), float(car_v_v[idx1])),
            (float(car_v_u[idx2]), float(car_v_v[idx2])),
        ]
        draw.polygon(tri_pts, fill=c_car[f_idx])

    return np.asarray(pil_img)
