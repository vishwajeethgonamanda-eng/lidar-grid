"""2.5D Elevation Map visualization using Plotly for RAIL-2.5D.

Renders exclusively grid cells (never raw point clouds), with physical cell widths,
clipped column heights, priority-based cell capping, road corridor range selection,
and a to-scale 3D ego car model.
"""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

from src.grid_engine import VarResGrid
from src.traversability import TraversabilityMap, UNKNOWN, DRIVABLE, NON_DRIVABLE, OBSTACLE
from src.raster import STATE_HEX_COLORS, SEMANTIC_HEX_COLORS
from src.car_model import (
    build_car_mesh,
    compute_sensing_rings,
    RING_TEAL,
    OTHER_CAR_BLUE,
    OTHER_CAR_DARK,
)


# Unit box vertices and faces for vectorized mesh generation
_UNIT_CORNERS = np.array([
    [-0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [ 0.5,  0.5, 0.0],
    [-0.5,  0.5, 0.0],
    [-0.5, -0.5, 1.0],
    [ 0.5, -0.5, 1.0],
    [ 0.5,  0.5, 1.0],
    [-0.5,  0.5, 1.0],
], dtype=np.float32)

_UNIT_TRIANGLES = np.array([
    [0, 1, 2], [0, 2, 3],  # bottom
    [4, 6, 5], [4, 7, 6],  # top
    [0, 5, 1], [0, 4, 5],  # front
    [2, 7, 3], [2, 6, 7],  # back
    [0, 3, 7], [0, 7, 4],  # left
    [1, 5, 6], [1, 6, 2],  # right
], dtype=np.int32)


def _build_batched_box_mesh(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    z_base: np.ndarray,
    z_top: np.ndarray,
    color: str,
    name: str,
    opacity: float = 0.9,
) -> Optional[go.Mesh3d]:
    """Vectorially builds a single disjoint go.Mesh3d trace containing extruded cell columns."""
    m = len(cx)
    if m == 0:
        return None

    # Vertices array: shape (M, 8, 3)
    verts = np.empty((m, 8, 3), dtype=np.float32)
    w_col = w[:, None] if w.ndim == 1 else w
    verts[:, :, 0] = cx[:, None] + _UNIT_CORNERS[:, 0] * w_col
    verts[:, :, 1] = cy[:, None] + _UNIT_CORNERS[:, 1] * w_col
    dz = z_top - z_base
    verts[:, :, 2] = z_base[:, None] + _UNIT_CORNERS[:, 2] * dz[:, None]

    flat_verts = verts.reshape(-1, 3)
    flat_i = (_UNIT_TRIANGLES[:, 0] + 8 * np.arange(m)[:, None]).ravel()
    flat_j = (_UNIT_TRIANGLES[:, 1] + 8 * np.arange(m)[:, None]).ravel()
    flat_k = (_UNIT_TRIANGLES[:, 2] + 8 * np.arange(m)[:, None]).ravel()

    return go.Mesh3d(
        x=flat_verts[:, 0],
        y=flat_verts[:, 1],
        z=flat_verts[:, 2],
        i=flat_i,
        j=flat_j,
        k=flat_k,
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        lighting=dict(ambient=0.65, diffuse=0.8, roughness=0.5, specular=0.2),
        showlegend=False,
    )


def _make_box_mesh(
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
    color: str,
    name: str = "car_part",
    opacity: float = 1.0,
) -> go.Mesh3d:
    """Builds a single axis-aligned 3D box Mesh3d trace."""
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
    return go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        lighting=dict(ambient=0.75, diffuse=0.85, specular=0.3),
        showlegend=False,
    )


def _build_scale_car_mesh() -> List[go.Mesh3d]:
    """Constructs a recognisable 3D car model to scale (4.5m x 1.8m x 1.5m) at the origin.

    Components:
    - Lower body: 4.5m long, 1.8m wide, 0.55m high (z in [0.20, 0.75])
    - Cabin: 2.55m long, 1.5m wide, 0.75m high (z in [0.75, 1.50]), shifted slightly back
    - Windshield: lighter glass strip on the front of cabin
    - 4 dark wheel blocks: 0.70m x 0.22m x 0.40m at ground level (z in [0.0, 0.40])
    """
    parts = []

    # 1. Lower body (blue)
    parts.append(_make_box_mesh(
        -2.25, 2.25, -0.90, 0.90, 0.20, 0.75,
        color="#1976d2", name="Car Body", opacity=0.95
    ))

    # 2. Cabin on top (shifted slightly back, darker blue)
    parts.append(_make_box_mesh(
        -1.50, 1.05, -0.75, 0.75, 0.75, 1.50,
        color="#1565c0", name="Car Cabin", opacity=0.95
    ))

    # 3. Windshield strip on front of cabin (light glass blue)
    parts.append(_make_box_mesh(
        1.02, 1.06, -0.70, 0.70, 0.80, 1.45,
        color="#b3e5fc", name="Windshield", opacity=0.98
    ))

    # 4. Four dark wheel blocks at ground level
    wheel_col = "#212121"
    # Front-left
    parts.append(_make_box_mesh(1.15, 1.85, 0.72, 0.94, 0.0, 0.40, color=wheel_col, name="Wheel FL"))
    # Front-right
    parts.append(_make_box_mesh(1.15, 1.85, -0.94, -0.72, 0.0, 0.40, color=wheel_col, name="Wheel FR"))
    # Rear-left
    parts.append(_make_box_mesh(-1.85, -1.15, 0.72, 0.94, 0.0, 0.40, color=wheel_col, name="Wheel RL"))
    # Rear-right
    parts.append(_make_box_mesh(-1.85, -1.15, -0.94, -0.72, 0.0, 0.40, color=wheel_col, name="Wheel RR"))

    return parts


def build_25d_elevation_figure(
    grid: VarResGrid,
    trav: TraversabilityMap,
    color_mode: str = "traversability",
    height_exaggeration: float = 3.0,
    draw_empty: bool = False,
    max_cells: int = 30000,
    view_range: str = "Road corridor",
    view_revision: int = 0,
    road_markings: bool = True,
    sensing_mode: str = "Pulse",
    current_time: float = 0.0,
    **kwargs: Any,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """Builds a polished Plotly 3D elevation map rendering ONLY grid cells (never raw points).

    Features:
    - View range filter: 'Road corridor' (-8m to 60m ahead, 18m lateral), '60 m', or 'Full 100 m'.
    - Priority-based cell cap: always preserves obstacle, non-drivable, focus-patch and fine-zone
      cells; thins low-priority coarse flat cells evenly across the map if budget is exceeded.
    - True physical cell footprint dimensions.
    - Capped column heights to eliminate needle spike artifacts.
    - Recognisable 3D car model to scale at origin.
    - Clean cockpit chase camera looking down the road.
    - Legend placed outside plot area in right panel.
    - Exact diagnostics showing 'cells drawn X of Y non-empty'.
    """
    t0 = time.perf_counter()
    fig = go.Figure()

    # 1. Helper to extract non-empty cells within selected view_range
    def is_in_view_range(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        if view_range == "Road corridor":
            return (x_arr >= -8.0) & (x_arr <= 60.0) & (y_arr >= -18.0) & (y_arr <= 18.0)
        elif view_range == "60 m":
            return np.hypot(x_arr, y_arr) <= 60.0
        else:  # "Full 100 m"
            return np.hypot(x_arr, y_arr) <= 100.0

    def extract_zone_cells(zone, zone_trav, zone_type: str):
        count = zone.count
        occ = count > 0
        if not np.any(occ):
            return None, 0

        rows, cols = np.where(occ)
        # In grid coordinates: x is forward (col offset from center), y is lateral (row offset)
        cell_x = zone.center_x - zone.half_extent + (cols.astype(np.float32) + 0.5) * zone.cell_size
        cell_y = zone.center_y - zone.half_extent + (rows.astype(np.float32) + 0.5) * zone.cell_size

        # Apply view range filter
        in_range = is_in_view_range(cell_x, cell_y)
        if not np.any(in_range):
            return None, 0

        rows = rows[in_range]
        cols = cols[in_range]
        cell_x = cell_x[in_range]
        cell_y = cell_y[in_range]

        z_min_raw = zone.z_min[rows, cols]
        z_max_raw = zone.z_max[rows, cols]
        z_min = np.where(np.isfinite(z_min_raw), z_min_raw, 0.0)
        z_max = np.where(np.isfinite(z_max_raw), z_max_raw, z_min + 0.05)

        # Apply height scaling and spike clipping (clip column height to max 3.0m)
        e = float(height_exaggeration)
        z_base_scaled = np.clip(z_min * e, -1.0 * e, 3.0 * e)
        col_dz = np.clip((z_max - z_min) * e, 0.04 * e, 3.0 * e)
        z_top_scaled = z_base_scaled + col_dz

        st = zone_trav.state[rows, cols]
        dom = zone.dominant_label[rows, cols]

        n_pts = len(cell_x)
        return {
            "x": cell_x,
            "y": cell_y,
            "w": np.full(n_pts, float(zone.cell_size), dtype=np.float32),
            "z_base": z_base_scaled,
            "z_top": z_top_scaled,
            "z_max_raw": z_max_raw,
            "state": st,
            "dominant_class": dom,
            "is_fine": np.full(n_pts, zone_type == "fine", dtype=bool),
            "is_patch": np.full(n_pts, zone_type.startswith("patch"), dtype=bool),
        }, n_pts

    zones_data = []
    fine_data, n_fine = extract_zone_cells(grid.fine, trav.fine, "fine")
    if fine_data:
        zones_data.append(fine_data)

    patch_counts = 0
    if hasattr(grid, "patches") and grid.patches:
        for idx, p_zone in enumerate(grid.patches):
            p_trav = trav.patches[idx] if (hasattr(trav, "patches") and idx < len(trav.patches)) else trav.fine
            p_data, p_cnt = extract_zone_cells(p_zone, p_trav, f"patch_{idx}")
            if p_data:
                zones_data.append(p_data)
                patch_counts += p_cnt

    coarse_data, n_coarse = extract_zone_cells(grid.coarse, trav.coarse, "coarse")
    if coarse_data:
        zones_data.append(coarse_data)

    total_non_empty = n_fine + patch_counts + n_coarse

    # 2. Merge all non-empty cells in the selected view range
    if zones_data:
        cx = np.concatenate([zd["x"] for zd in zones_data])
        cy = np.concatenate([zd["y"] for zd in zones_data])
        cw = np.concatenate([zd["w"] for zd in zones_data])
        z_base = np.concatenate([zd["z_base"] for zd in zones_data])
        z_top = np.concatenate([zd["z_top"] for zd in zones_data])
        z_raw = np.concatenate([zd["z_max_raw"] for zd in zones_data])
        c_state = np.concatenate([zd["state"] for zd in zones_data])
        c_dom = np.concatenate([zd["dominant_class"] for zd in zones_data])
        is_fine_all = np.concatenate([zd["is_fine"] for zd in zones_data])
        is_patch_all = np.concatenate([zd["is_patch"] for zd in zones_data])
    else:
        cx = cy = cw = z_base = z_top = z_raw = c_state = c_dom = np.array([], dtype=np.float32)
        is_fine_all = is_patch_all = np.array([], dtype=bool)

    total_candidates = len(cx)

    # 3. Priority-based cell cap
    # Always keep obstacle, non-drivable, focus-patch cells and all fine-zone cells.
    # If total exceeds max_cells, thin lowest-priority (far, flat, drivable/unknown) cells evenly.
    if total_candidates > max_cells:
        high_mask = is_fine_all | is_patch_all | (c_state == OBSTACLE) | (c_state == NON_DRIVABLE)
        low_mask = ~high_mask

        n_high = int(np.count_nonzero(high_mask))
        n_low = int(np.count_nonzero(low_mask))

        if n_high < max_cells:
            budget_low = max_cells - n_high
            idx_low = np.where(low_mask)[0]
            # Thin evenly across the spatial layout
            stride_sub = np.linspace(0, n_low - 1, budget_low, dtype=int)
            chosen_low = idx_low[stride_sub]
            chosen_high = np.where(high_mask)[0]
            sel_idx = np.concatenate([chosen_high, chosen_low])
        else:
            idx_high = np.where(high_mask)[0]
            stride_high = np.linspace(0, n_high - 1, max_cells, dtype=int)
            sel_idx = idx_high[stride_high]

        cx = cx[sel_idx]
        cy = cy[sel_idx]
        cw = cw[sel_idx]
        z_base = z_base[sel_idx]
        z_top = z_top[sel_idx]
        z_raw = z_raw[sel_idx]
        c_state = c_state[sel_idx]
        c_dom = c_dom[sel_idx]
        cells_drawn = len(cx)
    else:
        cells_drawn = total_candidates

    # 4. Render cell meshes according to color_mode
    if cells_drawn > 0:
        if color_mode == "traversability":
            groups = [
                ("Drivable", c_state == DRIVABLE, STATE_HEX_COLORS[DRIVABLE], 0.9),
                ("Non-drivable", c_state == NON_DRIVABLE, STATE_HEX_COLORS[NON_DRIVABLE], 0.9),
                ("Obstacle", c_state == OBSTACLE, STATE_HEX_COLORS[OBSTACLE], 0.95),
            ]
            for label, mask, col, op in groups:
                if np.any(mask):
                    mesh = _build_batched_box_mesh(
                        cx[mask], cy[mask], cw[mask], z_base[mask], z_top[mask],
                        color=col, name=label, opacity=op,
                    )
                    if mesh:
                        fig.add_trace(mesh)

        elif color_mode == "semantic class":
            v_mask = (c_dom == 4)
            if np.any(v_mask):
                v_coords = np.column_stack([cx[v_mask], cy[v_mask]])
                keys = np.round(v_coords / 4.5).astype(int)
                _, inv = np.unique(keys, axis=0, return_inverse=True)
                v_blue = v_mask.copy()
                v_blue[v_mask] = (inv % 2 == 0)
                v_dark = v_mask.copy()
                v_dark[v_mask] = (inv % 2 != 0)
            else:
                v_blue = np.zeros_like(c_dom, dtype=bool)
                v_dark = np.zeros_like(c_dom, dtype=bool)

            sem_groups = [
                ("Road (1)", c_dom == 1, SEMANTIC_HEX_COLORS[1]),
                ("Terrain (2)", c_dom == 2, SEMANTIC_HEX_COLORS[2]),
                ("Static Obstacle (3)", c_dom == 3, SEMANTIC_HEX_COLORS[3]),
                ("Vehicle Blue (4)", v_blue, OTHER_CAR_BLUE),
                ("Vehicle Dark (4)", v_dark, OTHER_CAR_DARK),
                ("Person (5)", c_dom == 5, SEMANTIC_HEX_COLORS[5]),
                ("Moving (6)", c_dom == 6, SEMANTIC_HEX_COLORS[6]),
                ("Non-drivable Ground (7)", c_dom == 7, SEMANTIC_HEX_COLORS[7]),
            ]
            for label, mask, col in sem_groups:
                if np.any(mask):
                    mesh = _build_batched_box_mesh(
                        cx[mask], cy[mask], cw[mask], z_base[mask], z_top[mask],
                        color=col, name=label, opacity=0.9,
                    )
                    if mesh:
                        fig.add_trace(mesh)

        else:  # "height"
            min_z = float(np.min(z_raw)) if len(z_raw) > 0 else 0.0
            max_z = float(np.max(z_raw)) if len(z_raw) > 0 else 2.0
            z_span = max(0.1, max_z - min_z)
            norm_z = np.clip((z_raw - min_z) / z_span, 0.0, 1.0)

            bands = [
                ("Low [Ground]", norm_z < 0.25, "#2980b9"),
                ("Medium-Low", (norm_z >= 0.25) & (norm_z < 0.50), "#27ae60"),
                ("Medium-High", (norm_z >= 0.50) & (norm_z < 0.75), "#f39c12"),
                ("High [Obstacles]", norm_z >= 0.75, "#e74c3c"),
            ]
            for label, mask, col in bands:
                if np.any(mask):
                    mesh = _build_batched_box_mesh(
                        cx[mask], cy[mask], cw[mask], z_base[mask], z_top[mask],
                        color=col, name=label, opacity=0.9,
                    )
                    if mesh:
                        fig.add_trace(mesh)

    # 5. Range Rings at 10m, 30m, 50m clipped to selected view range
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [10, 30, 50]:
        ring_x = r * np.cos(theta)
        ring_y = r * np.sin(theta)
        valid_ring = is_in_view_range(ring_x, ring_y)

        # Use NaN for points outside range to draw cleanly clipped line segments
        rx_clipped = np.where(valid_ring, ring_x, np.nan)
        ry_clipped = np.where(valid_ring, ring_y, np.nan)
        rz_clipped = np.zeros_like(rx_clipped)

        if np.any(valid_ring):
            fig.add_trace(
                go.Scatter3d(
                    x=rx_clipped,
                    y=ry_clipped,
                    z=rz_clipped,
                    mode="lines",
                    line=dict(color="#4e5668", width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Find a valid label point
            valid_idx = np.where(valid_ring)[0]
            if len(valid_idx) > 0:
                # Prefer lateral point if valid, else first valid
                lbl_idx = valid_idx[len(valid_idx) // 2]
                fig.add_trace(
                    go.Scatter3d(
                        x=[ring_x[lbl_idx]],
                        y=[ring_y[lbl_idx]],
                        z=[0.05],
                        mode="text",
                        text=[f"{r}m"],
                        textfont=dict(color="#8a94a6", size=10),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # 6. Fine-Zone Boundary (Thin cyan circle, clipped to view range)
    circle_x = grid.forward_offset + grid.fine_radius * np.cos(theta)
    circle_y = grid.fine_radius * np.sin(theta)
    valid_circle = is_in_view_range(circle_x, circle_y)
    if np.any(valid_circle):
        cx_clip = np.where(valid_circle, circle_x, np.nan)
        cy_clip = np.where(valid_circle, circle_y, np.nan)
        cz_clip = np.zeros_like(cx_clip) + 0.02
        fig.add_trace(
            go.Scatter3d(
                x=cx_clip,
                y=cy_clip,
                z=cz_clip,
                mode="lines",
                line=dict(color="#00d2ff", width=2.0),
                name=f"Fine Zone (r={grid.fine_radius:.0f}m)",
                hoverinfo="name",
                showlegend=False,
            )
        )

    # 7. Focus Patches Outlines in Yellow
    if hasattr(grid, "patches") and grid.patches:
        for idx, patch in enumerate(grid.patches):
            h = patch.half_extent
            cx_p, cy_p = patch.center_x, patch.center_y
            sq_x = [cx_p - h, cx_p + h, cx_p + h, cx_p - h, cx_p - h]
            sq_y = [cy_p - h, cy_p - h, cy_p + h, cy_p + h, cy_p - h]
            sq_z = [0.05, 0.05, 0.05, 0.05, 0.05]
            if is_in_view_range(np.array([cx_p]), np.array([cy_p]))[0]:
                fig.add_trace(
                    go.Scatter3d(
                        x=sq_x,
                        y=sq_y,
                        z=sq_z,
                        mode="lines",
                        line=dict(color="#f1c40f", width=3.0),
                        name=f"Focus Patch #{idx+1} (5cm)",
                        hoverinfo="name",
                        showlegend=False,
                    )
                )

    # 8. Optional Road Styling (road_markings=True): Amber center line, white lane edges, cyan edge glow
    if road_markings:
        x_min_rm = -8.0 if view_range == "Road corridor" else -60.0
        x_max_rm = 60.0 if view_range in ("Road corridor", "60 m") else 95.0
        x_rm = np.linspace(x_min_rm, x_max_rm, 120)
        z_rm = np.full_like(x_rm, 0.012)

        # Center amber line (#F2A900)
        fig.add_trace(go.Scatter3d(
            x=x_rm,
            y=np.zeros_like(x_rm),
            z=z_rm,
            mode="lines",
            line=dict(color="#F2A900", width=2.5),
            name="Center line",
            hoverinfo="skip",
            showlegend=False,
        ))

        # Lane edge lines in white
        for y_lane in [-3.5, 3.5]:
            fig.add_trace(go.Scatter3d(
                x=x_rm,
                y=np.full_like(x_rm, y_lane),
                z=z_rm,
                mode="lines",
                line=dict(color="#FFFFFF", width=2.0),
                name="Lane line",
                hoverinfo="skip",
                showlegend=False,
            ))

        # Cyan glow along drivable edges (#3FD0D4 at 0.5 alpha)
        for y_edge in [-3.7, 3.7]:
            fig.add_trace(go.Scatter3d(
                x=x_rm,
                y=np.full_like(x_rm, y_edge),
                z=z_rm,
                mode="lines",
                line=dict(color=RING_TEAL, width=3.5),
                opacity=0.5,
                name="Drivable Edge Glow",
                hoverinfo="skip",
                showlegend=False,
            ))

    # 9. Sensing rings on ground under car
    if sensing_mode != "Off":
        rings = compute_sensing_rings(mode=sensing_mode, current_time=current_time)
        theta_ring = np.linspace(0, 2 * np.pi, 48)
        for r_m, alpha in rings:
            rx = r_m * np.cos(theta_ring)
            ry = r_m * np.sin(theta_ring)
            rz = np.full_like(rx, 0.015)
            # Brighter boundary edge line
            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="lines",
                line=dict(color=RING_TEAL, width=2.5),
                opacity=min(1.0, alpha * 2.0),
                hoverinfo="skip",
                showlegend=False,
            ))
            # Translucent disc
            vx = np.concatenate([[0.0], rx])
            vy = np.concatenate([[0.0], ry])
            vz = np.full(len(theta_ring) + 1, 0.01)
            fi = np.zeros(len(theta_ring), dtype=int)
            fj = np.arange(1, len(theta_ring) + 1)
            fk = np.roll(fj, -1)
            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=fi, j=fj, k=fk,
                color=RING_TEAL,
                opacity=alpha,
                flatshading=True,
                hoverinfo="skip",
                showlegend=False,
            ))

    # 10. Recognisable Ego Crossover Model at origin using build_car_mesh()
    car_mesh_dict = build_car_mesh()
    v_car = car_mesh_dict["vertices"]
    f_car = car_mesh_dict["faces"]
    c_car = car_mesh_dict["colors"]

    fig.add_trace(go.Mesh3d(
        x=v_car[:, 0],
        y=v_car[:, 1],
        z=v_car[:, 2],
        i=f_car[:, 0],
        j=f_car[:, 1],
        k=f_car[:, 2],
        facecolor=c_car,
        flatshading=True,
        lighting=dict(ambient=0.75, diffuse=0.85, specular=0.05, roughness=0.5),
        name="Ego Vehicle",
        hoverinfo="name",
        showlegend=False,
    ))

    # 9. Camera Settings & Aspect Ratio customized from selected View Range
    if view_range == "Road corridor":
        # Road corridor is 68m long along x (-8 to 60) and 36m wide (-18 to 18)
        aspect = dict(x=2.5, y=1.2, z=0.45)
        camera_settings = dict(
            eye=dict(x=-2.2, y=0.0, z=1.25),
            center=dict(x=0.35, y=0.0, z=-0.12),
            up=dict(x=0.0, y=0.0, z=1.0),
        )
    elif view_range == "60 m":
        aspect = dict(x=1.8, y=1.5, z=0.40)
        camera_settings = dict(
            eye=dict(x=-2.5, y=0.0, z=1.50),
            center=dict(x=0.25, y=0.0, z=-0.10),
            up=dict(x=0.0, y=0.0, z=1.0),
        )
    else:  # "Full 100 m"
        aspect = dict(x=1.6, y=1.6, z=0.35)
        camera_settings = dict(
            eye=dict(x=-2.8, y=0.0, z=1.80),
            center=dict(x=0.20, y=0.0, z=-0.10),
            up=dict(x=0.0, y=0.0, z=1.0),
        )

    clean_axis = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        title="",
    )

    fig.update_layout(
        height=720,
        showlegend=False,  # Legend displayed cleanly in Streamlit right panel
        scene=dict(
            xaxis=clean_axis,
            yaxis=clean_axis,
            zaxis=clean_axis,
            camera=camera_settings,
            aspectmode="manual",
            aspectratio=aspect,
            bgcolor="#14161c",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#14161c",
        uirevision=str(view_revision),
    )

    t1 = time.perf_counter()
    render_ms = (t1 - t0) * 1000.0

    diag = {
        "cells_drawn": int(cells_drawn),
        "total_non_empty": int(total_non_empty),
        "max_cells": int(max_cells),
        "n_fine": int(n_fine),
        "n_patches": int(patch_counts),
        "n_coarse": int(n_coarse),
        "render_ms": render_ms,
        "height_exaggeration": height_exaggeration,
        "view_range": view_range,
    }

    return fig, diag
