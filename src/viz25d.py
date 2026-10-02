import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

from src.grid_engine import VarResGrid
from src.traversability import TraversabilityMap, UNKNOWN, DRIVABLE, NON_DRIVABLE, OBSTACLE


# Unit box definition for vectorized mesh building
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
    """Vectorially builds a single disjoint go.Mesh3d trace containing multiple extruded cell columns."""
    m = len(cx)
    if m == 0:
        return None

    # Vertices array: shape (M, 8, 3)
    verts = np.empty((m, 8, 3), dtype=np.float32)
    # w can be a scalar or array of shape (M,)
    w_col = w[:, None] if w.ndim == 1 else w
    verts[:, :, 0] = cx[:, None] + _UNIT_CORNERS[:, 0] * w_col
    verts[:, :, 1] = cy[:, None] + _UNIT_CORNERS[:, 1] * w_col
    dz = np.maximum(0.04, z_top - z_base)
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
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
        showlegend=True,
    )


def build_25d_elevation_figure(
    grid: VarResGrid,
    trav: TraversabilityMap,
    color_mode: str = "traversability",
    height_exaggeration: float = 3.0,
    draw_empty: bool = False,
    max_cells: int = 15000,
    view_revision: int = 0,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """Builds a Plotly 3D elevation map rendering ONLY the grid cells (never raw points).

    Args:
        grid: VarResGrid instance with fine, coarse, and optional patches.
        trav: TraversabilityMap containing state and confidence arrays.
        color_mode: 'traversability', 'semantic class', or 'height'.
        height_exaggeration: Vertical multiplier (default 3x, range 1x to 10x).
        draw_empty: If True, draws empty cells very faintly.
        max_cells: Maximum non-empty cells to draw to preserve responsive 60 FPS rotation.
        view_revision: Revision ID to manage camera angle persistence and reset.

    Returns:
        (plotly_figure, diagnostics_dict)
    """
    t0 = time.perf_counter()
    fig = go.Figure()

    # 1. Collect cells from all zones: fine, patches, coarse
    zones_data = []

    # Helper to extract cell geometry from a GridZone
    def extract_zone_cells(zone, zone_trav, zone_name: str, priority_mask=None):
        count = zone.count
        occ = count > 0
        if priority_mask is not None:
            occ = occ & priority_mask

        n_occ = int(np.count_nonzero(occ))
        if n_occ == 0:
            return None, 0

        rows, cols = np.where(occ)
        # World coordinates: x is forward, y is lateral
        cell_x = zone.center_x - zone.half_extent + (cols.astype(np.float32) + 0.5) * zone.cell_size
        cell_y = zone.center_y - zone.half_extent + (rows.astype(np.float32) + 0.5) * zone.cell_size

        z_min_raw = zone.z_min[rows, cols]
        z_max_raw = zone.z_max[rows, cols]
        # In case single point in cell has identical min/max
        z_min = np.where(np.isfinite(z_min_raw), z_min_raw, 0.0)
        z_max = np.where(np.isfinite(z_max_raw), z_max_raw, z_min + 0.05)

        st = zone_trav.state[rows, cols]
        cf = zone_trav.confidence[rows, cols]
        dom = zone.dominant_label[rows, cols]

        return {
            "name": zone_name,
            "x": cell_x,
            "y": cell_y,
            "w": float(zone.cell_size),
            "z_min": z_min * float(height_exaggeration),
            "z_max": z_max * float(height_exaggeration),
            "z_max_raw": z_max_raw,
            "state": st,
            "confidence": cf,
            "dominant_class": dom,
            "count": count[rows, cols],
        }, n_occ

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

    # Merge cell lists
    all_x = []
    all_y = []
    all_w = []
    all_z_base = []
    all_z_top = []
    all_z_max_raw = []
    all_state = []
    all_dom = []

    for zd in zones_data:
        all_x.append(zd["x"])
        all_y.append(zd["y"])
        all_w.append(np.full(len(zd["x"]), zd["w"], dtype=np.float32))
        all_z_base.append(zd["z_min"])
        all_z_top.append(zd["z_max"])
        all_z_max_raw.append(zd["z_max_raw"])
        all_state.append(zd["state"])
        all_dom.append(zd["dominant_class"])

    if all_x:
        cx = np.concatenate(all_x)
        cy = np.concatenate(all_y)
        cw = np.concatenate(all_w)
        z_base = np.concatenate(all_z_base)
        z_top = np.concatenate(all_z_top)
        z_raw = np.concatenate(all_z_max_raw)
        c_state = np.concatenate(all_state)
        c_dom = np.concatenate(all_dom)
    else:
        cx = cy = cw = z_base = z_top = z_raw = c_state = c_dom = np.array([])

    total_cells_to_draw = len(cx)

    # Subsample if exceeding max_cells budget to preserve responsive WebGL
    if total_cells_to_draw > max_cells:
        step = max(1, total_cells_to_draw // max_cells)
        sub_idx = np.arange(0, total_cells_to_draw, step)[:max_cells]
        cx = cx[sub_idx]
        cy = cy[sub_idx]
        cw = cw[sub_idx]
        z_base = z_base[sub_idx]
        z_top = z_top[sub_idx]
        z_raw = z_raw[sub_idx]
        c_state = c_state[sub_idx]
        c_dom = c_dom[sub_idx]
        cells_drawn = len(cx)
    else:
        cells_drawn = total_cells_to_draw

    # 2. Render cell meshes according to color_mode
    if cells_drawn > 0:
        if color_mode == "traversability":
            groups = [
                ("Drivable", c_state == DRIVABLE, "#2ecc71", 0.9),
                ("Non-drivable", c_state == NON_DRIVABLE, "#f39c12", 0.9),
                ("Obstacle", c_state == OBSTACLE, "#e74c3c", 0.95),
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
            sem_groups = [
                ("Road (1)", c_dom == 1, "#3498db"),
                ("Terrain (2)", c_dom == 2, "#27ae60"),
                ("Static Obstacle (3)", c_dom == 3, "#95a5a6"),
                ("Vehicle (4)", c_dom == 4, "#9b59b6"),
                ("Person (5)", c_dom == 5, "#e67e22"),
                ("Moving (6)", c_dom == 6, "#e74c3c"),
                ("Non-drivable Ground (7)", c_dom == 7, "#f1c40f"),
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
            # Height colored via elevation buckets
            min_z = float(np.min(z_raw)) if len(z_raw) > 0 else 0.0
            max_z = float(np.max(z_raw)) if len(z_raw) > 0 else 2.0
            z_span = max(0.1, max_z - min_z)
            norm_z = np.clip((z_raw - min_z) / z_span, 0.0, 1.0)

            # 4 discrete height color bands
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

    # 3. Optional Faint Empty Cells
    if draw_empty:
        # Faint sample of empty coarse grid cells near the vehicle
        empty_mask = (grid.coarse.count == 0)
        e_rows, e_cols = np.where(empty_mask)
        e_x = -100.0 + (e_cols[:1500].astype(np.float32) + 0.5) * 0.5
        e_y = -100.0 + (e_rows[:1500].astype(np.float32) + 0.5) * 0.5
        dist_filter = np.hypot(e_x, e_y) < 30.0
        if np.any(dist_filter):
            faint_mesh = _build_batched_box_mesh(
                e_x[dist_filter], e_y[dist_filter], np.full(int(np.count_nonzero(dist_filter)), 0.5, dtype=np.float32),
                np.full(int(np.count_nonzero(dist_filter)), -0.05 * height_exaggeration, dtype=np.float32),
                np.zeros(int(np.count_nonzero(dist_filter)), dtype=np.float32),
                color="#373c46", name="Empty (Faint)", opacity=0.25,
            )
            if faint_mesh:
                fig.add_trace(faint_mesh)

    # 4. Boundary Overlays: Fine-zone boundary circle (Blue) and Focus Patches (Yellow)
    # 4a. Fine circle on ground
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = grid.forward_offset + grid.fine_radius * np.cos(theta)
    circle_y = grid.fine_radius * np.sin(theta)
    circle_z = np.zeros_like(circle_x)
    fig.add_trace(
        go.Scatter3d(
            x=circle_x,
            y=circle_y,
            z=circle_z,
            mode="lines",
            line=dict(color="#00d2ff", width=4),
            name=f"Fine Zone (r={grid.fine_radius:.0f}m)",
            hoverinfo="name",
        )
    )

    # 4b. Focus patches outlined in yellow
    if hasattr(grid, "patches") and grid.patches:
        for idx, patch in enumerate(grid.patches):
            h = patch.half_extent
            cx_p, cy_p = patch.center_x, patch.center_y
            sq_x = [cx_p - h, cx_p + h, cx_p + h, cx_p - h, cx_p - h]
            sq_y = [cy_p - h, cy_p - h, cy_p + h, cy_p + h, cy_p - h]
            sq_z = [0.05, 0.05, 0.05, 0.05, 0.05]
            fig.add_trace(
                go.Scatter3d(
                    x=sq_x,
                    y=sq_y,
                    z=sq_z,
                    mode="lines",
                    line=dict(color="#f1c40f", width=5),
                    name=f"Focus Patch #{idx+1} (5cm)",
                    hoverinfo="name",
                )
            )

    # 5. Ego Car: Box 4.5m x 1.8m x 1.5m at origin (Marker, distinct cyan/silver colour)
    car_x = 0.0
    car_y = 0.0
    car_dx = 4.5
    car_dy = 1.8
    car_dz = 1.5 * float(height_exaggeration)
    car_mesh = _build_batched_box_mesh(
        np.array([car_x], dtype=np.float32),
        np.array([car_y], dtype=np.float32),
        np.array([car_dy], dtype=np.float32),  # lateral width
        np.array([0.0], dtype=np.float32),
        np.array([car_dz], dtype=np.float32),
        color="#00e5ff",
        name="Ego Car (4.5m x 1.8m)",
        opacity=0.85,
    )
    if car_mesh:
        # Scale length along x:
        car_mesh.x = np.array([
            -car_dx/2, car_dx/2, car_dx/2, -car_dx/2,
            -car_dx/2, car_dx/2, car_dx/2, -car_dx/2,
        ], dtype=np.float32)
        fig.add_trace(car_mesh)

    # 6. Camera Settings: Chase view behind and above car looking forward (+x)
    # In Plotly 3D: x forward, y lateral (left-to-right), z elevation
    camera_settings = dict(
        eye=dict(x=-1.5, y=0.0, z=0.8),       # Behind and above car
        center=dict(x=0.35, y=0.0, z=-0.05),  # Looking ahead forward
        up=dict(x=0.0, y=0.0, z=1.0),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X [Forward, m]", range=[-20, 60], backgroundcolor="#1e2026", gridcolor="#373c46"),
            yaxis=dict(title="Y [Lateral, m]", range=[-30, 30], backgroundcolor="#1e2026", gridcolor="#373c46"),
            zaxis=dict(title=f"Z [Elevation x{height_exaggeration:.0f}, m]", range=[-2, 15], backgroundcolor="#181a20", gridcolor="#373c46"),
            camera=camera_settings,
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=1.6, z=0.6),
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#14161c",
        uirevision=str(view_revision),  # Preserves interactive user rotation across frame updates
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(20, 22, 28, 0.85)",
            font=dict(color="#ecf0f1", size=11),
        ),
    )

    t1 = time.perf_counter()
    render_ms = (t1 - t0) * 1000.0

    diag = {
        "cells_drawn": cells_drawn,
        "total_non_empty": total_non_empty,
        "n_fine": n_fine,
        "n_patches": patch_counts,
        "n_coarse": n_coarse,
        "render_ms": render_ms,
        "height_exaggeration": height_exaggeration,
    }

    return fig, diag
