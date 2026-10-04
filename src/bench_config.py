"""Centralized benchmark and grid configuration for RAIL-2.5D.

Ensures bench_pipeline.py and run_pipeline.py use identical spatial extents,
cell resolutions, focus patch dimensions, and class configurations.
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark grid and execution configuration parameters."""

    range_m: float = 100.0             # Max LiDAR sensor range (radius 100 m, 200 m x 200 m area)
    fine_radius: float = 10.0         # Fine zone radius (m) around forward offset
    fine_cell_size: float = 0.05      # 5 cm fine cell size (0.05 m)
    fine_grid_size: int = 400         # 400 x 400 fine grid cells
    coarse_cell_size: float = 0.50    # 50 cm coarse cell size (0.50 m)
    coarse_grid_size: int = 400       # 400 x 400 coarse grid cells
    patch_size: float = 3.2           # 3.2 m x 3.2 m patch extent (half_extent = 1.6 m)
    patch_cell_size: float = 0.05     # 5 cm patch cell size (0.05 m)
    patch_grid_size: int = 64         # 64 x 64 cells per patch
    k_patches: int = 4                # Maximum focus patch count limit
    n_classes: int = 8                # Number of semantic classes (0 to 7)

    def format_summary(self) -> str:
        """Returns a single-line summary string of the benchmark configuration."""
        return (
            f"Fine-zone: radius {self.fine_radius:.1f} m, cell size {self.fine_cell_size:.2f} m | "
            f"Coarse-zone: cell size {self.coarse_cell_size:.2f} m | "
            f"Patches: {self.patch_size:.1f} m x {self.patch_size:.1f} m ({self.patch_cell_size:.2f} m cells, limit {self.k_patches}) | "
            f"Range: {self.range_m:.1f} m (Area: {self.range_m * 2:.0f} m x {self.range_m * 2:.0f} m)"
        )


_DEFAULT_CONFIG = BenchmarkConfig()


def get_bench_config() -> BenchmarkConfig:
    """Returns the standardized benchmark grid configuration."""
    return _DEFAULT_CONFIG
