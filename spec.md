# SPEC: Variable-Resolution 2.5D LiDAR Grid Engine

## Goal
Project labeled 3D LiDAR points into a 2.5D grid that is fine near the
sensor and coarse far away, with no points lost or double-counted.

## Coordinates
- Sensor at (0, 0). Points are (x, y, z) in metres, plus an integer label.
- r = sqrt(x^2 + y^2)

## Zones (decided by the POINT's own r, never by the cell)
- Fine:   r < 10 m,        5 cm cells, array 400 x 400 covering [-10, 10) m
- Coarse: 10 <= r < 100 m, 50 cm cells, array 400 x 400 covering [-100, 100) m
- Ignored: r >= 100 m (counted separately as `out_of_range`)
- Cell index: col = floor((x + half_extent) / cell_size), row likewise for y
  (half_extent = 10 for fine, 100 for coarse).
- Every in-range point goes to exactly ONE zone.

## Per-cell statistics (per zone, dense NumPy arrays)
- count
- z_min, z_max (init +inf / -inf)
- z_mean and z_var (use a vectorized sum and sum-of-squares, or Welford)
- label_hist: counts per label class (N_CLASSES configurable)
- Derived: dominant label = argmax of label_hist; empty cells = count 0

## Required API (src/grid_engine.py)
class VarResGrid:
    def __init__(self, n_classes: int)
    def reset(self) -> None
    def add_points(self, xyz: np.ndarray, labels: np.ndarray) -> None
    # xyz is (N, 3) float; labels is (N,) int. Must be fully vectorized
    # (no Python loops over points). Use np.add.at / np.minimum.at /
    # np.maximum.at or np.bincount.
    def stats(self) -> dict   # in_fine, in_coarse, out_of_range, total_input
    def memory_bytes(self) -> int
    def uniform_equivalent_bytes(self) -> int
    # same stats stored in a uniform 5 cm grid over the 100 m radius

## Definition of done
1. tests/test_conservation.py passes:
   sum(fine.count) + sum(coarse.count) == number of points with r < 100,
   and in_fine + in_coarse + out_of_range == total_input.
2. tests/test_seam.py passes: a synthetic 15 cm kerb running across the
   r = 10 m line has (z_max - z_min) >= 0.14 in the cells on BOTH sides.
3. A point at r = 9.999 goes to fine; a point at r = 10.0 goes to coarse.
4. Handles empty input, a single point, and all points out of range.
5. 100k points processed in well under 100 ms on a normal laptop.

## Rules
- NumPy only. No loops over points.
- Do not change the zone sizes or the API without asking.
- Write the tests FIRST, then the implementation.