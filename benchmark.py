import time
import numpy as np
from src.grid_engine import VarResGrid


def run_benchmark():
    n_points = 100_000
    n_classes = 4
    rng = np.random.default_rng(42)

    # Generate 100k points:
    # 30k in fine (r < 10m), 65k in coarse (10 <= r < 100m), 5k out of range (r >= 100m)
    r_fine = rng.uniform(0.0, 9.99, size=30_000)
    r_coarse = rng.uniform(10.0, 99.9, size=65_000)
    r_out = rng.uniform(100.0, 150.0, size=5_000)
    r = np.concatenate([r_fine, r_coarse, r_out])

    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.normal(loc=0.0, scale=1.0, size=n_points).astype(np.float32)
    xyz = np.column_stack([x, y, z]).astype(np.float32)
    labels = rng.integers(0, n_classes, size=n_points, dtype=np.int64)

    grid = VarResGrid(n_classes=n_classes)

    # Warmup
    grid.add_points(xyz[:1000], labels[:1000])
    grid.reset()

    # Benchmark timing across 10 iterations
    timings = []
    for _ in range(10):
        grid.reset()
        t0 = time.perf_counter()
        grid.add_points(xyz, labels)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms

    avg_time = np.mean(timings)
    min_time = np.min(timings)

    mem_bytes = grid.memory_bytes()
    uniform_bytes = grid.uniform_equivalent_bytes()
    ratio = uniform_bytes / mem_bytes

    print("=" * 60)
    print("Variable-Resolution 2.5D LiDAR Grid Engine Benchmark")
    print("=" * 60)
    print(f"Points processed:            {n_points:,}")
    print(f"Number of classes:           {n_classes}")
    print(f"Execution time (min):        {min_time:.2f} ms")
    print(f"Execution time (avg of 10):  {avg_time:.2f} ms")
    print(f"VarResGrid memory:           {mem_bytes:,} bytes ({mem_bytes / (1024 * 1024):.2f} MB)")
    print(f"Uniform 5cm equivalent:      {uniform_bytes:,} bytes ({uniform_bytes / (1024 * 1024):.2f} MB)")
    print(f"Compression ratio (uniform / var-res): {ratio:.1f}x")
    print("-" * 60)
    print(f"Grid stats: {grid.stats()}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
