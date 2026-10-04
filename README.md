# RAIL-2.5D: Risk-Adaptive Multi-Resolution 2.5D LiDAR Grid Mapping

RAIL-2.5D is a real-time, memory-efficient 2.5D LiDAR grid mapping pipeline designed for autonomous driving. It converts dense 3D LiDAR point clouds (100k+ points/frame) into a multi-resolution 2.5D elevation and traversability representation at >30 FPS.

The pipeline combines a fine-resolution grid (5 cm cells) in the immediate vicinity of the ego vehicle, a coarse-resolution grid (50 cm cells) across the wider 200 m × 200 m environment, and dynamically allocated high-resolution focus patches (5 cm cells) centered around safety-critical objects (pedestrians, vehicles).

---

## Architecture & Pipeline

```text
  LiDAR Scan (100k pts) + Semantic Labels
                     │
                     ▼
  ┌──────────────────────────────────────────────┐
  │ Pass 1: Initial Variable-Resolution Binning  │
  │ • Fine Zone (5 cm cells, r=10 m around ego)  │
  │ • Coarse Zone (50 cm cells, r=100 m)         │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Safety-Critical Risk Search                  │
  │ • Identify high-risk objects (pedestrians,   │
  │   vehicles) using spatial clustering         │
  │ • Prioritize top candidates by stopping risk │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Pass 2: High-Resolution Focus Patches        │
  │ • Up to 4 dynamic 3.2 m x 3.2 m patches @ 5cm│
  │ • Target safety bands (20–40 m, 40–60 m)     │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Traversability Analysis & 2.5D Elevation     │
  │ • Step-height evaluation (max 0.10 m)        │
  │ • Slope filter against elevation baselines   │
  │ • Dominant semantic labeling & confidence    │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  Drivable / Non-Drivable / Obstacle 2.5D Grid Map & Fast Raster (<30 ms)
```

---

## How to Run

### 1. Environment Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run app.py
```

The web dashboard provides top-down grid inspection, 2.5D road perspective rendering (<30 ms/frame), interactive 3D Plotly elevation view, and real-time playback control.

---

## Benchmark Performance

Evaluated over 300 frames with 100,000 points per frame (median of 3 runs, 10-frame warm-up excluded):

| Configuration | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Mean FPS | FPS at p95 | FPS at p99 | Target Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline Only (Patches OFF)** | 13.14 | 12.83 | 14.88 | 15.65 | 76.1 | 67.2 | 63.9 | **PASS** |
| **Pipeline Only (Patches ON)**  | 20.30 | 20.09 | 22.50 | 23.82 | 49.3 | 44.4 | 42.0 | **PASS** |

- **Real-Time Processing:** Meets the ≥30 FPS real-time deadline on mean (49.3 FPS), p95 (44.4 FPS), and p99 (42.0 FPS).
- **Memory Compression:** 48.2x reduction in allocated memory compared to a dense 5 cm uniform grid of the same 200 m × 200 m coverage.

---

## Limitations

- **Synthetic Data:** The bundled excerpt in `sample_data/` is synthetic data generated in SemanticKITTI format (containing a `README_FAKE.txt` marker).
- **Slope Rejection with 5 cm Cells:** With 5 cm cells, the 15-degree slope rule between immediately adjacent cells rejects about 30 percent of road cells because LiDAR range noise ($\sigma \approx 2\text{ cm}$) exceeds the 1.34 cm height difference threshold over 5 cm. Planned fix: slope evaluated over a 25 cm physical baseline.

---

## Dataset Discovery & Adding Real SemanticKITTI Data

The loader discovers sequences automatically in the following precedence order:
1. `data/` (recursively searched up to 3 levels, supporting both `data/sequences/<NN>` and `data/<folder>/sequences/<NN>`).
2. `sample_data/` (tracked in the repository, containing a 20-frame excerpt of sequence `99`).
3. Procedural synthetic scene generator (used when neither folder contains matching `.bin` and `.label` sequence files).

To evaluate on real SemanticKITTI scans:
1. Download a sequence (e.g. sequence `00`) from the official SemanticKITTI dataset.
2. Place the sequence folder under `data/sequences/00/` (or nested `data/<folder>/sequences/00/`):
   ```text
   data/
   └── sequences/
       └── 00/
           ├── velodyne/
           │   ├── 000000.bin
           │   └── ...
           ├── labels/
           │   ├── 000000.label
           │   └── ...
           └── poses.txt
   ```
3. The pipeline will prioritize `data/` over `sample_data/`, detect the real dataset (verifying the absence of `README_FAKE.txt`), display the green "Real SemanticKITTI" badge with origin `data/`, and use genuine vehicle poses for automatic speed estimation. Only genuine data without a marker file receives the official SemanticKITTI credit.

---

## Credits & License

- **SemanticKITTI:** Behley et al., *SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences*, ICCV 2019. CC BY-NC-SA 4.0 license.
