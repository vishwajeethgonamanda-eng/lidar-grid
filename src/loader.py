from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Merged class definitions (classes 0 to 7):
MERGED_CLASS_NAMES: Dict[int, str] = {
    0: "unlabeled",
    1: "road",
    2: "terrain",
    3: "static obstacle",
    4: "vehicle",
    5: "person",
    6: "moving objects",
    7: "non-drivable ground",
}

# Official SemanticKITTI raw label ID mapping to target merged classes:
# 0: unlabeled
# 1: road (road, parking, lane-marking)
# 2: terrain (vegetation, terrain)
# 3: static obstacle (building, fence, pole, traffic-sign, trunk, other-structure, other-object)
# 4: vehicle (car, truck, other-vehicle, bicycle, motorcycle, bus, on-rails)
# 5: person (person, bicyclist, motorcyclist)
# 6: moving objects (ids 252 to 259)
# 7: non-drivable ground (sidewalk, other-ground)
SEMANTIC_KITTI_TO_MERGED_CLASS: Dict[int, int] = {
    # Unlabeled / Outlier (0)
    0: 0,   # unlabeled
    1: 0,   # outlier
    # Vehicle (4)
    10: 4,  # car
    11: 4,  # bicycle
    13: 4,  # bus
    15: 4,  # motorcycle
    16: 4,  # on-rails
    18: 4,  # truck
    20: 4,  # other-vehicle
    # Person (5)
    30: 5,  # person
    31: 5,  # bicyclist
    32: 5,  # motorcyclist
    # Road / Ground (1)
    40: 1,  # road
    44: 1,  # parking
    60: 1,  # lane-marking
    # Non-drivable ground (7)
    48: 7,  # sidewalk
    49: 7,  # other-ground
    # Static Obstacles (3)
    50: 3,  # building
    51: 3,  # fence
    52: 3,  # other-structure
    71: 3,  # trunk
    80: 3,  # pole
    81: 3,  # traffic-sign
    99: 3,  # other-object
    # Terrain (2)
    70: 2,  # vegetation
    72: 2,  # terrain
    # Moving Objects (6)
    252: 6,  # moving-car
    253: 6,  # moving-bicyclist
    254: 6,  # moving-person
    255: 6,  # moving-motorcyclist
    256: 6,  # moving-on-rails
    257: 6,  # moving-bus
    258: 6,  # moving-truck
    259: 6,  # moving-other-vehicle
}

# Pre-computed vectorized lookup table (covers up to 16-bit semantic label IDs).
# All entries default to 0 (unlabeled/unknown). Only keys in SEMANTIC_KITTI_TO_MERGED_CLASS are populated.
_LOOKUP_TABLE = np.zeros(65536, dtype=np.int64)
for _raw_id, _target_class in SEMANTIC_KITTI_TO_MERGED_CLASS.items():
    if 0 <= _raw_id < len(_LOOKUP_TABLE):
        _LOOKUP_TABLE[_raw_id] = _target_class


def load_scan(bin_path: Union[str, Path]) -> np.ndarray:
    """Load SemanticKITTI .bin point cloud file.

    Returns:
        (N, 4) float32 array: x, y, z, intensity.
    """
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return data.reshape(-1, 4)


def load_labels(label_path: Union[str, Path]) -> np.ndarray:
    """Load SemanticKITTI .label file.

    Lower 16 bits contain semantic label; upper 16 bits contain instance id.

    Returns:
        (N,) int array containing only the lower 16 bits.
    """
    raw_data = np.fromfile(str(label_path), dtype=np.uint32)
    if raw_data.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return (raw_data & 0xFFFF).astype(np.int64)


def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Vectorized remapping of raw SemanticKITTI IDs into merged classes (0 to 7).

    Classes:
        0: unlabeled
        1: road (road, parking, lane-marking)
        2: terrain (vegetation, terrain)
        3: static obstacle (building, fence, pole, traffic-sign, trunk, other-structure, other-object)
        4: vehicle (car, truck, other-vehicle, bicycle, motorcycle, bus, on-rails)
        5: person (person, bicyclist, motorcyclist)
        6: moving objects (raw ids 252 to 259)
        7: non-drivable ground (sidewalk, other-ground)

    Any raw ID not present in the mapping table maps to 0.
    """
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros_like(labels)

    # In-table direct vectorized lookup; any unknown or out-of-table ID remains 0
    in_range_mask = (labels >= 0) & (labels < len(_LOOKUP_TABLE))
    out[in_range_mask] = _LOOKUP_TABLE[labels[in_range_mask]]
    return out


def make_synthetic_scene(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic LiDAR scene with known labels:
    - Flat ground at z=0 (class 1: road)
    - 15 cm kerb across r=10 m line (class 7: non-drivable ground / sidewalk)
    - Vertical pole (class 3: static obstacle)
    - 3D box (class 3: static obstacle)
    - Distant points beyond 100 m (class 0: unlabeled)

    Returns:
        (scan, labels) where scan is (N, 4) float32 [x, y, z, intensity]
        and labels is (N,) int64 merged class labels (0 to 7).
    """
    rng = np.random.default_rng(seed)

    # 1. Flat ground at z=0 (road, class 1)
    n_ground = 12000
    r_ground = rng.uniform(0.5, 85.0, size=n_ground)
    theta_ground = rng.uniform(0, 2 * np.pi, size=n_ground)
    x_ground = r_ground * np.cos(theta_ground)
    y_ground = r_ground * np.sin(theta_ground)
    z_ground = np.zeros(n_ground, dtype=np.float32)
    i_ground = rng.uniform(0.1, 0.4, size=n_ground).astype(np.float32)
    labels_ground = np.full(n_ground, 1, dtype=np.int64)

    # 2. 15 cm kerb across r=10 m line (sidewalk/non-drivable ground, class 7)
    n_kerb = 4000
    r_kerb = rng.uniform(9.7, 10.3, size=n_kerb)
    theta_kerb = rng.uniform(-0.3, 0.3, size=n_kerb)
    x_kerb = r_kerb * np.cos(theta_kerb)
    y_kerb = r_kerb * np.sin(theta_kerb)
    z_kerb = rng.choice([0.0, 0.15], size=n_kerb).astype(np.float32)
    i_kerb = rng.uniform(0.2, 0.5, size=n_kerb).astype(np.float32)
    labels_kerb = np.full(n_kerb, 7, dtype=np.int64)

    # 3. Pole at (x=5.0, y=2.0) (static obstacle, class 3)
    n_pole = 500
    pole_theta = rng.uniform(0, 2 * np.pi, size=n_pole)
    pole_r = rng.uniform(0.08, 0.12, size=n_pole)
    x_pole = 5.0 + pole_r * np.cos(pole_theta)
    y_pole = 2.0 + pole_r * np.sin(pole_theta)
    z_pole = rng.uniform(0.0, 3.0, size=n_pole).astype(np.float32)
    i_pole = rng.uniform(0.5, 0.9, size=n_pole).astype(np.float32)
    labels_pole = np.full(n_pole, 3, dtype=np.int64)

    # 4. Box at (x=15.0, y=5.0) (static obstacle, class 3)
    n_box = 800
    x_box = rng.uniform(14.0, 16.0, size=n_box)
    y_box = rng.uniform(4.0, 6.0, size=n_box)
    z_box = rng.uniform(0.0, 1.5, size=n_box).astype(np.float32)
    i_box = rng.uniform(0.4, 0.8, size=n_box).astype(np.float32)
    labels_box = np.full(n_box, 3, dtype=np.int64)

    # 5. Out of range points beyond 100 m (unlabeled, class 0)
    n_out = 100
    r_out = rng.uniform(102.0, 130.0, size=n_out)
    theta_out = rng.uniform(0, 2 * np.pi, size=n_out)
    x_out = r_out * np.cos(theta_out)
    y_out = r_out * np.sin(theta_out)
    z_out = rng.uniform(-1.0, 2.0, size=n_out).astype(np.float32)
    i_out = rng.uniform(0.1, 0.3, size=n_out).astype(np.float32)
    labels_out = np.zeros(n_out, dtype=np.int64)

    scan = np.vstack([
        np.column_stack([x_ground, y_ground, z_ground, i_ground]),
        np.column_stack([x_kerb, y_kerb, z_kerb, i_kerb]),
        np.column_stack([x_pole, y_pole, z_pole, i_pole]),
        np.column_stack([x_box, y_box, z_box, i_box]),
        np.column_stack([x_out, y_out, z_out, i_out]),
    ]).astype(np.float32)

    labels = np.concatenate([
        labels_ground,
        labels_kerb,
        labels_pole,
        labels_box,
        labels_out,
    ]).astype(np.int64)

    return scan, labels


class SourceDescription:
    """Descriptor object representing a discovered point cloud data source."""

    def __init__(
        self,
        kind: str,
        label: str,
        name: str,
        n_frames: int,
        path: Optional[Path] = None,
        origin: Optional[str] = None,
    ):
        self.kind = kind
        self.label = label
        self.name = name
        self.n_frames = n_frames
        self.path = path
        self.origin = origin

    def __iter__(self):
        return iter((self.kind, self.name, self.n_frames))

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.kind, self.name, self.n_frames)[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def as_tuple(self):
        return (self.kind, self.label, self.name, self.n_frames, self.path)

    def __repr__(self):
        return (
            f"SourceDescription(kind='{self.kind}', label='{self.label}', "
            f"name='{self.name}', n_frames={self.n_frames}, path={self.path}, origin={self.origin})"
        )

    def __eq__(self, other):
        if isinstance(other, SourceDescription):
            return (self.kind, self.label, self.name, self.n_frames, self.path) == (
                other.kind,
                other.label,
                other.name,
                other.n_frames,
                other.path,
            )
        if isinstance(other, tuple):
            if len(other) == 3:
                return (self.kind, self.name, self.n_frames) == other
            if len(other) == 5:
                return (self.kind, self.label, self.name, self.n_frames, self.path) == other
        return False


def _find_sequences_in_dir(base_path: Path) -> List[Tuple[Path, Path, int, List[Tuple[Path, Path]]]]:
    """Recursively search base_path (up to 3 levels deeper than base_path) for sequences/<NN>/velodyne
    with matching .bin and .label files.

    Handles both layouts:
      1. base_path/sequences/<NN>/velodyne
      2. base_path/<folder>/sequences/<NN>/velodyne

    Returns:
        List of tuples: (seq_root, velo_path, n_frames, pairs) sorted by seq_root path.
    """
    if not base_path.exists() or not base_path.is_dir():
        return []

    candidates = []
    for p in base_path.rglob("velodyne"):
        if not p.is_dir():
            continue
        try:
            rel = p.relative_to(base_path)
        except ValueError:
            continue

        # Depth restriction: sequences/<NN>/velodyne (3 parts) up to 3 levels nested (<= 5 parts)
        if len(rel.parts) > 5:
            continue

        seq_root = p.parent
        # Verify sequences/<NN>/velodyne structure or layout
        if seq_root.parent.name != "sequences" and "sequences" not in [part.lower() for part in rel.parts]:
            continue

        bin_files = sorted(list(p.glob("*.bin")))
        if not bin_files:
            continue

        labels_dir = seq_root / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            continue

        pairs = []
        for b in bin_files:
            lbl = labels_dir / f"{b.stem}.label"
            if lbl.exists():
                pairs.append((b, lbl))

        if not pairs:
            continue

        candidates.append((seq_root, p, len(pairs), pairs))

    candidates.sort(key=lambda c: str(c[0]))
    return candidates


def _has_fake_marker(seq_root: Path, stop_path: Path) -> bool:
    """Checks for README_FAKE.txt in seq_root or any parent folder up to stop_path (inclusive)."""
    curr = seq_root.resolve()
    stop = stop_path.resolve()
    while True:
        if (curr / "README_FAKE.txt").exists() or (curr / "readme_fake.txt").exists():
            return True
        if curr == stop or curr.parent == curr:
            break
        curr = curr.parent
    return False


def describe_source(
    data_dir: Union[str, Path] = "data",
    seq_id: Optional[str] = None,
    sample_dir: Optional[Union[str, Path]] = "sample_data",
) -> SourceDescription:
    """Discovers and identifies the nature of the LiDAR point cloud dataset.

    Searches in order:
      1. data_dir (default: 'data', recursively up to 3 levels)
      2. sample_dir (default: 'sample_data')

    Uses the first location that has sequences/<NN>/velodyne with matching .bin
    and .label files. Handles both layouts:
      - data/sequences/<NN>
      - data/<folder>/sequences/<NN>

    Labels:
      - "real_full": no README_FAKE.txt in sequence folder or any parent folder
        -> "Real SemanticKITTI"
      - "synthetic_kitti_format": README_FAKE.txt in sequence folder or a parent folder
        -> "Synthetic (SemanticKITTI format)"
      - "synthetic_scene": no files found, built-in generator used
        -> "Synthetic scene"
    """
    p_data = Path(data_dir)
    p_sample = Path(sample_dir) if sample_dir is not None else None

    # Determine candidate search roots in order of precedence:
    # 1. If data_dir has subdirectories "data" or "sample_data", caller passed a container root (e.g. tmp_path)
    # 2. If data_dir is the default "data" (or Path("data")): check Path("data"), then Path("sample_data")
    # 3. If data_dir is a custom path with sample_dir explicitly given: check data_dir, then sample_dir
    # 4. Otherwise: check data_dir directly.
    locations_to_search: List[Tuple[Path, str]] = []

    if (p_data / "data").exists() or (p_data / "sample_data").exists():
        if (p_data / "data").exists():
            locations_to_search.append((p_data / "data", "data/"))
        if (p_data / "sample_data").exists():
            locations_to_search.append((p_data / "sample_data", "sample_data/ (excerpt)"))
    elif p_data == Path("data") or str(p_data) == "data":
        locations_to_search.append((Path("data"), "data/"))
        if p_sample is not None:
            locations_to_search.append((p_sample, "sample_data/ (excerpt)"))
    else:
        origin_primary = "sample_data/ (excerpt)" if "sample_data" in p_data.parts else "data/"
        locations_to_search.append((p_data, origin_primary))
        if p_sample is not None and p_sample != Path("sample_data") and p_sample.resolve() != p_data.resolve():
            locations_to_search.append((p_sample, "sample_data/ (excerpt)"))

    chosen_location = None
    chosen_origin = None
    candidates = []

    for loc_path, origin_name in locations_to_search:
        cands = _find_sequences_in_dir(loc_path)
        if cands:
            chosen_location = loc_path
            chosen_origin = origin_name
            candidates = cands
            break

    if not candidates or chosen_location is None:
        return SourceDescription(
            kind="synthetic_scene",
            label="Synthetic scene",
            name="Synthetic Urban Scene",
            n_frames=0,
            path=None,
            origin=None,
        )

    # Select sequence
    selected = None
    if seq_id is not None:
        for seq_root, velo_p, n_frames, pairs in candidates:
            if seq_root.name == seq_id or seq_id in str(seq_root.relative_to(chosen_location)):
                selected = (seq_root, velo_p, n_frames, pairs)
                break

    if selected is None:
        selected = candidates[0]

    seq_root, velo_p, n_frames, pairs = selected

    # Determine if synthetic or real
    stop_dir = chosen_location
    if p_data.exists() and p_data.resolve() in seq_root.resolve().parents:
        stop_dir = p_data

    if _has_fake_marker(seq_root, stop_dir):
        kind = "synthetic_kitti_format"
        label = "Synthetic (SemanticKITTI format)"
    else:
        kind = "real_full"
        label = "Real SemanticKITTI"

    return SourceDescription(
        kind=kind,
        label=label,
        name=seq_root.name,
        n_frames=n_frames,
        path=seq_root,
        origin=chosen_origin,
    )


def scan_kitti_sequences(
    data_dir: Optional[Union[str, Path]] = None,
    sample_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Scans for SemanticKITTI sequences containing pairs of matching .bin and .label files.

    Searches in order: data/ (recursively, up to 3 levels), then sample_data/.
    Uses the first location that has sequences/<NN>/velodyne with matching files.
    Handles both layouts: data/sequences/<NN> and data/<folder>/sequences/<NN>.

    Returns:
        Dictionary mapping sequence display name to a dictionary containing:
          - "pairs": list of (bin_path, label_path) tuples
          - "poses_path": Path to poses.txt if present, else None
          - "desc": SourceDescription
          - "origin": "data/" or "sample_data/ (excerpt)"
    """
    p_data = Path(data_dir) if data_dir is not None else Path("data")
    p_sample = Path(sample_dir) if sample_dir is not None else Path("sample_data")

    locations_to_search: List[Tuple[Path, str]] = []

    if (p_data / "data").exists() or (p_data / "sample_data").exists():
        if (p_data / "data").exists():
            locations_to_search.append((p_data / "data", "data/"))
        if (p_data / "sample_data").exists():
            locations_to_search.append((p_data / "sample_data", "sample_data/ (excerpt)"))
    elif p_data == Path("data") or str(p_data) == "data":
        locations_to_search.append((Path("data"), "data/"))
        if p_sample is not None:
            locations_to_search.append((p_sample, "sample_data/ (excerpt)"))
    else:
        origin_primary = "sample_data/ (excerpt)" if "sample_data" in p_data.parts else "data/"
        locations_to_search.append((p_data, origin_primary))
        if p_sample is not None and p_sample != Path("sample_data") and p_sample.resolve() != p_data.resolve():
            locations_to_search.append((p_sample, "sample_data/ (excerpt)"))

    chosen_location = None
    chosen_origin = None
    candidates = []

    for loc_path, origin_name in locations_to_search:
        cands = _find_sequences_in_dir(loc_path)
        if cands:
            chosen_location = loc_path
            chosen_origin = origin_name
            candidates = cands
            break

    sequences: Dict[str, Dict[str, Any]] = {}
    if not candidates or chosen_location is None:
        return sequences

    for seq_root, velo_p, n_frames, pairs in candidates:
        try:
            seq_name = str(seq_root.relative_to(chosen_location))
        except ValueError:
            seq_name = seq_root.name

        cand_poses = seq_root / "poses.txt"
        if not cand_poses.exists():
            cand_poses = seq_root.parent / "poses.txt"

        desc = describe_source(data_dir=chosen_location, seq_id=seq_root.name)
        desc.origin = chosen_origin

        sequences[seq_name] = {
            "pairs": pairs,
            "poses_path": cand_poses if cand_poses.exists() else None,
            "desc": desc,
            "origin": chosen_origin,
        }

    return sequences

