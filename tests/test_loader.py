import numpy as np
import pytest
from src.loader import (
    load_scan,
    load_labels,
    remap_labels,
    make_synthetic_scene,
    SEMANTIC_KITTI_TO_MERGED_CLASS,
    MERGED_CLASS_NAMES,
)
from src.grid_engine import VarResGrid


def test_load_scan(tmp_path):
    bin_file = tmp_path / "000000.bin"
    # Write 5 points with (x, y, z, intensity)
    pts = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8],
        [-1.0, -2.0, 0.0, 0.1],
        [10.0, 20.0, -1.5, 0.9],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    pts.tofile(bin_file)

    loaded = load_scan(bin_file)
    assert loaded.shape == (5, 4)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, pts)

    # Empty scan
    empty_file = tmp_path / "empty.bin"
    np.array([], dtype=np.float32).tofile(empty_file)
    loaded_empty = load_scan(empty_file)
    assert loaded_empty.shape == (0, 4)
    assert loaded_empty.dtype == np.float32


def test_load_labels(tmp_path):
    label_file = tmp_path / "000000.label"
    # SemanticKITTI format: uint32
    # lower 16 bits = semantic label, upper 16 bits = instance id
    raw_semantics = np.array([0, 10, 40, 70, 252], dtype=np.uint32)
    instance_ids = np.array([0, 5, 12, 100, 1], dtype=np.uint32)
    raw_uint32 = raw_semantics | (instance_ids << 16)
    raw_uint32.tofile(label_file)

    loaded = load_labels(label_file)
    assert loaded.shape == (5,)
    # Verify upper 16 bits were masked out and lower 16 bits retained
    assert np.array_equal(loaded, raw_semantics)

    # Empty labels
    empty_file = tmp_path / "empty.label"
    np.array([], dtype=np.uint32).tofile(empty_file)
    loaded_empty = load_labels(empty_file)
    assert loaded_empty.shape == (0,)


def test_every_id_in_table_maps_to_expected_class():
    raw_ids = np.array(list(SEMANTIC_KITTI_TO_MERGED_CLASS.keys()), dtype=np.int64)
    expected_classes = np.array(list(SEMANTIC_KITTI_TO_MERGED_CLASS.values()), dtype=np.int64)
    remapped = remap_labels(raw_ids)
    assert np.array_equal(remapped, expected_classes)


def test_moving_objects_252_to_259():
    moving_ids = np.arange(252, 260, dtype=np.int64)
    remapped = remap_labels(moving_ids)
    assert np.all(remapped == 6)
    assert len(remapped) == 8


def test_unknown_ids_map_to_zero():
    unknown_ids = np.array([2, 5, 100, 65535], dtype=np.int64)
    remapped = remap_labels(unknown_ids)
    assert np.all(remapped == 0)


def test_shapes_and_empty_input():
    # Empty 1D input
    empty_1d = np.array([], dtype=np.int64)
    remapped_empty = remap_labels(empty_1d)
    assert remapped_empty.shape == empty_1d.shape

    # 1D input
    arr_1d = np.array([40, 48, 70], dtype=np.int64)
    assert remap_labels(arr_1d).shape == arr_1d.shape

    # 2D input
    arr_2d = np.array([[40, 48], [70, 10]], dtype=np.int64)
    remapped_2d = remap_labels(arr_2d)
    assert remapped_2d.shape == arr_2d.shape
    assert np.array_equal(remapped_2d, np.array([[1, 7], [2, 4]], dtype=np.int64))


def test_specific_ground_classes():
    # 48 and 49 map to 7 (non-drivable ground)
    non_drivable = np.array([48, 49], dtype=np.int64)
    assert np.array_equal(remap_labels(non_drivable), np.array([7, 7], dtype=np.int64))

    # 40, 44 and 60 map to 1 (road)
    road_classes = np.array([40, 44, 60], dtype=np.int64)
    assert np.array_equal(remap_labels(road_classes), np.array([1, 1, 1], dtype=np.int64))


def test_make_synthetic_scene_and_grid_integration():
    scan, labels = make_synthetic_scene()
    assert scan.ndim == 2 and scan.shape[1] == 4
    assert labels.ndim == 1 and len(labels) == len(scan)

    # Validate ground plane: at z=0, label=1 (road)
    ground_mask = (np.abs(scan[:, 2]) < 1e-4) & (np.hypot(scan[:, 0], scan[:, 1]) < 9.0)
    assert np.any(ground_mask)
    assert np.all(labels[ground_mask] == 1)

    # Validate kerb across r=10 m: has z near 0.15, label=7 (non-drivable ground)
    kerb_mask = (
        (scan[:, 2] > 0.14)
        & (scan[:, 2] < 0.16)
        & (np.hypot(scan[:, 0], scan[:, 1]) >= 9.6)
        & (np.hypot(scan[:, 0], scan[:, 1]) <= 10.4)
    )
    assert np.any(kerb_mask)
    assert np.all(labels[kerb_mask] == 7)

    # Validate pole: vertical structure with label=3
    pole_mask = labels == 3
    assert np.any(pole_mask)
    assert np.ptp(scan[pole_mask, 2]) > 1.5

    # Integration with VarResGrid: n_classes must be at least 8 (classes 0 to 7)
    grid = VarResGrid(n_classes=8)
    grid.add_points(scan[:, :3], labels)

    stats = grid.stats()
    assert stats["total_input"] == len(scan)
    assert stats["in_fine"] > 0
    assert stats["in_coarse"] > 0
    assert stats["out_of_range"] > 0
    assert stats["in_fine"] + stats["in_coarse"] + stats["out_of_range"] == stats["total_input"]


def test_describe_source_real_dataset(tmp_path):
    from src.loader import describe_source

    seq_dir = tmp_path / "sequences" / "00"
    velo_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    velo_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create dummy bin and label files
    np.zeros((10, 4), dtype=np.float32).tofile(velo_dir / "000000.bin")
    np.zeros(10, dtype=np.uint32).tofile(labels_dir / "000000.label")

    desc = describe_source(tmp_path)
    assert desc.kind == "real_full"
    assert desc.label == "Real SemanticKITTI"
    assert desc.name == "00"
    assert desc.n_frames == 1
    assert desc.path == seq_dir

    # Test unpacking
    k, n, nf = desc
    assert k == "real_full"
    assert n == "00"
    assert nf == 1

    k5, l5, n5, nf5, p5 = desc.as_tuple()
    assert k5 == "real_full"
    assert l5 == "Real SemanticKITTI"
    assert n5 == "00"
    assert nf5 == 1
    assert p5 == seq_dir


def test_describe_source_synthetic_marker_inside_sequence(tmp_path):
    from src.loader import describe_source

    seq_dir = tmp_path / "synthetic_kitti_sample" / "sequences" / "99"
    velo_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    velo_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.zeros((5, 4), dtype=np.float32).tofile(velo_dir / "000000.bin")
    np.zeros(5, dtype=np.uint32).tofile(labels_dir / "000000.label")

    # Place marker inside sequence folder
    marker = seq_dir / "README_FAKE.txt"
    marker.write_text("This dataset is synthetic.\n", encoding="utf-8")

    desc = describe_source(tmp_path)
    assert desc.kind == "synthetic_kitti_format"
    assert desc.label == "Synthetic (SemanticKITTI format)"
    assert desc.name == "99"
    assert desc.n_frames == 1


def test_describe_source_synthetic_marker_in_parent(tmp_path):
    from src.loader import describe_source

    dataset_root = tmp_path / "dataset"
    seq_dir = dataset_root / "sequences" / "01"
    velo_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    velo_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.zeros((5, 4), dtype=np.float32).tofile(velo_dir / "000000.bin")
    np.zeros(5, dtype=np.uint32).tofile(labels_dir / "000000.label")

    # Place marker in parent folder
    marker = dataset_root / "README_FAKE.txt"
    marker.write_text("Synthetic marker in parent\n", encoding="utf-8")

    desc = describe_source(tmp_path)
    assert desc.kind == "synthetic_kitti_format"
    assert desc.label == "Synthetic (SemanticKITTI format)"
    assert desc.name == "01"


def test_describe_source_synthetic_scene_fallback(tmp_path):
    from src.loader import describe_source

    # Empty folder
    desc = describe_source(tmp_path)
    assert desc.kind == "synthetic_scene"
    assert desc.label == "Synthetic scene"
    assert desc.n_frames == 0
    assert desc.path is None

    # Non-existent folder
    desc_missing = describe_source(tmp_path / "does_not_exist")
    assert desc_missing.kind == "synthetic_scene"
    assert desc_missing.label == "Synthetic scene"
    assert desc_missing.n_frames == 0
    assert desc_missing.path is None


def test_search_order_only_sample_data_present(tmp_path):
    from src.loader import describe_source

    sample_seq = tmp_path / "sample_data" / "sequences" / "99"
    velo_dir = sample_seq / "velodyne"
    labels_dir = sample_seq / "labels"
    velo_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    np.zeros((6, 4), dtype=np.float32).tofile(velo_dir / "000000.bin")
    np.zeros(6, dtype=np.uint32).tofile(labels_dir / "000000.label")
    (sample_seq / "README_FAKE.txt").write_text("fake data\n", encoding="utf-8")

    # 1. Calling describe_source with root tmp_path containing only sample_data
    desc1 = describe_source(tmp_path)
    assert desc1.kind == "synthetic_kitti_format"
    assert desc1.label == "Synthetic (SemanticKITTI format)"
    assert desc1.name == "99"
    assert desc1.n_frames == 1
    assert desc1.path == sample_seq
    assert desc1.origin == "sample_data/ (excerpt)"

    # 2. Calling describe_source with explicit data_dir and sample_dir
    desc2 = describe_source(data_dir=tmp_path / "data", sample_dir=tmp_path / "sample_data")
    assert desc2.kind == "synthetic_kitti_format"
    assert desc2.label == "Synthetic (SemanticKITTI format)"
    assert desc2.name == "99"
    assert desc2.n_frames == 1
    assert desc2.path == sample_seq
    assert desc2.origin == "sample_data/ (excerpt)"


def test_search_order_both_present_data_wins(tmp_path):
    from src.loader import describe_source

    # Real data in data/
    data_seq = tmp_path / "data" / "sequences" / "00"
    velo_data = data_seq / "velodyne"
    labels_data = data_seq / "labels"
    velo_data.mkdir(parents=True)
    labels_data.mkdir(parents=True)
    np.zeros((10, 4), dtype=np.float32).tofile(velo_data / "000000.bin")
    np.zeros(10, dtype=np.uint32).tofile(labels_data / "000000.label")

    # Synthetic sample in sample_data/
    sample_seq = tmp_path / "sample_data" / "sequences" / "99"
    velo_sample = sample_seq / "velodyne"
    labels_sample = sample_seq / "labels"
    velo_sample.mkdir(parents=True)
    labels_sample.mkdir(parents=True)
    np.zeros((5, 4), dtype=np.float32).tofile(velo_sample / "000000.bin")
    np.zeros(5, dtype=np.uint32).tofile(labels_sample / "000000.label")
    (sample_seq / "README_FAKE.txt").write_text("fake marker\n", encoding="utf-8")

    # 1. Calling describe_source with root tmp_path: data/ must win
    desc1 = describe_source(tmp_path)
    assert desc1.kind == "real_full"
    assert desc1.label == "Real SemanticKITTI"
    assert desc1.name == "00"
    assert desc1.n_frames == 1
    assert desc1.path == data_seq
    assert desc1.origin == "data/"

    # 2. Calling with explicit data_dir and sample_dir: data/ must win
    desc2 = describe_source(data_dir=tmp_path / "data", sample_dir=tmp_path / "sample_data")
    assert desc2.kind == "real_full"
    assert desc2.label == "Real SemanticKITTI"
    assert desc2.name == "00"
    assert desc2.n_frames == 1
    assert desc2.path == data_seq
    assert desc2.origin == "data/"


def test_search_order_neither_present(tmp_path):
    from src.loader import describe_source

    # Neither folder has sequence files
    empty_data = tmp_path / "data"
    empty_sample = tmp_path / "sample_data"
    empty_data.mkdir()
    empty_sample.mkdir()

    # 1. Root tmp_path
    desc1 = describe_source(tmp_path)
    assert desc1.kind == "synthetic_scene"
    assert desc1.label == "Synthetic scene"
    assert desc1.n_frames == 0
    assert desc1.path is None
    assert desc1.origin is None

    # 2. Explicit data_dir and sample_dir
    desc2 = describe_source(data_dir=empty_data, sample_dir=empty_sample)
    assert desc2.kind == "synthetic_scene"
    assert desc2.label == "Synthetic scene"
    assert desc2.n_frames == 0
    assert desc2.path is None
    assert desc2.origin is None


def test_search_nested_folder_layout(tmp_path):
    from src.loader import describe_source

    # data/<folder>/sequences/<NN>
    nested_seq = tmp_path / "data" / "kitti_dataset" / "sequences" / "03"
    velo_dir = nested_seq / "velodyne"
    labels_dir = nested_seq / "labels"
    velo_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    np.zeros((8, 4), dtype=np.float32).tofile(velo_dir / "000000.bin")
    np.zeros(8, dtype=np.uint32).tofile(labels_dir / "000000.label")

    desc = describe_source(tmp_path)
    assert desc.kind == "real_full"
    assert desc.label == "Real SemanticKITTI"
    assert desc.name == "03"
    assert desc.n_frames == 1
    assert desc.path == nested_seq
    assert desc.origin == "data/"

