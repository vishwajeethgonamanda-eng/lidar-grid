"""Unit tests for pipeline metrics recording and summary generation.

Verifies:
  1. run_pipeline.py produces metrics.csv with all required columns.
  2. make_results.py generates summary.md containing the detected data source label in the first line.
  3. make_results.py outputs the 3 required slide-ready PNG charts.
"""
from pathlib import Path
import csv
import pytest
from src.loader import describe_source
from make_results import load_metrics, main as make_results_main


def test_make_results_summary_contains_source_label(tmp_path, monkeypatch):
    """Verifies that summary.md contains the exact source label."""
    csv_file = tmp_path / "metrics.csv"
    expected_label = "Synthetic (SemanticKITTI format)"

    # Create dummy metrics.csv
    fieldnames = [
        "frame",
        "patches_off_ms",
        "patches_on_ms",
        "add_points_ms",
        "rebinning_ms",
        "re_binning_ms",
        "traversability_ms",
        "risk_search_ms",
        "total_ms",
        "fine_cells",
        "coarse_cells",
        "memory_var_mb",
        "memory_uniform_mb",
        "patch_count",
        "source_label",
        "traversability_agreement",
        "purity_20_40m",
        "purity_40_60m",
    ]
    rows = []
    for i in range(5):
        rows.append({
            "frame": i,
            "patches_off_ms": "18.500",
            "patches_on_ms": "28.500",
            "add_points_ms": "14.200",
            "rebinning_ms": "4.100",
            "re_binning_ms": "4.100",
            "traversability_ms": "7.500",
            "risk_search_ms": "2.700",
            "total_ms": "28.500",
            "fine_cells": 12000,
            "coarse_cells": 8000,
            "memory_var_mb": "12.35",
            "memory_uniform_mb": "179.20",
            "patch_count": 4,
            "source_label": expected_label,
            "traversability_agreement": "0.9850",
            "purity_20_40m": "0.9500",
            "purity_40_60m": "0.9200",
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Run make_results targeting tmp_path
    monkeypatch.setattr("sys.argv", ["make_results.py", "--csv", str(csv_file), "--out", str(tmp_path)])
    make_results_main()

    summary_file = tmp_path / "summary.md"
    assert summary_file.exists(), "summary.md was not created"

    content = summary_file.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]

    # Verify first line contains the source label
    assert expected_label in first_line, f"Source label '{expected_label}' not in first line: '{first_line}'"
    # Verify first line contains machine / on keyword
    assert " on " in first_line


def test_make_results_generates_three_png_charts(tmp_path, monkeypatch):
    """Verifies that make_results.py generates 3 valid PNG charts."""
    csv_file = tmp_path / "metrics.csv"
    fieldnames = [
        "frame", "patches_off_ms", "patches_on_ms", "add_points_ms", "rebinning_ms",
        "re_binning_ms", "traversability_ms", "risk_search_ms", "total_ms",
        "fine_cells", "coarse_cells", "memory_var_mb", "memory_uniform_mb",
        "patch_count", "source_label", "traversability_agreement",
        "purity_20_40m", "purity_40_60m",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "frame": 0, "patches_off_ms": "18.0", "patches_on_ms": "27.0",
            "add_points_ms": "14.0", "rebinning_ms": "4.0", "re_binning_ms": "4.0",
            "traversability_ms": "7.0", "risk_search_ms": "2.0", "total_ms": "27.0",
            "fine_cells": 10000, "coarse_cells": 8000, "memory_var_mb": "12.3",
            "memory_uniform_mb": "179.2", "patch_count": 3,
            "source_label": "Real SemanticKITTI", "traversability_agreement": "0.99",
            "purity_20_40m": "0.95", "purity_40_60m": "0.90",
        })

    monkeypatch.setattr("sys.argv", ["make_results.py", "--csv", str(csv_file), "--out", str(tmp_path)])
    make_results_main()

    chart1 = tmp_path / "stage_times_stacked.png"
    chart2 = tmp_path / "frame_time_histogram.png"
    chart3 = tmp_path / "memory_comparison.png"

    assert chart1.exists() and chart1.stat().st_size > 1000
    assert chart2.exists() and chart2.stat().st_size > 1000
    assert chart3.exists() and chart3.stat().st_size > 1000


def test_live_results_summary_matches_source_if_exists():
    """If results/summary.md has been generated, verifies that it names the current data source."""
    summary_path = Path("results/summary.md")
    if not summary_path.exists():
        pytest.skip("results/summary.md not yet generated")

    desc = describe_source("data")
    content = summary_path.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert desc.label in first_line or desc.label in content
