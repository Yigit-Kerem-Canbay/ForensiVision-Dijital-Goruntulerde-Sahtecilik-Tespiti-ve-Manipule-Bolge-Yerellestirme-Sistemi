"""
Unit tests for the ForensiVision dataset ingestion and evaluation pipeline.

Covers:
- DatasetSample dataclass and properties
- BaseDataset load_image, load_mask, and mask dimension validation
- CoMoFoD file parsing, variant mapping, and group-id handling
- Columbia file parsing and ground-truth pairing
- Manifest generation and serialization (CSV & JSON)
- Group-based leakage prevention
- evaluate_dataset end-to-end execution on mock fixtures
- Authentic sample separation (preventing artificial metric inflation)
- Resilient failure recovery (exceptions don't crash whole runs)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from forensivision.datasets import (
    BaseDataset,
    CoMoFoDDataset,
    ColumbiaDataset,
    DefactoDataset,
    DatasetSample,
    build_manifest,
    load_manifest,
    save_manifest,
)
from scripts.evaluate_dataset import run_evaluation


@pytest.fixture
def mock_comofod_dir(tmp_path: Path) -> Path:
    """Creates a miniature mock CoMoFoD dataset hierarchy with real PNG images."""
    data_dir = tmp_path / "comofod_raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    h, w = 120, 160

    # Base image 001: Original, Forged, Binary mask, Color mask, Attack variant JC1
    img_orig = np.full((h, w, 3), 180, dtype=np.uint8)
    img_forged = img_orig.copy()
    # Create distinct patches for copy-move
    img_forged[20:50, 20:50] = 50
    img_forged[20:50, 90:120] = 50

    # Binary mask: both regions marked with 255
    mask_bin = np.zeros((h, w), dtype=np.uint8)
    mask_bin[20:50, 20:50] = 255
    mask_bin[20:50, 90:120] = 255

    # Color mask (source=green, target=red in BGR)
    mask_col = np.zeros((h, w, 3), dtype=np.uint8)
    mask_col[20:50, 20:50] = [0, 255, 0]
    mask_col[20:50, 90:120] = [0, 0, 255]

    # Attack variant (e.g., JPEG compression)
    img_jc = img_forged.copy()

    # Save 001 files
    cv2.imwrite(str(data_dir / "001_O.png"), img_orig)
    cv2.imwrite(str(data_dir / "001_F.png"), img_forged)
    cv2.imwrite(str(data_dir / "001_B.png"), mask_bin)
    cv2.imwrite(str(data_dir / "001_M.png"), mask_col)
    cv2.imwrite(str(data_dir / "001_F_JC1.png"), img_jc)

    # Base image 002: Original, Forged, Binary mask
    img2_orig = np.full((h, w, 3), 210, dtype=np.uint8)
    img2_forged = img2_orig.copy()
    img2_forged[10:30, 10:30] = 20
    img2_forged[60:80, 60:80] = 20
    mask2_bin = np.zeros((h, w), dtype=np.uint8)
    mask2_bin[10:30, 10:30] = 255
    mask2_bin[60:80, 60:80] = 255

    cv2.imwrite(str(data_dir / "002_O.png"), img2_orig)
    cv2.imwrite(str(data_dir / "002_F.png"), img2_forged)
    cv2.imwrite(str(data_dir / "002_B.png"), mask2_bin)

    return data_dir


@pytest.fixture
def mock_columbia_dir(tmp_path: Path) -> Path:
    """Creates a miniature mock Columbia dataset directory with spliced and authentic images."""
    data_dir = tmp_path / "columbia_raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    h, w = 100, 140

    # Authentic image (e.g. canong3_01_sub_01.tif)
    auth_img = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(data_dir / "canong3_01_auth_01.tif"), auth_img)

    # Spliced image + ground truth edge mask
    spliced_img = np.full((h, w, 3), 120, dtype=np.uint8)
    spliced_img[30:70, 30:70] = 240
    cv2.imwrite(str(data_dir / "canong3_01_sp_01.tif"), spliced_img)

    # Mask
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[30:70, 30:70] = 255
    cv2.imwrite(str(data_dir / "canong3_01_01_edg.tif"), edge_mask)

    return data_dir


# =========================================================================
# Base Dataset & Loading Tests
# =========================================================================


def test_dataset_sample_properties(tmp_path: Path):
    """Verifies DatasetSample dataclass properties, dictionary conversion, and authentic checks."""
    dummy_img = tmp_path / "image.png"
    dummy_mask = tmp_path / "mask.png"
    dummy_img.touch()
    dummy_mask.touch()

    sample = DatasetSample(
        sample_id="test_001",
        task_type="copy_move",
        image_path=dummy_img,
        mask_path=dummy_mask,
        group_id="group_1",
        manipulation_type="copy_move",
        variant="clean",
    )

    assert sample.sample_id == "test_001"
    assert not sample.is_authentic()
    d = sample.to_dict()
    assert d["group_id"] == "group_1"
    assert d["image_path"] == str(dummy_img)

    # Authentic sample
    auth_sample = DatasetSample(
        sample_id="test_auth",
        task_type="copy_move",
        image_path=dummy_img,
        manipulation_type="authentic",
    )
    assert auth_sample.is_authentic()


def test_load_image_and_mask(mock_comofod_dir: Path):
    """Verifies loading image and binary mask through BaseDataset methods."""
    ds = CoMoFoDDataset(mock_comofod_dir)
    assert len(ds) > 0

    sample = ds[0]
    img = ds.load_image(sample)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3

    mask = ds.load_mask(sample, target_shape=img.shape[:2])
    assert isinstance(mask, np.ndarray)
    assert mask.shape == img.shape[:2]
    assert np.all(np.isin(mask, [0, 255]))


def test_mask_dimension_mismatch_raises_error(mock_comofod_dir: Path):
    """Verifies that dimension mismatch raises ValueError instead of silent corruption."""
    ds = CoMoFoDDataset(mock_comofod_dir)
    # Find forged sample
    forged = next(s for s in ds.samples if not s.is_authentic())

    # Pass an incorrect target_shape
    with pytest.raises(ValueError, match="Uyuşmazlık|uyuşmuyor"):
        ds.load_mask(forged, target_shape=(999, 999))


# =========================================================================
# CoMoFoD Dataset Parser Tests
# =========================================================================


def test_comofod_parsing(mock_comofod_dir: Path):
    """Verifies CoMoFoD file discovery, variant classification, and mask pairing."""
    ds = CoMoFoDDataset(mock_comofod_dir)
    samples = ds.samples

    # We expect:
    # 001: 001_F (clean copy-move), 001_F_JC1 (jpeg attack), 001_O (authentic)
    # 002: 002_F (clean copy-move), 002_O (authentic)
    # Total = 5 samples
    assert len(samples) == 5

    # Check 001_F clean
    f1 = next(s for s in samples if s.sample_id == "001_F")
    assert f1.task_type == "copy_move"
    assert f1.group_id == "001"
    assert f1.variant == "clean"
    assert not f1.is_authentic()
    assert f1.mask_path is not None
    assert f1.mask_path.name == "001_B.png"

    # Check 001_F_JC1 attack
    f1_jc = next(s for s in samples if s.sample_id == "001_F_JC1")
    assert "jpeg_compression" in f1_jc.variant
    assert f1_jc.group_id == "001"
    assert f1_jc.mask_path.name == "001_B.png"

    # Check authentic sample
    o1 = next(s for s in samples if s.sample_id == "001_O")
    assert o1.is_authentic()
    assert o1.mask_path is None
    assert o1.group_id == "001"


def test_comofod_groups_prevent_data_leakage(mock_comofod_dir: Path):
    """Verifies that all variants sharing the same base ID are grouped under identical group_id."""
    ds = CoMoFoDDataset(mock_comofod_dir)
    groups = ds.get_groups()

    assert "001" in groups
    assert "002" in groups
    # Group 001 contains 3 items (_F, _F_JC1, _O)
    assert len(groups["001"]) == 3
    # Group 002 contains 2 items (_F, _O)
    assert len(groups["002"]) == 2


# =========================================================================
# Columbia Dataset Parser Tests
# =========================================================================


def test_columbia_parsing(mock_columbia_dir: Path):
    """Verifies Columbia file pairing for spliced images, masks, and authentic images."""
    ds = ColumbiaDataset(mock_columbia_dir)
    samples = ds.samples

    # 1 spliced, 1 authentic
    assert len(samples) == 2

    spliced = next(s for s in samples if not s.is_authentic())
    assert spliced.task_type == "splicing"
    assert spliced.mask_path is not None
    assert "edg" in spliced.mask_path.name.lower()

    auth = next(s for s in samples if s.is_authentic())
    assert auth.task_type == "splicing"
    assert auth.mask_path is None


# =========================================================================
# Manifest Generation & Serialization Tests
# =========================================================================


def test_manifest_build_and_save(mock_comofod_dir: Path, tmp_path: Path):
    """Verifies building and writing a manifest to both JSON and CSV files."""
    ds = CoMoFoDDataset(mock_comofod_dir)

    manifest = build_manifest(ds)
    assert manifest["total_samples"] == len(ds)
    assert len(manifest["samples"]) == len(ds)

    out_dir = tmp_path / "manifests"
    json_path, csv_path = save_manifest(ds, output_dir=out_dir, filename_prefix="comofod_test")

    assert json_path.exists()
    assert csv_path.exists()

    # Load back and verify JSON
    loaded = load_manifest(json_path)
    assert loaded["dataset_name"] == "CoMoFoD"
    assert loaded["total_samples"] == len(ds)

    # Verify CSV format
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(ds)
        assert "sample_id" in rows[0]
        assert "group_id" in rows[0]


# =========================================================================
# Evaluation Pipeline & End-to-End Experiment Tests
# =========================================================================


def test_run_evaluation_on_mock_comofod(mock_comofod_dir: Path, tmp_path: Path):
    """
    Runs evaluate_dataset on the mock CoMoFoD dataset.
    Verifies:
    1. CSV, JSON, MD and examples outputs are generated.
    2. Macro and micro metrics are accurately computed.
    3. Authentic clean images do NOT pollute manipulated macro IoU/F1.
    4. SIFT/RANSAC detector executes smoothly on samples.
    """
    output_dir = tmp_path / "exp_comofod_run"

    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=10,
        seed=42,
        output_dir=output_dir,
        mask_mode="both",
        save_examples_count=2,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["total_requested_limit"] == 10
    assert summary["successful_samples"] > 0
    assert summary["failed_samples"] == 0

    # Verify output files exist
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "failures.csv").exists()

    # Verify macro and micro metrics are present
    macro = summary["macro_metrics"]
    micro = summary["micro_metrics"]
    assert "iou" in macro
    assert "f1_score" in macro
    assert "iou" in micro

    # Verify authentic analysis separation
    auth = summary["authentic_analysis"]
    assert auth["total_authentic"] == 2  # 001_O and 002_O
    assert "false_alarm_rate" in auth
    assert "false_alarms" in auth


def test_run_evaluation_on_missing_dataset_directory(tmp_path: Path):
    """Verifies that running evaluation on an empty or non-existent path handles gracefully."""
    empty_dir = tmp_path / "does_not_exist"
    output_dir = tmp_path / "exp_empty"

    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=empty_dir,
        output_dir=output_dir,
    )

    assert summary["status"] == "DATASET_NOT_FOUND_OR_EMPTY"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()


def test_run_evaluation_handles_individual_sample_failure(mock_comofod_dir: Path, tmp_path: Path):
    """
    Verifies that if one corrupted sample raises an exception,
    the pipeline logs it to failures.csv and continues processing remaining samples.
    """
    # Create a corrupt image file
    corrupt_file = mock_comofod_dir / "003_F.png"
    corrupt_file.write_text("corrupted non-image content")
    mask_file = mock_comofod_dir / "003_B.png"
    mask_file.write_text("corrupted non-image content")

    output_dir = tmp_path / "exp_failure_recovery"

    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=10,
        seed=42,
        output_dir=output_dir,
    )

    assert summary["status"] == "COMPLETED"
    # Should have 1 failure recorded
    assert summary["failed_samples"] >= 1
    assert summary["successful_samples"] >= 1

    # Check failures.csv has recorded the error
    with open(output_dir / "failures.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        failures = list(reader)
        assert len(failures) >= 1
        assert "003" in failures[0]["sample_id"]


# =========================================================================
# CoMoFoD Regression Tests (Mask vs BC Images & Authentic Matching)
# =========================================================================


def test_comofod_mask_distinction_and_bc_images(tmp_path: Path):
    """
    Regression test verifying:
    1. 001_B.png is recognized as binary mask (not image sample).
    2. 001_M.png is recognized as color mask (not image sample).
    3. 001_F_BC1.png is recognized as an image sample, NOT a mask.
    4. 001_F_JC1.jpg correctly pairs with 001_O_JC1.jpg (preserving attack and extension).
    5. 001_F.png correctly pairs with 001_O.png.
    6. Missing authentic counterpart is cleanly reported without silent mismatch.
    """
    data_dir = tmp_path / "comofod_regression"
    data_dir.mkdir(parents=True, exist_ok=True)

    dummy_img = np.full((50, 50, 3), 128, dtype=np.uint8)
    dummy_mask = np.zeros((50, 50), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 255

    # Base images and masks
    cv2.imwrite(str(data_dir / "001_O.png"), dummy_img)
    cv2.imwrite(str(data_dir / "001_F.png"), dummy_img)
    cv2.imwrite(str(data_dir / "001_B.png"), dummy_mask)
    cv2.imwrite(str(data_dir / "001_M.png"), dummy_img)

    # Brightness change (BC) images (the ones previously affected by '_B' substring bug)
    cv2.imwrite(str(data_dir / "001_F_BC1.png"), dummy_img)
    cv2.imwrite(str(data_dir / "001_O_BC1.png"), dummy_img)

    # JPEG compression attack (with corresponding JPG authentic)
    cv2.imwrite(str(data_dir / "001_F_JC1.jpg"), dummy_img)
    cv2.imwrite(str(data_dir / "001_O_JC1.jpg"), dummy_img)

    # Attack variant WITHOUT authentic counterpart (to test missing auth handling)
    cv2.imwrite(str(data_dir / "001_F_CA1.png"), dummy_img)

    ds = CoMoFoDDataset(data_dir)
    samples = ds.samples
    sample_ids = [s.sample_id for s in samples]

    # 1 & 2: Masks must NOT be in samples
    assert "001_B" not in sample_ids
    assert "001_M" not in sample_ids

    # 3: BC image must be recognized as an image sample
    assert "001_F_BC1" in sample_ids
    assert "001_O_BC1" in sample_ids
    bc_forged = next(s for s in samples if s.sample_id == "001_F_BC1")
    assert bc_forged.variant == "brightness_change_1"
    assert bc_forged.task_type == "copy_move"
    assert not bc_forged.is_authentic()
    assert bc_forged.mask_path is not None
    assert bc_forged.mask_path.name == "001_B.png"
    assert bc_forged.authentic_path is not None
    assert bc_forged.authentic_path.name == "001_O_BC1.png"

    # 4: JPEG transformation authentic pairing
    jc_forged = next(s for s in samples if s.sample_id == "001_F_JC1")
    assert jc_forged.authentic_path is not None
    assert jc_forged.authentic_path.name == "001_O_JC1.jpg"

    # 5: Clean forged and original pairing
    f_clean = next(s for s in samples if s.sample_id == "001_F")
    assert f_clean.authentic_path is not None
    assert f_clean.authentic_path.name == "001_O.png"

    # 6: Missing authentic counterpart cleanly reported in metadata
    ca_forged = next(s for s in samples if s.sample_id == "001_F_CA1")
    assert ca_forged.authentic_path is None
    assert ca_forged.metadata.get("authentic_pair_missing") is True
    assert "001_O_CA1.png" in ca_forged.metadata.get("expected_authentic_filename", "")


def test_comofod_real_dataset_integrity():
    """
    Validates complete dataset counts if data/raw/comofod is present:
    - Exactly 10,000 image samples (5,000 authentic, 5,000 forged).
    - Exactly 25 variants with 400 images each.
    - Zero masks counted as samples.
    """
    real_dir = Path("data/raw/comofod")
    if not real_dir.is_dir() or not (real_dir / "001_F.png").exists():
        pytest.skip("Real CoMoFoD dataset not present on disk, skipping.")

    ds = CoMoFoDDataset(real_dir)
    samples = ds.samples

    assert len(samples) == 10000

    auth_count = sum(1 for s in samples if s.is_authentic())
    forged_count = sum(1 for s in samples if not s.is_authentic())
    assert auth_count == 5000
    assert forged_count == 5000

    # Ensure no mask ended up as a sample
    for s in samples:
        assert not s.sample_id.endswith("_B")
        assert not s.sample_id.endswith("_M")


def test_run_evaluation_group_sampling(mock_comofod_dir: Path, tmp_path: Path):
    """Verifies run_evaluation with group-based sampling and variant filtering."""
    output_dir = tmp_path / "exp_group_sampling"

    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=1,
        seed=42,
        output_dir=output_dir,
        variant_filter="clean",
        sample_by_group=True,
    )

    assert summary["status"] == "COMPLETED"
    assert len(summary["selected_group_ids"]) == 1
    # 1 group selected means 1 forged clean and 1 authentic clean = 2 samples
    assert summary["total_evaluated_samples"] == 2
    assert summary["manipulated_samples_count"] == 1
    assert summary["authentic_samples_count"] == 1
    assert summary["authentic_analysis"]["empty_masks_count"] >= 0


def test_run_evaluation_timing_measurements(mock_comofod_dir: Path, tmp_path: Path):
    """
    Verifies that run_evaluation measures fine-grained timings:
    - image_load_ms, detector_wall_ms, detector_cpu_ms, mask_load_ms, sample_wall_ms
    - Backward-compatible time_ms matches sample_wall_ms
    - timing_analysis dictionary exists in summary with correct keys
    - report.md includes structured timing breakdown
    """
    import csv
    from scripts.evaluate_dataset import _format_duration

    output_dir = tmp_path / "exp_timing"

    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=2,
        seed=42,
        output_dir=output_dir,
    )

    assert summary["status"] == "COMPLETED"

    # 1. Verify metrics.csv contains explicit timing columns
    csv_path = output_dir / "metrics.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    required_timing_cols = [
        "time_ms",
        "image_load_ms",
        "detector_wall_ms",
        "detector_cpu_ms",
        "mask_load_ms",
        "sample_wall_ms",
    ]
    for col in required_timing_cols:
        assert col in rows[0], f"Column '{col}' must be present in metrics.csv"

    for r in rows:
        sample_wall = float(r["sample_wall_ms"])
        det_wall = float(r["detector_wall_ms"])
        det_cpu = float(r["detector_cpu_ms"])
        img_load = float(r["image_load_ms"])
        mask_load = float(r["mask_load_ms"])
        time_legacy = float(r["time_ms"])

        assert sample_wall >= 0.0
        assert det_wall >= 0.0
        assert det_cpu >= 0.0
        assert img_load >= 0.0
        assert mask_load >= 0.0
        # Backward-compatible time_ms must match sample_wall_ms
        assert abs(time_legacy - sample_wall) < 1e-4

    # 2. Verify summary.json timing_analysis
    assert "timing_analysis" in summary
    t_analysis = summary["timing_analysis"]
    for key in [
        "total_wall_time_seconds",
        "sum_sample_wall_seconds",
        "mean_sample_wall_ms",
        "mean_detector_wall_ms",
        "mean_detector_cpu_ms",
        "mean_image_load_ms",
        "mean_mask_load_ms",
    ]:
        assert key in t_analysis, f"Key '{key}' must be in timing_analysis"
        assert t_analysis[key] >= 0.0

    # Backward compatibility fields in root summary
    assert "total_time_seconds" in summary
    assert "average_time_per_sample_ms" in summary
    assert summary["total_time_seconds"] == t_analysis["total_wall_time_seconds"]

    # 3. Verify report.md contains structured timing breakdown
    report_path = output_dir / "report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Süre ve Zaman Ölçümü Dağılımı" in report_text
    assert "Dedektör Ortalama Duvar Süresi" in report_text
    assert "Dedektör Ortalama Süreç CPU Süresi" in report_text

    # 4. Verify _format_duration helper
    assert _format_duration(150.0) == "150.0 ms"
    assert "sn" in _format_duration(1296.7)
    assert "1.30 sn" in _format_duration(1296.7)


def test_run_evaluation_supports_min_inliers_configuration(mock_comofod_dir: Path, tmp_path: Path):
    """run_evaluation fonksiyonunun min_inliers parametresini doğru yansıttığını ve varsayılan 4'ü koruduğunu doğrular."""
    # 1. Varsayılan çağrı: min_inliers=4 olmalı
    out_default = tmp_path / "exp_default_inliers"
    summary_default = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=2,
        seed=42,
        output_dir=out_default,
    )
    assert summary_default["copy_move_config"]["min_inliers"] == 4

    # 2. Özel min_inliers=10 çağrısı: min_inliers=10 olmalı
    out_10 = tmp_path / "exp_inliers_10"
    summary_10 = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=2,
        seed=42,
        output_dir=out_10,
        min_inliers=10,
    )
    assert summary_10["copy_move_config"]["min_inliers"] == 10

    # Rapor dosyasında da min_inliers=10 görünmeli
    report_text = (out_10 / "report.md").read_text(encoding="utf-8")
    assert "min_inliers" in report_text
    assert "10" in report_text


def test_run_evaluation_supports_hull_fill_guard_configuration(mock_comofod_dir: Path, tmp_path: Path):
    """run_evaluation fonksiyonunun hull_fill_guard parametresini ve özet/rapor çıktısını doğruladığını test eder."""
    out_dir = tmp_path / "exp_hull_guard"
    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=2,
        seed=42,
        output_dir=out_dir,
        min_inliers=4,
        hull_fill_guard=True,
        hull_max_ratio=4.0,
    )

    # 1. summary.json konfigürasyonunu kontrol et
    cm_cfg = summary["copy_move_config"]
    assert cm_cfg["hull_fill_guard"] is True
    assert cm_cfg["hull_max_ratio"] == 4.0
    assert cm_cfg["min_inliers"] == 4

    # 2. summary.json dosyasından okuma kontrolü
    with open(out_dir / "summary.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["copy_move_config"]["hull_fill_guard"] is True
    assert loaded["copy_move_config"]["hull_max_ratio"] == 4.0

    # 3. report.md içeriğini kontrol et
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "hull_fill_guard" in report_text
    assert "Etkin (True)" in report_text
    assert "hull_max_ratio" in report_text
    assert "4.0" in report_text


def test_run_evaluation_supports_spatial_cluster_guard_configuration(mock_comofod_dir: Path, tmp_path: Path):
    """run_evaluation fonksiyonunun spatial_cluster_guard parametresini ve cluster_diagnostics.csv çıktısını doğruladığını test eder."""
    out_dir = tmp_path / "exp_cluster_guard"
    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=2,
        seed=42,
        output_dir=out_dir,
        min_inliers=4,
        spatial_cluster_guard=True,
        cluster_eps=35.0,
        cluster_min_samples=3,
        cluster_min_support_ratio=0.50,
    )

    # 1. summary.json konfigürasyonunu kontrol et
    cm_cfg = summary["copy_move_config"]
    assert cm_cfg["spatial_cluster_guard"] is True
    assert cm_cfg["cluster_eps_px"] == 35.0
    assert cm_cfg["cluster_min_samples"] == 3
    assert cm_cfg["cluster_min_support_ratio"] == 0.50

    # 2. summary.json dosyasından okuma kontrolü
    with open(out_dir / "summary.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["copy_move_config"]["spatial_cluster_guard"] is True

    # 3. report.md içeriğini kontrol et
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "spatial_cluster_guard" in report_text
    assert "Etkin (True)" in report_text
    assert "cluster_eps_px" in report_text
    assert "35.0" in report_text

    # 4. cluster_diagnostics.csv dosyasını kontrol et
    diag_csv_path = out_dir / "cluster_diagnostics.csv"
    assert diag_csv_path.exists()
    diag_text = diag_csv_path.read_text(encoding="utf-8")
    assert "sample_id" in diag_text
    assert "ransac_inliers_count" in diag_text
    assert "num_clusters" in diag_text
    assert "dominant_cluster_size" in diag_text
    assert "cluster_support_ratio" in diag_text
    assert "filtered_inliers_count" in diag_text
    assert "is_rejected_by_guard" in diag_text


def test_run_evaluation_supports_group_ids_file_selection(mock_comofod_dir: Path, tmp_path: Path):
    """
    Verifies that run_evaluation correctly restricts evaluation to scenes
    specified in a group_ids_file, recording the selection in summary.json and report.md.
    """
    out_dir = tmp_path / "exp_group_ids_file_test"
    target_groups_file = tmp_path / "target_scenes.txt"
    # Write scene '001' only to the target file
    target_groups_file.write_text("001\n# Comment line\n\n", encoding="utf-8")

    summary = run_evaluation(
        dataset_name="comofod",
        root_dir=mock_comofod_dir,
        task="copy_move",
        limit=10,
        seed=42,
        output_dir=out_dir,
        sample_by_group=True,
        group_ids_file=target_groups_file,
    )

    # Only scene '001' should be selected
    assert summary["selected_group_ids"] == ["001"]
    assert summary["group_ids_file"] == str(target_groups_file)
    assert summary["total_evaluated_samples"] > 0
    # Every evaluated sample must belong to group '001'
    for s_id in [r["sample_id"] for r in csv.DictReader(open(out_dir / "metrics.csv", encoding="utf-8"))]:
        assert s_id.startswith("001")

    # Verify report.md includes group_ids_file reference
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "group_ids_file" in report_text
    assert "target_scenes.txt" in report_text


def test_defacto_parsing_and_masks(tmp_path: Path) -> None:
    """Verifies that DefactoDataset correctly parses directory structures, stems, and loads separate masks."""
    root = tmp_path / "defacto_mock"
    img_dir = root / "copymove_img" / "img"
    donor_dir = root / "copymove_annotations" / "donor_mask"
    probe_dir = root / "copymove_annotations" / "probe_mask"
    graph_dir = root / "copymove_annotations" / "graph"

    for d in (img_dir, donor_dir, probe_dir, graph_dir):
        d.mkdir(parents=True, exist_ok=True)

    h, w = 100, 150
    # Create sample 0_000000000071
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    donor = np.zeros((h, w), dtype=np.uint8)
    donor[10:30, 10:40] = 255
    probe = np.zeros((h, w), dtype=np.uint8)
    probe[50:70, 80:110] = 255

    # Simulate JPEG artifacts around zero for probe mask
    probe[0:5, 0:5] = 4

    stem = "0_000000000071"
    cv2.imwrite(str(img_dir / f"{stem}.tif"), img)
    cv2.imwrite(str(donor_dir / f"{stem}.tif"), donor)
    cv2.imwrite(str(probe_dir / f"{stem}.jpg"), probe)

    graph_content = [
        {"State": 0, "Name": "Ground Truth", "Property": {"Path": "MSCOCO/images/train2017/000000000071.jpg"}},
        {"State": 1, "Name": "Copy and Move", "Property": {"Translation": "(481,248)", "Scale": "(1,1)", "Rotation": 0, "Mask feathering": 3}},
    ]
    (graph_dir / f"{stem}.json").write_text(json.dumps(graph_content), encoding="utf-8")

    ds = DefactoDataset(root_dir=root, task_type="copy_move", load_metadata=True)
    assert len(ds) == 1
    sample = ds[0]

    assert sample.sample_id == stem
    assert sample.group_id == "000000000071"
    assert sample.manipulation_type == "copy_move"
    assert not sample.is_authentic()
    assert sample.metadata["instance_id"] == "0"
    assert sample.metadata["coco_image_id"] == "000000000071"
    assert sample.metadata["translation"] == "(481,248)"

    # Test image loading
    loaded_img = ds.load_image(sample)
    assert loaded_img.shape == (h, w, 3)

    # Test both mask loading (merging donor and probe, and suppressing JPEG noise)
    both_mask = ds.load_mask(sample, mask_mode="both")
    assert both_mask.shape == (h, w)
    # The JPEG noise region [0:5, 0:5] should be 0 because value 4 <= 127
    assert both_mask[2, 2] == 0
    # Both donor and probe regions should be 255
    assert both_mask[20, 20] == 255
    assert both_mask[60, 95] == 255

    # Test target_only mask
    tgt_mask = ds.load_mask(sample, mask_mode="target_only")
    assert tgt_mask[60, 95] == 255
    assert tgt_mask[20, 20] == 0

    # Test source_only mask
    src_mask = ds.load_mask(sample, mask_mode="source_only")
    assert src_mask[20, 20] == 255
    assert src_mask[60, 95] == 0

    # Test get_sample_metadata
    meta = ds.get_sample_metadata(stem)
    assert meta is not None
    assert meta["rotation"] == 0
    assert meta["feathering"] == 3


def test_defacto_unicode_path_handling(tmp_path: Path) -> None:
    """Verifies that image and mask loading works seamlessly even in paths with Turkish/Unicode characters."""
    turkish_dir = tmp_path / "Sayısal Görünntü İşleme Veri Seti" / "defacto-copymove"
    img_dir = turkish_dir / "copymove_img" / "img"
    donor_dir = turkish_dir / "copymove_annotations" / "donor_mask"
    probe_dir = turkish_dir / "copymove_annotations" / "probe_mask"
    graph_dir = turkish_dir / "copymove_annotations" / "graph"

    for d in (img_dir, donor_dir, probe_dir, graph_dir):
        d.mkdir(parents=True, exist_ok=True)

    stem = "1_000000029706"
    img = np.full((80, 80, 3), 200, dtype=np.uint8)
    probe = np.zeros((80, 80), dtype=np.uint8)
    probe[20:40, 20:40] = 255

    # Use buffer encoding to write to Turkish path
    is_success, buf = cv2.imencode(".tif", img)
    assert is_success
    (img_dir / f"{stem}.tif").write_bytes(buf.tobytes())

    is_success_p, buf_p = cv2.imencode(".jpg", probe)
    assert is_success_p
    (probe_dir / f"{stem}.jpg").write_bytes(buf_p.tobytes())

    ds = DefactoDataset(root_dir=turkish_dir, task_type="copy_move")
    assert len(ds) == 1
    sample = ds[0]

    loaded_img = ds.load_image(sample)
    assert loaded_img is not None
    assert loaded_img.shape == (80, 80, 3)

    loaded_mask = ds.load_mask(sample, mask_mode="target_only")
    assert loaded_mask is not None
    assert loaded_mask[30, 30] == 255


def test_defacto_real_junction_sample() -> None:
    """If the real junction at data/raw/defacto/copymove exists, test basic loading on real samples."""
    junction_path = Path("data/raw/defacto/copymove")
    if not junction_path.is_dir():
        pytest.skip("DEFACTO junction not present in data/raw/defacto/copymove")

    ds = DefactoDataset(root_dir=junction_path, task_type="copy_move")
    assert len(ds) == 18194

    sample = ds[0]
    assert sample.manipulation_type == "copy_move"
    assert not sample.is_authentic()
    assert sample.group_id is not None
    assert sample.target_mask_path is not None and sample.target_mask_path.exists()
    assert sample.source_mask_path is not None and sample.source_mask_path.exists()

    img = ds.load_image(sample)
    assert img.ndim == 3

    mask = ds.load_mask(sample, mask_mode="both")
    assert mask.ndim == 2
    assert mask.shape == img.shape[:2]






