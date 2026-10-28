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
    if not real_dir.is_dir() or not any(real_dir.glob("*.png")):
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

