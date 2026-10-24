"""
Unit tests for the ForensiVision UI helpers and synthetic demo module.

Covers:
- Image decoding and color space conversions (BGR for algorithms, RGB for Streamlit)
- Corrupted, empty, or invalid file inputs
- Dimension limit safety
- Ground truth mask loading and strict dimension matching (mismatch raises ValueError)
- Honest metrics behavior (None when ground truth is absent)
- Copy-Move and Splicing analysis runners
- Export JSON payload generation
- PNG byte buffer encoding
- Synthetic demo generators
"""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest

from forensivision.ui.helpers import (
    compute_optional_metrics,
    generate_export_payload,
    image_to_png_bytes,
    load_uploaded_image,
    mask_to_png_bytes,
    run_copy_move_analysis,
    run_splicing_analysis,
    validate_and_load_ground_truth,
)
from forensivision.ui.synthetic import (
    generate_synthetic_copy_move_sample,
    generate_synthetic_splicing_sample,
    get_synthetic_demo_sample,
)


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Creates raw PNG bytes of a small test image."""
    img = np.full((60, 80, 3), 150, dtype=np.uint8)
    img[15:35, 20:40] = [200, 50, 50]
    success, buffer = cv2.imencode(".png", img)
    assert success
    return buffer.tobytes()


@pytest.fixture
def sample_mask_bytes() -> bytes:
    """Creates raw PNG bytes of a small test mask."""
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[15:35, 20:40] = 255
    success, buffer = cv2.imencode(".png", mask)
    assert success
    return buffer.tobytes()


# =========================================================================
# Image Loading & Validation Tests
# =========================================================================


def test_load_uploaded_image_valid(sample_png_bytes: bytes):
    """Verifies that valid image bytes produce correct BGR, RGB arrays and format."""
    img_bgr, img_rgb, fmt = load_uploaded_image(sample_png_bytes, filename="test.png")

    assert img_bgr.shape == (60, 80, 3)
    assert img_rgb.shape == (60, 80, 3)
    assert fmt == "PNG"
    assert img_bgr.dtype == np.uint8
    assert img_rgb.dtype == np.uint8

    # Verify BGR vs RGB channel ordering
    # Pixel [15, 20] was set to BGR [200, 50, 50]
    assert np.array_equal(img_bgr[15, 20], [200, 50, 50])
    assert np.array_equal(img_rgb[15, 20], [50, 50, 200])


def test_load_uploaded_image_from_bytesio(sample_png_bytes: bytes):
    """Verifies that BytesIO buffers are supported cleanly."""
    bio = io.BytesIO(sample_png_bytes)
    img_bgr, img_rgb, fmt = load_uploaded_image(bio)
    assert img_bgr.shape == (60, 80, 3)
    assert fmt == "PNG"


def test_load_uploaded_image_invalid_and_empty():
    """Verifies that invalid or empty inputs raise clear ValueErrors."""
    with pytest.raises(ValueError, match="boş veya None"):
        load_uploaded_image(None)

    with pytest.raises(ValueError, match="0 bayt"):
        load_uploaded_image(b"")

    with pytest.raises(ValueError, match="çözümlenemedi"):
        load_uploaded_image(b"corrupted non image bytes header")


# =========================================================================
# Ground Truth Loading & Dimension Validation Tests
# =========================================================================


def test_validate_and_load_ground_truth_valid(sample_mask_bytes: bytes):
    """Verifies that matching mask dimensions return binary uint8 mask [0, 255]."""
    mask = validate_and_load_ground_truth(sample_mask_bytes, expected_shape=(60, 80))
    assert mask is not None
    assert mask.shape == (60, 80)
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, [0, 255]))
    assert np.sum(mask == 255) == (20 * 20)


def test_validate_and_load_ground_truth_none():
    """Verifies that passing None returns None without errors."""
    assert validate_and_load_ground_truth(None, expected_shape=(60, 80)) is None
    assert validate_and_load_ground_truth(b"", expected_shape=(60, 80)) is None


def test_validate_and_load_ground_truth_dimension_mismatch(sample_mask_bytes: bytes):
    """Verifies that dimension mismatch raises ValueError and prevents silent resizing."""
    with pytest.raises(ValueError, match="uyuşmuyor"):
        validate_and_load_ground_truth(sample_mask_bytes, expected_shape=(100, 100))


# =========================================================================
# Honest Metrics Calculation Tests
# =========================================================================


def test_compute_optional_metrics_absent_ground_truth():
    """Honest Metrics Rule: if no ground truth mask is uploaded, returns None."""
    pred_mask = np.zeros((50, 50), dtype=np.uint8)
    res = compute_optional_metrics(pred_mask, gt_mask=None)
    assert res is None


def test_compute_optional_metrics_present_ground_truth():
    """Calculates metrics when ground truth is provided."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:30, 10:30] = 255

    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[10:30, 10:30] = 255

    res = compute_optional_metrics(pred, gt_mask=gt)
    assert res is not None
    assert res["ground_truth_provided"] is True
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["iou"] == 1.0


# =========================================================================
# Analysis Runners & Export Payload Tests
# =========================================================================


def test_run_copy_move_analysis_execution():
    """Verifies running copy-move detector via UI helper."""
    img, _, _, _ = generate_synthetic_copy_move_sample()
    result, elapsed = run_copy_move_analysis(img)

    assert elapsed > 0.0
    assert result.mask.shape == img.shape[:2]
    assert result.is_tampered is True
    assert result.num_inliers >= 4


def test_run_splicing_analysis_execution():
    """Verifies running splicing detector via UI helper."""
    img, _, _, _ = generate_synthetic_splicing_sample()
    result, elapsed = run_splicing_analysis(img)

    assert elapsed > 0.0
    assert result.mask.shape == img.shape[:2]
    assert result.noise_result is not None


def test_generate_export_payload_json():
    """Verifies structure and serialization of export JSON payload."""
    img, _, gt, _ = generate_synthetic_copy_move_sample()
    cm_res, cm_time = run_copy_move_analysis(img)
    cm_metrics = compute_optional_metrics(cm_res.mask, gt)

    payload_str = generate_export_payload(
        analysis_type="copy_move",
        image_shape=img.shape,
        detected_format="PNG",
        copy_move_result=cm_res,
        copy_move_time=cm_time,
        copy_move_metrics=cm_metrics,
        has_ground_truth=True,
    )

    data = json.loads(payload_str)
    assert data["system"] == "ForensiVision"
    assert data["analysis_type"] == "copy_move"
    assert data["input_metadata"]["has_ground_truth_mask"] is True
    assert "copy_move_analysis" in data
    assert data["copy_move_analysis"]["is_tampered"] is True
    assert "metrics" in data["copy_move_analysis"]


def test_buffer_encoding_helpers():
    """Verifies encoding mask and image arrays into PNG byte buffers."""
    mask = np.full((40, 40), 255, dtype=np.uint8)
    mask_bytes = mask_to_png_bytes(mask)
    assert isinstance(mask_bytes, bytes)
    assert len(mask_bytes) > 0

    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    img_bytes = image_to_png_bytes(img, is_rgb=False)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0


# =========================================================================
# Synthetic Demo Generator Tests
# =========================================================================


def test_synthetic_demo_generators():
    """Verifies both copy-move and splicing synthetic sample generators."""
    cm_bgr, cm_rgb, cm_gt, cm_desc = get_synthetic_demo_sample("copy_move")
    assert cm_bgr.shape == (400, 400, 3)
    assert cm_rgb.shape == (400, 400, 3)
    assert cm_gt.shape == (400, 400)
    assert "Copy-Move" in cm_desc
    assert np.count_nonzero(cm_gt) > 0

    sp_bgr, sp_rgb, sp_gt, sp_desc = get_synthetic_demo_sample("splicing")
    assert sp_bgr.shape == (400, 400, 3)
    assert sp_rgb.shape == (400, 400, 3)
    assert sp_gt.shape == (400, 400)
    assert "Splicing" in sp_desc
    assert np.count_nonzero(sp_gt) > 0
