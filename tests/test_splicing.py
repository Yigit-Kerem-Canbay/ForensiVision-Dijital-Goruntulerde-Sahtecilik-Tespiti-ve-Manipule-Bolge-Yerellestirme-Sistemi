"""
ForensiVision Image Splicing Modülü Birim ve Entegrasyon Testleri.

Bu test paketi; Error Level Analysis (ELA), Yerel Gürültü Tutarsızlığı (Noise Inconsistency),
füzyon stratejileri, girdi doğrulama kuralları ve değerlendirme motoru entegrasyonunu test eder.
"""

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from forensivision.evaluation import calculate_metrics
from forensivision.splicing import (
    ELAResult,
    NoiseAnalysisResult,
    SplicingConfig,
    SplicingResult,
    analyze_ela,
    analyze_noise_inconsistency,
    detect_splicing,
)


def _create_synthetic_splicing_image():
    """Testler için kontrollü bir sentetik splicing görüntüsü ve ground-truth maskesi üretir."""
    h, w = 240, 240
    np.random.seed(42)

    bg = np.ones((h, w, 3), dtype=np.float32) * 200
    bg += np.random.normal(0, 1.5, bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    box = (60, 160, 60, 160)  # 100x100 yama
    y1, y2, x1, x2 = box

    patch = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32) * 80
    patch += np.random.normal(0, 15.0, patch.shape)
    patch = np.clip(patch, 0, 255).astype(np.uint8)

    spliced = bg.copy()
    spliced[y1:y2, x1:x2] = patch

    gt = np.zeros((h, w), dtype=np.uint8)
    gt[y1:y2, x1:x2] = 255

    return spliced, gt


def test_ela_on_valid_jpeg_file():
    """ELA'nın gerçek bir JPEG dosyası üzerinde başarıyla çalıştığını ve geçerli çıktılar ürettiğini test eder."""
    img, _ = _create_synthetic_splicing_image()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cv2.imwrite(str(tmp_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        result = analyze_ela(tmp_path, quality=90)

        assert isinstance(result, ELAResult)
        assert result.is_applicable is True
        assert result.raw_difference.shape == img.shape[:2]
        assert result.normalized_map.shape == img.shape[:2]
        assert result.candidate_mask.shape == img.shape[:2]
        assert result.candidate_mask.dtype == np.uint8
        assert set(np.unique(result.candidate_mask)).issubset({0, 255})
        assert result.mean_difference >= 0.0
        assert result.std_difference >= 0.0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_ela_on_non_jpeg_file_handles_gracefully():
    """JPEG olmayan (örn. PNG) dosyalarda ELA'nın is_applicable=False döndürdüğünü test eder."""
    img, _ = _create_synthetic_splicing_image()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cv2.imwrite(str(tmp_path), img)

        result = analyze_ela(tmp_path)

        assert result.is_applicable is False
        assert "JPEG formatında değildir" in result.status_message
        assert np.count_nonzero(result.candidate_mask) == 0
        assert result.mean_difference == 0.0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_noise_analysis_output_properties():
    """Gürültü analiz haritalarının doğru boyutta, sonlu ve NaN/Inf içermediğini test eder."""
    img, _ = _create_synthetic_splicing_image()

    result = analyze_noise_inconsistency(img, window_size=15, threshold_std_factor=2.0)

    assert isinstance(result, NoiseAnalysisResult)
    assert result.residual_map.shape == img.shape[:2]
    assert result.local_variance_map.shape == img.shape[:2]
    assert result.normalized_map.shape == img.shape[:2]
    assert result.candidate_mask.shape == img.shape[:2]
    assert result.candidate_mask.dtype == np.uint8
    assert set(np.unique(result.candidate_mask)).issubset({0, 255})

    # Sayısal kararlılık kontrolleri
    assert not np.isnan(result.local_variance_map).any(), "Varyans haritasında NaN olamaz."
    assert not np.isinf(result.local_variance_map).any(), "Varyans haritasında sonsuz olamaz."
    assert not np.isnan(result.residual_map).any(), "Artık haritasında NaN olamaz."
    assert result.global_noise_mean >= 0.0
    assert result.global_noise_std >= 0.0


def test_noise_analysis_detects_noisy_spliced_region():
    """Gürültülü eklenmiş bölgenin gürültü aday maskesinde yakalandığını ve değerlendirilebildiğini test eder."""
    img, gt = _create_synthetic_splicing_image()

    result = analyze_noise_inconsistency(img, window_size=15, threshold_std_factor=1.5)
    metrics = calculate_metrics(result.candidate_mask, gt)

    # İstatistiksel gürültü analizi tespit beklentileri (kayar pencere sınır etkisi nedeniyle iç çekirdek yakalanır)
    assert metrics.tp > 0, "En azından manipüle bölgede doğru pozitif pikseller yakalanmalıdır."
    assert metrics.precision > 0.45, f"Gürültü hassasiyeti yeterli olmalıdır; alınan: {metrics.precision}"
    assert metrics.recall > 0.25, f"Gürültü yakalama oranı makul olmalıdır; alınan: {metrics.recall}"
    assert metrics.iou > 0.18, f"IoU kabul edilebilir olmalıdır; alınan: {metrics.iou}"




def test_splicing_fusion_modes():
    """Farklı füzyon stratejilerinin ('union', 'intersection', 'ela_only', 'noise_only') doğru çalıştığını test eder."""
    img, _ = _create_synthetic_splicing_image()

    for mode in ("union", "intersection", "ela_only", "noise_only"):
        cfg = SplicingConfig(fusion_mode=mode)
        res = detect_splicing(img, config=cfg)

        assert isinstance(res, SplicingResult)
        assert res.fusion_mode == mode
        assert res.mask.shape == img.shape[:2]
        assert res.mask.dtype == np.uint8
        assert set(np.unique(res.mask)).issubset({0, 255})


def test_splicing_on_different_image_dimensions():
    """Farklı boyut ve kanallardaki görüntülerin (2D gri ton, asimetrik boyutlar) desteklendiğini test eder."""
    # 2D Gri seviye
    gray = np.ones((180, 260), dtype=np.uint8) * 128
    res_gray = detect_splicing(gray)
    assert res_gray.mask.shape == (180, 260)

    # 3D Renkli asimetrik
    color = np.ones((210, 150, 3), dtype=np.uint8) * 150
    res_color = detect_splicing(color)
    assert res_color.mask.shape == (210, 150)


def test_invalid_inputs_raise_errors():
    """Geçersiz girdilerde (None, boş dizi, dosya bulunamadı, geçersiz kanal) uygun istisnaları test eder."""
    # None girdi
    with pytest.raises(ValueError, match="None olamaz"):
        analyze_ela(None)

    with pytest.raises(ValueError, match="None olamaz"):
        analyze_noise_inconsistency(None)

    with pytest.raises(ValueError, match="None olamaz"):
        detect_splicing(None)

    # Var olmayan dosya
    with pytest.raises(FileNotFoundError):
        analyze_ela("non_existent_file_xyz_123.jpg")

    # Boş dizi
    with pytest.raises(ValueError, match="boş bir dizi olamaz"):
        analyze_noise_inconsistency(np.array([], dtype=np.uint8))

    # Geçersiz kanal sayısı
    with pytest.raises(ValueError, match="Desteklenmeyen kanal sayısı"):
        analyze_noise_inconsistency(np.zeros((50, 50, 5), dtype=np.uint8))

    # Geçersiz fusion_mode
    with pytest.raises(ValueError, match="Geçersiz fusion_mode"):
        detect_splicing(np.zeros((50, 50, 3), dtype=np.uint8), config=SplicingConfig(fusion_mode="invalid_mode"))


def test_serialization_methods():
    """ELAResult, NoiseAnalysisResult ve SplicingResult nesnelerinin to_dict serileştirilmesini test eder."""
    img, _ = _create_synthetic_splicing_image()

    res = detect_splicing(img)
    d = res.to_dict()

    assert isinstance(d, dict)
    assert "is_suspicious" in d
    assert "fusion_mode" in d
    assert "mask_shape" in d
    assert "ela_summary" in d
    assert "noise_summary" in d
    assert "metadata" in d
