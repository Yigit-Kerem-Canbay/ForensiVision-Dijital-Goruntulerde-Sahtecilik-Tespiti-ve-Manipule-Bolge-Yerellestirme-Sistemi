"""
ForensiVision Copy-Move Tespit Modülü Birim ve Entegrasyon Testleri.

Bu test paketi; SIFT öznitelik çıkarımı, öz-eşleme filtrelemesi,
RANSAC geometrik doğrulaması, ikili maske üretimi, hata yönetimi ve
değerlendirme motoruyla (Precision, Recall, F1, IoU) entegrasyonu test eder.
"""

import cv2
import numpy as np
import pytest

from forensivision.copy_move import CopyMoveConfig, CopyMoveResult, detect_copy_move
from forensivision.evaluation import calculate_metrics


def _create_synthetic_test_case():
    """Deterministik ve zengin geometrik öznitelikli sentetik Copy-Move görüntüsü ve GT maskesi üretir."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 230

    # Arka plan dokusu
    for i in range(15, 290, 30):
        cv2.line(img, (i, 0), (i, 300), (210, 210, 210), 1)
        cv2.line(img, (0, i), (300, i), (210, 210, 210), 1)

    # Kaynak bölge: (40, 120, 40, 120)
    cv2.rectangle(img, (45, 45), (115, 115), (40, 80, 200), -1)
    cv2.circle(img, (80, 80), 25, (255, 255, 255), -1)
    cv2.circle(img, (80, 80), 12, (20, 20, 20), -1)
    cv2.putText(img, "FV", (55, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hedef bölgeye kopyala: (160, 240, 160, 240)
    patch = img[40:120, 40:120].copy()
    img[160:240, 160:240] = patch

    # Ground-truth: Her iki bölge de şüpheli (kaynak + hedef)
    gt = np.zeros((300, 300), dtype=np.uint8)
    gt[40:120, 40:120] = 255
    gt[160:240, 160:240] = 255

    return img, gt


def test_detect_synthetic_copy_move_success():
    """Sentetik Copy-Move görüntüsünde tespit akışının doğru çalıştığını ve beklenen çıktıları ürettiğini test eder."""
    img, _ = _create_synthetic_test_case()

    result = detect_copy_move(img)

    assert isinstance(result, CopyMoveResult)
    assert result.is_tampered is True
    assert result.num_keypoints > 20
    assert result.num_candidate_matches >= 4
    assert result.num_inliers >= 4
    assert result.transform_matrix is not None
    assert result.transform_matrix.shape == (2, 3)

    # Maske özellikleri
    assert result.mask.shape == (300, 300)
    assert result.mask.dtype == np.uint8
    unique_vals = set(np.unique(result.mask))
    assert unique_vals.issubset({0, 255}), f"Maske sadece 0 ve 255 içermeli, alınan: {unique_vals}"
    assert np.count_nonzero(result.mask) > 0, "Pozitif manipüle pikseller bulunmalıdır."

    # Koordinat eşleşmeleri
    assert len(result.matched_points_src) == result.num_inliers
    assert len(result.matched_points_dst) == result.num_inliers


def test_evaluation_integration_on_copy_move():
    """Tespit edilen maskenin değerlendirme motoru (calculate_metrics) ile ölçülebilirliğini test eder."""
    img, gt = _create_synthetic_test_case()

    result = detect_copy_move(img)
    metrics = calculate_metrics(result.mask, gt)

    # Akademik yerelleştirme beklentileri
    assert metrics.precision > 0.85, f"Precision yüksek olmalıdır; alınan: {metrics.precision}"
    assert metrics.recall > 0.50, f"Recall makul bir kapsama sahip olmalıdır; alınan: {metrics.recall}"
    assert metrics.f1_score > 0.60, f"F1-score yüksek olmalıdır; alınan: {metrics.f1_score}"
    assert metrics.iou > 0.50, f"IoU kabul edilebilir olmalıdır; alınan: {metrics.iou}"
    assert metrics.tp > 0
    assert metrics.fp < metrics.tp


def test_authentic_image_no_tampering():
    """Hiçbir kopyalama içermeyen orijinal dokulu bir görüntüde yanlış alarm üretilmediğini test eder."""
    # Birbirinden farklı şekiller içeren orijinal görüntü
    img = np.ones((250, 250, 3), dtype=np.uint8) * 240
    cv2.circle(img, (50, 50), 30, (10, 10, 10), -1)
    cv2.rectangle(img, (140, 140), (220, 220), (100, 20, 20), -1)
    cv2.line(img, (0, 250), (250, 0), (0, 100, 0), 3)

    result = detect_copy_move(img)

    assert result.is_tampered is False
    assert result.num_inliers < result.config.min_inliers
    assert np.count_nonzero(result.mask) == 0

    # Boş maske konvansiyonu ile boş GT karşılaştırması -> 1.0 olmalı
    empty_gt = np.zeros((250, 250), dtype=np.uint8)
    metrics = calculate_metrics(result.mask, empty_gt)
    assert metrics.iou == 1.0
    assert metrics.f1_score == 1.0


def test_low_texture_uniform_image():
    """Tamamen tek renk (dokuzuz) görüntülerde SIFT'in çökmeden kontrollü boş sonuç döndürdüğünü test eder."""
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    result = detect_copy_move(blank)

    assert result.is_tampered is False
    assert result.num_keypoints == 0
    assert result.num_candidate_matches == 0
    assert result.num_inliers == 0
    assert np.count_nonzero(result.mask) == 0


def test_grayscale_2d_image_support():
    """2D gri seviye görüntülerin sorunsuz işlendiğini test eder."""
    img_color, _ = _create_synthetic_test_case()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    result = detect_copy_move(img_gray)

    assert result.is_tampered is True
    assert result.mask.shape == (300, 300)
    assert result.num_inliers >= 4


def test_invalid_inputs_raise_appropriate_exceptions():
    """Geçersiz girdilerde (None, boş dizi, geçersiz boyut/kanal) uygun istisnaların fırlatıldığını test eder."""
    # None girdi
    with pytest.raises(ValueError, match="None olamaz"):
        detect_copy_move(None)

    # Boş dizi
    with pytest.raises(ValueError, match="boş bir dizi olamaz"):
        detect_copy_move(np.array([], dtype=np.uint8))

    # 1D dizi
    with pytest.raises(ValueError, match="Desteklenmeyen görüntü boyutu"):
        detect_copy_move(np.zeros((100,), dtype=np.uint8))

    # 4D dizi
    with pytest.raises(ValueError, match="Desteklenmeyen görüntü boyutu"):
        detect_copy_move(np.zeros((10, 10, 3, 2), dtype=np.uint8))

    # 3D fakat 2 kanallı dizi
    with pytest.raises(ValueError, match="Desteklenmeyen kanal sayısı"):
        detect_copy_move(np.zeros((50, 50, 2), dtype=np.uint8))

    # Sayısal olmayan girdi (metin)
    with pytest.raises(TypeError, match="NumPy dizisi olmalıdır"):
        detect_copy_move("invalid_image_path")


def test_custom_config_override():
    """Özelleştirilmiş konfigürasyon parametrelerinin dikkate alındığını test eder."""
    img, _ = _create_synthetic_test_case()

    custom_config = CopyMoveConfig(
        min_spatial_distance=50.0,
        ratio_threshold=0.70,
        min_inliers=10,
    )
    result = detect_copy_move(img, config=custom_config)

    assert result.config.min_spatial_distance == 50.0
    assert result.config.ratio_threshold == 0.70
    assert result.config.min_inliers == 10


def test_result_to_dict_serialization():
    """CopyMoveResult nesnesinin to_dict metodunun serileştirilebilir sözlük döndürdüğünü test eder."""
    img, _ = _create_synthetic_test_case()

    result = detect_copy_move(img)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert d["is_tampered"] is True
    assert "mask_shape" in d
    assert "num_inliers" in d
    assert d["mask_positive_pixels"] > 0
    assert "config" in d


def test_visualization_helpers():
    """Görselleştirme araçlarının (overlay ve eşleşme çizgileri) çalıştığını test eder."""
    from forensivision.visualization import create_mask_overlay, draw_copy_move_matches

    img, _ = _create_synthetic_test_case()
    result = detect_copy_move(img)

    overlay = create_mask_overlay(img, result.mask)
    assert overlay.shape == img.shape
    assert overlay.dtype == np.uint8

    matched_img = draw_copy_move_matches(
        img, result.matched_points_src, result.matched_points_dst
    )
    assert matched_img.shape == img.shape
    assert matched_img.dtype == np.uint8

