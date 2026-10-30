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
from forensivision.copy_move.detector import _apply_spatial_clustering, _build_tampered_mask
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


def test_default_copy_move_config_min_inliers_is_four():
    """Varsayılan CopyMoveConfig nesnesinin ve detect_copy_move çağrısının min_inliers=4 değerini koruduğunu doğrular."""
    config = CopyMoveConfig()
    assert config.min_inliers == 4

    img, _ = _create_synthetic_test_case()
    result = detect_copy_move(img)
    assert result.config.min_inliers == 4


def test_min_inliers_threshold_boundary_rejection_and_retention(monkeypatch):
    """
    Doğrular:
    - 9 inlier ile sonuç, num_inliers < min_inliers mantığı gereği reddedilir.
    - 10 inlier tek başına bu eşik kontrolünden dolayı reddedilmez;
      diğer geometrik kontroller (afin SVD, uzamsal benzersizlik) ayrıca uygulanır.
    """
    img, _ = _create_synthetic_test_case()
    cfg10 = CopyMoveConfig(min_inliers=10)

    # 1. 9 inlier senaryosu: min_inliers=10 iken reddedilmeli
    def mock_estimate_9_inliers(pts_src, pts_dst, method, ransacReprojThreshold):
        mask = np.zeros((len(pts_src), 1), dtype=np.uint8)
        mask[:9] = 1
        M = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32)
        return M, mask

    monkeypatch.setattr(cv2, "estimateAffine2D", mock_estimate_9_inliers)
    res9 = detect_copy_move(img, config=cfg10)
    assert res9.is_tampered is False
    assert res9.num_inliers == 9
    assert np.count_nonzero(res9.mask) == 0

    # 2. 10 inlier senaryosu: min_inliers=10 iken num_inliers < min_inliers kontrolünden geçer
    # A) Geçerli afin ve uzamsal benzersiz noktalarla: tespit başarılı olur
    def mock_estimate_10_inliers_valid(pts_src, pts_dst, method, ransacReprojThreshold):
        selected = []
        for i, pt in enumerate(pts_src):
            if not any(np.linalg.norm(pt - pts_src[j]) < 7.0 for j in selected):
                selected.append(i)
                if len(selected) == 10:
                    break
        mask = np.zeros((len(pts_src), 1), dtype=np.uint8)
        for idx in selected:
            mask[idx] = 1
        M = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32)
        return M, mask

    monkeypatch.setattr(cv2, "estimateAffine2D", mock_estimate_10_inliers_valid)
    res10_valid = detect_copy_move(img, config=cfg10)
    assert res10_valid.num_inliers == 10
    assert res10_valid.is_tampered is True
    assert np.count_nonzero(res10_valid.mask) > 0

    # B) 10 inlier olsa dahi afin matrisi tekil/çökmüşse,
    # diğer geometrik kontrol (_is_valid_affine_transform) devreye girip reddeder
    def mock_estimate_10_inliers_singular(pts_src, pts_dst, method, ransacReprojThreshold):
        selected = []
        for i, pt in enumerate(pts_src):
            if not any(np.linalg.norm(pt - pts_src[j]) < 7.0 for j in selected):
                selected.append(i)
                if len(selected) == 10:
                    break
        mask = np.zeros((len(pts_src), 1), dtype=np.uint8)
        for idx in selected:
            mask[idx] = 1
        M_singular = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]], dtype=np.float32)
        return M_singular, mask

    monkeypatch.setattr(cv2, "estimateAffine2D", mock_estimate_10_inliers_singular)
    res10_invalid = detect_copy_move(img, config=cfg10)
    assert res10_invalid.is_tampered is False
    assert res10_invalid.num_inliers == 0
    assert np.count_nonzero(res10_invalid.mask) == 0


def test_default_copy_move_config_hull_fill_guard_defaults():
    """CopyMoveConfig varsayılanlarının hull_fill_guard=False, hull_max_ratio=4.0 ve min_inliers=4 olduğunu doğrular."""
    config = CopyMoveConfig()
    assert config.hull_fill_guard is False
    assert config.hull_max_ratio == 4.0
    assert config.min_inliers == 4


def test_build_tampered_mask_guard_disabled_fills_sparse_hull():
    """Koruma kapalıyken (hull_fill_guard=False) seyrek noktalarda convex hull'ın doldurulduğunu doğrular."""
    shape = (500, 500)
    sparse_pts = np.array([[50.0, 50.0], [450.0, 50.0], [250.0, 450.0]], dtype=np.float32)
    empty_pts = np.empty((0, 2), dtype=np.float32)

    config_default = CopyMoveConfig(hull_fill_guard=False, point_radius=12)
    mask = _build_tampered_mask(shape, sparse_pts, empty_pts, config_default)

    # Üçgenin ağırlık merkezi (centroid) (250, 183) pikselleri dahil tüm iç alan doldurulmuş olmalıdır
    centroid_x, centroid_y = 250, 183
    assert mask[centroid_y, centroid_x] == 255
    # Alan 80.000 piksel civarındadır (> 50.000)
    assert np.count_nonzero(mask) > 50000


def test_build_tampered_mask_guard_enabled_suppresses_sparse_hull_preserves_circles():
    """Koruma etkinken (hull_fill_guard=True, ratio=4.0) seyrek noktalarda dışbükey zarfın doldurulmadığını,
    dairesel işaretlemelerin ise korunduğunu doğrular."""
    shape = (500, 500)
    sparse_pts = np.array([[50.0, 50.0], [450.0, 50.0], [250.0, 450.0]], dtype=np.float32)
    empty_pts = np.empty((0, 2), dtype=np.float32)

    config_guarded = CopyMoveConfig(hull_fill_guard=True, hull_max_ratio=4.0, point_radius=12)
    mask = _build_tampered_mask(shape, sparse_pts, empty_pts, config_guarded)

    # 1. Üçgenin ağırlık merkezi (centroid) boş kalmalıdır (fillConvexPoly uygulanmamış olmalı)
    centroid_x, centroid_y = 250, 183
    assert mask[centroid_y, centroid_x] == 0

    # 2. Her inlier anahtar noktası etrafındaki dairesel işaretlemeler korunmuş olmalıdır
    assert mask[50, 50] == 255
    assert mask[50, 450] == 255
    assert mask[450, 250] == 255

    # 3. Toplam aktif piksel sayısı yalnızca 3 dairenin alanı mertebesindedir (< 3000 piksel)
    # Bu da devasa sahte pozitif alanın (80.000 piksel) engellendiğini kanıtlar
    total_pixels = np.count_nonzero(mask)
    assert 1000 < total_pixels < 3000


def test_build_tampered_mask_guard_enabled_allows_compact_hull():
    """Yeterince kompakt nokta grubunda oran <= 4.0 kaldığından dışbükey zarfın doldurulmaya devam ettiğini doğrular."""
    shape = (500, 500)
    # Birbirine çok yakın (15-20 piksel mesafede) kompakt 3 nokta
    compact_pts = np.array([[200.0, 200.0], [220.0, 200.0], [210.0, 218.0]], dtype=np.float32)
    empty_pts = np.empty((0, 2), dtype=np.float32)

    config_guarded = CopyMoveConfig(hull_fill_guard=True, hull_max_ratio=4.0, point_radius=12)
    mask = _build_tampered_mask(shape, compact_pts, empty_pts, config_guarded)

    # Kompakt kümede dışbükey zarf alanı küçük olduğundan hull_amplification <= 4.0 kalır ve doldurulur
    centroid_x, centroid_y = 210, 206
    assert mask[centroid_y, centroid_x] == 255
    assert np.count_nonzero(mask) > 0


def test_default_copy_move_config_spatial_cluster_guard_defaults():
    """CopyMoveConfig varsayılanlarında spatial_cluster_guard=False, eps=35.0, min_samples=3, support_ratio=0.50 ve min_inliers=4 olduğunu doğrular."""
    config = CopyMoveConfig()
    assert config.spatial_cluster_guard is False
    assert config.cluster_eps_px == 35.0
    assert config.cluster_min_samples == 3
    assert config.cluster_min_support_ratio == 0.50
    assert config.min_inliers == 4


def test_spatial_cluster_guard_disabled_preserves_previous_behavior():
    """Koruma kapalıyken (spatial_cluster_guard=False) dağınık veya gürültülü eşleşmelerin elenmediğini doğrular."""
    # 2 nokta bir bölgede, 2 nokta çok uzakta (DBSCAN gürültü sayardı)
    pts_src = np.array([[10.0, 10.0], [12.0, 12.0], [300.0, 300.0], [305.0, 305.0]], dtype=np.float32)
    pts_dst = np.array([[50.0, 50.0], [52.0, 52.0], [350.0, 350.0], [355.0, 355.0]], dtype=np.float32)

    config = CopyMoveConfig(spatial_cluster_guard=False)
    filtered_src, filtered_dst, diag, is_rejected = _apply_spatial_clustering(pts_src, pts_dst, config)

    assert is_rejected is False
    assert len(filtered_src) == len(pts_src)
    assert len(filtered_dst) == len(pts_dst)
    assert diag["is_rejected_by_guard"] is False
    assert diag["filtered_inliers_count"] == len(pts_src)


def test_spatial_cluster_guard_single_dominant_cluster_accepted():
    """Tek bir baskın kümeye sahip sıkı eşleşmelerin koruma tarafından onaylandığını doğrular."""
    # 5 adet birbiriyle uyumlu, orta noktaları birbirine çok yakın (Öklid mesafesi < 10 px) eşleşme
    pts_src = np.array([
        [50.0, 50.0], [52.0, 51.0], [54.0, 49.0], [51.0, 53.0], [53.0, 52.0]
    ], dtype=np.float32)
    pts_dst = np.array([
        [150.0, 150.0], [152.0, 151.0], [154.0, 149.0], [151.0, 153.0], [153.0, 152.0]
    ], dtype=np.float32)

    config = CopyMoveConfig(spatial_cluster_guard=True, cluster_eps_px=35.0, cluster_min_samples=3, cluster_min_support_ratio=0.50, min_inliers=4)
    filtered_src, filtered_dst, diag, is_rejected = _apply_spatial_clustering(pts_src, pts_dst, config)

    assert is_rejected is False
    assert len(filtered_src) == 5
    assert diag["num_clusters"] == 1
    assert diag["dominant_cluster_size"] == 5
    assert diag["cluster_support_ratio"] == 1.0
    assert diag["is_rejected_by_guard"] is False


def test_spatial_cluster_guard_multiple_clusters_and_noise_selects_dominant():
    """Birden fazla küme ve gürültü içeren senaryoda yalnızca baskın kümenin doğru filtrelendiğini doğrular."""
    # Küme A (Baskın): 5 nokta, orta noktaları (100, 100) civarında
    src_A = np.array([[50.0, 50.0], [52.0, 50.0], [51.0, 52.0], [53.0, 51.0], [50.0, 53.0]], dtype=np.float32)
    dst_A = np.array([[150.0, 150.0], [152.0, 150.0], [151.0, 152.0], [153.0, 151.0], [150.0, 153.0]], dtype=np.float32)

    # Küme B (Azınlık): 2 nokta, orta noktaları (300, 300) civarında
    src_B = np.array([[250.0, 250.0], [252.0, 252.0]], dtype=np.float32)
    dst_B = np.array([[350.0, 350.0], [352.0, 352.0]], dtype=np.float32)

    # Gürültü: 1 nokta, orta noktası (500, 50)
    src_noise = np.array([[450.0, 20.0]], dtype=np.float32)
    dst_noise = np.array([[550.0, 80.0]], dtype=np.float32)

    pts_src = np.vstack([src_A, src_B, src_noise])
    pts_dst = np.vstack([dst_A, dst_B, dst_noise])
    total_inliers = 8

    config = CopyMoveConfig(spatial_cluster_guard=True, cluster_eps_px=35.0, cluster_min_samples=3, cluster_min_support_ratio=0.50, min_inliers=4)
    filtered_src, filtered_dst, diag, is_rejected = _apply_spatial_clustering(pts_src, pts_dst, config)

    assert is_rejected is False
    assert len(filtered_src) == 5
    assert np.allclose(filtered_src, src_A)
    assert np.allclose(filtered_dst, dst_A)
    assert diag["dominant_cluster_size"] == 5
    assert diag["cluster_support_ratio"] == 5 / 8
    assert diag["is_rejected_by_guard"] is False


def test_spatial_cluster_guard_rejects_when_dominant_under_min_inliers(monkeypatch):
    """Baskın küme min_inliers (4) altında kalırsa tespitin reddedildiğini ve maskenin boş üretildiğini doğrular."""
    # 3 eşleşmeli küme (destek oranı 3/3 = 1.0 > 0.50 ama boyut 3 < min_inliers 4)
    pts_src = np.array([[50.0, 50.0], [52.0, 50.0], [51.0, 52.0]], dtype=np.float32)
    pts_dst = np.array([[150.0, 150.0], [152.0, 150.0], [151.0, 152.0]], dtype=np.float32)

    config = CopyMoveConfig(spatial_cluster_guard=True, cluster_eps_px=35.0, cluster_min_samples=3, cluster_min_support_ratio=0.50, min_inliers=4)
    filtered_src, filtered_dst, diag, is_rejected = _apply_spatial_clustering(pts_src, pts_dst, config)

    assert is_rejected is True
    assert len(filtered_src) == 0
    assert diag["is_rejected_by_guard"] is True

    # detect_copy_move entegrasyonuyla boş maske üretildiğini doğrula
    img, _ = _create_synthetic_test_case()
    def mock_estimate_affine(pts_s, pts_d, method, ransacReprojThreshold):
        # 4 inlier ver ama 3'ü kümede 1'i uzakta olsun
        mask = np.zeros((len(pts_s), 1), dtype=np.uint8)
        mask[:4] = 1
        M = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32)
        return M, mask

    monkeypatch.setattr(cv2, "estimateAffine2D", mock_estimate_affine)
    # cluster_min_samples=4 yaparsak 3 noktalı küme oluşamaz (hepsi gürültü olur)
    cfg_reject = CopyMoveConfig(spatial_cluster_guard=True, cluster_eps_px=5.0, cluster_min_samples=4, min_inliers=4)
    res = detect_copy_move(img, config=cfg_reject)
    assert res.is_tampered is False
    assert np.count_nonzero(res.mask) == 0
    assert res.num_inliers == 0


def test_spatial_cluster_guard_rejects_when_support_ratio_under_50_percent():
    """Baskın küme min_inliers koşulunu sağlasa dahi destek oranı %50'nin altındaysa tespitin reddedildiğini doğrular."""
    # Küme A: 4 nokta (orta noktalar ~ 100, 100)
    src_A = np.array([[50.0, 50.0], [52.0, 50.0], [51.0, 52.0], [53.0, 51.0]], dtype=np.float32)
    dst_A = np.array([[150.0, 150.0], [152.0, 150.0], [151.0, 152.0], [153.0, 151.0]], dtype=np.float32)

    # Küme B: 4 nokta (orta noktalar ~ 300, 300)
    src_B = np.array([[250.0, 250.0], [252.0, 250.0], [251.0, 252.0], [253.0, 251.0]], dtype=np.float32)
    dst_B = np.array([[350.0, 350.0], [352.0, 350.0], [351.0, 352.0], [353.0, 351.0]], dtype=np.float32)

    # Küme C: 2 nokta (orta noktalar ~ 500, 500)
    src_C = np.array([[450.0, 450.0], [452.0, 452.0]], dtype=np.float32)
    dst_C = np.array([[550.0, 550.0], [552.0, 552.0]], dtype=np.float32)

    pts_src = np.vstack([src_A, src_B, src_C])
    pts_dst = np.vstack([dst_A, dst_B, dst_C])
    total_inliers = 10  # 4 + 4 + 2

    # Baskın küme boyutu = 4, destek oranı = 4 / 10 = 0.40 < 0.50
    config = CopyMoveConfig(spatial_cluster_guard=True, cluster_eps_px=35.0, cluster_min_samples=3, cluster_min_support_ratio=0.50, min_inliers=4)
    filtered_src, filtered_dst, diag, is_rejected = _apply_spatial_clustering(pts_src, pts_dst, config)

    assert is_rejected is True
    assert len(filtered_src) == 0
    assert diag["dominant_cluster_size"] == 4
    assert diag["cluster_support_ratio"] == 0.40
    assert diag["is_rejected_by_guard"] is True


def test_spatial_cluster_guard_exact_50_percent_boundary_condition():
    """Destek oranı tam olarak %50 (0.50) olduğunda sınır koşulunun doğru uygulanarak kabul edildiğini doğrular."""
    # Küme A: 5 nokta (orta noktalar ~ 100, 100)
    src_A = np.array([[50.0, 50.0], [52.0, 50.0], [51.0, 52.0], [53.0, 51.0], [50.0, 53.0]], dtype=np.float32)
    dst_A = np.array([[150.0, 150.0], [152.0, 150.0], [151.0, 152.0], [153.0, 151.0], [150.0, 153.0]], dtype=np.float32)

    # 5 adet birbirinden uzak gürültü noktası
    src_noise = np.array([
        [200.0, 20.0], [300.0, 40.0], [400.0, 60.0], [500.0, 80.0], [600.0, 100.0]
    ], dtype=np.float32)
    dst_noise = np.array([
        [210.0, 20.0], [310.0, 40.0], [410.0, 60.0], [510.0, 80.0], [610.0, 100.0]
    ], dtype=np.float32)

    pts_src = np.vstack([src_A, src_noise])
    pts_dst = np.vstack([dst_A, dst_noise])
    total_inliers = 10

    # Destek oranı: 5 / 10 = 0.50. cluster_min_support_ratio = 0.50 iken >= 0.50 sağlanmalı!
    config = CopyMoveConfig(spatial_cluster_guard=True, cluster_eps_px=35.0, cluster_min_samples=3, cluster_min_support_ratio=0.50, min_inliers=4)
    filtered_src, filtered_dst, diag, is_rejected = _apply_spatial_clustering(pts_src, pts_dst, config)

    assert is_rejected is False
    assert len(filtered_src) == 5
    assert np.allclose(filtered_src, src_A)
    assert diag["dominant_cluster_size"] == 5
    assert diag["cluster_support_ratio"] == 0.50
    assert diag["is_rejected_by_guard"] is False




