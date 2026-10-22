"""
ForensiVision Copy-Move Sahteciliği Tespit ve Yerelleştirme Modülü.

Bu modül; dijital bir görüntünün kendi içerisinden kopyalanıp başka bir bölgesine
yapıştırılmasıyla gerçekleştirilen tahrifatları (Copy-Move Forgery) tespit eder.

Algoritma Aşamaları:
    1. Ön İşleme: Görüntü uygun biçimde gri seviyeye (grayscale) dönüştürülür.
    2. Öznitelik Çıkarımı: SIFT algoritması ile anahtar noktalar ve 128 boyutlu yerel doku tanımlayıcıları çıkarılır.
    3. Öz-Eşleme Filtreleme (Self-Match Filter): Görüntü kendi tanımlayıcılarıyla eşleştirildiğinde,
       aynı anahtar noktanın kendisiyle veya yerel komşuluktaki noktalarla eşleşmesini önlemek için
       Öklid uzamsal mesafe eşiği uygulanır.
    4. Lowe Oran Testi: Kalan non-local adaylar arasında en yakın ve ikinci en yakın komşu mesafesi oranlanır.
    5. Geometrik Doğrulama (RANSAC): Eşleşmelerin rastgele doku benzerliği değil, tutarlı bir afin dönüşüm
       (öteleme, rotasyon, ölçekleme) oluşturup oluşturmadığı kontrol edilir.
    6. Bölge Yerelleştirme ve Maske Üretimi: RANSAC tarafından doğrulanan inlier anahtar noktalarının
       dışbükey zarfları (convex hull) ve etki alanları birleştirilerek şüpheli kaynak ve hedef bölgeleri
       içeren ikili tahmin maskesi (binary mask) üretilir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CopyMoveConfig:
    """Copy-Move tespit algoritması yapılandırma parametreleri.

    Attributes:
        min_keypoints: Tespiti başlatmak için gereken minimum anahtar nokta sayısı.
        ratio_threshold: Lowe oran testi eşiği (en yakın / ikinci en yakın).
        min_spatial_distance: Öz-eşlemeleri önlemek için gereken minimum piksel mesafesi.
        knn_k: KNN eşleştirmesinde incelenecek komşu sayısı.
        min_matches_for_ransac: RANSAC doğrulamasına girmek için gereken minimum eşleşme sayısı.
        ransac_reproj_threshold: RANSAC yeniden izdüşüm piksel hata eşiği.
        min_inliers: Tahrifat kararı vermek için gereken minimum RANSAC inlier sayısı.
        point_radius: Her inlier anahtar noktasının etrafında çizilecek taban yarıçap (piksel).
        morph_kernel_size: Maske kapatma (closing) işleminde kullanılacak eliptik çekirdek boyutu.
    """

    min_keypoints: int = 10
    ratio_threshold: float = 0.75
    min_spatial_distance: float = 25.0
    knn_k: int = 10
    min_matches_for_ransac: int = 4
    ransac_reproj_threshold: float = 5.0
    min_inliers: int = 4
    point_radius: int = 12
    morph_kernel_size: int = 5


@dataclass(frozen=True)
class CopyMoveResult:
    """Copy-Move tespit ve yerelleştirme sonuç nesnesi.

    Attributes:
        mask: Orijinal görüntüyle aynı boyutlu ikili tahmin maskesi (uint8, 0 ve 255).
              NOT: Bu maske, Copy-Move ile ilişkili hem kaynak hem hedef şüpheli bölgeleri içerir.
        is_tampered: Yeterli geometrik kanıt bulunup bulunmadığı (True: sahtecilik şüphesi var).
        num_keypoints: Çıkarılan toplam SIFT anahtar nokta sayısı.
        num_candidate_matches: Öz-eşleme ve Lowe filtresini geçen aday eşleşme sayısı.
        num_inliers: RANSAC afin geometrik doğrulamasından geçen tutarlı eşleşme sayısı.
        transform_matrix: Tespit edilen 2x3 afin dönüşüm matrisi (yoksa None).
        matched_points_src: Kaynak bölge inlier koordinatları (N, 2 float32).
        matched_points_dst: Hedef bölge inlier koordinatları (N, 2 float32).
        config: Kullanılan tespit yapılandırması.
    """

    mask: np.ndarray
    is_tampered: bool
    num_keypoints: int
    num_candidate_matches: int
    num_inliers: int
    transform_matrix: np.ndarray | None
    matched_points_src: np.ndarray
    matched_points_dst: np.ndarray
    config: CopyMoveConfig

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir özet sözlük döndürür (maske ve matris özet bilgileriyle)."""
        return {
            "is_tampered": self.is_tampered,
            "num_keypoints": self.num_keypoints,
            "num_candidate_matches": self.num_candidate_matches,
            "num_inliers": self.num_inliers,
            "has_transform_matrix": self.transform_matrix is not None,
            "mask_shape": list(self.mask.shape),
            "mask_positive_pixels": int(np.count_nonzero(self.mask)),
            "config": asdict(self.config),
        }


def _validate_and_prepare_image(image: Any) -> tuple[np.ndarray, tuple[int, int]]:
    """Girdi görüntüsünü doğrular ve gri seviye görüntü ile orijinal boyutları döndürür.

    Args:
        image: İncelenecek görüntü (2D veya 3D NumPy dizisi).

    Returns:
        tuple[np.ndarray, tuple[int, int]]: (gri_seviye_goruntu, (yukseklik, genislik))

    Raises:
        ValueError: Girdi None, boş dizi veya desteklenmeyen boyut/kanal içeriyorsa.
        TypeError: Girdi NumPy dizisi değilse veya sayısal değilse.
    """
    if image is None:
        raise ValueError("Girdi görüntüsü (image) None olamaz.")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Girdi görüntüsü bir NumPy dizisi olmalıdır; alınan tip: {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Girdi görüntüsü boş bir dizi olamaz.")

    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Görüntü sayısal veri tipinde olmalıdır; alınan tip: {image.dtype}")

    if image.ndim == 2:
        h, w = image.shape
        if image.dtype != np.uint8:
            gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            gray = image.copy()
        return gray, (h, w)

    if image.ndim == 3:
        h, w, c = image.shape
        if c not in (3, 4):
            raise ValueError(
                f"Desteklenmeyen kanal sayısı: {c}. Renkli görüntüler 3 (BGR/RGB) veya 4 (BGRA) kanallı olmalıdır."
            )
        img_uint8 = image if image.dtype == np.uint8 else cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if c == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2GRAY)
        return gray, (h, w)

    raise ValueError(
        f"Desteklenmeyen görüntü boyutu: {image.ndim}D. Yalnızca 2D (gri seviye) veya 3D (renkli) görüntüler desteklenir."
    )


def _build_tampered_mask(
    shape: tuple[int, int],
    src_points: np.ndarray,
    dst_points: np.ndarray,
    config: CopyMoveConfig,
) -> np.ndarray:
    """Doğrulanan kaynak ve hedef noktalarından ikili manipülasyon maskesi üretir.

    Kaynak ve hedef kümelerinin dışbükey zarfları (convex hull) doldurulur ve
    noktalar etrafındaki yerel pikseller morfolojik kapanışla birleştirilir.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for pts in (src_points, dst_points):
        if len(pts) == 0:
            continue

        # 1. Her inlier anahtar noktası etrafına etki alanı çiz
        for pt in pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(mask, (x, y), config.point_radius, 255, thickness=-1)

        # 2. 3 veya daha fazla nokta varsa dışbükey zarfı (convex hull) doldur
        if len(pts) >= 3:
            hull = cv2.convexHull(np.int32(pts))
            cv2.fillConvexPoly(mask, hull, 255)

    # 3. Küçük boşlukları ve kopuklukları gidermek için eliptik morfolojik kapanış
    if config.morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _count_unique_spatial_points(points: np.ndarray, tolerance: float = 6.0) -> int:
    """Verilen nokta kümesindeki fiziksel olarak farklı (birbirine çok yakın olmayan) nokta sayısını hesaplar.

    SIFT aynı (x, y) pikselinde farklı ölçek veya yönlerde birden fazla anahtar nokta çıkarabilir.
    Bu fonksiyon, aynı fiziksel köşenin tekrar sayılarak RANSAC'ı yanıltmasını önler.
    """
    if len(points) == 0:
        return 0

    unique: list[np.ndarray] = []
    for pt in points:
        if not any(float(np.linalg.norm(pt - u)) < tolerance for u in unique):
            unique.append(pt)
    return len(unique)


def _is_valid_affine_transform(
    transform_matrix: np.ndarray,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    max_distortion_ratio: float = 3.5,
) -> bool:
    """Afin dönüşümün fiziksel olarak gerçekçi bir kopyalama olup olmadığını denetler.

    Rastgele veya kolineer eşleşmelerde afin matris uzayı 1D çizgiye çökertebilir (tekil değer ~ 0).
    Gerçek Copy-Move manipülasyonlarında ise ölçekleme sınırlı ve iki eksen arasındaki deformasyon orantılıdır.
    """
    if transform_matrix is None or transform_matrix.shape != (2, 3):
        return False

    linear_part = transform_matrix[:, :2]
    # Tekil değer ayrışımı (Singular Value Decomposition - SVD)
    singular_values = np.linalg.svd(linear_part, compute_uv=False)
    sigma_max, sigma_min = float(singular_values[0]), float(singular_values[1])

    if sigma_min < 1e-4:
        return False  # Matris tekil (singular), uzay çizgiye çökmüş

    # Aşırı küçültme veya aşırı büyütme kontrolü
    if sigma_min < min_scale or sigma_max > max_scale:
        return False

    # İki eksen arasındaki aşırı orantısız bozulma (anizotropik germe) kontrolü
    distortion_ratio = sigma_max / sigma_min
    if distortion_ratio > max_distortion_ratio:
        return False

    return True


def detect_copy_move(
    image: Any,
    config: CopyMoveConfig | None = None,
) -> CopyMoveResult:
    """Görüntüde Copy-Move sahteciliğini SIFT ve RANSAC ile analiz eder ve yerelleştirir.

    Args:
        image: Analiz edilecek görüntü (2D gri tonlama veya 3D renkli NumPy dizisi).
        config: İsteğe bağlı algoritma yapılandırma nesnesi (CopyMoveConfig).

    Returns:
        CopyMoveResult: Tahmin maskesi, tespit durumu, eşleşme ve inlier sayılarını içeren sonuç.

    Raises:
        ValueError: Girdi görüntüsü None, boş veya geçersiz boyutluysa.
        TypeError: Girdi görüntüsü NumPy dizisi değilse.
    """
    if config is None:
        config = CopyMoveConfig()

    gray, (h, w) = _validate_and_prepare_image(image)
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    empty_pts = np.empty((0, 2), dtype=np.float32)

    # 1. SIFT ile öznitelik çıkarımı
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    num_keypoints = len(keypoints) if keypoints is not None else 0

    if descriptors is None or num_keypoints < config.min_keypoints:
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=0,
            num_inliers=0,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )

    # 2. Öz-Eşleştirme ve Aday Eşleşmelerin Filtrelenmesi (Self-Matching KNN)
    k = min(config.knn_k, len(descriptors))
    if k < 2:
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=0,
            num_inliers=0,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(descriptors, descriptors, k=k)

    candidate_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for i, m_list in enumerate(raw_matches):
        p_query = np.array(keypoints[i].pt, dtype=np.float32)

        # Uzamsal olarak yeterince uzak olan (non-local) eşleşmeleri ayıkla
        non_local: list[cv2.DMatch] = []
        for m in m_list:
            p_train = np.array(keypoints[m.trainIdx].pt, dtype=np.float32)
            dist_spatial = float(np.linalg.norm(p_query - p_train))
            if dist_spatial >= config.min_spatial_distance:
                non_local.append(m)

        # Lowe ratio testi: En yakın non-local komşu ile ikinci en yakın komşunun oranı
        if len(non_local) >= 2:
            m1, m2 = non_local[0], non_local[1]
            if m1.distance < config.ratio_threshold * m2.distance:
                # Simetrik çiftleri (i, j) ve (j, i) tekilleştir
                pair = tuple(sorted((i, m1.trainIdx)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidate_pairs.append(pair)

    num_candidates = len(candidate_pairs)

    # 3. Yetersiz aday eşleşme kontrolü
    if num_candidates < config.min_matches_for_ransac:
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=num_candidates,
            num_inliers=0,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )

    pts_src = np.float32([keypoints[p[0]].pt for p in candidate_pairs])
    pts_dst = np.float32([keypoints[p[1]].pt for p in candidate_pairs])

    # 4. RANSAC ile Geometrik Tutarlılık Analizi (Afin Dönüşüm)
    transform_matrix, inliers = cv2.estimateAffine2D(
        pts_src,
        pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
    )

    if transform_matrix is None or inliers is None:
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=num_candidates,
            num_inliers=0,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )

    inliers_mask = inliers.ravel() == 1
    num_inliers = int(np.sum(inliers_mask))

    if num_inliers < config.min_inliers:
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=num_candidates,
            num_inliers=num_inliers,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )

    inlier_pts_src = pts_src[inliers_mask]
    inlier_pts_dst = pts_dst[inliers_mask]

    # 5. Geometrik Tutarlılık ve Fiziksel Çeşitlilik Doğrulaması
    # A. Afin matris fiziksel olarak makul bir kopyalama mı (tekil/çökmüş dönüşüm değil)?
    if not _is_valid_affine_transform(transform_matrix):
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=num_candidates,
            num_inliers=0,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )

    # B. İnlier noktalar fiziksel olarak yeterli sayıda bağımsız konuma dağılmış mı?
    unique_src_count = _count_unique_spatial_points(inlier_pts_src, tolerance=6.0)
    unique_dst_count = _count_unique_spatial_points(inlier_pts_dst, tolerance=6.0)

    if unique_src_count < config.min_inliers or unique_dst_count < config.min_inliers:
        return CopyMoveResult(
            mask=empty_mask,
            is_tampered=False,
            num_keypoints=num_keypoints,
            num_candidate_matches=num_candidates,
            num_inliers=0,
            transform_matrix=None,
            matched_points_src=empty_pts,
            matched_points_dst=empty_pts,
            config=config,
        )


    # 6. Şüpheli Bölge Maskesi Üretimi
    tampered_mask = _build_tampered_mask(
        shape=(h, w),
        src_points=inlier_pts_src,
        dst_points=inlier_pts_dst,
        config=config,
    )

    return CopyMoveResult(
        mask=tampered_mask,
        is_tampered=True,
        num_keypoints=num_keypoints,
        num_candidate_matches=num_candidates,
        num_inliers=num_inliers,
        transform_matrix=transform_matrix,
        matched_points_src=inlier_pts_src,
        matched_points_dst=inlier_pts_dst,
        config=config,
    )

