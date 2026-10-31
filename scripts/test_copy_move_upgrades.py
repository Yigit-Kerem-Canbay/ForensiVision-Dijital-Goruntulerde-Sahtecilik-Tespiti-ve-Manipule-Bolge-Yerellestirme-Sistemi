"""
Copy-Move Gelişmiş Hibrit Mimari Deney Betiği.

Bu betik:
1. SIFT + AKAZE hibrit öznitelik çıkarımı,
2. CLAHE yerel kontrast adaptasyonu,
3. Akıllı Vektör Oylaması ve Çoklu Kümeleme (Multi-cluster validation)
yöntemlerini 50 Temiz, 50 CoMoFoD Kirli ve 50 DEFACTO Kirli üzerinde test eder.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from forensivision.datasets import CoMoFoDDataset, DefactoDataset
from forensivision.datasets.base import _safe_read_image
from forensivision.copy_move.detector import (
    CopyMoveConfig, _validate_and_prepare_image, _is_valid_affine_transform,
    _count_unique_spatial_points
)


def extract_hybrid_features(gray: np.ndarray, use_akaze: bool = True, use_clahe: bool = True):
    """SIFT ve AKAZE özelliklerini birleştirerek hem dokulu hem pürüzsüz nesneleri yakalar."""
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc_gray = clahe.apply(gray)
    else:
        proc_gray = gray

    # 1. SIFT (Hassas eşiklerle)
    sift = cv2.SIFT_create(contrastThreshold=0.025, edgeThreshold=15)
    kp_sift, desc_sift = sift.detectAndCompute(proc_gray, None)

    if not use_akaze:
        return kp_sift, desc_sift

    # 2. AKAZE (Sınır ve pürüzsüz dokular için)
    akaze = cv2.AKAZE_create(threshold=0.0015)
    kp_akaze, desc_akaze = akaze.detectAndCompute(proc_gray, None)

    # İki farklı tanımlayıcı türünü bağımsız veya birleşik değerlendireceğiz
    return (kp_sift, desc_sift), (kp_akaze, desc_akaze)


def match_and_verify(
    keypoints: list[cv2.KeyPoint],
    descriptors: np.ndarray,
    norm_type: int = cv2.NORM_L2,
    ratio_thresh: float = 0.78,
    min_spatial_dist: float = 20.0,
    ransac_thresh: float = 4.0,
    min_inliers: int = 4,
):
    """Öz-eşleme, RANSAC ve uzamsal filtreleme boru hattı."""
    if descriptors is None or len(keypoints) < 8:
        return False, 0, None, None

    bf = cv2.BFMatcher(norm_type)
    k = min(8, len(descriptors))
    raw_matches = bf.knnMatch(descriptors, descriptors, k=k)

    candidate_pairs = []
    seen = set()

    for i, m_list in enumerate(raw_matches):
        p_q = np.array(keypoints[i].pt, dtype=np.float32)
        non_local = []
        for m in m_list:
            p_t = np.array(keypoints[m.trainIdx].pt, dtype=np.float32)
            if np.linalg.norm(p_q - p_t) >= min_spatial_dist:
                non_local.append(m)

        if len(non_local) >= 2:
            m1, m2 = non_local[0], non_local[1]
            if m1.distance < ratio_thresh * m2.distance:
                pair = tuple(sorted((i, m1.trainIdx)))
                if pair not in seen:
                    seen.add(pair)
                    candidate_pairs.append(pair)

    if len(candidate_pairs) < min_inliers:
        return False, 0, None, None

    pts_src = np.float32([keypoints[p[0]].pt for p in candidate_pairs])
    pts_dst = np.float32([keypoints[p[1]].pt for p in candidate_pairs])

    M, inliers = cv2.estimateAffine2D(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if M is None or inliers is None:
        return False, 0, None, None

    inliers_mask = inliers.ravel() == 1
    num_inl = int(np.sum(inliers_mask))
    if num_inl < min_inliers:
        return False, 0, None, None

    if not _is_valid_affine_transform(M):
        return False, 0, None, None

    inlier_src = pts_src[inliers_mask]
    inlier_dst = pts_dst[inliers_mask]

    # Akıllı Vektör Tutarlılığı:
    # 1. Ya displacement vektörleri tutarlı olmalı (öteleme)
    # 2. Ya da afin dönüşümün hata varyansı çok düşük olmalı (rotasyon/ölçekleme)
    displacements = inlier_dst - inlier_src
    disp_std = np.std(displacements, axis=0)

    # Fiziksel olarak farklı nokta sayısı denetimi
    u_src = _count_unique_spatial_points(inlier_src, tolerance=7.0)
    u_dst = _count_unique_spatial_points(inlier_dst, tolerance=7.0)

    if u_src < min_inliers or u_dst < min_inliers:
        return False, 0, None, None

    # DBSCAN ile sahte alarm filtreleme (fakat çok katı %50 yerine en az 4 noktalı küme kontrolü)
    midpoints = (inlier_src + inlier_dst) / 2.0
    db = DBSCAN(eps=45.0, min_samples=3).fit(midpoints)
    valid_labels = [l for l in np.unique(db.labels_) if l != -1]

    # Eğer saf öteleme varsa displacement std çok küçüktür (maks 10 piksel)
    is_pure_translation = float(np.max(disp_std)) < 12.0

    if is_pure_translation and num_inl >= min_inliers and u_src >= min_inliers:
        return True, num_inl, inlier_src, inlier_dst

    if valid_labels:
        max_cluster_size = max(np.sum(db.labels_ == l) for l in valid_labels)
        if max_cluster_size >= min_inliers:
            return True, max_cluster_size, inlier_src, inlier_dst

    return False, 0, None, None


def evaluate_detector(
    clean_samples: list,
    comofod_dirty_samples: list,
    defacto_dirty_samples: list,
    tag: str,
    use_akaze: bool = False,
    use_clahe: bool = True,
    ratio_thresh: float = 0.78,
    min_spatial_dist: float = 20.0,
    min_inliers: int = 4,
):
    # 1. Clean
    clean_correct = 0
    for s in clean_samples:
        img = _safe_read_image(s.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Test SIFT
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            p_gray = clahe.apply(gray)
        else:
            p_gray = gray
            
        sift = cv2.SIFT_create(contrastThreshold=0.025, edgeThreshold=15)
        kp_s, desc_s = sift.detectAndCompute(p_gray, None)
        flagged_s, _, _, _ = match_and_verify(kp_s, desc_s, cv2.NORM_L2, ratio_thresh, min_spatial_dist, min_inliers=min_inliers)

        flagged_a = False
        if use_akaze and not flagged_s:
            akaze = cv2.AKAZE_create(threshold=0.0015)
            kp_a, desc_a = akaze.detectAndCompute(p_gray, None)
            flagged_a, _, _, _ = match_and_verify(kp_a, desc_a, cv2.NORM_HAMMING, ratio_thresh=0.80, min_spatial_dist=min_spatial_dist, min_inliers=min_inliers)

        is_flagged = flagged_s or flagged_a
        if not is_flagged:
            clean_correct += 1

    clean_acc = (clean_correct / len(clean_samples)) * 100.0

    # 2. CoMoFoD Dirty
    cm_detected = 0
    for s in comofod_dirty_samples:
        img = _safe_read_image(s.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if use_clahe else gray
        
        sift = cv2.SIFT_create(contrastThreshold=0.025, edgeThreshold=15)
        kp_s, desc_s = sift.detectAndCompute(p_gray, None)
        flagged_s, _, _, _ = match_and_verify(kp_s, desc_s, cv2.NORM_L2, ratio_thresh, min_spatial_dist, min_inliers=min_inliers)
        
        flagged_a = False
        if use_akaze and not flagged_s:
            akaze = cv2.AKAZE_create(threshold=0.0015)
            kp_a, desc_a = akaze.detectAndCompute(p_gray, None)
            flagged_a, _, _, _ = match_and_verify(kp_a, desc_a, cv2.NORM_HAMMING, ratio_thresh=0.80, min_spatial_dist=min_spatial_dist, min_inliers=min_inliers)
            
        if flagged_s or flagged_a:
            cm_detected += 1
            
    cm_det_rate = (cm_detected / len(comofod_dirty_samples)) * 100.0

    # 3. DEFACTO Dirty
    df_detected = 0
    for s in defacto_dirty_samples:
        img = _safe_read_image(s.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if use_clahe else gray
        
        sift = cv2.SIFT_create(contrastThreshold=0.025, edgeThreshold=15)
        kp_s, desc_s = sift.detectAndCompute(p_gray, None)
        flagged_s, _, _, _ = match_and_verify(kp_s, desc_s, cv2.NORM_L2, ratio_thresh, min_spatial_dist, min_inliers=min_inliers)
        
        flagged_a = False
        if use_akaze and not flagged_s:
            akaze = cv2.AKAZE_create(threshold=0.0015)
            kp_a, desc_a = akaze.detectAndCompute(p_gray, None)
            flagged_a, _, _, _ = match_and_verify(kp_a, desc_a, cv2.NORM_HAMMING, ratio_thresh=0.80, min_spatial_dist=min_spatial_dist, min_inliers=min_inliers)
            
        if flagged_s or flagged_a:
            df_detected += 1
            
    df_det_rate = (df_detected / len(defacto_dirty_samples)) * 100.0

    print(f"[{tag:32s}] Temiz: %{clean_acc:5.1f} | CoMoFoD Kirli: %{cm_det_rate:5.1f} | DEFACTO Kirli: %{df_det_rate:5.1f}")


def main():
    cmfd = CoMoFoDDataset("data/raw/comofod")
    clean_50 = [s for s in cmfd if s.is_authentic()][:50]
    cm_dirty_50 = [s for s in cmfd if not s.is_authentic()][:50]

    df_cm = DefactoDataset("data/raw/defacto/copymove", task_type="copy_move")
    df_dirty_50 = list(df_cm.samples)[:50]

    print("=" * 85)
    print("COPY-MOVE HİBRİT GELİŞTİRME PİLOT TESTLERİ (50 Temiz + 50 CoMoFoD + 50 DEFACTO)")
    print("=" * 85)

    # 1. Temel SIFT (Eski Ayarlar)
    evaluate_detector(clean_50, cm_dirty_50, df_dirty_50, "1. Eski Baseline", use_akaze=False, use_clahe=False, ratio_thresh=0.75, min_inliers=4)

    # 2. CLAHE + Hassas SIFT + Vektör Oylaması
    evaluate_detector(clean_50, cm_dirty_50, df_dirty_50, "2. CLAHE + SIFT + Vektör Oylama", use_akaze=False, use_clahe=True, ratio_thresh=0.78, min_inliers=4)

    # 3. CLAHE + SIFT + AKAZE Hibrit
    evaluate_detector(clean_50, cm_dirty_50, df_dirty_50, "3. SIFT + AKAZE Hibrit (MinInl=4)", use_akaze=True, use_clahe=True, ratio_thresh=0.78, min_inliers=4)

    # 4. SIFT + AKAZE Hibrit (MinInl=5 Güvenlik)
    evaluate_detector(clean_50, cm_dirty_50, df_dirty_50, "4. SIFT + AKAZE Hibrit (MinInl=5)", use_akaze=True, use_clahe=True, ratio_thresh=0.78, min_inliers=5)


if __name__ == "__main__":
    main()
