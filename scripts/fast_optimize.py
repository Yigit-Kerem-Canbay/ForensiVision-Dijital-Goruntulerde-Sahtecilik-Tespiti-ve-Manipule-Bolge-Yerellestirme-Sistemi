"""
Ultra Hızlı Parametre Optimizasyonu (Öznitelikleri Bir Kez Çıkar - Bellekte Test Et).

Bu betik:
- 50 Temiz, 50 CoMoFoD, 50 DEFACTO görselinin SIFT özelliklerini sadece BİR KEZ çıkarır (~15 saniye).
- 54 farklı filtre ve kümeleme kombinasyonunu bu hazır özellikler üzerinde saniyeler içinde test eder.
- Temiz görsel başarısını >= %90 tutarken en yüksek kirli başarısını veren konfigürasyonu anında bulur.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from forensivision.datasets import CoMoFoDDataset, DefactoDataset
from forensivision.datasets.base import _safe_read_image
from forensivision.copy_move.detector import (
    CopyMoveConfig, _is_valid_affine_transform, _count_unique_spatial_points
)


def extract_cached_data(samples):
    cached = []
    sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=12)
    for s in samples:
        img = _safe_read_image(s.image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        cached.append((kp, desc))
    return cached


def test_combination(cached_list, ratio, spatial_dist, eps, support_ratio, min_inliers):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    flagged_count = 0

    for kp, desc in cached_list:
        if desc is None or len(kp) < 8:
            continue

        k = min(8, len(desc))
        raw_matches = bf.knnMatch(desc, desc, k=k)

        candidate_pairs = []
        seen = set()

        for i, m_l in enumerate(raw_matches):
            p_q = np.array(kp[i].pt, dtype=np.float32)
            non_local = []
            for m in m_l:
                p_t = np.array(kp[m.trainIdx].pt, dtype=np.float32)
                if np.linalg.norm(p_q - p_t) >= spatial_dist:
                    non_local.append(m)

            if len(non_local) >= 2:
                m1, m2 = non_local[0], non_local[1]
                if m1.distance < ratio * m2.distance:
                    pair = tuple(sorted((i, m1.trainIdx)))
                    if pair not in seen:
                        seen.add(pair)
                        candidate_pairs.append(pair)

        if len(candidate_pairs) < min_inliers:
            continue

        pts_src = np.float32([kp[p[0]].pt for p in candidate_pairs])
        pts_dst = np.float32([kp[p[1]].pt for p in candidate_pairs])

        M, inliers = cv2.estimateAffine2D(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=4.5)
        if M is None or inliers is None:
            continue

        inliers_mask = inliers.ravel() == 1
        num_inl = int(np.sum(inliers_mask))
        if num_inl < min_inliers or not _is_valid_affine_transform(M):
            continue

        inlier_src = pts_src[inliers_mask]
        inlier_dst = pts_dst[inliers_mask]

        midpoints = (inlier_src + inlier_dst) / 2.0
        db = DBSCAN(eps=eps, min_samples=3).fit(midpoints)
        valid_labels = [l for l in np.unique(db.labels_) if l != -1]

        if not valid_labels:
            continue

        dominant_size = max(np.sum(db.labels_ == l) for l in valid_labels)
        ratio_in_dom = dominant_size / num_inl

        if dominant_size >= min_inliers and ratio_in_dom >= support_ratio:
            # Unique spatial points check
            best_l = max(valid_labels, key=lambda l: np.sum(db.labels_ == l))
            u_src = _count_unique_spatial_points(inlier_src[db.labels_ == best_l], tolerance=6.0)
            u_dst = _count_unique_spatial_points(inlier_dst[db.labels_ == best_l], tolerance=6.0)
            if u_src >= min_inliers and u_dst >= min_inliers:
                flagged_count += 1

    return flagged_count


def main():
    print("Örnekler taranıyor...")
    cmfd = CoMoFoDDataset("data/raw/comofod")
    clean_samples = [s for s in cmfd if s.is_authentic()][:50]
    cmfd_dirty = [s for s in cmfd if not s.is_authentic()][:50]

    df_cm = DefactoDataset("data/raw/defacto/copymove", task_type="copy_move")
    df_dirty = list(df_cm.samples)[:50]

    print("SIFT özellikleri çıkarılıyor (sadece 1 kez)...")
    clean_cached = extract_cached_data(clean_samples)
    cmfd_cached = extract_cached_data(cmfd_dirty)
    df_cached = extract_cached_data(df_dirty)
    print("Özellikler hazır! Izgara araması başlıyor...\n")

    ratios = [0.75, 0.78, 0.80]
    spatial_dists = [18.0, 22.0, 25.0]
    cluster_eps_list = [35.0, 42.0]
    support_ratios = [0.45, 0.50]
    min_inliers_list = [4, 5]

    best_score = -1.0
    best_combo = None

    for r in ratios:
        for dist in spatial_dists:
            for eps in cluster_eps_list:
                for sup in support_ratios:
                    for min_inl in min_inliers_list:
                        # Clean
                        c_flagged = test_combination(clean_cached, r, dist, eps, sup, min_inl)
                        clean_acc = ((len(clean_cached) - c_flagged) / len(clean_cached)) * 100.0

                        if clean_acc < 88.0:
                            continue

                        # CoMoFoD
                        cm_det = test_combination(cmfd_cached, r, dist, eps, sup, min_inl)
                        cm_acc = (cm_det / len(cmfd_cached)) * 100.0

                        # DEFACTO
                        df_det = test_combination(df_cached, r, dist, eps, sup, min_inl)
                        df_acc = (df_det / len(df_cached)) * 100.0

                        score = (clean_acc * 2.0) + cm_acc + df_acc
                        print(
                            f"Ratio={r:.2f}, Dist={dist:4.1f}, EPS={eps:4.1f}, Sup={sup:.2f}, MinInl={min_inl} "
                            f"-> Temiz: %{clean_acc:5.1f} | CoMoFoD: %{cm_acc:5.1f} | DEFACTO: %{df_acc:5.1f} | Skor: {score:.1f}"
                        )

                        if score > best_score:
                            best_score = score
                            best_combo = (r, dist, eps, sup, min_inl, clean_acc, cm_acc, df_acc)

    print("\n" + "=" * 70)
    print("OPTİMİZASYON TAMAMLANDI!")
    print(f"En İyi Sonuçlar:")
    print(f"  - Temiz Başarı:  %{best_combo[5]:.1f}")
    print(f"  - CoMoFoD Kirli: %{best_combo[6]:.1f}")
    print(f"  - DEFACTO Kirli: %{best_combo[7]:.1f}")
    print(f"En İyi Parametreler: Ratio={best_combo[0]}, Dist={best_combo[1]}, EPS={best_combo[2]}, Sup={best_combo[3]}, MinInliers={best_combo[4]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
