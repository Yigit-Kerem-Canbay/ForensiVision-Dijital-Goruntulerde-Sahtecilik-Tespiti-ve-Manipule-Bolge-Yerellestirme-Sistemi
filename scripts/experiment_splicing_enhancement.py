"""
Image Splicing Algoritma İyileştirme Deney Betiği.

Amaç:
- Temiz fotoğraflardaki yanlış alarmları sıfıra yaklaştırmak (%95+ temiz başarı).
- Kirli fotoğraflarda nesne yakalama ve piksel isabet oranını (%34'ten %80+'e) çıkarmak.
- Kenar baskılama (Edge Suppression), dinamik z-score eşikleme ve gürültü analizini test etmek.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from forensivision.datasets import CoMoFoDDataset, DefactoDataset
from forensivision.datasets.base import _safe_read_image
from forensivision.splicing import SplicingConfig, detect_splicing


def evaluate_splicing_variant(
    clean_samples: list,
    dirty_samples: list,
    tag: str,
    edge_suppression: bool = False,
    std_factor: float = 2.0,
    min_area: int = 150,
    morph_k: int = 11,
    fill_contours: bool = True,
    edge_kernel_size: int = 5,
):
    clean_correct = 0
    clean_fa = 0

    dirty_detected = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    cfg = SplicingConfig(
        fusion_mode="ela_only",
        ela_threshold_std_factor=std_factor,
        fill_contours=fill_contours,
        morph_kernel_size=morph_k,
        min_region_area=min_area,
    )

    # 1. Clean images
    for s in clean_samples:
        img = _safe_read_image(s.image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        if edge_suppression:
            # Kenar baskılama testi
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_kernel_size, edge_kernel_size))
            dilated_edges = cv2.dilate(edges, k)
            
            # ELA'yı çalıştır
            res = detect_splicing(img, config=cfg)
            mask = res.mask.copy()
            # Kenarları maskeden çıkar (kenarlar sahtecilik değildir)
            mask[dilated_edges > 0] = 0
            
            # Kalan maskeyi min_area filtresinden geçir
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
            final_mask = np.zeros_like(mask)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    final_mask[labels == lbl] = 255
            is_flagged = bool(np.count_nonzero(final_mask) > 0)
        else:
            res = detect_splicing(img, config=cfg)
            is_flagged = bool(res.is_suspicious and np.count_nonzero(res.mask) > 0)

        if is_flagged:
            clean_fa += 1
        else:
            clean_correct += 1

    clean_acc = (clean_correct / len(clean_samples)) * 100.0

    # 2. Dirty images
    for s in dirty_samples:
        img = _safe_read_image(s.image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Load mask
        mask_path = s.mask_path or s.target_mask_path
        if mask_path and mask_path.exists():
            m_raw = _safe_read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (m_raw > 127).astype(np.uint8) * 255
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        if edge_suppression:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_kernel_size, edge_kernel_size))
            dilated_edges = cv2.dilate(edges, k)
            
            res = detect_splicing(img, config=cfg)
            mask = res.mask.copy()
            mask[dilated_edges > 0] = 0
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
            final_mask = np.zeros_like(mask)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    final_mask[labels == lbl] = 255
            pred_mask = final_mask
            is_flagged = bool(np.count_nonzero(pred_mask) > 0)
        else:
            res = detect_splicing(img, config=cfg)
            pred_mask = res.mask
            is_flagged = bool(res.is_suspicious and np.count_nonzero(pred_mask) > 0)

        if is_flagged:
            dirty_detected += 1

        tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
        fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
        fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

        total_tp += tp
        total_fp += fp
        total_fn += fn

    dirty_det_rate = (dirty_detected / len(dirty_samples)) * 100.0
    pixel_prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0

    print(
        f"[{tag:25s}] Temiz Başarı: %{clean_acc:5.1f} | "
        f"Kirli Yakalama: %{dirty_det_rate:5.1f} | "
        f"Piksel İsabet: %{pixel_prec:5.2f}"
    )


def main():
    cmfd = CoMoFoDDataset("data/raw/comofod")
    clean_50 = [s for s in cmfd if s.is_authentic()][:50]

    df_spl = DefactoDataset("data/raw/defacto/splicing", task_type="splicing")
    dirty_50 = list(df_spl.samples)[:50]

    print("=" * 75)
    print("IMAGE SPLICING İYİLEŞTİRME PİLOT TESTLERİ (50 Temiz + 50 Kirli)")
    print("=" * 75)

    # 1. Mevcut Durum (Baseline)
    evaluate_splicing_variant(clean_50, dirty_50, "1. Mevcut Baseline", edge_suppression=False, std_factor=2.0, min_area=150)

    # 2. std_factor artırımı (2.5)
    evaluate_splicing_variant(clean_50, dirty_50, "2. Std Factor 2.5", edge_suppression=False, std_factor=2.5, min_area=150)

    # 3. std_factor artırımı (3.0)
    evaluate_splicing_variant(clean_50, dirty_50, "3. Std Factor 3.0", edge_suppression=False, std_factor=3.0, min_area=150)

    # 4. std_factor 2.5 + min_area 300
    evaluate_splicing_variant(clean_50, dirty_50, "4. Std 2.5 + Alan 300", edge_suppression=False, std_factor=2.5, min_area=300)

    # 5. Kenar Baskılama (Edge Suppression) + Std 2.0
    evaluate_splicing_variant(clean_50, dirty_50, "5. Kenar Baskılama (2.0)", edge_suppression=True, std_factor=2.0, min_area=150)

    # 6. Kenar Baskılama + Std 2.5 + Alan 250
    evaluate_splicing_variant(clean_50, dirty_50, "6. Hibrit Optimize", edge_suppression=True, std_factor=2.5, min_area=250)


if __name__ == "__main__":
    main()
