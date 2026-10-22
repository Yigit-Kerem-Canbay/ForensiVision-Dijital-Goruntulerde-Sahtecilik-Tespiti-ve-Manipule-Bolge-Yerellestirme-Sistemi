"""
ForensiVision Sentetik Copy-Move Deney ve Görselleştirme Betiği.

Bu betik; deterministik ve dokulu bir sentetik görüntü oluşturur, görüntünün bir
bölgesini kopyalayıp başka bir konuma yapıştırarak (Copy-Move) gerçek zemin
(ground-truth) maskesini ve manipüle görüntüyü üretir.

Ardından ForensiVision Copy-Move tespit motorunu çalıştırır, üretilen tahmin maskesini
değerlendirme motoruyla karşılaştırır ve sonuçları görsel olarak 'data/output/' dizinine kaydeder.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from forensivision.copy_move import CopyMoveConfig, detect_copy_move
from forensivision.evaluation import calculate_metrics
from forensivision.visualization import create_mask_overlay, draw_copy_move_matches


def generate_synthetic_copy_move(
    width: int = 400,
    height: int = 400,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Deterministik ve zengin öznitelikli sentetik bir Copy-Move test görüntüsü üretir.

    Returns:
        tuple: (manipule_goruntu, gercek_maske, kaynak_kutu, hedef_kutu)
               kutu formati: (y1, y2, x1, x2)
    """
    # 1. Açık gri dokulu arka plan
    img = np.ones((height, width, 3), dtype=np.uint8) * 235

    # Arka plan ızgara çizgileri (hafif kontrast)
    for x in range(20, width, 40):
        cv2.line(img, (x, 0), (x, height), (215, 215, 215), 1)
    for y in range(20, height, 40):
        cv2.line(img, (0, y), (width, y), (215, 215, 215), 1)

    # 2. Kaynak bölge (100x100 boyutunda zengin geometrik şekiller ve yazı)
    src_box = (50, 150, 50, 150)  # y1, y2, x1, x2
    sy1, sy2, sx1, sx2 = src_box

    # Kaynak içeriği çizimi
    cv2.rectangle(img, (sx1 + 5, sy1 + 5), (sx2 - 5, sy2 - 5), (50, 90, 210), -1)
    cv2.circle(img, (sx1 + 50, sy1 + 50), 30, (255, 255, 255), -1)
    cv2.circle(img, (sx1 + 50, sy1 + 50), 15, (30, 30, 30), -1)
    cv2.putText(
        img,
        "COPY",
        (sx1 + 18, sy1 + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 3. Hedef bölgeye kopyalama (Hedef: y: 220-320, x: 220-320)
    dst_box = (220, 320, 220, 320)
    dy1, dy2, dx1, dx2 = dst_box

    patch = img[sy1:sy2, sx1:sx2].copy()
    img[dy1:dy2, dx1:dx2] = patch

    # 4. İkili Gerçek Zemin (Ground-Truth) Maskesi
    # NOT: Prototipimiz hem kaynağı hem hedefi şüpheli bölge olarak işaretlediğinden
    # ground-truth maskesi de iki bölgeyi kapsayacak şekilde tanımlanır.
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask[sy1:sy2, sx1:sx2] = 255
    gt_mask[dy1:dy2, dx1:dx2] = 255

    return img, gt_mask, src_box, dst_box


def main() -> None:
    print("=" * 60)
    print("ForensiVision — Sentetik Copy-Move Tespit ve Doğrulama Demosu")
    print("=" * 60)

    # 1. Sentetik manipülasyon örneğini oluştur
    img, gt_mask, src_box, dst_box = generate_synthetic_copy_move()
    print(f"[1] Sentetik test görüntüsü üretildi: {img.shape}")
    print(f"    - Kaynak bölge kutusu: {src_box}")
    print(f"    - Hedef bölge kutusu : {dst_box}")

    # 2. Tespit algoritmasını çalıştır
    config = CopyMoveConfig(
        min_spatial_distance=30.0,
        ratio_threshold=0.75,
        min_inliers=4,
    )
    result = detect_copy_move(img, config=config)

    print("\n[2] Algoritma Çıktıları:")
    print(f"    - Manipülasyon Tespit Edildi mi: {result.is_tampered}")
    print(f"    - Bulunan SIFT Anahtar Noktaları: {result.num_keypoints}")
    print(f"    - Aday Eşleşmeler (Lowe + Mesafe): {result.num_candidate_matches}")
    print(f"    - RANSAC Doğrulamalı Inlier Sayısı: {result.num_inliers}")

    # 3. Değerlendirme motoru ile metrikleri hesapla
    metrics = calculate_metrics(result.mask, gt_mask)
    print("\n[3] Akademik Yerelleştirme Metrikleri:")
    print(f"    - Precision (Kesinlik) : {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
    print(f"    - Recall (Duyarlılık)  : {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
    print(f"    - F1-Score             : {metrics.f1_score:.4f} ({metrics.f1_score*100:.2f}%)")
    print(f"    - IoU (Jaccard)        : {metrics.iou:.4f} ({metrics.iou*100:.2f}%)")
    print(f"    - Piksel Sayıları      : TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}")

    # 4. Görselleri hazırla ve dışa aktar
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = create_mask_overlay(img, result.mask, color=(0, 0, 255), alpha=0.45)
    matched_vis = draw_copy_move_matches(
        img, result.matched_points_src, result.matched_points_dst
    )

    # 4 panelli karşılaştırma ızgarası:
    # [ Orijinal Manipüle ]  [ Eşleşme Çizgileri ]
    # [ Gerçek Maske (GT) ]  [ Tespit Bindirmesi ]
    gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack([img, matched_vis])
    bottom_row = np.hstack([gt_bgr, overlay])
    grid = np.vstack([top_row, bottom_row])

    output_path = output_dir / "synthetic_copy_move_demo.png"
    cv2.imwrite(str(output_path), grid)
    print(f"\n[4] Karşılaştırma görseli kaydedildi: {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
