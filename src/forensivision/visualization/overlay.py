"""
ForensiVision Görselleştirme ve Maske Bindirme (Overlay) Araçları.

Bu modül; adli tespit maskelerinin orijinal görüntü üzerine yarı saydam bindirilmesini,
kaynak ve hedef eşleşme çizgilerinin çizilmesini ve çoklu karşılaştırma panellerinin
oluşturulmasını sağlar.
"""

from __future__ import annotations

import cv2
import numpy as np


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """İkili manipülasyon maskesini orijinal görüntü üzerine renkli yarı saydam katman olarak bindirir.

    Args:
        image: Orijinal görüntü (2D gri tonlama veya 3D BGR/RGB).
        mask: İkili maske (H, W boyutunda, uint8 veya boolean).
        color: Bindirme rengi (B, G, R formatında, varsayılan kırmızı: (0, 0, 255)).
        alpha: Saydamlık oranı [0.0 - 1.0].

    Returns:
        np.ndarray: Bindirme uygulanmış 3 kanallı BGR renkli görüntü.
    """
    if image is None or mask is None:
        raise ValueError("Görüntü ve maske None olamaz.")

    # Görüntüyü 3 kanallı renkli formata hazırla
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        base = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Desteklenmeyen görüntü formatı: {image.shape}")

    bin_mask = (mask > 0).astype(np.uint8)

    colored_overlay = base.copy()
    colored_overlay[bin_mask == 1] = color

    # Yarı saydam birleştirme: overlay = alpha * color + (1 - alpha) * original
    output = cv2.addWeighted(colored_overlay, alpha, base, 1.0 - alpha, 0)

    # İsteğe bağlı olarak maskelenen bölgenin sınırına belirgin bir kontur çiz
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2)

    return output


def draw_copy_move_matches(
    image: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    line_color: tuple[int, int, int] = (0, 255, 0),
    point_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Tespit edilen kaynak ve hedef inlier noktalarını ve aralarındaki eşleşme çizgilerini çizer."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for p1, p2 in zip(pts_src, pts_dst):
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))

        cv2.line(canvas, pt1, pt2, line_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 4, point_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 4, (255, 0, 0), -1, cv2.LINE_AA)

    return canvas
