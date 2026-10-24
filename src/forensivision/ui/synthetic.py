"""
ForensiVision Sentetik Demo Örnekleri Üretici Modülü (Synthetic Demo Generator).

Bu modül; gerçek veri setleri yerel ortamda bulunmadığında kullanıcı arayüzü üzerinde
sistemin kabiliyetlerini, analiz aşamalarını ve metrik hesaplama motorunu
deneyimleyebilmek için kontrollü sentetik örnekler üretir.

UYARI:
    Bu sentetik örnekler yalnızca geliştirme, test ve görsel arayüz doğrulama
    amaçlıdır; gerçek dünya performansı veya adli kanıt teşkil etmez.
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_synthetic_copy_move_sample(
    width: int = 400,
    height: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Deterministik ve dokulu bir sentetik Copy-Move örneği üretir.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, str]:
            - image_bgr: Manipüle edilmiş BGR renkli görüntü.
            - image_rgb: Streamlit için RGB renkli görüntü.
            - ground_truth_mask: Kaynak ve hedef bölgeleri içeren (H, W) ikili maske (uint8 0 ve 255).
            - description: Örnek hakkında açıklayıcı metin.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 235

    # Arka plan ızgara çizgileri
    for x in range(20, width, 40):
        cv2.line(img, (x, 0), (x, height), (215, 215, 215), 1)
    for y in range(20, height, 40):
        cv2.line(img, (0, y), (width, y), (215, 215, 215), 1)

    # Kaynak bölge (100x100 boyutunda şekil ve yazı)
    sy1, sy2, sx1, sx2 = 50, 150, 50, 150
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

    # Hedef bölgeye kopyalama
    dy1, dy2, dx1, dx2 = 220, 320, 220, 320
    patch = img[sy1:sy2, sx1:sx2].copy()
    img[dy1:dy2, dx1:dx2] = patch

    # İkili Gerçek Zemin (Ground-Truth) Maskesi: Hem kaynak hem hedef 255
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask[sy1:sy2, sx1:sx2] = 255
    gt_mask[dy1:dy2, dx1:dx2] = 255

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    description = (
        "Sentetik Copy-Move Senaryosu: Görüntünün sol üst köşesindeki (50, 50, 150, 150) "
        "dokulu geometrik nesne kopyalanarak sağ alt köşedeki (220, 220, 320, 320) koordinatlarına taşınmıştır."
    )
    return img, image_rgb, gt_mask, description


def generate_synthetic_splicing_sample(
    width: int = 400,
    height: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Belirgin yerel gürültü tutarsızlığına sahip sentetik bir Splicing örneği üretir.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, str]:
            - image_bgr: Manipüle edilmiş BGR renkli görüntü.
            - image_rgb: Streamlit için RGB renkli görüntü.
            - ground_truth_mask: Splicing eklenen bölgeyi içeren (H, W) ikili maske (uint8 0 ve 255).
            - description: Örnek hakkında açıklayıcı metin.
    """
    np.random.seed(42)

    # Düşük gürültülü temel arka plan
    img = np.ones((height, width, 3), dtype=np.float32) * 140.0
    bg_noise = np.random.normal(0, 1.5, (height, width, 3)).astype(np.float32)
    img = np.clip(img + bg_noise, 0, 255)

    # Spliced yama konumu: (120:280, 120:280)
    y1, y2, x1, x2 = 120, 280, 120, 280

    # Farklı sensör/yüksek gürültülü yabancı nesne yaması
    patch = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32) * 165.0
    spliced_noise = np.random.normal(0, 14.0, (y2 - y1, x2 - x1, 3)).astype(np.float32)
    patch = np.clip(patch + spliced_noise, 0, 255)

    # Yamayı yerleştir
    img[y1:y2, x1:x2] = patch

    # Doku ve kenar bilgisi
    img_uint8 = img.astype(np.uint8)
    cv2.rectangle(img_uint8, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (90, 90, 220), 2)
    cv2.putText(
        img_uint8,
        "SPLICE",
        (x1 + 25, y1 + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Gerçek Zemin Maskesi (Sadece eklenen yama 255)
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask[y1:y2, x1:x2] = 255

    image_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    description = (
        "Sentetik Splicing Senaryosu: Arka plana göre çok daha yüksek sensör gürültüsüne "
        f"ve farklı parlaklığa sahip bağımsız bir yama ({x1}:{x2}, {y1}:{y2}) görüntünün merkezine monte edilmiştir."
    )
    return img_uint8, image_rgb, gt_mask, description


def get_synthetic_demo_sample(
    scenario_type: str = "copy_move",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Belirtilen senaryo türüne göre sentetik demo verisini döndürür."""
    if scenario_type.lower() == "splicing":
        return generate_synthetic_splicing_sample()
    return generate_synthetic_copy_move_sample()
