"""
ForensiVision Sentetik Image Splicing Deney ve Doğrulama Betiği.

Bu betik; başka bir görüntüden alınan nesnelerin hedef görüntüye eklenmesiyle
oluşturulan Image Splicing tahrifatını temsil eden kontrollü sentetik senaryolar üretir:
    1. Senaryo A: Yüksek Gürültü Tutarsızlığı (High Noise Discrepancy)
    2. Senaryo B: Farklı JPEG Sıkıştırma Geçmişi / ELA Senaryosu (JPEG Recompression)
    3. Senaryo C: Düşük / Zorlayıcı Gürültü Tutarsızlığı (Subtle Noise Discrepancy)

Her senaryo için ELA, Gürültü Analizi ve Birleşik (Füzyon) tespit yöntemleri
çalıştırılır; metrikler hesaplanır ve 8 panelli adli karşılaştırma görseli üretilir.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from forensivision.evaluation import calculate_metrics
from forensivision.splicing import (
    SplicingConfig,
    analyze_ela,
    analyze_noise_inconsistency,
    detect_splicing,
)
from forensivision.visualization import create_mask_overlay


def generate_scenario_high_noise(
    size: tuple[int, int] = (360, 360),
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Senaryo A: Arka plan ile eklenen nesne arasında belirgin sensör gürültüsü farkı."""
    h, w = size
    np.random.seed(42)

    # Düşük gürültülü pürüzsüz arka plan (sigma=1.5)
    bg = np.ones((h, w, 3), dtype=np.float32) * 210
    # Yumuşak dikey gradyan
    for y in range(h):
        bg[y, :, :] -= (y / h) * 40.0
    bg += np.random.normal(0, 1.5, bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # Eklenen nesne: Bağımsız ve belirgin gürültüye sahip doku (sigma=16.0)
    box = (90, 230, 90, 230)  # y1, y2, x1, x2 (140x140)
    y1, y2, x1, x2 = box

    patch = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32) * 90
    cv2.circle(patch, (70, 70), 45, (160, 160, 160), -1)
    cv2.circle(patch, (70, 70), 20, (50, 50, 50), -1)
    patch += np.random.normal(0, 16.0, patch.shape)
    patch = np.clip(patch, 0, 255).astype(np.uint8)

    spliced = bg.copy()
    spliced[y1:y2, x1:x2] = patch

    gt = np.zeros((h, w), dtype=np.uint8)
    gt[y1:y2, x1:x2] = 255

    return spliced, gt, box


def generate_scenario_jpeg_recompression(
    size: tuple[int, int] = (360, 360),
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Senaryo B: Farklı JPEG sıkıştırma geçmişine sahip bölgelerin birleşimi (ELA odaklı)."""
    h, w = size

    # 1. Arka plan düşük kalitede (Q=55) sıkıştırılıp açılmış
    bg = np.ones((h, w, 3), dtype=np.uint8) * 195
    cv2.circle(bg, (180, 180), 120, (230, 230, 230), -1)
    buf_bg = io.BytesIO()
    Image.fromarray(bg).save(buf_bg, format="JPEG", quality=55)
    buf_bg.seek(0)
    bg_compressed = np.array(Image.open(buf_bg))

    # 2. Eklenen nesne sıkıştırılmamış / yüksek kaliteli bir kaynaktan
    box = (80, 200, 80, 200)
    y1, y2, x1, x2 = box
    patch = np.ones((y2 - y1, x2 - x1, 3), dtype=np.uint8) * 75
    cv2.rectangle(patch, (15, 15), (105, 105), (240, 90, 90), -1)
    cv2.putText(patch, "SPLICE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 3. Bileşik görüntü oluşturulup orta kalitede (Q=90) tekrar JPEG kaydedilir
    composite = bg_compressed.copy()
    composite[y1:y2, x1:x2] = patch

    buf_final = io.BytesIO()
    Image.fromarray(composite).save(buf_final, format="JPEG", quality=90)
    buf_final.seek(0)
    final_img = np.array(Image.open(buf_final))

    gt = np.zeros((h, w), dtype=np.uint8)
    gt[y1:y2, x1:x2] = 255

    return final_img, gt, box


def generate_scenario_subtle_noise(
    size: tuple[int, int] = (360, 360),
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Senaryo C: Düşük / Zorlayıcı gürültü farkı (Arka plan sigma=3.0, nesne sigma=6.0)."""
    h, w = size
    np.random.seed(123)

    bg = np.ones((h, w, 3), dtype=np.float32) * 180
    bg += np.random.normal(0, 3.0, bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    box = (100, 220, 100, 220)
    y1, y2, x1, x2 = box
    patch = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32) * 110
    patch += np.random.normal(0, 6.0, patch.shape)
    patch = np.clip(patch, 0, 255).astype(np.uint8)

    spliced = bg.copy()
    spliced[y1:y2, x1:x2] = patch

    gt = np.zeros((h, w), dtype=np.uint8)
    gt[y1:y2, x1:x2] = 255

    return spliced, gt, box


def _format_metrics_line(name: str, m: Any) -> str:
    return (
        f"{name:<20} | P: {m.precision*100:5.2f}% | R: {m.recall*100:5.2f}% | "
        f"F1: {m.f1_score*100:5.2f}% | IoU: {m.iou*100:5.2f}% | "
        f"(TP={m.tp}, FP={m.fp}, FN={m.fn})"
    )


def build_comparison_board(
    image: np.ndarray,
    gt_mask: np.ndarray,
    ela_norm: np.ndarray,
    noise_norm: np.ndarray,
    ela_candidate: np.ndarray,
    noise_candidate: np.ndarray,
    fused_mask: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """8 panelli karşılaştırma görseli oluşturur (2 satır x 4 sütun)."""
    # 3 kanallı renkli formatlara dönüştür
    gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    ela_norm_bgr = cv2.cvtColor(ela_norm, cv2.COLOR_GRAY2BGR)
    noise_norm_bgr = cv2.cvtColor(noise_norm, cv2.COLOR_GRAY2BGR)
    ela_cand_bgr = cv2.cvtColor(ela_candidate, cv2.COLOR_GRAY2BGR)
    noise_cand_bgr = cv2.cvtColor(noise_candidate, cv2.COLOR_GRAY2BGR)
    fused_bgr = cv2.cvtColor(fused_mask, cv2.COLOR_GRAY2BGR)

    # Başlık ekleme yardımcısı
    def add_title(panel: np.ndarray, text: str) -> np.ndarray:
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (p.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(p, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return p

    p1 = add_title(image, "1. Girdi Goruntusu (Spliced)")
    p2 = add_title(gt_bgr, "2. Gercek Maske (Ground-Truth)")
    p3 = add_title(ela_norm_bgr, "3. ELA Fark Haritasi")
    p4 = add_title(noise_norm_bgr, "4. Gurultu Varyans Haritasi")

    p5 = add_title(ela_cand_bgr, "5. ELA Aday Maskesi")
    p6 = add_title(noise_cand_bgr, "6. Gurultu Aday Maskesi")
    p7 = add_title(fused_bgr, "7. Birlesik Aday Maske (Fuzyon)")
    p8 = add_title(overlay, "8. Nihai Adli Bindirme (Overlay)")

    row1 = np.hstack([p1, p2, p3, p4])
    row2 = np.hstack([p5, p6, p7, p8])
    return np.vstack([row1, row2])


def run_experiment() -> None:
    print("=" * 80)
    print("ForensiVision — Aşama 4: Image Splicing Analiz ve Doğrulama Deneyleri")
    print("=" * 80)

    scenarios = [
        ("Senaryo A: Belirgin Gürültü Tutarsızlığı", generate_scenario_high_noise()),
        ("Senaryo B: JPEG Sıkıştırma Tutarsızlığı (ELA Odaklı)", generate_scenario_jpeg_recompression()),
        ("Senaryo C: Düşük / Zorlayıcı Gürültü Tutarsızlığı", generate_scenario_subtle_noise()),
    ]

    config = SplicingConfig(
        ela_quality=90,
        ela_threshold_std_factor=2.0,
        noise_window_size=15,
        noise_threshold_std_factor=1.8,
        fusion_mode="union",
        min_region_area=60,
    )

    primary_board: np.ndarray | None = None

    for idx, (title, (img, gt, box)) in enumerate(scenarios, 1):
        print(f"\n[{idx}] {title}")
        print(f"    - Görüntü Boyutu: {img.shape[:2]}, Eklenen Bölge: {box}")

        # Analizleri çalıştır
        res_ela = analyze_ela(img, quality=config.ela_quality, threshold_std_factor=config.ela_threshold_std_factor)
        res_noise = analyze_noise_inconsistency(img, window_size=config.noise_window_size, threshold_std_factor=config.noise_threshold_std_factor)
        res_splicing = detect_splicing(img, config=config)

        # Metrikleri hesapla
        m_ela = calculate_metrics(res_ela.candidate_mask, gt)
        m_noise = calculate_metrics(res_noise.candidate_mask, gt)
        m_fused = calculate_metrics(res_splicing.mask, gt)

        print(f"    {_format_metrics_line('ELA Yöntemi', m_ela)}")
        print(f"    {_format_metrics_line('Gürültü Analizi', m_noise)}")
        print(f"    {_format_metrics_line('Birleşik (Füzyon)', m_fused)}")

        # İlk senaryo (Senaryo A) için görsel paneli oluştur
        if idx == 1:
            overlay = create_mask_overlay(img, res_splicing.mask, color=(0, 0, 255), alpha=0.45)
            primary_board = build_comparison_board(
                image=img,
                gt_mask=gt,
                ela_norm=res_ela.normalized_map,
                noise_norm=res_noise.normalized_map,
                ela_candidate=res_ela.candidate_mask,
                noise_candidate=res_noise.candidate_mask,
                fused_mask=res_splicing.mask,
                overlay=overlay,
            )

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    if primary_board is not None:
        out_path = output_dir / "synthetic_splicing_demo.png"
        cv2.imwrite(str(out_path), primary_board)
        print(f"\n[+] 8 Panelli Karşılaştırma Panosu Kaydedildi: {out_path.resolve()}")

    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
