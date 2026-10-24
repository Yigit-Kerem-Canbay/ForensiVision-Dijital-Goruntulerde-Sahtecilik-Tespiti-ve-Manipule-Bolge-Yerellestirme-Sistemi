"""
ForensiVision Kullanıcı Arayüzü Yardımcı Fonksiyonları (UI Helpers).

Bu modül; Streamlit arayüzünden bağımsız olarak girdi doğrulama, format dönüşümleri,
analiz motoru yürütme, metrik hesaplama ve dışa aktarım işlemlerini yönetir.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from forensivision.copy_move import CopyMoveConfig, CopyMoveResult, detect_copy_move
from forensivision.evaluation import calculate_metrics
from forensivision.splicing import SplicingConfig, SplicingResult, detect_splicing
from forensivision.visualization import create_mask_overlay, draw_copy_move_matches

# CPU güvenliği için maksimum piksel boyutu (Genişlik veya Yükseklik)
MAX_IMAGE_DIMENSION = 4096


def load_uploaded_image(
    file_bytes_or_buffer: bytes | io.BytesIO,
    filename: str = "",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Yüklenen ham dosya baytlarını okur, doğrular ve BGR ile RGB dizilerine dönüştürür.

    Args:
        file_bytes_or_buffer: Yüklenen dosyanın ham baytları veya BytesIO nesnesi.
        filename: Dosya adı (uzantı ve format tespiti için).

    Returns:
        tuple[np.ndarray, np.ndarray, str]:
            - image_bgr: OpenCV ve analiz motorları için 3 kanallı BGR NumPy dizisi (uint8).
            - image_rgb: Streamlit ve PIL görselleştirmesi için 3 kanallı RGB NumPy dizisi (uint8).
            - detected_format: 'JPEG', 'PNG', 'BMP' vb. format adı.

    Raises:
        ValueError: Dosya okunamıyorsa, boşsa veya desteklenen bir görsel formatı değilse.
    """
    if file_bytes_or_buffer is None:
        raise ValueError("Yüklenen dosya içeriği boş veya None.")

    raw_bytes: bytes
    if isinstance(file_bytes_or_buffer, io.BytesIO):
        raw_bytes = file_bytes_or_buffer.getvalue()
    elif isinstance(file_bytes_or_buffer, bytes):
        raw_bytes = file_bytes_or_buffer
    else:
        raise TypeError(f"Beklenen tip bytes veya BytesIO; alınan: {type(file_bytes_or_buffer).__name__}")

    if len(raw_bytes) == 0:
        raise ValueError("Yüklenen dosya 0 bayt (boş).")

    # 1. PIL ile format ve geçerlilik denetimi
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        detected_format = pil_img.format or "UNKNOWN"
    except Exception as e:
        raise ValueError(f"Yüklenen dosya geçerli bir görüntü dosyası olarak çözümlenemedi: {e}") from e

    # 2. OpenCV ile baytlardan okuma
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Görüntü OpenCV ile çözümlenemedi. Dosya bozuk veya desteklenmeyen bir biçimde.")

    h, w = image_bgr.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Görüntü boyutları ({w}x{h}) CPU işlem sınırını ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}) aşıyor. "
            "Lütfen daha uygun çözünürlükte bir görüntü yükleyiniz."
        )

    # 3. RGB dönüşümü
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_bgr, image_rgb, detected_format.upper()


def validate_and_load_ground_truth(
    mask_bytes_or_buffer: bytes | io.BytesIO | None,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    """İsteğe bağlı yüklenen doğruluk (ground-truth) maskesini okur ve boyutlarını denetler.

    Args:
        mask_bytes_or_buffer: Maske dosyasının ham baytları (yüklenmediyse None).
        expected_shape: Görüntünün (yükseklik, genişlik) boyutları.

    Returns:
        np.ndarray | None: (H, W) boyutunda 0 veya 255 ikili uint8 maske (yüklenmediyse None).

    Raises:
        ValueError: Maske okunamıyorsa veya boyutları görüntüyle uyuşmuyorsa.
    """
    if mask_bytes_or_buffer is None:
        return None

    raw_bytes: bytes
    if isinstance(mask_bytes_or_buffer, io.BytesIO):
        raw_bytes = mask_bytes_or_buffer.getvalue()
    elif isinstance(mask_bytes_or_buffer, bytes):
        raw_bytes = mask_bytes_or_buffer
    else:
        raise TypeError(f"Beklenen tip bytes veya BytesIO; alınan: {type(mask_bytes_or_buffer).__name__}")

    if len(raw_bytes) == 0:
        return None

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    mask_raw = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if mask_raw is None or mask_raw.size == 0:
        raise ValueError("Doğruluk maskesi okunamadı veya bozuk.")

    exp_h, exp_w = expected_shape[:2]
    mask_h, mask_w = mask_raw.shape[:2]

    if (mask_h, mask_w) != (exp_h, exp_w):
        raise ValueError(
            f"Doğruluk maskesi boyutları ({mask_w}x{mask_h}) orijinal görüntü boyutları ({exp_w}x{exp_h}) "
            "ile uyuşmuyor. Adli değerlendirmede maskeler sessizce yeniden boyutlandırılamaz."
        )

    # İkili formata çevir: 0 dışındaki tüm pikseller 255 (ön plan / manipüle bölge)
    binary_mask = (mask_raw > 0).astype(np.uint8) * 255
    return binary_mask


def run_copy_move_analysis(
    image_bgr: np.ndarray,
    config: CopyMoveConfig | None = None,
) -> tuple[CopyMoveResult, float]:
    """Copy-Move tespit motorunu çalıştırır ve işlem süresini ölçer."""
    t0 = time.perf_counter()
    result = detect_copy_move(image_bgr, config=config)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_splicing_analysis(
    image_bgr: np.ndarray,
    config: SplicingConfig | None = None,
) -> tuple[SplicingResult, float]:
    """Image Splicing analiz motorunu çalıştırır ve işlem süresini ölçer."""
    t0 = time.perf_counter()
    result = detect_splicing(image_bgr, config=config)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def compute_optional_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
) -> dict[str, Any] | None:
    """Gerçek doğruluk maskesi varsa piksel metriklerini hesaplar; yoksa None döner.

    Dürüst Metrik İlkesi:
        Kullanıcı bir doğruluk maskesi sağlamadığı sürece metrikler tahmin edilemez
        veya uydurulamaz. Bu durumda None döndürülerek arayüzde açık bir açıklama gösterilir.
    """
    if gt_mask is None:
        return None

    res = calculate_metrics(pred_mask, gt_mask)
    return {
        "precision": float(res.precision),
        "recall": float(res.recall),
        "f1_score": float(res.f1_score),
        "iou": float(res.iou),
        "tp": int(res.tp),
        "fp": int(res.fp),
        "fn": int(res.fn),
        "ground_truth_provided": True,
    }


def generate_export_payload(
    analysis_type: str,
    image_shape: tuple[int, ...],
    detected_format: str,
    copy_move_result: CopyMoveResult | None = None,
    splicing_result: SplicingResult | None = None,
    copy_move_time: float | None = None,
    splicing_time: float | None = None,
    copy_move_metrics: dict[str, Any] | None = None,
    splicing_metrics: dict[str, Any] | None = None,
    has_ground_truth: bool = False,
) -> str:
    """Analiz sonuçları, yapılandırma ve metrik durumunu içeren yapılandırılmış JSON metni üretir."""
    payload: dict[str, Any] = {
        "system": "ForensiVision",
        "version": "0.1.0",
        "analysis_type": analysis_type,
        "input_metadata": {
            "dimensions": {
                "height": image_shape[0],
                "width": image_shape[1],
                "channels": image_shape[2] if len(image_shape) > 2 else 1,
            },
            "format": detected_format,
            "has_ground_truth_mask": has_ground_truth,
        },
        "disclaimer": (
            "ForensiVision deneysel bir adli analiz prototipidir. Bu sonuçlar kesin adli kanıt teşkil etmez; "
            "uzman incelemesi gerektiren istatistiksel ve geometrik göstergelerdir."
        ),
    }

    if copy_move_result is not None:
        payload["copy_move_analysis"] = {
            "is_tampered": copy_move_result.is_tampered,
            "num_keypoints": copy_move_result.num_keypoints,
            "num_candidate_matches": copy_move_result.num_candidate_matches,
            "num_inliers": copy_move_result.num_inliers,
            "has_affine_matrix": copy_move_result.transform_matrix is not None,
            "execution_time_seconds": round(copy_move_time, 4) if copy_move_time else None,
            "metrics": copy_move_metrics if copy_move_metrics else "Doğruluk maskesi yüklenmediği için hesaplanamadı.",
            "config": copy_move_result.to_dict().get("config"),
        }

    if splicing_result is not None:
        payload["splicing_analysis"] = {
            "is_suspicious": splicing_result.is_suspicious,
            "fusion_mode": splicing_result.fusion_mode,
            "suspicious_pixel_count": int(np.count_nonzero(splicing_result.mask)),
            "execution_time_seconds": round(splicing_time, 4) if splicing_time else None,
            "ela_summary": splicing_result.ela_result.to_dict(),
            "noise_summary": splicing_result.noise_result.to_dict(),
            "metrics": splicing_metrics if splicing_metrics else "Doğruluk maskesi yüklenmediği için hesaplanamadı.",
            "config": splicing_result.to_dict().get("config"),
        }

    return json.dumps(payload, indent=2, ensure_ascii=False)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """İkili maske dizisini PNG formatında bayt dizisine dönüştürür."""
    success, buffer = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("Maske PNG olarak kodlanamadı.")
    return buffer.tobytes()


def image_to_png_bytes(image: np.ndarray, is_rgb: bool = False) -> bytes:
    """Görüntü dizisini PNG formatında bayt dizisine dönüştürür."""
    img_to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    success, buffer = cv2.imencode(".png", img_to_encode)
    if not success:
        raise RuntimeError("Görüntü PNG olarak kodlanamadı.")
    return buffer.tobytes()
