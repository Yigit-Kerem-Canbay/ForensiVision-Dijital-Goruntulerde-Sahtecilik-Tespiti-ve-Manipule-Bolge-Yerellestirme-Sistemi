"""
ForensiVision Image Splicing Tespit ve Füzyon Modülü.

Bu modül; Error Level Analysis (ELA) ve Gürültü Tutarsızlığı (Noise Inconsistency)
yöntemlerini bağımsız veya birleşik (füzyon) olarak yürüterek görüntüye dışarıdan
eklenen şüpheli bölgeleri yerelleştirir.

Önemli Uyarı:
    Bu analizler kesin bir sahtecilik kanıtı değil; olası istatistiksel ve sıkıştırma
    tutarsızlıklarını gösteren deneysel bir adli analiz aracıdır.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from forensivision.splicing.ela import ELAResult, analyze_ela
from forensivision.splicing.noise import NoiseAnalysisResult, analyze_noise_inconsistency


@dataclass
class SplicingConfig:
    """Image Splicing analiz motoru yapılandırma parametreleri.

    Attributes:
        ela_quality: ELA için yeniden sıkıştırma JPEG kalite faktörü (varsayılan 90).
        ela_multiplier: ELA görsel haritası için parlaklık çarpanı (varsayılan 15.0).
        ela_threshold_std_factor: ELA anomali eşiği için standart sapma çarpanı (varsayılan 2.0).
        noise_blur_kernel: Gürültü analizi Gaussian filtre çekirdek boyutu (varsayılan 5).
        noise_blur_sigma: Gürültü analizi Gaussian filtre standart sapması (varsayılan 1.0).
        noise_window_size: Yerel varyans hesaplanan kayar pencere boyutu (varsayılan 15).
        noise_threshold_std_factor: Gürültü anomali eşiği için standart sapma çarpanı (varsayılan 2.0).
        noise_deviation_mode: Gürültü anomali yönü ('both', 'higher', 'lower').
        fusion_mode: Maske birleştirme stratejisi ('union', 'intersection', 'ela_only', 'noise_only').
        min_region_area: Küçük izole sahte alarmları eleyen minimum piksel alanı (varsayılan 80).
        morph_kernel_size: Maske toparlama için morfolojik çekirdek boyutu (varsayılan 5).
    """

    # ELA parametreleri
    ela_quality: int = 90
    ela_multiplier: float = 15.0
    ela_threshold_std_factor: float = 2.0

    # Gürültü parametreleri
    noise_blur_kernel: int = 5
    noise_blur_sigma: float = 1.0
    noise_window_size: int = 15
    noise_threshold_std_factor: float = 2.0
    noise_deviation_mode: str = "both"

    # Füzyon ve temizleme parametreleri
    fusion_mode: str = "union"
    min_region_area: int = 80
    morph_kernel_size: int = 5


@dataclass(frozen=True)
class SplicingResult:
    """Image Splicing analiz sonuçları veri yapısı.

    Attributes:
        mask: Birleşik veya seçili adli aday maske (uint8 0 veya 255, H x W).
        is_suspicious: Yeterli tutarsızlık/şüphe alanı bulunup bulunmadığı.
        ela_result: Error Level Analysis detay sonuç nesnesi.
        noise_result: Gürültü tutarsızlığı detay sonuç nesnesi.
        fusion_mode: Kullanılan maske birleştirme stratejisi.
        config: Kullanılan yapılandırma ayarları.
        metadata: Analiz hakkında açıklayıcı adli metaveriler.
    """

    mask: np.ndarray
    is_suspicious: bool
    ela_result: ELAResult
    noise_result: NoiseAnalysisResult
    fusion_mode: str
    config: SplicingConfig
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir özet sözlük döndürür."""
        return {
            "is_suspicious": self.is_suspicious,
            "fusion_mode": self.fusion_mode,
            "mask_shape": list(self.mask.shape),
            "suspicious_pixel_count": int(np.count_nonzero(self.mask)),
            "ela_summary": self.ela_result.to_dict(),
            "noise_summary": self.noise_result.to_dict(),
            "metadata": self.metadata,
            "config": asdict(self.config),
        }


def _filter_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Belirli bir piksel alanından küçük bağımsız adacıkları (bağlantılı bileşenler) eler."""
    if min_area <= 0 or np.count_nonzero(mask) == 0:
        return mask

    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # 0 arka plandır
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == label] = 255

    return filtered_mask


def detect_splicing(
    image_input: str | Path | np.ndarray,
    config: SplicingConfig | None = None,
) -> SplicingResult:
    """Görüntüde Image Splicing tahrifatını ELA ve Gürültü Tutarsızlığı ile inceler.

    Args:
        image_input: Dosya yolu (str/Path) veya görüntü matrisi (np.ndarray).
        config: İsteğe bağlı yapılandırma nesnesi (SplicingConfig).

    Returns:
        SplicingResult: Aday maske, ELA ve gürültü analizleri ile durum metaverileri.

    Raises:
        FileNotFoundError: Dosya yolu bulunamazsa.
        ValueError: Girdi boş veya geçersiz boyutluysa.
        TypeError: Desteklenmeyen girdi türü ise.
    """
    if config is None:
        config = SplicingConfig()

    valid_fusion_modes = ("union", "intersection", "ela_only", "noise_only")
    if config.fusion_mode not in valid_fusion_modes:
        raise ValueError(
            f"Geçersiz fusion_mode: '{config.fusion_mode}'. Beklenen: {valid_fusion_modes}"
        )

    # 1. ELA Analizini Çalıştır
    ela_res = analyze_ela(
        image_input=image_input,
        quality=config.ela_quality,
        scale_multiplier=config.ela_multiplier,
        threshold_std_factor=config.ela_threshold_std_factor,
    )

    # 2. Görüntüyü Dizi Olarak Al ve Gürültü Analizini Çalıştır
    if isinstance(image_input, (str, Path)):
        img_arr = cv2.imread(str(image_input))
        if img_arr is None:
            raise ValueError(f"Görüntü dosyası okunamadı: {image_input}")
    else:
        img_arr = image_input

    noise_res = analyze_noise_inconsistency(
        image=img_arr,
        blur_kernel_size=config.noise_blur_kernel,
        blur_sigma=config.noise_blur_sigma,
        window_size=config.noise_window_size,
        threshold_std_factor=config.noise_threshold_std_factor,
        deviation_mode=config.noise_deviation_mode,
    )

    h, w = noise_res.candidate_mask.shape

    # 3. Maske Birleştirme (Füzyon) Stratejisi
    ela_mask_active = ela_res.is_applicable and (np.count_nonzero(ela_res.candidate_mask) > 0)

    if config.fusion_mode == "ela_only":
        if ela_res.is_applicable:
            fused_mask = ela_res.candidate_mask.copy()
        else:
            fused_mask = np.zeros((h, w), dtype=np.uint8)
    elif config.fusion_mode == "noise_only":
        fused_mask = noise_res.candidate_mask.copy()
    elif config.fusion_mode == "intersection":
        if ela_res.is_applicable:
            # Hem ELA hem gürültü tutarsızlığı aynı anda görülen kesişim pikselleri
            fused_mask = cv2.bitwise_and(ela_res.candidate_mask, noise_res.candidate_mask)
        else:
            # ELA uygulanamıyorsa güvenli geri çekilme olarak gürültü maskesini baz al
            fused_mask = noise_res.candidate_mask.copy()
    else:  # 'union' (varsayılan)
        if ela_res.is_applicable:
            fused_mask = cv2.bitwise_or(ela_res.candidate_mask, noise_res.candidate_mask)
        else:
            fused_mask = noise_res.candidate_mask.copy()

    # 4. Morfolojik Toparlama ve Küçük Parçaları Filtreleme
    if config.morph_kernel_size > 1 and np.count_nonzero(fused_mask) > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
        )
        fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, kernel)

    final_mask = _filter_small_regions(fused_mask, config.min_region_area)
    is_suspicious = bool(np.count_nonzero(final_mask) > 0)

    metadata: dict[str, Any] = {
        "ela_applicable": ela_res.is_applicable,
        "ela_status": ela_res.status_message,
        "noise_status": noise_res.status_message,
        "warning": (
            "Bu analiz kesin sahtecilik hükmü değil; olası istatistiksel ve sıkıştırma "
            "tutarsızlıklarını gösteren deneysel bir adli tespittir."
        ),
    }

    return SplicingResult(
        mask=final_mask,
        is_suspicious=is_suspicious,
        ela_result=ela_res,
        noise_result=noise_res,
        fusion_mode=config.fusion_mode,
        config=config,
        metadata=metadata,
    )
