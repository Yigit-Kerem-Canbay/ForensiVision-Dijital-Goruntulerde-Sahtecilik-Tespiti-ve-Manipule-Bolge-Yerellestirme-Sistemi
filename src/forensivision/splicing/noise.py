"""
ForensiVision Gürültü Tutarsızlığı (Noise Inconsistency) Analiz Modülü.

Bu modül; görüntünün farklı bölgelerindeki yüksek frekanslı artık gürültü (noise residual)
istatistiklerini ve yerel varyans tutarsızlıklarını inceleyerek, başka bir kaynaktan
eklenen (spliced) nesne veya bölgeleri tespit etmeyi hedefler.

Teorik Temel:
    Her dijital kamera sensörü, ISO ayarı ve görüntü işleme boru hattı kendine has
    bir gürültü parmak izi (sensor pattern noise) bırakır. Farklı kameralardan çekilmiş
    veya farklı yazılımlarla işlenmiş iki görüntü birleştirildiğinde (splicing),
    eklenen bölgenin yerel gürültü varyansı arka plandan istatistiksel olarak sapma gösterir.

Önemli Sınırlamalar:
    - Gürültü farkları tek başına kesin sahtecilik kanıtı değildir.
    - Doğal nesne sınırları, yüksek kontrastlı dokular (örneğin saç, yapraklar, kum)
      ve keskin kenarlar da doğal olarak yüksek yerel varyans üretir.
    - Düz ve dokusuz alanlar (örneğin açık gökyüzü) ise düşük varyans sergiler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class NoiseAnalysisResult:
    """Gürültü tutarsızlığı analiz sonuçları veri yapısı.

    Attributes:
        residual_map: Orijinal görüntüden düşük frekans bileşeni çıkarılarak elde edilen yüksek frekanslı artık (float32, H x W).
        local_variance_map: Kayar pencere ile hesaplanan yerel gürültü standart sapma haritası (float32, H x W).
        normalized_map: İnsan gözüyle inceleme için ölçeklendirilmiş görsel gürültü haritası (uint8 0-255, H x W).
        candidate_mask: İstatiksel anomali eşiğiyle üretilmiş ikili aday manipülasyon maskesi (uint8 0 veya 255, H x W).
        window_size: Yerel istatistiklerin hesaplandığı kayar pencere kenar boyutu (piksel).
        global_noise_mean: Görüntü genelindeki ortalama yerel gürültü seviyesi.
        global_noise_std: Yerel gürültü seviyelerinin standart sapması.
        threshold_value: Aday maske üretiminde kullanılan sayısal sapma eşiği.
        status_message: Analiz durumu, kısıtlar ve istatistiksel özet.
    """

    residual_map: np.ndarray
    local_variance_map: np.ndarray
    normalized_map: np.ndarray
    candidate_mask: np.ndarray
    window_size: int
    global_noise_mean: float
    global_noise_std: float
    threshold_value: float
    status_message: str

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir özet sözlük döndürür."""
        return {
            "window_size": self.window_size,
            "global_noise_mean": round(self.global_noise_mean, 4),
            "global_noise_std": round(self.global_noise_std, 4),
            "threshold_value": round(self.threshold_value, 4),
            "candidate_pixels": int(np.count_nonzero(self.candidate_mask)),
            "status_message": self.status_message,
        }


def _prepare_grayscale_image(image: Any) -> np.ndarray:
    """Görüntüyü doğrular ve 2D float32 gri seviye matrisine dönüştürür."""
    if image is None:
        raise ValueError("Girdi görüntüsü (image) None olamaz.")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Girdi görüntüsü bir NumPy dizisi olmalıdır; alınan tip: {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Girdi görüntüsü boş bir dizi olamaz.")

    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Görüntü sayısal veri tipinde olmalıdır; alınan tip: {image.dtype}")

    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2GRAY).astype(np.float32)
        else:
            raise ValueError(f"Desteklenmeyen kanal sayısı: {image.shape[2]}")
    else:
        raise ValueError(f"Desteklenmeyen görüntü boyutu: {image.ndim}D. 2D veya 3D olmalıdır.")

    return gray


def analyze_noise_inconsistency(
    image: Any,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
    window_size: int = 15,
    threshold_std_factor: float = 2.0,
    deviation_mode: str = "both",
) -> NoiseAnalysisResult:
    """Görüntüdeki yüksek frekanslı artık gürültü varyansını inceleyerek olası tutarsızlıkları analiz eder.

    Args:
        image: Analiz edilecek görüntü (2D gri tonlama veya 3D renkli NumPy dizisi).
        blur_kernel_size: Düşük frekans kestirimi için Gaussian bulanıklaştırma çekirdek boyutu (tek sayı).
        blur_sigma: Gaussian bulanıklaştırma standart sapması.
        window_size: Yerel varyans hesaplanan kayar pencere boyutu (tek sayı, varsayılan 15).
        threshold_std_factor: Aday maske eşiği için standart sapma çarpanı (|noise - mean| > factor * std).
        deviation_mode: Anomali arama yönü: 'both' (hem aşırı gürültülü hem aşırı pürüzsüz),
                        'higher' (yalnızca daha gürültülü), 'lower' (yalnızca daha pürüzsüz).

    Returns:
        NoiseAnalysisResult: Artık haritası, yerel varyans, normalize görsel ve aday maskeyi içeren sonuç.

    Raises:
        ValueError: Girdi geçersiz, boş veya parametreler uygunsuzsa.
        TypeError: Girdi sayısal bir NumPy dizisi değilse.
    """
    if blur_kernel_size % 2 == 0 or blur_kernel_size < 3:
        raise ValueError(f"blur_kernel_size 3 veya daha büyük tek sayı olmalıdır; alınan: {blur_kernel_size}")

    if window_size % 2 == 0 or window_size < 3:
        raise ValueError(f"window_size 3 veya daha büyük tek sayı olmalıdır; alınan: {window_size}")

    if deviation_mode not in ("both", "higher", "lower"):
        raise ValueError(f"Geçersiz deviation_mode: '{deviation_mode}'. 'both', 'higher' veya 'lower' beklenir.")

    gray = _prepare_grayscale_image(image)
    h, w = gray.shape

    # 1. Düşük Frekans Bileşeni Kestirimi ve Yüksek Frekans Artık Haritası
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), blur_sigma)
    residual = gray - blurred  # (H, W) float32

    # Sayısal kararlılık: NaN veya sonsuz değerleri sıfırla
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Kayar Pencere (BoxFilter) ile Yerel Standart Sapma Haritası
    ksize = (window_size, window_size)
    local_mean = cv2.boxFilter(residual, -1, ksize, borderType=cv2.BORDER_REFLECT)
    local_mean_sq = cv2.boxFilter(residual**2, -1, ksize, borderType=cv2.BORDER_REFLECT)

    local_variance = np.maximum(local_mean_sq - (local_mean**2), 0.0)
    local_std = np.sqrt(local_variance)
    local_std = np.nan_to_num(local_std, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. İstatistiksel Dağılım Hesabı
    global_mean = float(np.mean(local_std))
    global_std = float(np.std(local_std))

    # 4. Görselleştirme İçin Normalize Edilmiş Harita [0, 255]
    max_val = float(np.percentile(local_std, 99.0)) if local_std.size > 0 else 1.0
    if max_val <= 1e-6:
        max_val = 1.0
    normalized_map = np.clip((local_std / max_val) * 255.0, 0.0, 255.0).astype(np.uint8)

    # 5. Anomali Adayı Eşikleme
    threshold_diff = threshold_std_factor * global_std
    effective_threshold = max(threshold_diff, 1.5)  # Aşırı düşük gürültülü görüntülerde taban eşik

    if deviation_mode == "higher":
        raw_mask = (local_std > (global_mean + effective_threshold))
    elif deviation_mode == "lower":
        raw_mask = (local_std < max(global_mean - effective_threshold, 0.0))
    else:  # 'both'
        raw_mask = np.abs(local_std - global_mean) > effective_threshold

    candidate_mask = raw_mask.astype(np.uint8) * 255

    # Kenarlardaki izole piksel gürültülerini temizle ve bölgeyi toparla
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    status = (
        f"Gürültü analizi tamamlandı (Pencere={window_size}x{window_size}, Mod='{deviation_mode}'). "
        f"Ortalama yerel gürültü: {global_mean:.2f}, Std: {global_std:.2f}, Eşik sapması: {effective_threshold:.2f}. "
        "UYARI: Gürültü tutarsızlıkları yalnızca yapıştırma sahteciliğinden değil; "
        "doğal yüksek dokulu veya keskin nesnelerden de kaynaklanabilir."
    )

    return NoiseAnalysisResult(
        residual_map=residual,
        local_variance_map=local_std,
        normalized_map=normalized_map,
        candidate_mask=candidate_mask,
        window_size=window_size,
        global_noise_mean=global_mean,
        global_noise_std=global_std,
        threshold_value=effective_threshold,
        status_message=status,
    )
