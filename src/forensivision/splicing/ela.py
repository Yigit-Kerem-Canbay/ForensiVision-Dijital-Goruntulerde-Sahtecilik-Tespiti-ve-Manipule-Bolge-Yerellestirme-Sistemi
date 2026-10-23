"""
ForensiVision Error Level Analysis (ELA) Modülü.

Error Level Analysis (Hata Seviyesi Analizi), bir görüntünün belirli bir kalitede
tekrar JPEG formatında sıkıştırılmasıyla orijinal görüntü arasındaki piksel farklarını
inceleyen adli bir yöntemdir.

Teorik Temel:
    JPEG sıkıştırması kayıplı (lossy) bir algoritmadır ve her sıkıştırma döngüsünde
    hata oranı belirli bir dengeye (equilibrium) ulaşır. Farklı bir kaynaktan veya
    farklı sıkıştırma geçmişine sahip bir görüntüden eklenen (spliced) nesneler,
    yeniden sıkıştırıldığında görüntünün geri kalanına göre daha yüksek veya farklı
    bir hata seviyesi sergileyebilir.

Önemli Sınırlamalar:
    - ELA tek başına kesin bir sahtecilik kanıtı değildir.
    - Yüksek kontrastlı doğal kenarlar veya parlak dokular da doğal olarak yüksek ELA değeri üretebilir.
    - Görüntünün JPEG formatında olmaması veya birden çok kez yeniden kaydedilmiş olması
      analizin güvenilirliğini doğrudan etkiler.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ELAResult:
    """Error Level Analysis (ELA) analiz sonuçları veri yapısı.

    Attributes:
        raw_difference: Orijinal ile yeniden sıkıştırılmış görüntü arasındaki ham mutlak fark (float32, H x W).
        normalized_map: İnsan gözüyle inceleme için ölçeklendirilmiş fark haritası (uint8 0-255, H x W).
        candidate_mask: Eşikleme ile belirlenmiş olası anomali adayı ikili maske (uint8 0 veya 255, H x W).
        is_applicable: ELA yönteminin bu girdi için geçerli olup olmadığı (JPEG doğrulaması).
        quality: Analizde kullanılan JPEG yeniden sıkıştırma kalite faktörü (örn. 90).
        scale_multiplier: Görsel harita için kullanılan parlaklık çarpanı.
        mean_difference: Görüntü genelindeki ortalama mutlak hata.
        std_difference: Hata seviyesinin standart sapması.
        threshold_value: Aday maske üretiminde kullanılan sayısal eşik.
        status_message: Analiz durumu, kısıtlar ve uyarılar hakkında açıklama.
    """

    raw_difference: np.ndarray
    normalized_map: np.ndarray
    candidate_mask: np.ndarray
    is_applicable: bool
    quality: int
    scale_multiplier: float
    mean_difference: float
    std_difference: float
    threshold_value: float
    status_message: str

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir özet sözlük döndürür."""
        return {
            "is_applicable": self.is_applicable,
            "quality": self.quality,
            "scale_multiplier": self.scale_multiplier,
            "mean_difference": round(self.mean_difference, 4),
            "std_difference": round(self.std_difference, 4),
            "threshold_value": round(self.threshold_value, 4),
            "candidate_pixels": int(np.count_nonzero(self.candidate_mask)),
            "status_message": self.status_message,
        }


def _is_jpeg_file(file_path: Path) -> bool:
    """Dosyanın gerçek bir JPEG formatında olup olmadığını sihirli baytlar (magic bytes) ile denetler."""
    if not file_path.is_file():
        return False
    try:
        with open(file_path, "rb") as f:
            header = f.read(3)
            return header == b"\xff\xd8\xff"
    except Exception:
        return False


def analyze_ela(
    image_input: str | Path | np.ndarray,
    quality: int = 90,
    scale_multiplier: float = 15.0,
    threshold_std_factor: float = 2.0,
    is_jpeg: bool | None = None,
) -> ELAResult:
    """Görüntü üzerinde Error Level Analysis (ELA) gerçekleştirir.

    Args:
        image_input: Görüntü dosya yolu (str/Path) veya görüntü dizisi (np.ndarray).
        quality: Yeniden sıkıştırma JPEG kalite faktörü (1-100 arası, varsayılan 90).
        scale_multiplier: Görsel harita için parlaklık ölçek çarpanı (varsayılan 15.0).
        threshold_std_factor: Aday maske eşiği için standart sapma katsayısı (mean + factor * std).
        is_jpeg: Girdinin JPEG olup olmadığını manuel belirtmek için (None ise otomatik tespit edilir).

    Returns:
        ELAResult: Ham fark, normalize harita, aday maske ve istatistikleri içeren sonuç.

    Raises:
        FileNotFoundError: Belirtilen dosya yolu mevcut değilse.
        ValueError: Girdi boş, geçersiz boyutlu veya None ise.
        TypeError: Girdi türü desteklenmiyorsa.
    """
    if image_input is None:
        raise ValueError("image_input None olamaz.")

    if not (1 <= quality <= 100):
        raise ValueError(f"quality parametresi 1 ile 100 arasında olmalıdır; alınan: {quality}")

    file_path: Path | None = None
    orig_bgr: np.ndarray

    # 1. Girdi Türü Kontrolü ve Görüntünün Yüklenmesi
    if isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {file_path}")

        detected_jpeg = _is_jpeg_file(file_path)
        actual_is_jpeg = detected_jpeg if is_jpeg is None else is_jpeg

        # Görüntüyü oku
        loaded_img = cv2.imread(str(file_path))
        if loaded_img is None:
            raise ValueError(f"Görüntü dosyası OpenCV ile okunamadı veya bozuk: {file_path}")
        orig_bgr = loaded_img
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("image_input boş bir dizi olamaz.")
        if not np.issubdtype(image_input.dtype, np.number):
            raise TypeError(f"Görüntü dizisi sayısal olmalıdır; alınan tip: {image_input.dtype}")
        if image_input.ndim not in (2, 3):
            raise ValueError(f"Desteklenmeyen görüntü boyutu: {image_input.ndim}D. 2D veya 3D olmalıdır.")

        if image_input.ndim == 2:
            orig_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            orig_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        elif image_input.shape[2] == 3:
            orig_bgr = image_input.copy()
        else:
            raise ValueError(f"Desteklenmeyen kanal sayısı: {image_input.shape[2]}")

        actual_is_jpeg = True if is_jpeg is None else is_jpeg
    else:
        raise TypeError(f"Desteklenmeyen girdi türü: {type(image_input).__name__}. Dosya yolu veya np.ndarray beklenir.")

    h, w = orig_bgr.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    empty_diff = np.zeros((h, w), dtype=np.float32)

    # 2. JPEG Olmayan Girdilerde Kontrollü Dönüş
    if not actual_is_jpeg:
        return ELAResult(
            raw_difference=empty_diff,
            normalized_map=empty_mask,
            candidate_mask=empty_mask,
            is_applicable=False,
            quality=quality,
            scale_multiplier=scale_multiplier,
            mean_difference=0.0,
            std_difference=0.0,
            threshold_value=0.0,
            status_message=(
                "ELA uygulanamaz: Girdi JPEG formatında değildir veya dosya kodlama geçmişi "
                "doğrulanamamıştır. Kayıpsız formatlarda (PNG, BMP vb.) ELA yanıltıcı sonuçlar üretir."
            ),
        )

    # 3. Bellek İçi Yeniden Sıkıştırma (Memory Buffer Recompression)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(orig_rgb)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed_rgb = np.array(Image.open(buffer), dtype=np.float32)

    # 4. Mutlak Piksel Farkı (ELA Fark Haritası)
    # Kanallar bazında mutlak farkın ortalamasını al
    diff_channels = np.abs(orig_rgb.astype(np.float32) - recompressed_rgb)
    raw_diff = np.mean(diff_channels, axis=2)  # (H, W) float32

    mean_diff = float(np.mean(raw_diff))
    std_diff = float(np.std(raw_diff))

    # 5. Görselleştirme İçin Normalize Edilmiş Kopya
    # Belirli bir çarpan ile ölçeklendir ve [0, 255] uint8 aralığına sıkıştır
    normalized_map = np.clip(raw_diff * scale_multiplier, 0.0, 255.0).astype(np.uint8)

    # 6. Anomali Adayı Eşikleme (Aday Maske)
    # Ortalama hata seviyesinin 'threshold_std_factor' standart sapma üzerindeki pikselleri şüpheli kabul et
    threshold_val = mean_diff + (threshold_std_factor * std_diff)

    # Çok düşük genel hata seviyelerinde gereksiz sahte alarmı önlemek için taban eşik (en az 4.0 piksel farkı)
    effective_threshold = max(threshold_val, 4.0)

    candidate_mask = (raw_diff > effective_threshold).astype(np.uint8) * 255

    # Küçük tekil gürültü piksellerini elemek için morfolojik açma (opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)

    status = (
        f"ELA başarıyla uygulandı (Kalite={quality}). Ortalama hata: {mean_diff:.2f}, "
        f"Std: {std_diff:.2f}, Eşik: {effective_threshold:.2f}. "
        "UYARI: ELA parlaklığı tek başına sahtecilik kanıtı değildir; yüksek dokulu doğal bölgeleri de işaretleyebilir."
    )

    return ELAResult(
        raw_difference=raw_diff,
        normalized_map=normalized_map,
        candidate_mask=candidate_mask,
        is_applicable=True,
        quality=quality,
        scale_multiplier=scale_multiplier,
        mean_difference=mean_diff,
        std_difference=std_diff,
        threshold_value=effective_threshold,
        status_message=status,
    )
