"""
ForensiVision Veri Seti Temel Modülü (Dataset Abstractions).

Bu modül; farklı akademik adli bilişim veri setlerini (CoMoFoD, Columbia, CASIA, DEFACTO vb.)
ortak bir veri modeli altında birleştiren `DatasetSample` ve `BaseDataset` sınıflarını tanımlar.

Tasarım İlkeleri:
    - Veri sızıntısını (data leakage) önlemek için aynı temel görüntüye ait varyantları
      gruplayan `group_id` desteği.
    - Copy-Move ve Splicing için açık `task_type` ('copy_move' | 'splicing').
    - Kaynak ve hedef bölgelerin semantiğini yöneten `mask_mode` ('both', 'target_only', 'source_only').
    - Görüntü ve maske boyut uyuşmazlığında sessizce boyut değiştirmek yerine açıklayıcı hata denetimi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DatasetSample:
    """Tekil bir adli veri seti örneği veri yapısı.

    Attributes:
        sample_id: Örnek için benzersiz tanımlayıcı (örn. 'comofod_001_F_JC1').
        task_type: Manipülasyon kategorisi ('copy_move' veya 'splicing').
        image_path: İncelenecek görüntünün dosya yolu.
        mask_path: Gerçek manipülasyon maskesinin dosya yolu (özgün görüntülerde None olabilir).
        authentic_path: Varsa orijinal/temiz karşılık görüntüsünün dosya yolu.
        source_mask_path: Varsa yalnızca kopyalanan (kaynak/donor) bölge maskesinin yolu.
        target_mask_path: Varsa yalnızca yapıştırılan (hedef/probe) bölge maskesinin yolu.
        manipulation_type: Örnek türü ('copy_move', 'splicing', 'authentic').
        variant: Uygulanan saldırı/dönüşüm türü (örn. 'jpeg_compression', 'rotation', 'clean').
        group_id: Aynı temel görselden türetilen varyantları bağlayan grup kimliği (örn. '001').
        metadata: Ek veri seti metaverileri.
    """

    sample_id: str
    task_type: str
    image_path: Path
    mask_path: Path | None = None
    authentic_path: Path | None = None
    source_mask_path: Path | None = None
    target_mask_path: Path | None = None
    manipulation_type: str = "manipulated"
    variant: str | None = None
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_authentic(self) -> bool:
        """Örneğin manipüle edilmemiş temiz/orijinal bir görüntü olup olmadığını döndürür."""
        return self.manipulation_type == "authentic"

    def to_dict(self) -> dict[str, Any]:
        """Serileştirilebilir sözlük çıktısı üretir."""
        res = asdict(self)
        res["image_path"] = str(self.image_path)
        res["mask_path"] = str(self.mask_path) if self.mask_path else None
        res["authentic_path"] = str(self.authentic_path) if self.authentic_path else None
        res["source_mask_path"] = str(self.source_mask_path) if self.source_mask_path else None
        res["target_mask_path"] = str(self.target_mask_path) if self.target_mask_path else None
        return res


def _safe_read_image(file_path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Windows ve POSIX sistemlerde Türkçe veya Unicode karakterli dosya yollarını güvenle okur."""
    p = Path(file_path)
    if not p.is_file():
        return None
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            return None
        decoded = cv2.imdecode(buf, flags)
        if decoded is not None:
            return decoded
    except Exception:
        pass
    return cv2.imread(str(p), flags)


class BaseDataset(ABC):
    """Tüm adli veri seti okuyucuları için ortak soyut temel sınıf."""

    def __init__(self, root_dir: str | Path, name: str, task_type: str) -> None:
        self.root_dir = Path(root_dir)
        self.name = name
        self.task_type = task_type
        self._samples: list[DatasetSample] = []
        self._is_loaded = False

    @abstractmethod
    def load_samples(self) -> list[DatasetSample]:
        """Veri setindeki dosya yapısını tarar ve DatasetSample listesini döndürür."""
        pass

    @property
    def samples(self) -> list[DatasetSample]:
        """Taranan veri örneklerinin listesi."""
        if not self._is_loaded:
            self._samples = self.load_samples()
            self._is_loaded = True
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetSample:
        return self.samples[idx]

    def get_sample(self, sample_id: str) -> DatasetSample:
        """Kimliğe göre tek bir örneği bulur."""
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        raise KeyError(f"Örnek kimliği bulunamadı: {sample_id}")

    def load_image(self, sample: DatasetSample) -> np.ndarray:
        """Görüntü dosyasını 3 kanallı BGR NumPy dizisi olarak okur."""
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Görüntü dosyası mevcut değil: {sample.image_path}")

        img = _safe_read_image(sample.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Görüntü dosyası okunamadı veya bozuk: {sample.image_path}")
        return img

    def load_mask(
        self,
        sample: DatasetSample,
        mask_mode: str = "both",
        target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Doğruluk (ground-truth) maskesini ikili formda (uint8, 0 ve 255) yükler.

        Args:
            sample: İncelenen veri örneği.
            mask_mode: 'both' (kaynak + hedef), 'target_only' (yalnızca yapıştırılan) veya
                       'source_only' (yalnızca kopyalanan).
            target_shape: Görüntü ile boyut doğrulaması için (H, W) tuple.

        Returns:
            np.ndarray: 2D uint8 ikili maske (0 arka plan, 255 manipüle).

        Raises:
            ValueError: Boyut uyuşmazlığı varsa veya maske okunamıyorsa.
        """
        has_any_mask = bool(sample.mask_path or sample.target_mask_path or sample.source_mask_path)
        # Temiz/Orijinal görüntülerde tüm pikseller 0'dır
        if sample.is_authentic() or not has_any_mask:
            if target_shape is None:
                img = self.load_image(sample)
                target_shape = img.shape[:2]
            return np.zeros(target_shape, dtype=np.uint8)

        # DEFACTO gibi hem kaynak hem hedef maskesi ayrı dosyalar olan veri setlerinde 'both' modu:
        if (
            mask_mode == "both"
            and sample.target_mask_path
            and sample.source_mask_path
            and sample.target_mask_path.exists()
            and sample.source_mask_path.exists()
        ):
            tgt_raw = _safe_read_image(sample.target_mask_path, cv2.IMREAD_GRAYSCALE)
            src_raw = _safe_read_image(sample.source_mask_path, cv2.IMREAD_GRAYSCALE)
            if tgt_raw is None or src_raw is None:
                raise ValueError("Kaynak veya hedef maske dosyası gri seviyede okunamadı.")

            # JPEG sıkıştırma artefaktlarını (probe) veya yumuşatmayı (donor feathering) temizle
            tgt_bin = (tgt_raw > 127) if tgt_raw.max() > 127 else (tgt_raw > 0)
            src_bin = (src_raw > 127) if src_raw.max() > 127 else (src_raw > 0)
            bin_mask = ((tgt_bin | src_bin).astype(np.uint8)) * 255

        else:
            # Tek bir maske dosyası yükleme (CoMoFoD birleşik maskesi veya tekil maske seçimi)
            mask_file_to_load = sample.mask_path
            if mask_mode == "target_only" and sample.target_mask_path and sample.target_mask_path.exists():
                mask_file_to_load = sample.target_mask_path
            elif mask_mode == "source_only" and sample.source_mask_path and sample.source_mask_path.exists():
                mask_file_to_load = sample.source_mask_path
            elif mask_file_to_load is None:
                mask_file_to_load = sample.target_mask_path or sample.source_mask_path

            if mask_file_to_load is None or not mask_file_to_load.exists():
                raise FileNotFoundError(f"Maske dosyası mevcut değil: {mask_file_to_load}")

            mask_raw = _safe_read_image(mask_file_to_load, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                raise ValueError(f"Maske dosyası gri seviyede okunamadı: {mask_file_to_load}")

            # JPEG sıkıştırma artefaktları varsa (>127), yoksa pozitif piksel (>0)
            if mask_raw.max() > 127:
                bin_mask = (mask_raw > 127).astype(np.uint8) * 255
            else:
                bin_mask = (mask_raw > 0).astype(np.uint8) * 255

        # Boyut uyuşmazlığı denetimi
        if target_shape is not None and bin_mask.shape != target_shape:
            raise ValueError(
                f"Görüntü boyutu ile maske boyutu uyuşmuyor: "
                f"Görüntü={target_shape}, Maske={bin_mask.shape}. "
                "Adli değerlendirmede maskeler sessizce yeniden boyutlandırılamaz."
            )

        return bin_mask

    def get_groups(self) -> dict[str, list[DatasetSample]]:
        """Örnekleri `group_id` bazında sözlükte toplar (veri sızıntısını önlemek için)."""
        groups: dict[str, list[DatasetSample]] = {}
        for s in self.samples:
            gid = s.group_id or s.sample_id
            groups.setdefault(gid, []).append(s)
        return groups

    def summary(self) -> dict[str, Any]:
        """Veri setinin dağılımı hakkında özet bilgi döndürür."""
        total = len(self.samples)
        authentic_count = sum(1 for s in self.samples if s.is_authentic())
        manipulated_count = total - authentic_count
        groups = self.get_groups()

        return {
            "name": self.name,
            "task_type": self.task_type,
            "root_dir": str(self.root_dir),
            "is_available": self.root_dir.is_dir() and total > 0,
            "total_samples": total,
            "authentic_samples": authentic_count,
            "manipulated_samples": manipulated_count,
            "unique_groups": len(groups),
        }
