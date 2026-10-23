"""
ForensiVision CoMoFoD Veri Seti Okuyucusu (University of Zagreb).

CoMoFoD (Copy-Move Forgery Database), Copy-Move sahteciliği tespiti için
akademik referans veri setlerinden biridir.

Dosya Adlandırma Kuralları:
    - {ID}_O.png     : Orijinal / Temiz görüntü (Authentic)
    - {ID}_F.png     : Temel Copy-Move uygulanmış sahte görüntü (Forged - Clean)
    - {ID}_B.png     : İkili doğruluk maskesi (Binary ground-truth; kaynak ve hedefi kapsar)
    - {ID}_M.png     : Renkli doğruluk maskesi (Color ground-truth)

Dönüşüm ve Saldırı (Post-processing) Varyantları:
    - BC (Brightness Change) : Parlaklık değişimi
    - CA (Contrast Adjustment): Kontrast ayarı
    - CR (Color Reduction)   : Renk indirgeme
    - IB (Image Blurring)    : Bulanıklaştırma (Gaussian vb.)
    - JC (JPEG Compression)  : JPEG sıkıştırması
    - NA (Noise Addition)    : Gürültü ekleme
    - RT (Rotation)          : Döndürme
    - SC (Scale)             : Ölçekleme
    - DS (Distortion)        : Bozulma
    - CB (Combination)       : Karma dönüşümler
"""

from __future__ import annotations

import re
from pathlib import Path

from forensivision.datasets.base import BaseDataset, DatasetSample

# Varyant kodları ile insan tarafından okunabilir etiketlerin eşlemesi
COMOFOD_ATTACK_MAP = {
    "BC": "brightness_change",
    "CA": "contrast_adjustment",
    "CR": "color_reduction",
    "IB": "image_blurring",
    "JC": "jpeg_compression",
    "NA": "noise_addition",
    "RT": "rotation",
    "SC": "scaling",
    "DS": "distortion",
    "CB": "combination",
}


class CoMoFoDDataset(BaseDataset):
    """CoMoFoD Copy-Move veri seti okuyucusu."""

    def __init__(self, root_dir: str | Path) -> None:
        super().__init__(root_dir=root_dir, name="CoMoFoD", task_type="copy_move")

    def load_samples(self) -> list[DatasetSample]:
        """Kök dizindeki CoMoFoD dosyalarını tarar ve örnek listesini oluşturur."""
        if not self.root_dir.is_dir():
            return []

        # Desteklenen görüntü uzantılarını tara (.png, .jpg, .tif)
        all_files = [
            f for f in self.root_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp")
        ]

        # 1. Maskeleri indeksle: {group_id: Path}
        # Örnek: '001_B.png' veya '001_B_RT1.png'
        binary_masks: dict[str, Path] = {}
        color_masks: dict[str, Path] = {}

        for f in all_files:
            stem = f.stem
            # _B (ikili maske) veya _M (renkli maske)
            if "_B" in stem:
                binary_masks[stem] = f
            elif "_M" in stem:
                color_masks[stem] = f

        samples: list[DatasetSample] = []

        # 2. Görüntü dosyalarını ayrıştır
        # Regex formatı: {ID}_{TYPE}[_{ATTACK}]
        # Örnek: '001_O', '001_F', '001_F_JC1', '001_O_BC2'
        pattern = re.compile(r"^(\d+)_([OF])(?:_([A-Z]{2}\d*))?$", re.IGNORECASE)

        for f in all_files:
            stem = f.stem
            # Maske dosyalarını ana görüntü olarak sayma
            if "_B" in stem or "_M" in stem:
                continue

            match = pattern.match(stem)
            if not match:
                continue

            group_id = match.group(1)       # '001'
            type_code = match.group(2).upper()  # 'O' veya 'F'
            attack_code = match.group(3)   # 'JC1', 'RT2' veya None

            is_auth = (type_code == "O")
            manipulation_type = "authentic" if is_auth else "copy_move"

            # Saldırı türünü çözümle
            variant_name = "clean"
            if attack_code:
                prefix = attack_code[:2].upper()
                variant_name = COMOFOD_ATTACK_MAP.get(prefix, attack_code.lower())
                if len(attack_code) > 2:
                    variant_name = f"{variant_name}_{attack_code[2:]}"

            # İlgili maskeyi bul
            mask_path: Path | None = None
            if not is_auth:
                # Önce spesifik maske ara (örn. 001_B_RT1.png), yoksa temel maskeyi (001_B.png) al
                variant_mask_key = f"{group_id}_B_{attack_code}" if attack_code else f"{group_id}_B"
                mask_path = binary_masks.get(variant_mask_key) or binary_masks.get(f"{group_id}_B")

            # Orijinal karşılık dosyasını bul (varsa)
            auth_path = None
            auth_candidate = f.parent / f"{group_id}_O{f.suffix}"
            if auth_candidate.exists():
                auth_path = auth_candidate

            sample = DatasetSample(
                sample_id=stem,
                task_type="copy_move",
                image_path=f,
                mask_path=mask_path,
                authentic_path=auth_path,
                manipulation_type=manipulation_type,
                variant=variant_name,
                group_id=group_id,
                metadata={
                    "attack_code": attack_code,
                    "dataset_source": "University of Zagreb CoMoFoD",
                },
            )
            samples.append(sample)

        # Deterministik sıralama (sample_id bazında)
        samples.sort(key=lambda s: s.sample_id)
        return samples
