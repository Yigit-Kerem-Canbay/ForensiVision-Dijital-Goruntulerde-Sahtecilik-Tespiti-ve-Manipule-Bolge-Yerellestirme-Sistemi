"""
ForensiVision Columbia Color Splicing Veri Seti Okuyucusu (Columbia University DVMM).

Columbia Color Splicing Dataset, sıkıştırılmamış TIFF formatında gerçek ve ekleme (spliced)
görüntüleri içeren akademik bir kıyaslama kaynağıdır.

Dosya Yapısı:
    - Özgün fotoğraflar genellikle kamera modeliyle başlar (örn. canong3_*, nikond70_* veya auth_*).
    - Spliced fotoğraflar 'sp' veya 'spliced' etiketini barındırır (örn. canong3_nikond70_sub_01.tif).
    - Doğruluk maskeleri genellikle '_edg' veya '_mask' ekiyle veya ayrı 'masks/' klasöründe saklanır.
"""

from __future__ import annotations

from pathlib import Path

from forensivision.datasets.base import BaseDataset, DatasetSample


class ColumbiaDataset(BaseDataset):
    """Columbia Splicing veri seti okuyucusu."""

    def __init__(self, root_dir: str | Path) -> None:
        super().__init__(root_dir=root_dir, name="Columbia", task_type="splicing")

    def load_samples(self) -> list[DatasetSample]:
        """Kök dizindeki Columbia dosyalarını tarar ve örnek listesini oluşturur."""
        if not self.root_dir.is_dir():
            return []

        all_files = [
            f for f in self.root_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".bmp")
        ]

        # Maskeleri indeksle
        masks: dict[str, Path] = {}
        for f in all_files:
            stem = f.stem.lower()
            if "_edg" in stem or "_mask" in stem or "mask" in f.parent.name.lower():
                clean_key = stem.replace("_edg", "").replace("_mask", "")
                masks[clean_key] = f

        samples: list[DatasetSample] = []

        for f in all_files:
            stem = f.stem
            stem_lower = stem.lower()

            # Maske dosyalarını ana görüntü olarak sayma
            if "_edg" in stem_lower or "_mask" in stem_lower or "mask" in f.parent.name.lower():
                continue

            # Spliced vs Authentic ayrımı
            is_spliced = (
                "_sp" in stem_lower or
                "splice" in stem_lower or
                "spliced" in f.parent.name.lower() or
                "sub_" in stem_lower
            )

            manipulation_type = "splicing" if is_spliced else "authentic"
            clean_key = stem_lower.replace("_sp", "")
            mask_path = masks.get(clean_key) if is_spliced else None

            sample = DatasetSample(
                sample_id=stem,
                task_type="splicing",
                image_path=f,
                mask_path=mask_path,
                authentic_path=None,
                manipulation_type=manipulation_type,
                variant="clean",
                group_id=clean_key,
                metadata={"dataset_source": "Columbia University DVMM"},
            )
            samples.append(sample)

        samples.sort(key=lambda s: s.sample_id)
        return samples
