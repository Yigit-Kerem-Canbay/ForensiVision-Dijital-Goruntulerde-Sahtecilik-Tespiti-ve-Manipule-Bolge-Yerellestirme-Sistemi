"""
ForensiVision v2 — CASIA v2.0 ve Temiz (MS-COCO / Saha) Veri Seti Okuyucuları.

Bu modül:
1. CASIA v2.0 veri setini (Au: Temiz, Tp: Manipüle) doğru maske eşleştirmesiyle okur.
   ÖNEMLİ: Eski v1'deki maskesiz Tp görsellerini 'sıfır maske' ile eğitime sokma hatasını
   kesin olarak engeller; maskesi bulunamayan Tp görselleri 'missing_mask=True' olarak işaretlenir
   ve piksel kaybı (segmentation loss) hesabından hariç tutulur.
2. MS-COCO veya kullanıcının çektiği gerçek dünya temiz fotoğraflarını `COCO ID` / `scene_id`
   bilgisiyle okuyarak DEFACTO ile aynı sahneyi paylaşan görsellerin sızıntı yapmasını önler.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from forensivision.datasets.base import BaseDataset, DatasetSample


class CasiaV2Dataset(BaseDataset):
    """CASIA v2.0 (Authentic + Tampered + Ground-Truth Mask) veri seti okuyucusu."""

    def __init__(self, root_dir: str | Path, strict_masks: bool = True) -> None:
        """CASIA v2.0 okuyucusunu başlatır.

        Args:
            root_dir: CASIA2 kök dizini (Au, Tp ve Gt/mask alt klasörlerini içeren dizin).
            strict_masks: True ise maskesi bulunamayan Tp (kirli) örnekleri eğitim listesine almaz.
        """
        super().__init__(root_dir=root_dir, name="CASIA_v2", task_type="mixed")
        self.strict_masks = strict_masks

    def load_samples(self) -> list[DatasetSample]:
        if not self.root_dir.is_dir():
            return []

        all_files = [
            f for f in self.root_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".bmp")
        ]

        # 1. Önce tüm Ground-Truth (GT / mask) dosyalarını indeksle
        gt_masks: dict[str, Path] = {}
        for f in all_files:
            parent_lower = f.parent.name.lower()
            stem_lower = f.stem.lower()
            if "gt" in parent_lower or "mask" in parent_lower or "groundtruth" in parent_lower or stem_lower.endswith("_gt"):
                clean_key = re.sub(r"_gt$", "", stem_lower)
                gt_masks[clean_key] = f

        samples: list[DatasetSample] = []

        # CASIA Tp adlandırma deseni: Tp_D_CRN_M_N_nat00059_nat00059_10045
        # İlk 'D' (Different source -> Splicing) veya 'S' (Same source -> Copy-Move)
        tp_pattern = re.compile(
            r"^tp_([ds])_[a-z0-9]+_[a-z0-9]+_[a-z0-9]+_([a-z]+\d+)_([a-z]+\d+)_(\d+)$",
            re.IGNORECASE,
        )
        au_pattern = re.compile(r"^au_([a-z]+)_(\d+)$", re.IGNORECASE)

        for f in all_files:
            parent_lower = f.parent.name.lower()
            stem = f.stem
            stem_lower = stem.lower()

            # Maske klasöründeki dosyaları ana görüntü olarak ekleme
            if "gt" in parent_lower or "mask" in parent_lower or "groundtruth" in parent_lower or stem_lower.endswith("_gt"):
                continue

            if stem_lower.startswith("au_") or parent_lower == "au":
                au_match = au_pattern.match(stem_lower)
                if au_match:
                    scene_id = f"casia_{au_match.group(1)}{au_match.group(2)}"
                else:
                    scene_id = f"casia_{stem_lower}"

                samples.append(
                    DatasetSample(
                        sample_id=f"casia_{stem}",
                        task_type="authentic",
                        image_path=f,
                        mask_path=None,
                        manipulation_type="authentic",
                        variant=f.suffix.lower().lstrip("."),
                        group_id=scene_id,
                        metadata={"dataset_source": "CASIA_v2", "has_valid_mask": True},
                    )
                )

            elif stem_lower.startswith("tp_") or parent_lower == "tp":
                mask_path = gt_masks.get(stem_lower)
                if mask_path is None and self.strict_masks:
                    # v1'deki hatayı önle: Maskesi olmayan kirli görseli sıfır maskeyle eğitime sokma!
                    continue

                tp_match = tp_pattern.match(stem_lower)
                if tp_match:
                     manip_code = tp_match.group(1).upper()
                     src_scene = tp_match.group(2)
                     manipulation_type = "splicing" if manip_code == "D" else "copy_move"
                     group_id = f"casia_{src_scene}"
                else:
                     manipulation_type = "splicing" if "_d_" in stem_lower else "copy_move"
                     group_id = f"casia_{stem_lower}"

                samples.append(
                    DatasetSample(
                        sample_id=f"casia_{stem}",
                        task_type=manipulation_type,
                        image_path=f,
                        mask_path=mask_path,
                        manipulation_type=manipulation_type,
                        variant=f.suffix.lower().lstrip("."),
                        group_id=group_id,
                        metadata={
                            "dataset_source": "CASIA_v2",
                            "has_valid_mask": mask_path is not None,
                        },
                    )
                )

        samples.sort(key=lambda s: s.sample_id)
        return samples


class CleanImageDataset(BaseDataset):
    """MS-COCO veya Gerçek Dünya (Kamera/WhatsApp) temiz görüntü veri seti okuyucusu.

    MS-COCO dosya isimlerinden (örn. '000000000071.jpg') 12 haneli COCO ID'yi çıkarır
    ve `group_id` olarak atar. Böylece DEFACTO'daki '0_000000000071.tif' ile aynı
    `group_id` ('000000000071') değerini paylaşır ve eğitim/test ayrımında sızıntı olmaz.
    """

    def __init__(self, root_dir: str | Path, dataset_name: str = "CleanPool") -> None:
        super().__init__(root_dir=root_dir, name=dataset_name, task_type="authentic")

    def load_samples(self) -> list[DatasetSample]:
        if not self.root_dir.is_dir():
            return []

        all_files = [
            f for f in self.root_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".bmp")
        ]
        all_files.sort(key=lambda p: p.name)

        coco_id_pattern = re.compile(r"(\d{12})$")
        samples: list[DatasetSample] = []

        for f in all_files:
            stem = f.stem
            m = coco_id_pattern.search(stem)
            group_id = m.group(1) if m else f"{self.name.lower()}_{stem}"

            samples.append(
                DatasetSample(
                    sample_id=f"{self.name.lower()}_{stem}",
                    task_type="authentic",
                    image_path=f,
                    mask_path=None,
                    manipulation_type="authentic",
                    variant=f.suffix.lower().lstrip("."),
                    group_id=group_id,
                    metadata={
                        "dataset_source": self.name,
                        "coco_image_id": m.group(1) if m else None,
                    },
                )
            )

        return samples

