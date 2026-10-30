"""
ForensiVision DEFACTO Veri Seti Okuyucusu (Large-scale Image Manipulation Detection Benchmark).

DEFACTO (DEFacto: An End-to-End Large-Scale Dataset for Image Manipulation Detection),
adli görüntü analizinde derin öğrenme ve geleneksel yöntemleri test etmek amacıyla
MS-COCO 2017 temel görüntüleri üzerine inşa edilmiş geniş ölçekli bir akademik veri setidir.

Klasör Mimarisi (Copy-Move):
    ├── copymove_annotations/
    │   ├── donor_mask/    : Kopyalanan kaynak bölge (source mask; .tif, yumuşatılmış kenar)
    │   ├── probe_mask/    : Yapıştırılan hedef bölge (target mask; .jpg, JPEG sıkıştırma artefaktlı)
    │   └── graph/         : Her manipülasyonun afin dönüşüm parametrelerini içeren .json dosyaları
    └── copymove_img/
        └── img/           : Manipüle edilmiş sahte görüntüler (.tif, 18.194 örnek)

Dosya İsimlendirme Formatı:
    {INSTANCE_ID}_{COCO_IMAGE_ID}.tif (örn. 0_000000000071.tif, 8_000000029706.tif)
    - INSTANCE_ID   : MS-COCO nesne etiketleme sıra numarası (kopyalanan nesne indeksi)
    - COCO_IMAGE_ID : 12 basamaklı orijinal MS-COCO görüntü kimliği (sahne/arka plan kimliği)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forensivision.datasets.base import BaseDataset, DatasetSample


class DefactoDataset(BaseDataset):
    """DEFACTO adli görüntü manipülasyonu veri seti okuyucusu."""

    def __init__(
        self,
        root_dir: str | Path,
        task_type: str = "copy_move",
        load_metadata: bool = False,
    ) -> None:
        """DEFACTO veri setini başlatır.

        Args:
            root_dir: DEFACTO kök dizini (örn. 'data/raw/defacto/copymove' veya 'data/raw/defacto').
            task_type: 'copy_move' veya 'splicing'.
            load_metadata: Her örnek oluşturulurken graph JSON dosyalarından afin dönüşüm
                           metaverilerinin okunup okunmayacağı.
        """
        super().__init__(root_dir=root_dir, name="DEFACTO", task_type=task_type)
        self.load_metadata = load_metadata
        self._graph_dir: Path | None = None

    def _resolve_paths(self) -> tuple[Path | None, Path | None, Path | None, Path | None]:
        """Kök dizin yapısını analiz ederek görüntü, kaynak maske, hedef maske ve grafik dizinlerini belirler."""
        candidates = [
            self.root_dir,
            self.root_dir / "copymove" if self.task_type == "copy_move" else self.root_dir / "splicing",
        ]

        for candidate in candidates:
            if not candidate.is_dir():
                continue

            if self.task_type == "copy_move":
                img_dir = candidate / "copymove_img" / "img"
                if not img_dir.is_dir():
                    img_dir = candidate / "img"

                annot_dir = candidate / "copymove_annotations"
                donor_dir = annot_dir / "donor_mask"
                probe_dir = annot_dir / "probe_mask"
                graph_dir = annot_dir / "graph"

                if img_dir.is_dir():
                    return (
                        img_dir,
                        donor_dir if donor_dir.is_dir() else None,
                        probe_dir if probe_dir.is_dir() else None,
                        graph_dir if graph_dir.is_dir() else None,
                    )

            elif self.task_type == "splicing":
                img_dir = candidate / "splicing_img" / "img"
                if not img_dir.is_dir():
                    img_dir = candidate / "img"

                annot_dir = candidate / "splicing_annotations"
                probe_dir = annot_dir / "probe_mask"
                donor_dir = annot_dir / "donor_mask"
                graph_dir = annot_dir / "graph"

                if img_dir.is_dir():
                    return (
                        img_dir,
                        donor_dir if donor_dir.is_dir() else None,
                        probe_dir if probe_dir.is_dir() else None,
                        graph_dir if graph_dir.is_dir() else None,
                    )

        return None, None, None, None

    def load_samples(self) -> list[DatasetSample]:
        """Dizin hiyerarşisini tarayarak DEFACTO örnek listesini oluşturur."""
        img_dir, donor_dir, probe_dir, graph_dir = self._resolve_paths()
        self._graph_dir = graph_dir

        if img_dir is None or not img_dir.is_dir():
            return []

        # Desteklenen uzantılar (.tif, .png, .jpg)
        image_files: list[Path] = []
        for ext in ("*.tif", "*.png", "*.jpg", "*.jpeg", "*.TIFF"):
            image_files.extend(img_dir.glob(ext))

        # Deterministik sıralama (adli tekrarlanabilirlik)
        image_files.sort(key=lambda p: p.name)

        samples: list[DatasetSample] = []
        for img_path in image_files:
            stem = img_path.stem
            parts = stem.split("_", 1)

            # group_id: Sahne düzeyinde sızıntıyı önlemek için MS-COCO temel görüntü kimliği
            # Örn: '0_000000000071' -> '000000000071'
            group_id = parts[1] if len(parts) > 1 else stem

            # Kaynak maskesi (donor) tespiti (.tif, .png, .jpg)
            src_mask_path: Path | None = None
            if donor_dir:
                for ext in (".tif", ".png", ".jpg"):
                    cand = donor_dir / f"{stem}{ext}"
                    if cand.is_file():
                        src_mask_path = cand
                        break

            # Hedef maskesi (probe) tespiti (.jpg, .png, .tif)
            tgt_mask_path: Path | None = None
            if probe_dir:
                for ext in (".jpg", ".png", ".tif"):
                    cand = probe_dir / f"{stem}{ext}"
                    if cand.is_file():
                        tgt_mask_path = cand
                        break

            # Birleşik veya birincil maske yolu
            primary_mask = tgt_mask_path or src_mask_path

            # İsteğe bağlı JSON metaverisini oku
            metadata: dict[str, Any] = {
                "instance_id": parts[0] if len(parts) > 1 else "0",
                "coco_image_id": group_id,
            }
            if self.load_metadata and graph_dir:
                json_path = graph_dir / f"{stem}.json"
                if json_path.is_file():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            graph_data = json.load(f)
                            if isinstance(graph_data, list) and len(graph_data) > 1:
                                prop = graph_data[1].get("Property", {})
                                metadata["translation"] = prop.get("Translation")
                                metadata["scale"] = prop.get("Scale")
                                metadata["rotation"] = prop.get("Rotation")
                                metadata["feathering"] = prop.get("Mask feathering")
                    except Exception:
                        pass

            sample = DatasetSample(
                sample_id=stem,
                task_type=self.task_type,
                image_path=img_path,
                mask_path=primary_mask,
                manipulation_type=self.task_type,
                variant="clean",
                group_id=group_id,
                source_mask_path=src_mask_path,
                target_mask_path=tgt_mask_path,
                metadata=metadata,
            )
            samples.append(sample)

        return samples

    def get_sample_metadata(self, sample_id: str) -> dict[str, Any] | None:
        """Belirtilen örneğin graph JSON dosyasındaki afin dönüşüm parametrelerini döndürür."""
        if self._graph_dir is None:
            _, _, _, self._graph_dir = self._resolve_paths()

        if self._graph_dir is None:
            return None

        json_path = self._graph_dir / f"{sample_id}.json"
        if not json_path.is_file():
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                res: dict[str, Any] = {"raw": data}
                if isinstance(data, list) and len(data) > 1:
                    prop = data[1].get("Property", {})
                    res["translation"] = prop.get("Translation")
                    res["scale"] = prop.get("Scale")
                    res["rotation"] = prop.get("Rotation")
                    res["feathering"] = prop.get("Mask feathering")
                return res
        except Exception:
            return None
