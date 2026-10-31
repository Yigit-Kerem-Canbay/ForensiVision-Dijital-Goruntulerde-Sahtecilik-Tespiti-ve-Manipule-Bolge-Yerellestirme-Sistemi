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

Klasör Mimarisi (Splicing):
    Splicing veri seti 7 anlamsal kategoriye bölünmüş 7 alt parçadan (Part 1 - Part 7) oluşur:
    - splicing_1 : Airplane (10.765 örnek)
    - splicing_2 : Birds (12.187 örnek)
    - splicing_3 : Clock (15.899 örnek)
    - splicing_4 : Frisbee (19.305 örnek)
    - splicing_5 : Sports Ball (17.767 örnek)
    - splicing_6 : Stop Sign (14.356 örnek)
    - splicing_7 : Traffic Light (16.113 örnek)
    Toplam: 106.392 manipüle edilmiş görüntü.
    Her parça {splicing_X_img} ve {splicing_X_annotations} çifti içerir.

Dosya İsimlendirme Formatı:
    {INSTANCE_ID}_{COCO_IMAGE_ID}.tif
    - INSTANCE_ID   : Kopyalanan / yapıştırılan nesne veya eşleştirme indeksi
    - COCO_IMAGE_ID : 12 basamaklı orijinal MS-COCO arka plan görüntü kimliği
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forensivision.datasets.base import BaseDataset, DatasetSample

# Splicing parçalarının kategori adları
SPLICING_CATEGORY_MAP = {
    1: "airplane",
    2: "birds",
    3: "clock",
    4: "frisbee",
    5: "sports_ball",
    6: "stop_sign",
    7: "traffic_light",
}


class DefactoDataset(BaseDataset):
    """DEFACTO adli görüntü manipülasyonu veri seti okuyucusu."""

    def __init__(
        self,
        root_dir: str | Path,
        task_type: str = "copy_move",
        part: int | str | None = None,
        load_metadata: bool = False,
    ) -> None:
        """DEFACTO veri setini başlatır.

        Args:
            root_dir: DEFACTO kök dizini (örn. 'data/raw/defacto/copymove', 'data/raw/defacto/splicing' veya 'data/raw/defacto').
            task_type: 'copy_move' veya 'splicing'.
            part: Splicing için isteğe bağlı parça numarası (1-7) veya None (tüm parçalar).
            load_metadata: Her örnek oluşturulurken graph JSON dosyalarından afin dönüşüm
                           metaverilerinin okunup okunmayacağı.
        """
        super().__init__(root_dir=root_dir, name="DEFACTO", task_type=task_type)
        self.part = int(part) if part is not None else None
        self.load_metadata = load_metadata
        self._graph_dirs: list[Path] = []

    def _resolve_subsets(
        self,
    ) -> list[tuple[int | None, Path, Path | None, Path | None, Path | None]]:
        """Kök dizin yapısını analiz ederek (part_idx, img_dir, donor_dir, probe_dir, graph_dir) listesini döndürür."""
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
                    return [(
                        None,
                        img_dir,
                        donor_dir if donor_dir.is_dir() else None,
                        probe_dir if probe_dir.is_dir() else None,
                        graph_dir if graph_dir.is_dir() else None,
                    )]

            elif self.task_type == "splicing":
                # Çok parçalı (Part 1 - 7) yapıyı kontrol et
                subsets = []
                part_range = [self.part] if self.part is not None else range(1, 8)

                for p_idx in part_range:
                    img_dir = candidate / f"splicing_{p_idx}_img" / "img"
                    if not img_dir.is_dir():
                        img_dir = candidate / f"splicing_{p_idx}_img"
                    annot_dir = candidate / f"splicing_{p_idx}_annotations"

                    donor_dir = annot_dir / "donor_mask"
                    probe_dir = annot_dir / "probe_mask"
                    graph_dir = annot_dir / "graph"

                    if img_dir.is_dir():
                        subsets.append((
                            p_idx,
                            img_dir,
                            donor_dir if donor_dir.is_dir() else None,
                            probe_dir if probe_dir.is_dir() else None,
                            graph_dir if graph_dir.is_dir() else None,
                        ))

                if subsets:
                    return subsets

                # Tekil splicing klasörü yedeği
                img_dir = candidate / "splicing_img" / "img"
                if not img_dir.is_dir():
                    img_dir = candidate / "img"
                annot_dir = candidate / "splicing_annotations"
                donor_dir = annot_dir / "donor_mask"
                probe_dir = annot_dir / "probe_mask"
                graph_dir = annot_dir / "graph"

                if img_dir.is_dir():
                    return [(
                        1,
                        img_dir,
                        donor_dir if donor_dir.is_dir() else None,
                        probe_dir if probe_dir.is_dir() else None,
                        graph_dir if graph_dir.is_dir() else None,
                    )]

        return []

    def load_samples(self) -> list[DatasetSample]:
        """Dizin hiyerarşisini tarayarak DEFACTO örnek listesini oluşturur."""
        subsets = self._resolve_subsets()
        if not subsets:
            return []

        self._graph_dirs = [g for _, _, _, _, g in subsets if g is not None]

        samples: list[DatasetSample] = []
        is_multi_part = len(subsets) > 1

        for p_idx, img_dir, donor_dir, probe_dir, graph_dir in subsets:
            cat_name = SPLICING_CATEGORY_MAP.get(p_idx, f"part_{p_idx}") if p_idx else "copy_move"

            image_files: list[Path] = []
            for ext in ("*.tif", "*.png", "*.jpg", "*.jpeg", "*.TIFF"):
                image_files.extend(img_dir.glob(ext))

            image_files.sort(key=lambda p: p.name)

            for img_path in image_files:
                stem = img_path.stem
                parts = stem.split("_", 1)

                group_id = parts[1] if len(parts) > 1 else stem
                # Çok parçalı splicing'de çakışmayı önlemek için sample_id öneki
                sample_id = f"p{p_idx}_{stem}" if is_multi_part else stem

                # Kaynak maskesi (donor)
                donor_mask_found: Path | None = None
                if donor_dir:
                    for ext in (".tif", ".png", ".jpg"):
                        cand = donor_dir / f"{stem}{ext}"
                        if cand.is_file():
                            donor_mask_found = cand
                            break

                # Hedef maskesi (probe)
                tgt_mask_path: Path | None = None
                if probe_dir:
                    for ext in (".jpg", ".png", ".tif"):
                        cand = probe_dir / f"{stem}{ext}"
                        if cand.is_file():
                            tgt_mask_path = cand
                            break

                # Splicing'de manipüle edilen hedef bölge strictly probe_mask'tır!
                # Çünkü kaynak nesne başka bir fotoğraftan getirilmiştir (aynı resimde kaynak bölge yoktur).
                # Copy-Move'da ise hem kaynak hem hedef aynı görseldedir.
                if self.task_type == "splicing":
                    primary_mask = tgt_mask_path
                    actual_src_mask = None
                    donor_path_meta = str(donor_mask_found) if donor_mask_found else None
                else:
                    primary_mask = tgt_mask_path or donor_mask_found
                    actual_src_mask = donor_mask_found
                    donor_path_meta = None

                metadata: dict[str, Any] = {
                    "instance_id": parts[0] if len(parts) > 1 else "0",
                    "coco_image_id": group_id,
                    "category": cat_name,
                    "part": p_idx,
                }
                if donor_path_meta:
                    metadata["donor_mask_path"] = donor_path_meta

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
                                    metadata["external_image"] = prop.get("External image")
                        except Exception:
                            pass

                sample = DatasetSample(
                    sample_id=sample_id,
                    task_type=self.task_type,
                    image_path=img_path,
                    mask_path=primary_mask,
                    manipulation_type=self.task_type,
                    variant=cat_name,
                    group_id=group_id,
                    source_mask_path=actual_src_mask,
                    target_mask_path=tgt_mask_path,
                    metadata=metadata,
                )
                samples.append(sample)

        return samples

    def get_sample_metadata(self, sample_id: str) -> dict[str, Any] | None:
        """Belirtilen örneğin graph JSON dosyasındaki afin dönüşüm parametrelerini döndürür."""
        # sample_id p1_0_000000195755 şeklinde ise stem'i ayıkla
        clean_stem = sample_id.split("_", 1)[1] if sample_id.startswith("p") and "_" in sample_id else sample_id

        for g_dir in self._graph_dirs:
            json_path = g_dir / f"{clean_stem}.json"
            if json_path.is_file():
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
                            res["external_image"] = prop.get("External image")
                        return res
                except Exception:
                    return None

        return None
