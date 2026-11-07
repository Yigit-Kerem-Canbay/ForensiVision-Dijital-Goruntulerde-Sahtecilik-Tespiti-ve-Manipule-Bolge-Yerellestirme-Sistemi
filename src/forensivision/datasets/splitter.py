"""
ForensiVision v2 — Sahne-Gruplu (Scene-Grouped) ve Sızıntısız Veri Ayırıcı (Splitter).

Bu modül, v1'de yaşanan en büyük iki veri hatasını kesin olarak engeller:
1. Sahne Sızıntısı (Scene Leakage):
   - MS-COCO temiz fotoğrafı (örn. '000000000071.jpg'),
   - DEFACTO Copy-Move türevi ('0_000000000071.tif') ve
   - DEFACTO Splicing türevleri ('p1_0_000000000071.tif')
   aynı `canonical_scene_id` ('coco_000000000071') altında kilitlenir.
   Bir sahne hangi kümeye (train / val / calibration / test) atanmışsa,
   o sahneden türetilen TÜM temiz ve sahte görseller yalnızca o kümede yer alır.
2. CoMoFoD Küme Sızıntısı:
   - Aynı temel görselden (örn. '001') türetilen 25 temiz ('001_O', '001_O_BC1'...)
     ve 25 sahte ('001_F', '001_F_BC1'...) varyantın tamamı 'comofod_001' sahnesi
     altında tek bir kümeye atanır.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forensivision.datasets.base import BaseDataset, DatasetSample

COCO_12_PATTERN = re.compile(r"^(\d{12})$")


def compute_canonical_scene_id(sample: DatasetSample, dataset_name: str) -> str:
    """Örnek için veri setleri arası ortak (canonical) sahne kimliği üretir.

    Eğer `group_id` 12 haneli bir MS-COCO kimliği ise (DEFACTO Copy-Move, DEFACTO Splicing
    ve MS-COCO Clean ortaklığı), 'coco_{12_DIGIT_ID}' döndürür.
    Diğer veri setlerinde '{dataset}_{group_id}' döndürür.
    """
    gid = str(sample.group_id or sample.sample_id).strip()
    if COCO_12_PATTERN.match(gid):
        return f"coco_{gid}"

    coco_meta = sample.metadata.get("coco_image_id") if sample.metadata else None
    if coco_meta and COCO_12_PATTERN.match(str(coco_meta)):
        return f"coco_{coco_meta}"

    ds_prefix = dataset_name.lower().replace(" ", "_")
    if gid.startswith(ds_prefix):
        return gid
    return f"{ds_prefix}_{gid}"


def _deterministic_bucket(scene_id: str, seed: int = 42) -> float:
    """Sahne kimliğinden [0.0, 1.0) aralığında deterministik bir olasılık değeri üretir."""
    key = f"{seed}:{scene_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    int_val = int(digest[:12], 16)
    return int_val / float(0xFFFFFFFFFFFF + 1)


@dataclass(frozen=True)
class IndexedSampleRecord:
    """Manifest tablosunda yer alan zenginleştirilmiş v2 veri kaydı."""

    sample_id: str
    dataset_name: str
    canonical_scene_id: str
    split: str  # 'train' | 'val' | 'calibration' | 'test'
    is_tampered: int  # 0: Temiz (Authentic), 1: Kirli (Copy-Move veya Splicing)
    manipulation_type: str  # 'authentic' | 'copy_move' | 'splicing'
    variant: str
    file_ext: str
    image_path: str
    mask_path: str | None
    source_mask_path: str | None
    target_mask_path: str | None
    has_valid_mask: bool


class SceneGroupedManifestBuilder:
    """Çoklu adli veri setlerini sızıntısız olarak birleştirip bölen v2 oluşturucu."""

    def __init__(
        self,
        train_ratio: float = 0.75,
        val_ratio: float = 0.10,
        calibration_ratio: float = 0.05,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        total = train_ratio + val_ratio + calibration_ratio + test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Bölme oranları toplamı 1.0 olmalıdır, verilen: {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.calibration_ratio = calibration_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def assign_split(self, canonical_scene_id: str) -> str:
        """Sahne kimliğini deterministik olarak train/val/calibration/test kümesine atar."""
        u = _deterministic_bucket(canonical_scene_id, self.seed)
        t1 = self.train_ratio
        t2 = t1 + self.val_ratio
        t3 = t2 + self.calibration_ratio

        if u < t1:
            return "train"
        elif u < t2:
            return "val"
        elif u < t3:
            return "calibration"
        else:
            return "test"

    def build_records(
        self,
        datasets: list[BaseDataset],
        require_mask_for_tampered: bool = True,
    ) -> tuple[list[IndexedSampleRecord], dict[str, Any]]:
        """Verilen veri setlerini tarar, etiket ve maske doğrulaması yapar ve böler."""
        records: list[IndexedSampleRecord] = []
        skipped_missing_mask = 0

        for ds in datasets:
            for s in ds.samples:
                is_auth = s.is_authentic()
                has_mask = bool(
                    (s.mask_path and Path(s.mask_path).is_file())
                    or (s.target_mask_path and Path(s.target_mask_path).is_file())
                )

                if not is_auth and not has_mask and require_mask_for_tampered:
                    skipped_missing_mask += 1
                    continue

                scene_id = compute_canonical_scene_id(s, ds.name)
                split = self.assign_split(scene_id)
                ext = Path(s.image_path).suffix.lower()

                rec = IndexedSampleRecord(
                    sample_id=f"{ds.name.lower()}_{s.sample_id}",
                    dataset_name=ds.name,
                    canonical_scene_id=scene_id,
                    split=split,
                    is_tampered=0 if is_auth else 1,
                    manipulation_type=s.manipulation_type,
                    variant=s.variant or "clean",
                    file_ext=ext,
                    image_path=str(s.image_path),
                    mask_path=str(s.mask_path) if s.mask_path else None,
                    source_mask_path=str(s.source_mask_path) if s.source_mask_path else None,
                    target_mask_path=str(s.target_mask_path) if s.target_mask_path else None,
                    has_valid_mask=True if is_auth else has_mask,
                )
                records.append(rec)

        audit_report = self.verify_zero_leakage(records, skipped_missing_mask)
        return records, audit_report

    @staticmethod
    def verify_zero_leakage(
        records: list[IndexedSampleRecord],
        skipped_missing_mask: int = 0,
    ) -> dict[str, Any]:
        """Kümeler arasında tek bir ortak `canonical_scene_id` bile olmadığını matematiksel olarak doğrular."""
        scenes_by_split: dict[str, set[str]] = {
            "train": set(),
            "val": set(),
            "calibration": set(),
            "test": set(),
        }
        counts_by_split: dict[str, dict[str, int]] = {
            sp: {"total": 0, "authentic": 0, "copy_move": 0, "splicing": 0}
            for sp in scenes_by_split
        }
        counts_by_dataset: dict[str, dict[str, int]] = {}

        for r in records:
            scenes_by_split[r.split].add(r.canonical_scene_id)
            counts_by_split[r.split]["total"] += 1
            if r.manipulation_type in counts_by_split[r.split]:
                counts_by_split[r.split][r.manipulation_type] += 1

            ds_stat = counts_by_dataset.setdefault(
                r.dataset_name,
                {"total": 0, "authentic": 0, "copy_move": 0, "splicing": 0},
            )
            ds_stat["total"] += 1
            if r.manipulation_type in ds_stat:
                ds_stat[r.manipulation_type] += 1

        # Çapraz kesişim kontrolü
        split_names = list(scenes_by_split.keys())
        overlaps: dict[str, int] = {}
        total_leaked_scenes = 0
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                inter = scenes_by_split[s1].intersection(scenes_by_split[s2])
                overlaps[f"{s1}_vs_{s2}"] = len(inter)
                total_leaked_scenes += len(inter)

        if total_leaked_scenes > 0:
            raise RuntimeError(
                f"KRİTİK VERİ SIZINTISI TESPİT EDİLDİ! Kümeler arası ortak sahne sayısı: {overlaps}"
            )

        return {
            "zero_scene_leakage_verified": total_leaked_scenes == 0,
            "total_records": len(records),
            "skipped_missing_mask_records": skipped_missing_mask,
            "unique_scenes_per_split": {k: len(v) for k, v in scenes_by_split.items()},
            "split_overlaps": overlaps,
            "counts_by_split": counts_by_split,
            "counts_by_dataset": counts_by_dataset,
        }

    @staticmethod
    def save_v2_manifest(
        records: list[IndexedSampleRecord],
        audit_report: dict[str, Any],
        output_dir: str | Path = "data/manifests",
        prefix: str = "forensivision_v2",
    ) -> tuple[Path, Path]:
        """Doğrulanmış v2 manifest tablosunu CSV ve denetim raporunu JSON olarak kaydeder."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{prefix}_manifest.csv"
        report_path = out_dir / f"{prefix}_audit_report.json"

        fieldnames = [
            "sample_id",
            "dataset_name",
            "canonical_scene_id",
            "split",
            "is_tampered",
            "manipulation_type",
            "variant",
            "file_ext",
            "image_path",
            "mask_path",
            "source_mask_path",
            "target_mask_path",
            "has_valid_mask",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.__dict__)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)

        return csv_path, report_path

