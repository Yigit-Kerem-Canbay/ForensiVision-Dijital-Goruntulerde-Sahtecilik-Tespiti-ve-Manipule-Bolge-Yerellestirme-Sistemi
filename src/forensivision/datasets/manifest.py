"""
ForensiVision Veri Seti Manifest ve İndeksleme Modülü.

Bu modül; taranan veri setlerinin metaverilerini, dosya yollarını ve grup ilişkilerini
JSON ve CSV formatlarında dışa aktararak deneylerin tam tekrarlanabilirliğini (reproducibility) sağlar.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from forensivision.datasets.base import BaseDataset, DatasetSample


def build_manifest(dataset: BaseDataset) -> dict[str, Any]:
    """Veri setinden tam metaveri manifest sözlüğü oluşturur."""
    samples_data = [s.to_dict() for s in dataset.samples]
    summary_data = dataset.summary()

    return {
        "dataset_name": dataset.name,
        "task_type": dataset.task_type,
        "root_dir": str(dataset.root_dir),
        "total_samples": len(samples_data),
        "summary": summary_data,
        "samples": samples_data,
    }


def save_manifest(
    dataset: BaseDataset,
    output_dir: str | Path = "data/manifests",
    filename_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Veri seti manifestini hem JSON hem CSV formatında kaydeder.

    Returns:
        tuple[Path, Path]: (json_dosya_yolu, csv_dosya_yolu)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = filename_prefix or dataset.name.lower()
    json_path = out_dir / f"{prefix}_manifest.json"
    csv_path = out_dir / f"{prefix}_manifest.csv"

    manifest_dict = build_manifest(dataset)

    # 1. JSON Kaydı
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2, ensure_ascii=False)

    # 2. CSV Kaydı
    samples = manifest_dict["samples"]
    fieldnames = [
        "sample_id",
        "task_type",
        "manipulation_type",
        "variant",
        "group_id",
        "image_path",
        "mask_path",
        "authentic_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in samples:
            writer.writerow(row)

    return json_path, csv_path


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Kaydedilmiş bir JSON manifest dosyasını okur."""
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest dosyası bulunamadı: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
