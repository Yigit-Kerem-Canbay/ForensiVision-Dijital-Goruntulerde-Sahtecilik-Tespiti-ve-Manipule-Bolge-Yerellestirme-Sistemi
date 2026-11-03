"""
ForensiVision Kaggle Veri Seti Paketleme Betiği.

Bu betik:
- Masaüstündeki 'Sayısal Görünntü İşleme Veri Seti' klasörünü
- 3 adet bağımsız ve temiz ZIP kolisine dönüştürür:
    1. comofod.zip (~3 GB)
    2. defacto_copymove.zip (~10 GB)
    3. defacto_splicing.zip (~56 GB)
- Hedef: C:\\Users\\yigit\\Desktop\\kaggle_upload\\
- Hızlı 'STORE' modu kullanır (yeniden sıkıştırma yapmaz, orijinal pikselleri %100 korur ve dakikalar içinde bitirir).
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path


def find_source_dir() -> Path:
    desktop = Path("C:/Users/yigit/Desktop")
    for p in desktop.iterdir():
        if p.is_dir() and ("görünntü" in p.name.lower() or "grnnt" in p.name.lower()):
            return p
    # Varsayılan deneme
    fallback = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError("Masaüstünde 'Sayısal Görünntü İşleme Veri Seti' klasörü bulunamadı!")


def zip_subfolder(source_folder: Path, output_zip_path: Path, title: str):
    print(f"\n=======================================================")
    print(f"PAKETLENİYOR: {title}")
    print(f"Kaynak : {source_folder}")
    print(f"Hedef  : {output_zip_path}")
    print(f"=======================================================", flush=True)

    t0 = time.perf_counter()
    file_list: list[Path] = []
    total_bytes = 0

    print("Dosya listesi taranıyor...", flush=True)
    for root, _, files in os.walk(source_folder):
        root_path = Path(root)
        for f in files:
            p = root_path / f
            file_list.append(p)
            total_bytes += p.stat().st_size

    total_files = len(file_list)
    print(f"Toplam Dosya Sayısı: {total_files:,} | Toplam Boyut: {total_bytes / (1024**3):.2f} GB", flush=True)

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    written_bytes = 0
    with zipfile.ZipFile(
        output_zip_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as zf:
        for idx, file_path in enumerate(file_list, 1):
            arcname = file_path.relative_to(source_folder.parent)
            zf.write(file_path, arcname=str(arcname))
            written_bytes += file_path.stat().st_size

            if idx % 10000 == 0 or idx == total_files:
                pct = (idx / total_files) * 100.0
                elapsed = time.perf_counter() - t0
                speed = (written_bytes / (1024 * 1024)) / (elapsed + 1e-5)
                print(f"  İlerleme: {idx:,} / {total_files:,} (%{pct:.1f}) | Hız: {speed:.1f} MB/sn", flush=True)

    elapsed_total = time.perf_counter() - t0
    final_size_gb = output_zip_path.stat().st_size / (1024**3)
    print(f"TAMAMLANDI: {output_zip_path.name} ({final_size_gb:.2f} GB) - Süre: {elapsed_total:.1f} sn", flush=True)


def main():
    source_root = find_source_dir()
    output_dir = Path("C:/Users/yigit/Desktop/kaggle_upload")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("*" * 80)
    print("FORENSIVISION KAGGLE PAKETLEME BAŞLADI")
    print(f"Kaynak Klasör : {source_root}")
    print(f"Çıktı Klasörü : {output_dir}")
    print("*" * 80, flush=True)

    # 1. CoMoFoD Paketi
    comofod_dir = source_root / "CoMoFoD_small_v2"
    if comofod_dir.is_dir():
        zip_subfolder(comofod_dir, output_dir / "comofod.zip", "1/3: CoMoFoD Veri Seti")

    # 2. DEFACTO Copy-Move Paketi
    cm_dir = source_root / "defacto-copymove"
    if cm_dir.is_dir():
        zip_subfolder(cm_dir, output_dir / "defacto_copymove.zip", "2/3: DEFACTO Copy-Move Veri Seti")

    # 3. DEFACTO Splicing Paketi
    spl_dir = source_root / "defacto-splicing"
    if spl_dir.is_dir():
        zip_subfolder(spl_dir, output_dir / "defacto_splicing.zip", "3/3: DEFACTO Splicing Veri Seti")

    print("\n" + "=" * 80)
    print("TÜM PAKETLEME İŞLEMİ BAŞARIYLA TAMAMLANDI!")
    print(f"Kaggle için hazır zip dosyaların: {output_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
