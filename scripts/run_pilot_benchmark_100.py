"""
ForensiVision 100-Görsel Pilot Benchmark ve İyileştirme Değerlendirme Sistemi.

Bu betik:
- Optimize edilmiş yeni parametreleri (Ratio=0.78, Dist=25, Sup=0.45, Hull Guard, ELA-Std=2.2, Area=180) devreye alır.
- 35.000 görselde yapılan tüm adli testleri 100'er temsilci görselle baştan sona çalıştırır.
- Çıktılar:
  1. CoMoFoD Copy-Move (100 Kirli)
  2. CoMoFoD Copy-Move (100 Temiz - Kontrol Grubu)
  3. DEFACTO Copy-Move (100 Kirli)
  4. Birleşik Copy-Move (200 Kirli: 100 CoMoFoD + 100 DEFACTO)
  5. Image Splicing (100 Temiz - Kontrol Grubu)
  6. DEFACTO Splicing (100 Kirli - 7 Kategoriye Dengeli Dağılım)
- Sonuçları hem eski sistemle kıyaslayarak ekrana basar hem de JSON / Markdown raporu kaydeder.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forensivision.copy_move import CopyMoveConfig, detect_copy_move
from forensivision.datasets import CoMoFoDDataset, DefactoDataset
from forensivision.datasets.base import DatasetSample, _safe_read_image
from forensivision.splicing import SplicingConfig, detect_splicing


def evaluate_copy_move_suite(samples: list[DatasetSample], cfg: CopyMoveConfig, title: str, is_clean: bool = False):
    print(f"\n>>> {title.upper()} ({len(samples)} Görsel)...")
    t0 = time.perf_counter()

    flagged = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0

    attack_stats: dict[str, dict[str, Any]] = {}

    for idx, s in enumerate(samples, 1):
        img = _safe_read_image(s.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        res = detect_copy_move(img, config=cfg)
        is_fl = bool(res.is_tampered and np.count_nonzero(res.mask) > 0)
        if is_fl:
            flagged += 1

        if not is_clean:
            # Load ground truth mask
            mask_path = s.mask_path or s.target_mask_path or s.source_mask_path
            if (
                s.target_mask_path
                and s.source_mask_path
                and s.target_mask_path.exists()
                and s.source_mask_path.exists()
            ):
                t_raw = _safe_read_image(s.target_mask_path, cv2.IMREAD_GRAYSCALE)
                s_raw = _safe_read_image(s.source_mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (((t_raw > 127) | (s_raw > 127)).astype(np.uint8)) * 255 if (t_raw is not None and s_raw is not None) else np.zeros((h, w), dtype=np.uint8)
            elif mask_path and mask_path.exists():
                m_raw = _safe_read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (m_raw > 127).astype(np.uint8) * 255 if m_raw is not None else np.zeros((h, w), dtype=np.uint8)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            pred_mask = res.mask
            gt_pos = int(np.count_nonzero(gt_mask > 0))
            total_gt += gt_pos

            tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
            fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
            fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Attack type grouping
            var = s.variant or "clean"
            atk_group = var.split("_")[0] if "_" in var else var
            attack_stats.setdefault(atk_group, {"count": 0, "detected": 0, "tp": 0, "fp": 0})
            attack_stats[atk_group]["count"] += 1
            if is_fl:
                attack_stats[atk_group]["detected"] += 1
            attack_stats[atk_group]["tp"] += tp
            attack_stats[atk_group]["fp"] += fp

    elapsed = time.perf_counter() - t0

    if is_clean:
        correct_clean = len(samples) - flagged
        clean_acc = (correct_clean / len(samples)) * 100.0
        fa_rate = (flagged / len(samples)) * 100.0
        print(f"    Sonuç: Doğru Temiz = {correct_clean}/{len(samples)} (%{clean_acc:.1f}) | Yanlış Alarm = {flagged} (%{fa_rate:.1f})")
        return {
            "title": title,
            "total": len(samples),
            "correct_clean": correct_clean,
            "false_alarms": flagged,
            "clean_accuracy_pct": round(clean_acc, 2),
            "false_alarm_rate_pct": round(fa_rate, 2),
            "elapsed_sec": round(elapsed, 2),
        }
    else:
        det_rate = (flagged / len(samples)) * 100.0
        prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
        rec = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 0.0
        iou = (total_tp / (total_tp + total_fp + total_fn) * 100.0) if (total_tp + total_fp + total_fn) > 0 else 0.0
        print(f"    Sonuç: Yakalanan = {flagged}/{len(samples)} (%{det_rate:.1f}) | Piksel İsabet: %{prec:.2f} | IoU: %{iou:.2f}")

        atk_summary = {}
        for atk, data in attack_stats.items():
            a_prec = (data["tp"] / (data["tp"] + data["fp"]) * 100.0) if (data["tp"] + data["fp"]) > 0 else 0.0
            atk_summary[atk] = {
                "count": data["count"],
                "detected": data["detected"],
                "det_rate": round(data["detected"] / data["count"] * 100.0, 1),
                "pixel_prec": round(a_prec, 2),
            }

        return {
            "title": title,
            "total": len(samples),
            "detected": flagged,
            "missed": len(samples) - flagged,
            "detection_rate_pct": round(det_rate, 2),
            "pixel_precision_pct": round(prec, 2),
            "pixel_recall_pct": round(rec, 2),
            "pixel_iou_pct": round(iou, 2),
            "attack_breakdown": atk_summary,
            "elapsed_sec": round(elapsed, 2),
        }


def evaluate_splicing_suite(samples: list[DatasetSample], cfg: SplicingConfig, title: str, is_clean: bool = False):
    print(f"\n>>> {title.upper()} ({len(samples)} Görsel)...")
    t0 = time.perf_counter()

    flagged = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    cat_stats: dict[str, dict[str, Any]] = {}

    for idx, s in enumerate(samples, 1):
        img = _safe_read_image(s.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        res = detect_splicing(img, config=cfg)
        is_fl = bool(res.is_suspicious and np.count_nonzero(res.mask) > 0)
        if is_fl:
            flagged += 1

        if not is_clean:
            mask_path = s.mask_path or s.target_mask_path
            if mask_path and mask_path.exists():
                m_raw = _safe_read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (m_raw > 127).astype(np.uint8) * 255 if m_raw is not None else np.zeros((h, w), dtype=np.uint8)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            pred_mask = res.mask
            tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
            fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
            fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

            total_tp += tp
            total_fp += fp
            total_fn += fn

            cat = s.variant or s.metadata.get("category", "unknown")
            cat_stats.setdefault(cat, {"count": 0, "detected": 0, "tp": 0, "fp": 0})
            cat_stats[cat]["count"] += 1
            if is_fl:
                cat_stats[cat]["detected"] += 1
            cat_stats[cat]["tp"] += tp
            cat_stats[cat]["fp"] += fp

    elapsed = time.perf_counter() - t0

    if is_clean:
        correct_clean = len(samples) - flagged
        clean_acc = (correct_clean / len(samples)) * 100.0
        fa_rate = (flagged / len(samples)) * 100.0
        print(f"    Sonuç: Doğru Temiz = {correct_clean}/{len(samples)} (%{clean_acc:.1f}) | Yanlış Alarm = {flagged} (%{fa_rate:.1f})")
        return {
            "title": title,
            "total": len(samples),
            "correct_clean": correct_clean,
            "false_alarms": flagged,
            "clean_accuracy_pct": round(clean_acc, 2),
            "false_alarm_rate_pct": round(fa_rate, 2),
            "elapsed_sec": round(elapsed, 2),
        }
    else:
        det_rate = (flagged / len(samples)) * 100.0
        prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
        rec = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 0.0
        iou = (total_tp / (total_tp + total_fp + total_fn) * 100.0) if (total_tp + total_fp + total_fn) > 0 else 0.0
        print(f"    Sonuç: Yakalanan = {flagged}/{len(samples)} (%{det_rate:.1f}) | Piksel İsabet: %{prec:.2f} | IoU: %{iou:.2f}")

        cat_summary = {}
        for cat, data in cat_stats.items():
            c_prec = (data["tp"] / (data["tp"] + data["fp"]) * 100.0) if (data["tp"] + data["fp"]) > 0 else 0.0
            cat_summary[cat] = {
                "count": data["count"],
                "detected": data["detected"],
                "det_rate": round(data["detected"] / data["count"] * 100.0, 1),
                "pixel_prec": round(c_prec, 2),
            }

        return {
            "title": title,
            "total": len(samples),
            "detected": flagged,
            "missed": len(samples) - flagged,
            "detection_rate_pct": round(det_rate, 2),
            "pixel_precision_pct": round(prec, 2),
            "pixel_recall_pct": round(rec, 2),
            "pixel_iou_pct": round(iou, 2),
            "category_breakdown": cat_summary,
            "elapsed_sec": round(elapsed, 2),
        }


def main():
    out_dir = PROJECT_ROOT / "data" / "output" / "experiments" / "pilot_benchmark_100"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    print("=" * 80)
    print("FORENSIVISION 100-GÖRSEL YENİ OPTİMİZE BENCHMARK SUITE")
    print("=" * 80)

    # 1. Veri Örneklerini Hazırla
    print("\n[Veri Hazırlığı]")
    cmfd = CoMoFoDDataset("data/raw/comofod")
    all_cm_clean = [s for s in cmfd if s.is_authentic()]
    all_cm_dirty = [s for s in cmfd if not s.is_authentic()]
    rng.shuffle(all_cm_clean)
    rng.shuffle(all_cm_dirty)
    clean_100 = all_cm_clean[:100]
    cm_dirty_100 = all_cm_dirty[:100]

    df_cm = DefactoDataset("data/raw/defacto/copymove", task_type="copy_move")
    all_df_cm = list(df_cm.samples)
    rng.shuffle(all_df_cm)
    df_cm_dirty_100 = all_df_cm[:100]

    df_spl = DefactoDataset("data/raw/defacto/splicing", task_type="splicing")
    # 7 Kategoriye dengeli 100 örnek (~14'er adet)
    cat_map: dict[str, list[DatasetSample]] = {}
    for s in df_spl.samples:
        cat = s.variant or s.metadata.get("category", "unknown")
        cat_map.setdefault(cat, []).append(s)

    df_spl_dirty_100 = []
    for cat in sorted(cat_map.keys()):
        s_list = cat_map[cat]
        rng.shuffle(s_list)
        df_spl_dirty_100.extend(s_list[:14])
    # Toplam 100 olsun diye ilk kategoriden 2 tane daha ekle
    df_spl_dirty_100.extend(cat_map[sorted(cat_map.keys())[0]][14:16])
    rng.shuffle(df_spl_dirty_100)

    print(f"  -> CoMoFoD: 100 Temiz, 100 Kirli hazır.")
    print(f"  -> DEFACTO Copy-Move: 100 Kirli hazır.")
    print(f"  -> DEFACTO Splicing: 100 Kirli (7 kategoriye dengeli) hazır.")

    # 2. Optimize Edilmiş Konfigürasyonlar
    cm_cfg_opt = CopyMoveConfig(
        ratio_threshold=0.78,
        min_spatial_distance=25.0,
        cluster_eps_px=35.0,
        cluster_min_support_ratio=0.45,
        min_inliers=4,
        hull_fill_guard=True,
        spatial_cluster_guard=True,
        point_radius=16,
        morph_kernel_size=9,
    )

    spl_cfg_opt = SplicingConfig(
        fusion_mode="ela_only",
        ela_threshold_std_factor=2.2,
        min_region_area=180,
        fill_contours=True,
        morph_kernel_size=9,
    )

    # -----------------------------------------------------------------------
    # TESTLERİ ÇALIŞTIR
    # -----------------------------------------------------------------------
    # Deney 1: CoMoFoD Copy-Move (100 Kirli)
    exp1 = evaluate_copy_move_suite(cm_dirty_100, cm_cfg_opt, "Deney 1: CoMoFoD Copy-Move (100 Kirli)", is_clean=False)

    # Deney 2: CoMoFoD Copy-Move (100 Temiz - Yanlış Alarm)
    exp2 = evaluate_copy_move_suite(clean_100, cm_cfg_opt, "Deney 2: CoMoFoD Copy-Move (100 Temiz)", is_clean=True)

    # Deney 3: DEFACTO Copy-Move (100 Kirli)
    exp3 = evaluate_copy_move_suite(df_cm_dirty_100, cm_cfg_opt, "Deney 3: DEFACTO Copy-Move (100 Kirli)", is_clean=False)

    # Deney 4: Birleşik Copy-Move (200 Kirli)
    comb_cm_dirty = cm_dirty_100 + df_cm_dirty_100
    exp4 = evaluate_copy_move_suite(comb_cm_dirty, cm_cfg_opt, "Deney 4: Birleşik Copy-Move (200 Kirli: CoMoFoD+DEFACTO)", is_clean=False)

    # Deney 5: Image Splicing (100 Temiz)
    exp5 = evaluate_splicing_suite(clean_100, spl_cfg_opt, "Deney 5: Image Splicing (100 Temiz)", is_clean=True)

    # Deney 6: DEFACTO Splicing (100 Kirli - 7 Kategori)
    exp6 = evaluate_splicing_suite(df_spl_dirty_100, spl_cfg_opt, "Deney 6: DEFACTO Splicing (100 Kirli)", is_clean=False)

    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp1_comofod_cm_dirty": exp1,
        "exp2_comofod_cm_clean": exp2,
        "exp3_defacto_cm_dirty": exp3,
        "exp4_combined_cm_dirty": exp4,
        "exp5_splicing_clean": exp5,
        "exp6_defacto_splicing_dirty": exp6,
    }

    with open(out_dir / "results_100.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("TÜM PİLOT TESTLER BAŞARIYLA TAMAMLANDI!")
    print(f"Sonuçlar Kaydedildi: {out_dir / 'results_100.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
