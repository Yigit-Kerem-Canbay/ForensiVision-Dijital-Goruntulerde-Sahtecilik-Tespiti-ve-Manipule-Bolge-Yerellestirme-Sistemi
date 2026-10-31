"""
ForensiVision Sade ve Kapsamlı Benchmark Değerlendirme Betiği.

Bu betik; kullanıcının talebi doğrultusunda:
- Copy-Move için 500 Temiz + 500 Kirli görseli,
- Image Splicing için 500 Temiz + 500 Kirli görseli
baştan sona adli teste tabi tutar.

Çıktılar:
- Temiz görsellerde başarı (Doğru Negatif ve Yanlış Alarm)
- Kirli görsellerde tespit oranı (Görsel bazlı yakalama)
- Piksel düzeyinde isabet analizi (Şans eseri mi yoksa tam isabet mi?)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import cv2
import numpy as np

from forensivision.copy_move import CopyMoveConfig, detect_copy_move
from forensivision.datasets import BaseDataset, CoMoFoDDataset, DefactoDataset
from forensivision.splicing import SplicingConfig, detect_splicing


def evaluate_copy_move(
    clean_ds: BaseDataset,
    dirty_ds: BaseDataset,
    clean_samples: list,
    dirty_samples: list,
    output_dir: Path,
) -> dict:
    print("\n" + "=" * 65)
    print("1. BÖLÜM: COPY-MOVE (KOPYALA-KAYDIR) ANALİZİ BAŞLIYOR")
    print(f"Toplam Test: {len(clean_samples)} Temiz + {len(dirty_samples)} Kirli = {len(clean_samples) + len(dirty_samples)} Görsel")
    print("=" * 65)

    cfg = CopyMoveConfig(
        hull_fill_guard=True,
        spatial_cluster_guard=True,
        min_spatial_distance=20.0,
        ratio_threshold=0.78,
        point_radius=16,
        morph_kernel_size=9,
    )

    # A. 500 TEMİZ GÖRSEL ANALİZİ
    print("\n>>> [1/2] 500 Temiz Görsel Test Ediliyor (Yanlış Alarm Denetimi)...")
    clean_correct = 0
    clean_false_alarms = 0
    clean_times = []

    for idx, sample in enumerate(clean_samples, 1):
        t0 = time.perf_counter()
        img = clean_ds.load_image(sample)
        res = detect_copy_move(img, config=cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        clean_times.append(elapsed_ms)

        is_flagged = res.is_tampered and (np.count_nonzero(res.mask) > 0)
        if is_flagged:
            clean_false_alarms += 1
        else:
            clean_correct += 1

        if idx % 100 == 0 or idx == len(clean_samples):
            print(f"    - İşlenen: {idx}/{len(clean_samples)} | Doğru Temiz: {clean_correct} | Hatalı Alarm: {clean_false_alarms}")

    clean_accuracy = (clean_correct / len(clean_samples)) * 100.0
    clean_fa_rate = (clean_false_alarms / len(clean_samples)) * 100.0

    # B. 500 KİRLİ GÖRSEL ANALİZİ
    print("\n>>> [2/2] 500 Kirli Görsel Test Ediliyor (Piksel Hassasiyeti Denetimi)...")
    dirty_detected = 0
    dirty_missed = 0

    total_tp_pixels = 0
    total_fp_pixels = 0
    total_fn_pixels = 0
    total_gt_tampered_pixels = 0

    sample_ious = []
    sample_precisions = []
    sample_recalls = []
    dirty_times = []

    for idx, sample in enumerate(dirty_samples, 1):
        t0 = time.perf_counter()
        img = dirty_ds.load_image(sample)
        h, w = img.shape[:2]
        gt_mask = dirty_ds.load_mask(sample, target_shape=(h, w))
        res = detect_copy_move(img, config=cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dirty_times.append(elapsed_ms)

        pred_mask = res.mask
        is_flagged = res.is_tampered and (np.count_nonzero(pred_mask) > 0)

        if is_flagged:
            dirty_detected += 1
        else:
            dirty_missed += 1

        gt_pos = np.count_nonzero(gt_mask > 0)
        total_gt_tampered_pixels += gt_pos

        tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
        fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
        fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

        total_tp_pixels += tp
        total_fp_pixels += fp
        total_fn_pixels += fn

        union = tp + fp + fn
        iou = (tp / union) if union > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        sample_ious.append(iou)
        sample_precisions.append(prec)
        sample_recalls.append(rec)

        if idx % 100 == 0 or idx == len(dirty_samples):
            current_micro_prec = (total_tp_pixels / (total_tp_pixels + total_fp_pixels) * 100.0) if (total_tp_pixels + total_fp_pixels) > 0 else 0.0
            print(f"    - İşlenen: {idx}/{len(dirty_samples)} | Yakalanan Kirli: {dirty_detected} | Piksel İsabet: %{current_micro_prec:.2f}")

    dirty_detection_rate = (dirty_detected / len(dirty_samples)) * 100.0
    pixel_precision = (total_tp_pixels / (total_tp_pixels + total_fp_pixels) * 100.0) if (total_tp_pixels + total_fp_pixels) > 0 else 0.0
    pixel_recall = (total_tp_pixels / (total_tp_pixels + total_fn_pixels) * 100.0) if (total_tp_pixels + total_fn_pixels) > 0 else 0.0
    pixel_iou = (total_tp_pixels / (total_tp_pixels + total_fp_pixels + total_fn_pixels) * 100.0) if (total_tp_pixels + total_fp_pixels + total_fn_pixels) > 0 else 0.0

    return {
        "task": "copy_move",
        "clean_summary": {
            "total_tested": len(clean_samples),
            "correct_clean": clean_correct,
            "false_alarms": clean_false_alarms,
            "accuracy_percentage": round(clean_accuracy, 2),
            "false_alarm_rate_percentage": round(clean_fa_rate, 2),
            "mean_time_ms": round(float(np.mean(clean_times)), 1),
        },
        "dirty_summary": {
            "total_tested": len(dirty_samples),
            "detected_as_dirty": dirty_detected,
            "missed_as_clean": dirty_missed,
            "detection_rate_percentage": round(dirty_detection_rate, 2),
            "mean_time_ms": round(float(np.mean(dirty_times)), 1),
        },
        "pixel_summary": {
            "total_manipulated_pixels": total_gt_tampered_pixels,
            "correctly_marked_pixels_tp": total_tp_pixels,
            "wrongly_marked_pixels_fp": total_fp_pixels,
            "missed_tampered_pixels_fn": total_fn_pixels,
            "pixel_precision_percentage": round(pixel_precision, 2),
            "pixel_recall_percentage": round(pixel_recall, 2),
            "pixel_iou_percentage": round(pixel_iou, 2),
            "macro_avg_iou_percentage": round(float(np.mean(sample_ious)) * 100.0, 2),
        },
    }


def evaluate_splicing(
    clean_ds: BaseDataset,
    dirty_ds: BaseDataset,
    clean_samples: list,
    dirty_samples: list,
    output_dir: Path,
) -> dict:
    print("\n" + "=" * 65)
    print("2. BÖLÜM: IMAGE SPLICING (GÖRÜNTÜ EKLEME) ANALİZİ BAŞLIYOR")
    print(f"Toplam Test: {len(clean_samples)} Temiz + {len(dirty_samples)} Kirli = {len(clean_samples) + len(dirty_samples)} Görsel")
    print("=" * 65)

    cfg = SplicingConfig(
        fusion_mode="ela_only",
        fill_contours=True,
        morph_kernel_size=11,
        min_region_area=150,
    )

    # A. 500 TEMİZ GÖRSEL ANALİZİ
    print("\n>>> [1/2] 500 Temiz Görsel Test Ediliyor (Yanlış Alarm Denetimi)...")
    clean_correct = 0
    clean_false_alarms = 0
    clean_times = []

    for idx, sample in enumerate(clean_samples, 1):
        t0 = time.perf_counter()
        img = clean_ds.load_image(sample)
        res = detect_splicing(img, config=cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        clean_times.append(elapsed_ms)

        is_flagged = res.is_suspicious and (np.count_nonzero(res.mask) > 0)
        if is_flagged:
            clean_false_alarms += 1
        else:
            clean_correct += 1

        if idx % 100 == 0 or idx == len(clean_samples):
            print(f"    - İşlenen: {idx}/{len(clean_samples)} | Doğru Temiz: {clean_correct} | Hatalı Alarm: {clean_false_alarms}")

    clean_accuracy = (clean_correct / len(clean_samples)) * 100.0
    clean_fa_rate = (clean_false_alarms / len(clean_samples)) * 100.0

    # B. 500 KİRLİ GÖRSEL ANALİZİ
    print("\n>>> [2/2] 500 Kirli Görsel Test Ediliyor (Piksel Hassasiyeti Denetimi)...")
    dirty_detected = 0
    dirty_missed = 0

    total_tp_pixels = 0
    total_fp_pixels = 0
    total_fn_pixels = 0
    total_gt_tampered_pixels = 0

    sample_ious = []
    sample_precisions = []
    sample_recalls = []
    dirty_times = []

    category_stats: dict[str, dict] = {}

    for idx, sample in enumerate(dirty_samples, 1):
        t0 = time.perf_counter()
        img = dirty_ds.load_image(sample)
        h, w = img.shape[:2]
        gt_mask = dirty_ds.load_mask(sample, target_shape=(h, w))
        res = detect_splicing(img, config=cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dirty_times.append(elapsed_ms)

        pred_mask = res.mask
        is_flagged = res.is_suspicious and (np.count_nonzero(pred_mask) > 0)

        if is_flagged:
            dirty_detected += 1
        else:
            dirty_missed += 1

        gt_pos = np.count_nonzero(gt_mask > 0)
        total_gt_tampered_pixels += gt_pos

        tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
        fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
        fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

        total_tp_pixels += tp
        total_fp_pixels += fp
        total_fn_pixels += fn

        union = tp + fp + fn
        iou = (tp / union) if union > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        sample_ious.append(iou)
        sample_precisions.append(prec)
        sample_recalls.append(rec)

        cat = sample.manipulation_type
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "detected": 0, "tps": 0, "fps": 0, "fns": 0, "ious": []}
        category_stats[cat]["count"] += 1
        if is_flagged:
            category_stats[cat]["detected"] += 1
        category_stats[cat]["tps"] += tp
        category_stats[cat]["fps"] += fp
        category_stats[cat]["fns"] += fn
        category_stats[cat]["ious"].append(iou)

        if idx % 100 == 0 or idx == len(dirty_samples):
            current_micro_prec = (total_tp_pixels / (total_tp_pixels + total_fp_pixels) * 100.0) if (total_tp_pixels + total_fp_pixels) > 0 else 0.0
            print(f"    - İşlenen: {idx}/{len(dirty_samples)} | Yakalanan Kirli: {dirty_detected} | Piksel İsabet: %{current_micro_prec:.2f}")

    dirty_detection_rate = (dirty_detected / len(dirty_samples)) * 100.0
    pixel_precision = (total_tp_pixels / (total_tp_pixels + total_fp_pixels) * 100.0) if (total_tp_pixels + total_fp_pixels) > 0 else 0.0
    pixel_recall = (total_tp_pixels / (total_tp_pixels + total_fn_pixels) * 100.0) if (total_tp_pixels + total_fn_pixels) > 0 else 0.0
    pixel_iou = (total_tp_pixels / (total_tp_pixels + total_fp_pixels + total_fn_pixels) * 100.0) if (total_tp_pixels + total_fp_pixels + total_fn_pixels) > 0 else 0.0

    cat_summary = {}
    for cat, data in category_stats.items():
        c_tp = data["tps"]
        c_fp = data["fps"]
        c_fn = data["fns"]
        c_prec = (c_tp / (c_tp + c_fp) * 100.0) if (c_tp + c_fp) > 0 else 0.0
        c_rec = (c_tp / (c_tp + c_fn) * 100.0) if (c_tp + c_fn) > 0 else 0.0
        c_iou = (c_tp / (c_tp + c_fp + c_fn) * 100.0) if (c_tp + c_fp + c_fn) > 0 else 0.0
        cat_summary[cat] = {
            "count": data["count"],
            "detected": data["detected"],
            "detection_rate": round(data["detected"] / data["count"] * 100.0, 1),
            "precision": round(c_prec, 2),
            "recall": round(c_rec, 2),
            "iou": round(c_iou, 2),
            "mean_iou": round(float(np.mean(data["ious"])) * 100.0, 2),
        }

    return {
        "task": "splicing",
        "clean_summary": {
            "total_tested": len(clean_samples),
            "correct_clean": clean_correct,
            "false_alarms": clean_false_alarms,
            "accuracy_percentage": round(clean_accuracy, 2),
            "false_alarm_rate_percentage": round(clean_fa_rate, 2),
            "mean_time_ms": round(float(np.mean(clean_times)), 1),
        },
        "dirty_summary": {
            "total_tested": len(dirty_samples),
            "detected_as_dirty": dirty_detected,
            "missed_as_clean": dirty_missed,
            "detection_rate_percentage": round(dirty_detection_rate, 2),
            "mean_time_ms": round(float(np.mean(dirty_times)), 1),
        },
        "pixel_summary": {
            "total_manipulated_pixels": total_gt_tampered_pixels,
            "correctly_marked_pixels_tp": total_tp_pixels,
            "wrongly_marked_pixels_fp": total_fp_pixels,
            "missed_tampered_pixels_fn": total_fn_pixels,
            "pixel_precision_percentage": round(pixel_precision, 2),
            "pixel_recall_percentage": round(pixel_recall, 2),
            "pixel_iou_percentage": round(pixel_iou, 2),
            "macro_avg_iou_percentage": round(float(np.mean(sample_ious)) * 100.0, 2),
        },
        "category_summary": cat_summary,
    }


def main():
    out_dir = Path("data/output/experiments/plain_benchmark_500_500")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    # 1. TEMİZ GÖRSELLERİ HAZIRLA (CoMoFoD'dan 500 temiz görsel)
    print("Veri setleri taranıyor...")
    cmfd = CoMoFoDDataset("data/raw/comofod")
    all_clean = [s for s in cmfd if s.is_authentic()]
    rng.shuffle(all_clean)
    clean_500 = all_clean[:500]

    # 2. COPY-MOVE KİRLİ GÖRSELLERİ HAZIRLA (DEFACTO Copy-Move'dan 500 kirli görsel)
    df_cm = DefactoDataset("data/raw/defacto/copymove", task_type="copy_move")
    all_df_cm = list(df_cm.samples)
    rng.shuffle(all_df_cm)
    cm_dirty_500 = all_df_cm[:500]

    # 3. SPLICING KİRLİ GÖRSELLERİ HAZIRLA (DEFACTO Splicing'den 500 kirli görsel)
    df_spl = DefactoDataset("data/raw/defacto/splicing", task_type="splicing")
    all_df_spl = list(df_spl.samples)
    rng.shuffle(all_df_spl)
    spl_dirty_500 = all_df_spl[:500]

    # A. COPY-MOVE TESTİ
    cm_report = evaluate_copy_move(cmfd, df_cm, clean_500, cm_dirty_500, out_dir)

    # B. SPLICING TESTİ
    spl_report = evaluate_splicing(cmfd, df_spl, clean_500, spl_dirty_500, out_dir)

    # C. JSON RAPORU KAYDET
    full_report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "copy_move": cm_report,
        "splicing": spl_report,
    }

    def _numpy_json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=_numpy_json_default)


    print("\n" + "=" * 65)
    print("TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
    print(f"Rapor Kaydedildi: {out_dir / 'report.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
