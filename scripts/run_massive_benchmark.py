"""
ForensiVision Büyük Ölçekli ve Kapsamlı Adli Manipülasyon Benchmark Motoru.

Bu betik:
- Bilgisayarın konforunu ve kararlılığını korumak için 8 paralel worker kullanır (20 çekirdeğin 12'si boş kalır).
- Windows süreç önceliğini 'BELOW_NORMAL_PRIORITY_CLASS' seviyesine ayarlar (farede veya sistemde en ufak kasma olmaz).
- RAM taşmasını önlemek için görselleri akış (streaming) halinde açar, analiz eder ve anında bellekten boşaltır.
- Her testin ara sonuçlarını anında diske JSON olarak kaydeder.

Deneyler:
1. CoMoFoD Copy-Move (5.000 Kirli)
2. CoMoFoD Copy-Move (5.000 Temiz)
3. DEFACTO Copy-Move (5.000 Kirli)
4. Birleşik Copy-Move (10.000 Kirli = 5.000 CoMoFoD + 5.000 DEFACTO)
5. Image Splicing (5.000 Temiz - CoMoFoD Kontrol Grubu)
6. DEFACTO Splicing (10.000 Kirli - 7 Anlamsal Kategoriye Dengeli Dağılım)
Bonus: CoMoFoD Saldırı Türü ve Dayanıklılık Kırılımı (JPEG, Blur, Noise vb.)
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Proje kök dizinini Python arama yoluna ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forensivision.copy_move import CopyMoveConfig, detect_copy_move
from forensivision.datasets import BaseDataset, CoMoFoDDataset, DefactoDataset
from forensivision.datasets.base import DatasetSample, _safe_read_image
from forensivision.splicing import SplicingConfig, detect_splicing


# ---------------------------------------------------------------------------
# GÜVENLİ ÇOKLU İŞLEMCİ İNİTİALİZER (WINDOWS ÖNCELİK SINIRLAYICI)
# ---------------------------------------------------------------------------
def init_worker():
    """Her alt worker sürecinin Windows önceliğini 'BELOW_NORMAL' olarak ayarlar."""
    try:
        # 0x00004000 = BELOW_NORMAL_PRIORITY_CLASS
        h_proc = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(h_proc), 0x00004000)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GÜVENLİ MASKE VE GÖRÜNTÜ YÜKLEYİCİ
# ---------------------------------------------------------------------------
def _load_mask_for_sample(sample: DatasetSample, target_shape: tuple[int, int]) -> np.ndarray:
    """Tekil bir örnek için ikili (0-255) zemin doğruluk maskesini yükler."""
    if sample.is_authentic():
        return np.zeros(target_shape, dtype=np.uint8)

    # 1. DEFACTO Copy-Move gibi hem hedef hem kaynak maskesi olan veri setleri
    if (
        sample.target_mask_path
        and sample.source_mask_path
        and sample.target_mask_path.exists()
        and sample.source_mask_path.exists()
    ):
        tgt_raw = _safe_read_image(sample.target_mask_path, cv2.IMREAD_GRAYSCALE)
        src_raw = _safe_read_image(sample.source_mask_path, cv2.IMREAD_GRAYSCALE)
        if tgt_raw is None or src_raw is None:
            return np.zeros(target_shape, dtype=np.uint8)
        tgt_bin = (tgt_raw > 127) if tgt_raw.max() > 127 else (tgt_raw > 0)
        src_bin = (src_raw > 127) if src_raw.max() > 127 else (src_raw > 0)
        bin_mask = ((tgt_bin | src_bin).astype(np.uint8)) * 255
    else:
        # 2. CoMoFoD veya DEFACTO Splicing (tekil maske)
        mask_path = sample.mask_path or sample.target_mask_path or sample.source_mask_path
        if mask_path is None or not mask_path.exists():
            return np.zeros(target_shape, dtype=np.uint8)
        mask_raw = _safe_read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            return np.zeros(target_shape, dtype=np.uint8)
        if mask_raw.max() > 127:
            bin_mask = (mask_raw > 127).astype(np.uint8) * 255
        else:
            bin_mask = (mask_raw > 0).astype(np.uint8) * 255

    if bin_mask.shape != target_shape:
        bin_mask = cv2.resize(bin_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return bin_mask


# ---------------------------------------------------------------------------
# WORKER ANALİZ FONKSİYONLARI
# ---------------------------------------------------------------------------
def _worker_copy_move(sample: DatasetSample) -> dict[str, Any]:
    """Copy-Move tespit motorunu tek bir görsel üzerinde çalıştırır."""
    t0 = time.perf_counter()
    img = _safe_read_image(sample.image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"sample_id": sample.sample_id, "error": "Image read error"}

    h, w = img.shape[:2]
    cfg = CopyMoveConfig(
        hull_fill_guard=True,
        spatial_cluster_guard=True,
        min_spatial_distance=20.0,
        ratio_threshold=0.78,
        point_radius=16,
        morph_kernel_size=9,
    )

    res = detect_copy_move(img, config=cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred_mask = res.mask
    is_flagged = bool(res.is_tampered and (np.count_nonzero(pred_mask) > 0))

    if sample.is_authentic():
        # Temiz Görsel Mantığı
        return {
            "sample_id": sample.sample_id,
            "is_authentic": True,
            "is_flagged": is_flagged,
            "elapsed_ms": elapsed_ms,
            "variant": sample.variant or "authentic",
            "group_id": sample.group_id,
        }
    else:
        # Kirli Görsel Mantığı
        gt_mask = _load_mask_for_sample(sample, target_shape=(h, w))
        gt_pos = int(np.count_nonzero(gt_mask > 0))

        tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
        fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
        fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            "sample_id": sample.sample_id,
            "is_authentic": False,
            "is_flagged": is_flagged,
            "elapsed_ms": elapsed_ms,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_pos": gt_pos,
            "iou": iou,
            "prec": prec,
            "rec": rec,
            "variant": sample.variant or "unknown",
            "group_id": sample.group_id,
        }


def _worker_splicing(sample: DatasetSample) -> dict[str, Any]:
    """Splicing tespit motorunu tek bir görsel üzerinde çalıştırır."""
    t0 = time.perf_counter()
    img = _safe_read_image(sample.image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"sample_id": sample.sample_id, "error": "Image read error"}

    h, w = img.shape[:2]
    cfg = SplicingConfig(
        fusion_mode="ela_only",
        fill_contours=True,
        morph_kernel_size=11,
        min_region_area=150,
    )

    res = detect_splicing(img, config=cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred_mask = res.mask
    is_flagged = bool(res.is_suspicious and (np.count_nonzero(pred_mask) > 0))

    category = sample.variant or sample.metadata.get("category", "unknown")

    if sample.is_authentic():
        # Temiz Görsel Mantığı
        return {
            "sample_id": sample.sample_id,
            "is_authentic": True,
            "is_flagged": is_flagged,
            "elapsed_ms": elapsed_ms,
            "category": "authentic",
        }
    else:
        # Kirli Görsel Mantığı
        gt_mask = _load_mask_for_sample(sample, target_shape=(h, w))
        gt_pos = int(np.count_nonzero(gt_mask > 0))

        tp = int(np.count_nonzero((pred_mask > 0) & (gt_mask > 0)))
        fp = int(np.count_nonzero((pred_mask > 0) & (gt_mask == 0)))
        fn = int(np.count_nonzero((pred_mask == 0) & (gt_mask > 0)))

        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            "sample_id": sample.sample_id,
            "is_authentic": False,
            "is_flagged": is_flagged,
            "elapsed_ms": elapsed_ms,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_pos": gt_pos,
            "iou": iou,
            "prec": prec,
            "rec": rec,
            "category": category,
        }


# ---------------------------------------------------------------------------
# BATCH YÖNETİCİSİ VE METRİK TOPLAYICI
# ---------------------------------------------------------------------------
def run_evaluation_pool(
    samples: list[DatasetSample],
    task_func,
    test_title: str,
    max_workers: int = 8,
    is_clean_test: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verilen örnek listesini ProcessPoolExecutor ile güvenle ve paralel çalıştırır."""
    print("\n" + "=" * 70)
    print(f">>> {test_title.upper()}")
    print(f"    Toplam İncelenecek Görsel: {len(samples)} adet | Çalışan Çekirdek: {max_workers}")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    # İlerleme sayaçları
    completed = 0
    flagged_count = 0
    tp_acc = 0
    fp_acc = 0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        futures = {executor.submit(task_func, s): s for s in samples}

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1

            if res.get("is_flagged", False):
                flagged_count += 1

            if not is_clean_test:
                tp_acc += res.get("tp", 0)
                fp_acc += res.get("fp", 0)

            if completed % 500 == 0 or completed == len(samples):
                pct = (completed / len(samples)) * 100.0
                if is_clean_test:
                    correct_clean = completed - flagged_count
                    acc_clean = (correct_clean / completed) * 100.0
                    print(
                        f"    [İlerleme: {completed:5d}/{len(samples)} (%{pct:5.1f})] "
                        f"Doğru Temiz: {correct_clean:5d} (%{acc_clean:5.1f}) | Yanlış Alarm: {flagged_count:4d}"
                    )
                else:
                    det_rate = (flagged_count / completed) * 100.0
                    cur_prec = (tp_acc / (tp_acc + fp_acc) * 100.0) if (tp_acc + fp_acc) > 0 else 0.0
                    print(
                        f"    [İlerleme: {completed:5d}/{len(samples)} (%{pct:5.1f})] "
                        f"Yakalanan Kirli: {flagged_count:5d} (%{det_rate:5.1f}) | Piksel İsabet: %{cur_prec:5.2f}"
                    )

    total_time_sec = time.perf_counter() - t_start

    # Metrikleri Hesapla
    if is_clean_test:
        correct_clean = len(samples) - flagged_count
        clean_acc = (correct_clean / len(samples)) * 100.0
        fa_rate = (flagged_count / len(samples)) * 100.0
        summary = {
            "test_title": test_title,
            "total_samples": len(samples),
            "correct_clean": correct_clean,
            "false_alarms": flagged_count,
            "clean_accuracy_pct": round(clean_acc, 2),
            "false_alarm_rate_pct": round(fa_rate, 2),
            "total_time_sec": round(total_time_sec, 1),
            "time_per_image_ms": round((total_time_sec / len(samples)) * 1000.0, 2),
        }
    else:
        det_rate = (flagged_count / len(samples)) * 100.0
        total_tp = sum(r.get("tp", 0) for r in results)
        total_fp = sum(r.get("fp", 0) for r in results)
        total_fn = sum(r.get("fn", 0) for r in results)
        total_gt = sum(r.get("gt_pos", 0) for r in results)

        pixel_prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
        pixel_rec = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 0.0
        pixel_iou = (total_tp / (total_tp + total_fp + total_fn) * 100.0) if (total_tp + total_fp + total_fn) > 0 else 0.0
        mean_iou = float(np.mean([r.get("iou", 0.0) for r in results])) * 100.0

        summary = {
            "test_title": test_title,
            "total_samples": len(samples),
            "detected_as_dirty": flagged_count,
            "missed_as_clean": len(samples) - flagged_count,
            "detection_rate_pct": round(det_rate, 2),
            "pixel_metrics": {
                "total_tampered_pixels_gt": total_gt,
                "correctly_marked_tp": total_tp,
                "wrongly_marked_fp": total_fp,
                "missed_pixels_fn": total_fn,
                "pixel_precision_pct": round(pixel_prec, 2),
                "pixel_recall_pct": round(pixel_rec, 2),
                "pixel_iou_pct": round(pixel_iou, 2),
                "mean_sample_iou_pct": round(mean_iou, 2),
            },
            "total_time_sec": round(total_time_sec, 1),
            "time_per_image_ms": round((total_time_sec / len(samples)) * 1000.0, 2),
        }

    return summary, results


# ---------------------------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------------------------
def main():
    out_dir = PROJECT_ROOT / "data" / "output" / "experiments" / "massive_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    max_workers = 8  # 20 çekirdeğin 8'i kullanılır; kalan 12 çekirdek sisteme ayrılır

    print("\n" + "#" * 70)
    print("FORENSIVISION BÜYÜK ÖLÇEKLİ ADLİ TEST VE BENCHMARK SÜRECİ")
    print(f"İşlemci Gücü: 20 Çekirdek | Ayrılan Güvenli Worker Sayısı: {max_workers}")
    print(f"Çıktı Dizini: {out_dir}")
    print("#" * 70)

    # 1. VERİ SETLERİNİ TARA VE ÖRNEKLERİ HAZIRLA
    print("\n[Adım 1/4] CoMoFoD Veri Seti Yükleniyor...")
    cmfd = CoMoFoDDataset(PROJECT_ROOT / "data" / "raw" / "comofod")
    cmfd_clean = [s for s in cmfd if s.is_authentic()]
    cmfd_dirty = [s for s in cmfd if not s.is_authentic()]
    print(f"    -> CoMoFoD: {len(cmfd_clean)} Temiz, {len(cmfd_dirty)} Kirli hazır.")

    print("\n[Adım 2/4] DEFACTO Copy-Move Veri Seti Yükleniyor...")
    df_cm = DefactoDataset(PROJECT_ROOT / "data" / "raw" / "defacto" / "copymove", task_type="copy_move")
    df_cm_samples = list(df_cm.samples)
    rng.shuffle(df_cm_samples)
    df_cm_dirty_5000 = df_cm_samples[:5000]
    print(f"    -> DEFACTO Copy-Move: Toplam {len(df_cm_samples)} örnekten 5.000 kirli seçildi.")

    print("\n[Adım 3/4] DEFACTO Splicing Veri Seti Yükleniyor ve 7 Kategoriye Dağıtılıyor...")
    df_spl = DefactoDataset(PROJECT_ROOT / "data" / "raw" / "defacto" / "splicing", task_type="splicing")
    
    # 7 Kategoriye göre grupla
    cat_buckets: dict[str, list[DatasetSample]] = {}
    for s in df_spl.samples:
        cat = s.variant or s.metadata.get("category", "unknown")
        cat_buckets.setdefault(cat, []).append(s)

    df_spl_dirty_10000: list[DatasetSample] = []
    # Her kategoriye ~1.428 örnek düşecek şekilde dengeli seç
    categories = sorted(cat_buckets.keys())
    target_per_cat = {
        "airplane": 1428,
        "birds": 1428,
        "clock": 1428,
        "frisbee": 1428,
        "sports_ball": 1428,
        "stop_sign": 1428,
        "traffic_light": 1432,  # Toplam tam 10.000 olsun
    }

    for cat in categories:
        samples_in_cat = cat_buckets[cat]
        rng.shuffle(samples_in_cat)
        needed = target_per_cat.get(cat, 1428)
        selected = samples_in_cat[:needed]
        df_spl_dirty_10000.extend(selected)
        print(f"    -> Kategori '{cat:13s}': {len(samples_in_cat):6d} arasından {len(selected):5d} görsel seçildi.")

    rng.shuffle(df_spl_dirty_10000)
    print(f"    -> DEFACTO Splicing: Toplam {len(df_spl_dirty_10000)} dengeli kirli görsel hazır.")

    # -----------------------------------------------------------------------
    # TEST 1: CoMoFoD Copy-Move (5.000 Kirli)
    # -----------------------------------------------------------------------
    exp1_sum, exp1_raw = run_evaluation_pool(
        samples=cmfd_dirty,
        task_func=_worker_copy_move,
        test_title="Deney 1: CoMoFoD Copy-Move (5.000 Kirli Görsel)",
        max_workers=max_workers,
        is_clean_test=False,
    )
    with open(out_dir / "exp1_comofod_cm_dirty.json", "w", encoding="utf-8") as f:
        json.dump(exp1_sum, f, indent=2, ensure_ascii=False)

    # CoMoFoD Saldırı Türü Kırılımı (Bonus Analiz)
    comofod_attacks: dict[str, dict[str, Any]] = {}
    for r in exp1_raw:
        var = r.get("variant", "unknown")
        # JC1, B1, NA1 vs. saldırı kategorisini al
        atk_group = var.split("_")[0] if "_" in var else var
        comofod_attacks.setdefault(atk_group, {"count": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0})
        comofod_attacks[atk_group]["count"] += 1
        if r.get("is_flagged", False):
            comofod_attacks[atk_group]["detected"] += 1
        comofod_attacks[atk_group]["tp"] += r.get("tp", 0)
        comofod_attacks[atk_group]["fp"] += r.get("fp", 0)
        comofod_attacks[atk_group]["fn"] += r.get("fn", 0)

    attack_breakdown = {}
    for atk, data in comofod_attacks.items():
        c_tp = data["tp"]
        c_fp = data["fp"]
        c_fn = data["fn"]
        c_prec = (c_tp / (c_tp + c_fp) * 100.0) if (c_tp + c_fp) > 0 else 0.0
        c_rec = (c_tp / (c_tp + c_fn) * 100.0) if (c_tp + c_fn) > 0 else 0.0
        c_iou = (c_tp / (c_tp + c_fp + c_fn) * 100.0) if (c_tp + c_fp + c_fn) > 0 else 0.0
        attack_breakdown[atk] = {
            "count": data["count"],
            "detected": data["detected"],
            "detection_rate_pct": round((data["detected"] / data["count"]) * 100.0, 1),
            "pixel_precision_pct": round(c_prec, 2),
            "pixel_recall_pct": round(c_rec, 2),
            "pixel_iou_pct": round(c_iou, 2),
        }

    exp1_sum["attack_breakdown"] = attack_breakdown
    with open(out_dir / "exp1_comofod_cm_dirty.json", "w", encoding="utf-8") as f:
        json.dump(exp1_sum, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # TEST 2: CoMoFoD Copy-Move (5.000 Temiz)
    # -----------------------------------------------------------------------
    exp2_sum, _ = run_evaluation_pool(
        samples=cmfd_clean,
        task_func=_worker_copy_move,
        test_title="Deney 2: CoMoFoD Copy-Move (5.000 Temiz Görsel - Yanlış Alarm)",
        max_workers=max_workers,
        is_clean_test=True,
    )
    with open(out_dir / "exp2_comofod_cm_clean.json", "w", encoding="utf-8") as f:
        json.dump(exp2_sum, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # TEST 3: DEFACTO Copy-Move (5.000 Kirli)
    # -----------------------------------------------------------------------
    exp3_sum, exp3_raw = run_evaluation_pool(
        samples=df_cm_dirty_5000,
        task_func=_worker_copy_move,
        test_title="Deney 3: DEFACTO Copy-Move (5.000 Kirli Görsel)",
        max_workers=max_workers,
        is_clean_test=False,
    )
    with open(out_dir / "exp3_defacto_cm_dirty.json", "w", encoding="utf-8") as f:
        json.dump(exp3_sum, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # TEST 4: BİRLEŞİK Copy-Move (10.000 Kirli = 5.000 CoMoFoD + 5.000 DEFACTO)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> DENEY 4: BİRLEŞİK COPY-MOVE (10.000 KİRLİ GÖRSEL ANALİZİ)")
    print("    5.000 CoMoFoD + 5.000 DEFACTO sonuçları konsolide ediliyor...")
    print("=" * 70)

    comb_raw = exp1_raw + exp3_raw
    comb_flagged = sum(1 for r in comb_raw if r.get("is_flagged", False))
    comb_tp = sum(r.get("tp", 0) for r in comb_raw)
    comb_fp = sum(r.get("fp", 0) for r in comb_raw)
    comb_fn = sum(r.get("fn", 0) for r in comb_raw)
    comb_gt = sum(r.get("gt_pos", 0) for r in comb_raw)

    comb_det_rate = (comb_flagged / len(comb_raw)) * 100.0
    comb_prec = (comb_tp / (comb_tp + comb_fp) * 100.0) if (comb_tp + comb_fp) > 0 else 0.0
    comb_rec = (comb_tp / (comb_tp + comb_fn) * 100.0) if (comb_tp + comb_fn) > 0 else 0.0
    comb_iou = (comb_tp / (comb_tp + comb_fp + comb_fn) * 100.0) if (comb_tp + comb_fp + comb_fn) > 0 else 0.0
    comb_mean_iou = float(np.mean([r.get("iou", 0.0) for r in comb_raw])) * 100.0

    exp4_sum = {
        "test_title": "Deney 4: Birleşik Copy-Move (10.000 Kirli: 5.000 CoMoFoD + 5.000 DEFACTO)",
        "total_samples": len(comb_raw),
        "detected_as_dirty": comb_flagged,
        "missed_as_clean": len(comb_raw) - comb_flagged,
        "detection_rate_pct": round(comb_det_rate, 2),
        "pixel_metrics": {
            "total_tampered_pixels_gt": comb_gt,
            "correctly_marked_tp": comb_tp,
            "wrongly_marked_fp": comb_fp,
            "missed_pixels_fn": comb_fn,
            "pixel_precision_pct": round(comb_prec, 2),
            "pixel_recall_pct": round(comb_rec, 2),
            "pixel_iou_pct": round(comb_iou, 2),
            "mean_sample_iou_pct": round(comb_mean_iou, 2),
        },
        "dataset_comparison": {
            "comofod_detection_rate_pct": exp1_sum["detection_rate_pct"],
            "comofod_pixel_precision_pct": exp1_sum["pixel_metrics"]["pixel_precision_pct"],
            "defacto_detection_rate_pct": exp3_sum["detection_rate_pct"],
            "defacto_pixel_precision_pct": exp3_sum["pixel_metrics"]["pixel_precision_pct"],
        },
    }
    with open(out_dir / "exp4_combined_cm_dirty.json", "w", encoding="utf-8") as f:
        json.dump(exp4_sum, f, indent=2, ensure_ascii=False)
    print(f"    -> 10.000 Kirli Görselde Yakalama: %{comb_det_rate:.2f} | Piksel İsabet: %{comb_prec:.2f}")

    # -----------------------------------------------------------------------
    # TEST 5: Splicing Temiz Kontrol Grubu (5.000 Temiz Görsel)
    # -----------------------------------------------------------------------
    exp5_sum, _ = run_evaluation_pool(
        samples=cmfd_clean,
        task_func=_worker_splicing,
        test_title="Deney 5: Image Splicing (5.000 Temiz Görsel - Yanlış Alarm)",
        max_workers=max_workers,
        is_clean_test=True,
    )
    with open(out_dir / "exp5_splicing_clean.json", "w", encoding="utf-8") as f:
        json.dump(exp5_sum, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # TEST 6: DEFACTO Splicing (10.000 Kirli Görsel - 7 Kategori)
    # -----------------------------------------------------------------------
    exp6_sum, exp6_raw = run_evaluation_pool(
        samples=df_spl_dirty_10000,
        task_func=_worker_splicing,
        test_title="Deney 6: DEFACTO Splicing (10.000 Kirli Görsel - 7 Kategori Dağılımı)",
        max_workers=max_workers,
        is_clean_test=False,
    )

    # 7 Kategori Kırılımı
    spl_cat_stats: dict[str, dict[str, Any]] = {}
    for r in exp6_raw:
        cat = r.get("category", "unknown")
        spl_cat_stats.setdefault(cat, {"count": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []})
        spl_cat_stats[cat]["count"] += 1
        if r.get("is_flagged", False):
            spl_cat_stats[cat]["detected"] += 1
        spl_cat_stats[cat]["tp"] += r.get("tp", 0)
        spl_cat_stats[cat]["fp"] += r.get("fp", 0)
        spl_cat_stats[cat]["fn"] += r.get("fn", 0)
        spl_cat_stats[cat]["ious"].append(r.get("iou", 0.0))

    cat_breakdown = {}
    for cat, data in spl_cat_stats.items():
        c_tp = data["tp"]
        c_fp = data["fp"]
        c_fn = data["fn"]
        c_prec = (c_tp / (c_tp + c_fp) * 100.0) if (c_tp + c_fp) > 0 else 0.0
        c_rec = (c_tp / (c_tp + c_fn) * 100.0) if (c_tp + c_fn) > 0 else 0.0
        c_iou = (c_tp / (c_tp + c_fp + c_fn) * 100.0) if (c_tp + c_fp + c_fn) > 0 else 0.0
        cat_breakdown[cat] = {
            "count": data["count"],
            "detected": data["detected"],
            "detection_rate_pct": round((data["detected"] / data["count"]) * 100.0, 1),
            "pixel_precision_pct": round(c_prec, 2),
            "pixel_recall_pct": round(c_rec, 2),
            "pixel_iou_pct": round(c_iou, 2),
            "mean_iou_pct": round(float(np.mean(data["ious"])) * 100.0, 2),
        }

    exp6_sum["category_breakdown"] = cat_breakdown
    with open(out_dir / "exp6_defacto_splicing_dirty.json", "w", encoding="utf-8") as f:
        json.dump(exp6_sum, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # KONSOLİDE RAPOR
    # -----------------------------------------------------------------------
    full_report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_profile": {
            "total_logical_cpus": os.cpu_count(),
            "workers_allocated": max_workers,
            "priority": "BELOW_NORMAL_PRIORITY_CLASS (Safe Desktop Experience)",
        },
        "experiments": {
            "exp1_comofod_cm_dirty": exp1_sum,
            "exp2_comofod_cm_clean": exp2_sum,
            "exp3_defacto_cm_dirty": exp3_sum,
            "exp4_combined_cm_dirty": exp4_sum,
            "exp5_splicing_clean": exp5_sum,
            "exp6_defacto_splicing_dirty": exp6_sum,
        },
    }

    def _numpy_json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(out_dir / "consolidated_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=_numpy_json_default)

    print("\n" + "#" * 70)
    print("TÜM ADLİ TESTLER VE BENCHMARK BAŞARIYLA TAMAMLANDI!")
    print(f"Bütün Sonuçlar Diske Kaydedildi: {out_dir / 'consolidated_report.json'}")
    print("#" * 70)


if __name__ == "__main__":
    main()
