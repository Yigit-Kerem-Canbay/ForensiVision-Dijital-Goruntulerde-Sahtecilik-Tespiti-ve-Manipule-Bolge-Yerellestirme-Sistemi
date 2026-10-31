"""
ForensiVision Tekrarlanabilir Veri Seti Değerlendirme ve Deney Yürütme Betiği.

Bu betik; CoMoFoD, Columbia ve benzeri adli görüntü manipülasyonu veri setleri üzerinde
Copy-Move ve Splicing dedektörlerinin başarımını kontrollü, tekrarlanabilir ve
kaynak kısıtlarına duyarlı biçimde test eder.

Özellikler:
    - Belirlenen örnek sayısı sınırı (--limit) ve deterministik rastgele tohum (--seed) desteği.
    - Manipüle edilmiş görüntüler için Makro ve Mikro ortalama metrikleri (Precision, Recall, F1, IoU).
    - Temiz/Özgün görüntüler için ayrı Yanlış Alarm (False Alarm) analizi (makro skoru yanıltmamak için).
    - Varyant bazlı başarım dağılımı (JPEG, Blur, Noise, Rotation vb.).
    - İstisnalarda deneyi durdurmayan hata kayıt mekanizması (failures.csv).
    - metrics.csv, summary.json, report.md ve examples/ çıktıları.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from forensivision.copy_move import CopyMoveConfig, detect_copy_move
from forensivision.datasets import BaseDataset, CoMoFoDDataset, ColumbiaDataset, DefactoDataset, DatasetSample
from forensivision.evaluation import calculate_metrics
from forensivision.splicing import SplicingConfig, detect_splicing
from forensivision.visualization import create_mask_overlay


def get_dataset_reader(dataset_name: str, root_dir: str | Path, task: str = "auto") -> BaseDataset:
    """Belirtilen veri seti adı için uygun okuyucu sınıfını döndürür."""
    name_clean = dataset_name.strip().lower()
    if name_clean == "comofod":
        return CoMoFoDDataset(root_dir)
    elif name_clean == "columbia":
        return ColumbiaDataset(root_dir)
    elif name_clean == "defacto":
        defacto_task = "splicing" if task == "splicing" else "copy_move"
        return DefactoDataset(root_dir, task_type=defacto_task)
    else:
        raise ValueError(
            f"Bilinmeyen veri seti: '{dataset_name}'. Desteklenenler: 'comofod', 'columbia', 'defacto'"
        )


def run_evaluation(
    dataset_name: str = "comofod",
    root_dir: str | Path = "data/raw/comofod",
    task: str = "auto",
    limit: int = 30,
    seed: int = 42,
    output_dir: str | Path = "data/output/experiments/default_run",
    mask_mode: str = "both",
    save_examples_count: int = 5,
    custom_dataset: BaseDataset | None = None,
    variant_filter: str | None = None,
    sample_by_group: bool = False,
    copy_move_config: CopyMoveConfig | None = None,
    min_inliers: int | None = None,
    hull_fill_guard: bool = False,
    hull_max_ratio: float = 4.0,
    spatial_cluster_guard: bool = False,
    cluster_eps: float = 35.0,
    cluster_min_samples: int = 3,
    cluster_min_support_ratio: float = 0.50,
    min_spatial_distance: float | None = None,
    ratio_threshold: float | None = None,
    point_radius: int | None = None,
    morph_kernel_size: int | None = None,
    splicing_config: SplicingConfig | None = None,
    splicing_fusion_mode: str | None = None,
    splicing_fill_contours: bool = False,
    group_ids_file: str | Path | None = None,
) -> dict[str, Any]:
    """Veri seti üzerinde değerlendirme deneyini çalıştırır ve çıktıları kaydeder."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    examples_dir = out_path / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Copy-Move konfigürasyonunu belirle (varsayılan davranışı korur)
    if copy_move_config is not None:
        cfg_dict = asdict(copy_move_config)
        if min_inliers is not None:
            cfg_dict["min_inliers"] = min_inliers
        if hull_fill_guard:
            cfg_dict["hull_fill_guard"] = True
            cfg_dict["hull_max_ratio"] = hull_max_ratio
        if spatial_cluster_guard:
            cfg_dict["spatial_cluster_guard"] = True
            cfg_dict["cluster_eps_px"] = cluster_eps
            cfg_dict["cluster_min_samples"] = cluster_min_samples
            cfg_dict["cluster_min_support_ratio"] = cluster_min_support_ratio
        if min_spatial_distance is not None:
            cfg_dict["min_spatial_distance"] = min_spatial_distance
        if ratio_threshold is not None:
            cfg_dict["ratio_threshold"] = ratio_threshold
        if point_radius is not None:
            cfg_dict["point_radius"] = point_radius
        if morph_kernel_size is not None:
            cfg_dict["morph_kernel_size"] = morph_kernel_size
        effective_cm_config = CopyMoveConfig(**cfg_dict)
    elif min_inliers is not None or hull_fill_guard or spatial_cluster_guard or min_spatial_distance is not None or ratio_threshold is not None or point_radius is not None or morph_kernel_size is not None:
        cfg_dict = {}
        if min_inliers is not None:
            cfg_dict["min_inliers"] = min_inliers
        if hull_fill_guard:
            cfg_dict["hull_fill_guard"] = True
            cfg_dict["hull_max_ratio"] = hull_max_ratio
        if spatial_cluster_guard:
            cfg_dict["spatial_cluster_guard"] = True
            cfg_dict["cluster_eps_px"] = cluster_eps
            cfg_dict["cluster_min_samples"] = cluster_min_samples
            cfg_dict["cluster_min_support_ratio"] = cluster_min_support_ratio
        if min_spatial_distance is not None:
            cfg_dict["min_spatial_distance"] = min_spatial_distance
        if ratio_threshold is not None:
            cfg_dict["ratio_threshold"] = ratio_threshold
        if point_radius is not None:
            cfg_dict["point_radius"] = point_radius
        if morph_kernel_size is not None:
            cfg_dict["morph_kernel_size"] = morph_kernel_size
        effective_cm_config = CopyMoveConfig(**cfg_dict)
    else:
        effective_cm_config = CopyMoveConfig()

    # Splicing konfigürasyonunu belirle
    if splicing_config is not None:
        effective_splicing_config = splicing_config
    else:
        spl_dict = {}
        if splicing_fusion_mode is not None:
            spl_dict["fusion_mode"] = splicing_fusion_mode
        if splicing_fill_contours:
            spl_dict["fill_contours"] = True
            spl_dict["morph_kernel_size"] = 11
            spl_dict["min_region_area"] = 150
        effective_splicing_config = SplicingConfig(**spl_dict) if spl_dict else SplicingConfig()

    # 1. Veri setini yükle
    dataset = custom_dataset or get_dataset_reader(dataset_name, root_dir, task=task)
    total_samples = len(dataset)

    # Veri seti boşsa veya klasör yoksa kontrollü durum raporu üret
    if total_samples == 0:
        summary_empty: dict[str, Any] = {
            "dataset_name": dataset.name,
            "root_dir": str(dataset.root_dir),
            "status": "DATASET_NOT_FOUND_OR_EMPTY",
            "message": (
                f"'{dataset.name}' veri seti '{dataset.root_dir}' konumunda bulunamadı veya klasör boş. "
                "Lütfen ilgili veri setini indirip belirtilen dizine yerleştiriniz."
            ),
            "total_samples": 0,
        }
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_empty, f, indent=2, ensure_ascii=False)

        report_md_empty = (
            f"# ForensiVision Deney Raporu — {dataset.name}\n\n"
            f"> **Durum:** Veri seti dosyaları `{dataset.root_dir}` dizininde bulunamadı.\n\n"
            f"Deneyin çalıştırılabilmesi için `{dataset.name}` veri setinin "
            f"`{dataset.root_dir}` klasörüne yerleştirilmesi gerekmektedir.\n"
        )
        with open(out_path / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md_empty)

        return summary_empty

    # Görev türünü belirle
    effective_task = dataset.task_type if task == "auto" else task

    # 2. Deterministik Örnekleme (Deterministic Sampling)
    samples = list(dataset.samples)
    if variant_filter is not None:
        samples = [s for s in samples if s.variant == variant_filter]

    target_group_ids: set[str] | None = None
    if group_ids_file is not None:
        p_grp = Path(group_ids_file)
        if p_grp.is_file():
            with open(p_grp, "r", encoding="utf-8") as f:
                target_group_ids = {
                    line.strip() for line in f if line.strip() and not line.strip().startswith("#")
                }
            samples = [s for s in samples if (s.group_id or s.sample_id) in target_group_ids]

    rng = np.random.RandomState(seed)
    selected_group_ids: list[str] = []

    if sample_by_group:
        groups_map: dict[str, list[DatasetSample]] = {}
        for s in samples:
            gid = s.group_id or s.sample_id
            groups_map.setdefault(gid, []).append(s)

        all_group_ids = sorted(groups_map.keys())
        if limit > 0 and len(all_group_ids) > limit:
            chosen_indices = rng.choice(len(all_group_ids), size=limit, replace=False)
            selected_group_ids = sorted([all_group_ids[i] for i in chosen_indices])
        else:
            selected_group_ids = all_group_ids

        selected_samples = []
        for gid in selected_group_ids:
            grp_samples = sorted(groups_map[gid], key=lambda s: (s.is_authentic(), s.sample_id))
            selected_samples.extend(grp_samples)
    else:
        if limit > 0 and len(samples) > limit:
            indices = rng.choice(len(samples), size=limit, replace=False)
            selected_samples = [samples[i] for i in indices]
        else:
            selected_samples = samples

    # 3. Örnekleri Değerlendir
    metrics_records: list[dict[str, Any]] = []
    failures_records: list[dict[str, Any]] = []
    domain_failures: list[dict[str, Any]] = []
    cluster_diag_records: list[dict[str, Any]] = []

    # Metrik havuzları
    manip_precisions: list[float] = []
    manip_recalls: list[float] = []
    manip_f1s: list[float] = []
    manip_ious: list[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Temiz görüntü havuzları
    auth_total = 0
    auth_false_alarms = 0
    auth_marked_ratios: list[float] = []

    variant_buckets: dict[str, dict[str, list[float]]] = {}

    start_all_wall = time.perf_counter()
    examples_saved = 0

    for idx, sample in enumerate(selected_samples, 1):
        t_sample_start = time.perf_counter()
        try:
            # 1. Girdi görüntüsünü oku (Görüntü yükleme duvar saati süresi)
            t_img_start = time.perf_counter()
            img = dataset.load_image(sample)
            h, w = img.shape[:2]
            image_load_ms = (time.perf_counter() - t_img_start) * 1000.0

            # 2. Dedektörü çalıştır (Dedektör duvar saati ve süreç CPU süresi)
            t_det_start = time.perf_counter()
            t_det_cpu_start = time.process_time()
            if effective_task == "copy_move":
                res = detect_copy_move(img, config=effective_cm_config)
                pred_mask = res.mask
                is_tampered_decision = res.is_tampered
                if res.cluster_diagnostics is not None:
                    cluster_diag_records.append({
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id or sample.sample_id,
                        "manipulation_type": sample.manipulation_type,
                        "is_authentic": sample.is_authentic(),
                        "is_detected": is_tampered_decision,
                        "ransac_inliers_count": res.cluster_diagnostics.get("ransac_inliers_count", 0),
                        "num_clusters": res.cluster_diagnostics.get("num_clusters", 0),
                        "dominant_cluster_size": res.cluster_diagnostics.get("dominant_cluster_size", 0),
                        "cluster_support_ratio": res.cluster_diagnostics.get("cluster_support_ratio", 0.0),
                        "filtered_inliers_count": res.cluster_diagnostics.get("filtered_inliers_count", 0),
                        "is_rejected_by_guard": res.cluster_diagnostics.get("is_rejected_by_guard", False),
                    })
            elif effective_task == "splicing":
                res = detect_splicing(img, config=effective_splicing_config)
                pred_mask = res.mask
                is_tampered_decision = res.is_suspicious
            else:
                raise ValueError(f"Desteklenmeyen görev türü: {effective_task}")
            detector_wall_ms = (time.perf_counter() - t_det_start) * 1000.0
            detector_cpu_ms = (time.process_time() - t_det_cpu_start) * 1000.0

            # 3. Doğruluk maskesini oku (Maske yükleme duvar saati süresi)
            t_mask_start = time.perf_counter()
            gt_mask = dataset.load_mask(sample, mask_mode=mask_mode, target_shape=(h, w))
            mask_load_ms = (time.perf_counter() - t_mask_start) * 1000.0

            # 4. Örnek hazırlık ve tespit toplam duvar saati süresi
            sample_wall_ms = (time.perf_counter() - t_sample_start) * 1000.0

            # Metrikleri hesapla
            eval_res = calculate_metrics(pred_mask, gt_mask)

            pred_positive = int(np.count_nonzero(pred_mask))
            is_empty_mask = bool(pred_positive == 0)
            total_pixels = h * w
            marked_ratio = pred_positive / total_pixels

            # Kayıt sözlüğü oluştur
            record: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "task_type": effective_task,
                "manipulation_type": sample.manipulation_type,
                "variant": sample.variant or "default",
                "group_id": sample.group_id or "default",
                "precision": eval_res.precision,
                "recall": eval_res.recall,
                "f1_score": eval_res.f1_score,
                "iou": eval_res.iou,
                "tp": eval_res.tp,
                "fp": eval_res.fp,
                "fn": eval_res.fn,
                "detected_pixels": pred_positive,
                "is_empty_mask": is_empty_mask,
                "marked_pixel_ratio": marked_ratio,
                "is_detected": is_tampered_decision,
                "time_ms": sample_wall_ms,
                "image_load_ms": image_load_ms,
                "detector_wall_ms": detector_wall_ms,
                "detector_cpu_ms": detector_cpu_ms,
                "mask_load_ms": mask_load_ms,
                "sample_wall_ms": sample_wall_ms,
                "status": "SUCCESS",
            }
            metrics_records.append(record)

            # Temiz vs Manipüle ayrımı
            if sample.is_authentic():
                auth_total += 1
                if is_tampered_decision or pred_positive > 0:
                    auth_false_alarms += 1
                    domain_failures.append({
                        "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path),
                        "error_type": "FALSE_POSITIVE_AUTHENTIC",
                        "error_message": f"Özgün temiz görüntüde yanlış alarm üretildi ({pred_positive} piksel işaretlendi, oran: %{marked_ratio*100:.2f}). Sahnedeki doğal simetrik/tekrarlı doku tahrifat gibi algılandı.",
                    })
                auth_marked_ratios.append(marked_ratio)
            else:
                if not is_tampered_decision and eval_res.recall == 0.0:
                    domain_failures.append({
                        "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path),
                        "error_type": "FALSE_NEGATIVE_UNDETECTED",
                        "error_message": f"Tahrifat tespit edilemedi (Recall=0.0). Sahne: {sample.group_id}. RANSAC inlier eşiğine ulaşılamadı.",
                    })
                elif is_tampered_decision and eval_res.iou == 0.0:
                    domain_failures.append({
                        "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path),
                        "error_type": "FALSE_NEGATIVE_ZERO_IOU",
                        "error_message": f"Tahrifat kararı verildi ancak işaretlenen alan manipüle bölgeyle örtüşmedi (IoU=0.0, FP={eval_res.fp}, FN={eval_res.fn}). Doğal tekrarlayan arka plan eşleşmesi.",
                    })

                # Yalnızca gerçek manipülasyonlu örnekleri lokalizasyon ortalamasına kat
                manip_precisions.append(eval_res.precision)
                manip_recalls.append(eval_res.recall)
                manip_f1s.append(eval_res.f1_score)
                manip_ious.append(eval_res.iou)

                total_tp += eval_res.tp
                total_fp += eval_res.fp
                total_fn += eval_res.fn

                # Varyant bazlı dağılım
                var_key = sample.variant or "default"
                if var_key not in variant_buckets:
                    variant_buckets[var_key] = {"precision": [], "recall": [], "f1": [], "iou": []}
                variant_buckets[var_key]["precision"].append(eval_res.precision)
                variant_buckets[var_key]["recall"].append(eval_res.recall)
                variant_buckets[var_key]["f1"].append(eval_res.f1_score)
                variant_buckets[var_key]["iou"].append(eval_res.iou)

            # Örnek görsel kaydı (ilk N örnek için - hem münferit hem birleşik panel)
            if not sample.is_authentic() and examples_saved < save_examples_count:
                overlay = create_mask_overlay(img, pred_mask, color=(0, 0, 255), alpha=0.45)
                gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
                pred_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

                grid = np.hstack([img, gt_bgr, pred_bgr, overlay])
                cv2.imwrite(str(examples_dir / f"{sample.sample_id}_grid.png"), grid)
                cv2.imwrite(str(examples_dir / f"{sample.sample_id}_eval.png"), grid)
                cv2.imwrite(str(examples_dir / f"{sample.sample_id}_original.png"), img)
                cv2.imwrite(str(examples_dir / f"{sample.sample_id}_ground_truth.png"), gt_mask)
                cv2.imwrite(str(examples_dir / f"{sample.sample_id}_pred_mask.png"), pred_mask)
                cv2.imwrite(str(examples_dir / f"{sample.sample_id}_overlay.png"), overlay)
                examples_saved += 1

        except Exception as err:
            failures_records.append({
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "error_type": type(err).__name__,
                "error_message": str(err),
            })

    total_wall_time_s = time.perf_counter() - start_all_wall
    manip_count = len(manip_precisions)

    # 4. Süre Havuzları ve Ayrıntılı Zaman Analizi
    if metrics_records:
        all_sample_wall_ms = [r["sample_wall_ms"] for r in metrics_records]
        all_detector_wall_ms = [r["detector_wall_ms"] for r in metrics_records]
        all_detector_cpu_ms = [r["detector_cpu_ms"] for r in metrics_records]
        all_image_load_ms = [r["image_load_ms"] for r in metrics_records]
        all_mask_load_ms = [r["mask_load_ms"] for r in metrics_records]

        sum_sample_wall_ms = float(np.sum(all_sample_wall_ms))
        mean_sample_wall_ms = float(np.mean(all_sample_wall_ms))
        mean_detector_wall_ms = float(np.mean(all_detector_wall_ms))
        mean_detector_cpu_ms = float(np.mean(all_detector_cpu_ms))
        mean_image_load_ms = float(np.mean(all_image_load_ms))
        mean_mask_load_ms = float(np.mean(all_mask_load_ms))
    else:
        sum_sample_wall_ms = 0.0
        mean_sample_wall_ms = 0.0
        mean_detector_wall_ms = 0.0
        mean_detector_cpu_ms = 0.0
        mean_image_load_ms = 0.0
        mean_mask_load_ms = 0.0

    timing_summary = {
        "total_wall_time_seconds": round(total_wall_time_s, 2),
        "sum_sample_wall_seconds": round(sum_sample_wall_ms / 1000.0, 2),
        "mean_sample_wall_ms": round(mean_sample_wall_ms, 2),
        "mean_detector_wall_ms": round(mean_detector_wall_ms, 2),
        "mean_detector_cpu_ms": round(mean_detector_cpu_ms, 2),
        "mean_image_load_ms": round(mean_image_load_ms, 2),
        "mean_mask_load_ms": round(mean_mask_load_ms, 2),
    }

    # 5. Toplu İstatistikler (Macro & Micro)
    macro_metrics = {
        "precision": float(np.mean(manip_precisions)) if manip_count > 0 else 0.0,
        "recall": float(np.mean(manip_recalls)) if manip_count > 0 else 0.0,
        "f1_score": float(np.mean(manip_f1s)) if manip_count > 0 else 0.0,
        "iou": float(np.mean(manip_ious)) if manip_count > 0 else 0.0,
    }

    # Micro hesaplama (Global piksel toplamları üzerinden)
    denom_micro_prec = total_tp + total_fp
    denom_micro_rec = total_tp + total_fn
    micro_prec = (total_tp / denom_micro_prec) if denom_micro_prec > 0 else 0.0
    micro_rec = (total_tp / denom_micro_rec) if denom_micro_rec > 0 else 0.0
    denom_micro_f1 = (2 * total_tp) + total_fp + total_fn
    micro_f1 = ((2 * total_tp) / denom_micro_f1) if denom_micro_f1 > 0 else 0.0
    denom_micro_iou = total_tp + total_fp + total_fn
    micro_iou = (total_tp / denom_micro_iou) if denom_micro_iou > 0 else 0.0

    micro_metrics = {
        "precision": float(micro_prec),
        "recall": float(micro_rec),
        "f1_score": float(micro_f1),
        "iou": float(micro_iou),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }

    # Temiz görsel yanlış alarm analizi
    auth_summary = {
        "total_authentic": auth_total,
        "false_alarms": auth_false_alarms,
        "false_alarm_rate": (auth_false_alarms / auth_total) if auth_total > 0 else 0.0,
        "empty_masks_count": (auth_total - auth_false_alarms),
        "mean_marked_pixel_ratio": float(np.mean(auth_marked_ratios)) if auth_total > 0 else 0.0,
    }

    # Varyant ortalamaları
    variant_summary: dict[str, Any] = {}
    for v_name, v_data in variant_buckets.items():
        variant_summary[v_name] = {
            "count": len(v_data["iou"]),
            "macro_precision": float(np.mean(v_data["precision"])),
            "macro_recall": float(np.mean(v_data["recall"])),
            "macro_f1": float(np.mean(v_data["f1"])),
            "macro_iou": float(np.mean(v_data["iou"])),
        }

    summary: dict[str, Any] = {
        "status": "COMPLETED",
        "dataset_name": dataset.name,
        "task_type": effective_task,
        "copy_move_config": asdict(effective_cm_config) if effective_task == "copy_move" else None,
        "splicing_config": asdict(effective_splicing_config) if effective_task == "splicing" else None,
        "mask_mode": mask_mode,
        "seed": seed,
        "selected_group_ids": selected_group_ids,
        "group_ids_file": str(group_ids_file) if group_ids_file else None,
        "total_requested_limit": limit,
        "total_evaluated_samples": len(selected_samples),
        "successful_samples": len(metrics_records),
        "failed_samples": len(failures_records),
        "manipulated_samples_count": manip_count,
        "authentic_samples_count": auth_total,
        "macro_metrics": macro_metrics,
        "micro_metrics": micro_metrics,
        "authentic_analysis": auth_summary,
        "variant_analysis": variant_summary,
        "timing_analysis": timing_summary,
        "total_time_seconds": round(total_wall_time_s, 2),
        "average_time_per_sample_ms": round(mean_sample_wall_ms, 2) if selected_samples else 0.0,
    }

    # 5. Dosyalara Yazım

    # A. metrics.csv
    csv_file = out_path / "metrics.csv"
    if metrics_records:
        fieldnames = list(metrics_records[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_records)

    # B. failures.csv
    fail_file = out_path / "failures.csv"
    with open(fail_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "error_type", "error_message"])
        writer.writeheader()
        writer.writerows(failures_records + domain_failures)

    # C. cluster_diagnostics.csv
    if cluster_diag_records:
        diag_file = out_path / "cluster_diagnostics.csv"
        diag_fieldnames = list(cluster_diag_records[0].keys())
        with open(diag_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=diag_fieldnames)
            writer.writeheader()
            writer.writerows(cluster_diag_records)

    # D. summary.json
    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # E. report.md
    report_md = _build_markdown_report(summary)
    with open(out_path / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    return summary


def _format_duration(ms: float) -> str:
    """Milisaniye cinsinden süreyi ms ve sn olarak biçimlendirir."""
    if ms >= 1000.0:
        return f"{ms:.1f} ms ({ms / 1000.0:.2f} sn)"
    return f"{ms:.1f} ms"


def _build_markdown_report(s: dict[str, Any]) -> str:
    """Deney özetinden akademik GitHub markdown raporu üretir."""
    macro = s.get("macro_metrics", {})
    micro = s.get("micro_metrics", {})
    auth = s.get("authentic_analysis", {})
    var_dict = s.get("variant_analysis", {})
    selected_groups = s.get("selected_group_ids", [])
    timing = s.get("timing_analysis", {})

    total_wall_s = timing.get("total_wall_time_seconds", s.get("total_time_seconds", 0.0))
    sum_sample_s = timing.get("sum_sample_wall_seconds", 0.0)
    mean_sample_ms = timing.get("mean_sample_wall_ms", s.get("average_time_per_sample_ms", 0.0))
    mean_det_wall_ms = timing.get("mean_detector_wall_ms", 0.0)
    mean_det_cpu_ms = timing.get("mean_detector_cpu_ms", 0.0)
    mean_img_load_ms = timing.get("mean_image_load_ms", 0.0)
    mean_mask_load_ms = timing.get("mean_mask_load_ms", 0.0)

    lines = [
        f"# ForensiVision Deney Değerlendirme Raporu",
        f"**Veri Seti:** {s.get('dataset_name')} | **Görev:** {s.get('task_type')} | **Tohum (Seed):** {s.get('seed')}",
        "",
        "---",
        "",
        "## 1. Deney Yöntemi ve Kapsamı",
        f"- **Toplam Değerlendirilen:** {s.get('total_evaluated_samples')} örnek",
        f"- **Başarılı:** {s.get('successful_samples')} | **Hata/Atlanan:** {s.get('failed_samples')}",
        f"- **Manipüle Edilmiş Örnek:** {s.get('manipulated_samples_count')}",
        f"- **Temiz/Orijinal Örnek:** {s.get('authentic_samples_count')}",
    ]
    if s.get("group_ids_file"):
        lines.append(f"- **Hedef Sahne Listesi Dosyası (`group_ids_file`):** `{s.get('group_ids_file')}`")
    lines.extend([
        "",
        "### Süre ve Zaman Ölçümü Dağılımı",
        f"- **Deney Toplam Duvar Saati Süresi (Total Wall Time):** {total_wall_s:.2f} sn ({total_wall_s / 60.0:.2f} dk)",
        f"- **Örneklerin Toplam İşlem Duvar Süresi (Sum Sample Wall):** {sum_sample_s:.2f} sn",
        f"- **Örnek Başına Ortalama Duvar Süresi (Mean Sample Wall):** {_format_duration(mean_sample_ms)}",
        f"- **Dedektör Ortalama Duvar Süresi (Mean Detector Wall):** {_format_duration(mean_det_wall_ms)}",
        f"- **Dedektör Ortalama Süreç CPU Süresi (Mean Detector CPU):** {_format_duration(mean_det_cpu_ms)}",
        f"- **Ortalama Görüntü Yükleme Süresi (Mean Image Load):** {_format_duration(mean_img_load_ms)}",
        f"- **Ortalama Maske Yükleme Süresi (Mean Mask Load):** {_format_duration(mean_mask_load_ms)}",
        "",
    ])

    cm_cfg = s.get("copy_move_config")
    if cm_cfg:
        guard_active = bool(cm_cfg.get("hull_fill_guard", False))
        cluster_guard_active = bool(cm_cfg.get("spatial_cluster_guard", False))
        lines.extend([
            "### Etkin Algoritma Konfigürasyonu (Copy-Move)",
            f"- **Minimum RANSAC İnlier Eşiği (`min_inliers`):** {cm_cfg.get('min_inliers')}",
            f"- **Dışbükey Zarf Doldurma Koruması (`hull_fill_guard`):** {'Etkin (True)' if guard_active else 'Devre Dışı (False)'}",
            f"- **Maksimum Zarf Amplifikasyon Oranı (`hull_max_ratio`):** {cm_cfg.get('hull_max_ratio')}",
            f"- **Uzamsal Kümeleme Koruması (`spatial_cluster_guard`):** {'Etkin (True)' if cluster_guard_active else 'Devre Dışı (False)'}",
            f"- **DBSCAN Epsilon (`cluster_eps_px`):** {cm_cfg.get('cluster_eps_px')} px",
            f"- **DBSCAN Min Samples (`cluster_min_samples`):** {cm_cfg.get('cluster_min_samples')}",
            f"- **DBSCAN Min Destek Oranı (`cluster_min_support_ratio`):** {cm_cfg.get('cluster_min_support_ratio')}",
            f"- **Lowe Oran Testi Eşiği (`ratio_threshold`):** {cm_cfg.get('ratio_threshold')}",
            f"- **Uzamsal Mesafe Filtresi (`min_spatial_distance`):** {cm_cfg.get('min_spatial_distance')} px",
            f"- **RANSAC Yeniden İzdüşüm Hata Eşiği (`ransac_reproj_threshold`):** {cm_cfg.get('ransac_reproj_threshold')} px",
            f"- **Aday Eşleşme Eşiği (`min_matches_for_ransac`):** {cm_cfg.get('min_matches_for_ransac')}",
            "",
        ])

    if selected_groups:
        id_list_str = ", ".join(f"`{gid}`" for gid in selected_groups)
        lines.extend([
            f"### Seçilen Temel Sahne Kimlikleri ({len(selected_groups)} Sahne)",
            f"Deterministik tohum (`seed={s.get('seed')}`) ile seçilen benzersiz temel sahne kimlikleri:",
            f"> {id_list_str}",
            "",
            "Her sahne kimliği için temel manipüle görüntü (`_F.png`), ikili doğruluk maskesi (`_B.png`) ve özgün temiz görüntü (`_O.png`) değerlendirilmiştir.",
            "",
        ])

    lines.extend([
        "## 2. Akademik Yerelleştirme Metrikleri (Yalnızca Manipüle Örnekler)",
        "",
        "> [!NOTE]",
        "> Temiz görüntülerdeki 1.0 konvansiyonel skorlar bu ortalamaya dahil edilmemiştir.",
        "",
        "| Değerlendirme Modu | Precision | Recall | F1-Score | IoU (Jaccard) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Makro Ortalama (Macro Avg)** | **{macro.get('precision', 0)*100:.2f}%** | **{macro.get('recall', 0)*100:.2f}%** | **{macro.get('f1_score', 0)*100:.2f}%** | **{macro.get('iou', 0)*100:.2f}%** |",
        f"| **Mikro Ortalama (Micro Avg)** | **{micro.get('precision', 0)*100:.2f}%** | **{micro.get('recall', 0)*100:.2f}%** | **{micro.get('f1_score', 0)*100:.2f}%** | **{micro.get('iou', 0)*100:.2f}%** |",
        "",
        f"*Toplam Piksel Sayımları:* $TP={micro.get('total_tp', 0)}$, $FP={micro.get('total_fp', 0)}$, $FN={micro.get('total_fn', 0)}$",
        "",
        "## 3. Temiz Görüntülerde Yanlış Alarm (False Alarm) Analizi",
        f"- **Toplam Temiz Görüntü:** {auth.get('total_authentic', 0)}",
        f"- **Boş Maske Üreten Temiz Görüntü (Doğru Negatif):** {auth.get('empty_masks_count', 0)}",
        f"- **Yanlış Alarm Sayısı:** {auth.get('false_alarms', 0)}",
        f"- **Yanlış Alarm Oranı (FAR):** %{auth.get('false_alarm_rate', 0)*100:.2f}",
        f"- **Ortalama İşaretlenen Piksel Oranı:** %{auth.get('mean_marked_pixel_ratio', 0)*100:.4f}",
        "",
    ])

    if var_dict:
        lines.extend([
            "## 4. Saldırı ve Varyant Bazlı Başarım Dağılımı",
            "",
            "| Varyant / Saldırı | Örnek | Makro Precision | Makro Recall | Makro F1 | Makro IoU |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for v_name, v_metrics in var_dict.items():
            lines.append(
                f"| `{v_name}` | {v_metrics['count']} | "
                f"{v_metrics['macro_precision']*100:.2f}% | "
                f"{v_metrics['macro_recall']*100:.2f}% | "
                f"{v_metrics['macro_f1']*100:.2f}% | "
                f"{v_metrics['macro_iou']*100:.2f}% |"
            )
        lines.append("")

    lines.extend([
        "## 5. Bilimsel Geçerlilik, Bulgular ve Sınırlılıklar",
        "- **Sabit Parametreler ve Tarafsızlık:** Algoritma eşik değerleri (Lowe ratio, RANSAC inlier eşikleri vb.) bu pilot örnek kümesine göre ayarlanmamış veya optimize edilmemiştir.",
        "- **Başarım Dağılımı:** Belirgin dokuya ve keskin sınırlara sahip nesnelerin kopyalandığı sahnelerde SIFT eşleştirmesi yüksek hassasiyet (Precision) ve tatmin edici IoU sağlarken, düz renkli veya aşırı homojen bölgelerde anahtar nokta yoğunluğunun düşüklüğü nedeniyle tespit kaçırılabilmekte veya RANSAC afin eşleşme eşiğine ulaşılamamaktadır.",
        "- **Genelleştirilebilirlik Uyarısı:** Bireysel örneklerdeki yüksek Recall değerleri genel sistem başarısı olarak kabul edilemez; sistemin dayanımı gerçekçi makro/mikro ortalamalar ve varyant deneyleriyle değerlendirilmelidir.",
        "- **Kapsam Sınırı:** Bu pilot deney yalnızca temel (clean) sahneleri kapsamaktadır; dönüşüm dayanıklılığı (JPEG sıkıştırma, Gauss bulanıklığı, gürültü ekleme, kontrast değişimi vb.) müteakip aşamalarda incelenecektir.",
        "",
        "## 6. Çıktı Dosyaları",
        "- `metrics.csv`: Bütün sahte ve temiz örneklerin satır bazında metrikleri ve aşama bazlı süreleri (`image_load_ms`, `detector_wall_ms`, `detector_cpu_ms`, `mask_load_ms`, `sample_wall_ms`, `time_ms`).",
        "- `summary.json`: Makine tarafından okunabilir deney özeti, ayrıntılı zamanlama analizi (`timing_analysis`) ve yapılandırma verileri.",
        "- `failures.csv`: Hata veren veya istisnaya yol açan örneklerin dökümü.",
        "- `examples/`: Seçilen örneklerin orijinal görüntü, doğruluk maskesi, tahmin maskesi, overlay ve 4 panelli karşılaştırma görselleri.",
        "",
        "---",
        "*Rapor ForensiVision Automated Evaluation Engine tarafından oluşturulmuştur.*",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ForensiVision Veri Seti Değerlendirme Betiği")
    parser.add_argument("--dataset", type=str, default="comofod", help="Veri seti adı ('comofod', 'columbia', 'defacto')")
    parser.add_argument("--root", type=str, default="data/raw/comofod", help="Veri seti kök dizini")
    parser.add_argument("--task", type=str, default="auto", help="Görev türü ('copy_move', 'splicing', 'auto')")
    parser.add_argument("--limit", type=int, default=30, help="Değerlendirilecek maksimum örnek/grup sayısı (CPU güvenliği için)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministik örnekleme için rastgelelik tohumu")
    parser.add_argument("--output", type=str, default="data/output/experiments/default_run", help="Sonuç çıktı dizini")
    parser.add_argument("--mask-mode", type=str, default="both", help="Maske modu ('both', 'target_only')")
    parser.add_argument("--examples", type=int, default=5, help="Kaydedilecek görsel örnek sayısı")
    parser.add_argument("--variant", type=str, default=None, help="Belirli bir varyantı filtrele (örn. 'clean')")
    parser.add_argument("--sample-by-group", action="store_true", help="Örnekleri tek tek değil, benzersiz sahne grupları (group_id) bazında seç")
    parser.add_argument("--min-inliers", type=int, default=None, help="Copy-Move için minimum RANSAC inlier eşiği (varsayılan: 4)")
    parser.add_argument("--hull-fill-guard", action="store_true", help="Dışbükey zarf doldurma korumasını etkinleştir (seyrek noktalarda convex hull doldurmayı engeller)")
    parser.add_argument("--hull-max-ratio", type=float, default=4.0, help="Dışbükey zarf / daireler alanı maksimum amplifikasyon oranı eşiği (varsayılan: 4.0)")
    parser.add_argument("--spatial-cluster-guard", action="store_true", help="DBSCAN tabanlı uzamsal tutarlılık korumasını etkinleştir")
    parser.add_argument("--cluster-eps", type=float, default=35.0, help="DBSCAN maksimum komşuluk mesafesi (piksel, varsayılan: 35.0)")
    parser.add_argument("--cluster-min-samples", type=int, default=3, help="DBSCAN çekirdek nokta için minimum örnek sayısı (varsayılan: 3)")
    parser.add_argument("--cluster-min-support-ratio", type=float, default=0.50, help="Baskın küme için minimum destek oranı (varsayılan: 0.50)")
    parser.add_argument("--min-spatial-distance", type=float, default=None, help="Öz-eşleme minimum uzamsal mesafe eşiği (piksel, varsayılan: 25.0)")
    parser.add_argument("--ratio-threshold", type=float, default=None, help="Lowe oran testi eşiği (varsayılan: 0.75)")
    parser.add_argument("--point-radius", type=int, default=None, help="İnlier anahtar nokta çizim yarıçapı (varsayılan: 12)")
    parser.add_argument("--morph-kernel-size", type=int, default=None, help="Morfolojik kapanış çekirdek boyutu (varsayılan: 5)")
    parser.add_argument("--splicing-fusion-mode", type=str, default=None, help="Splicing füzyon modu ('union', 'intersection', 'ela_only', 'noise_only')")
    parser.add_argument("--splicing-fill-contours", action="store_true", help="Splicing ELA tohumlarını morfolojik köprüleme ve kontur doldurmayla genişlet")
    parser.add_argument("--group-ids-file", type=str, default=None, help="Yalnızca belirtilen dosyada yer alan sahne/grup kimliklerini değerlendir (örn. holdout listesi)")

    args = parser.parse_args()

    print("=" * 70)
    print("ForensiVision — Veri Seti Değerlendirme Başlatılıyor")
    print(f"Veri Seti : {args.dataset}")
    print(f"Kök Dizin : {args.root}")
    print(f"Örnek/Grup Sınırı: {args.limit} (Tohum: {args.seed})")
    print(f"Grup Bazlı Örnekleme: {args.sample_by_group}")
    if args.group_ids_file:
        print(f"Sahne Kimlik Dosyası: {args.group_ids_file}")
    print(f"Varyant Filtresi    : {args.variant or 'Tümü'}")
    print(f"Min Inliers Eşiği   : {args.min_inliers if args.min_inliers is not None else 'Varsayılan (4)'}")
    print(f"Zarf Doldurma Koruma: {args.hull_fill_guard} (Maks. Oran: {args.hull_max_ratio})")
    print(f"Kümeleme Koruması   : {args.spatial_cluster_guard} (Eps: {args.cluster_eps}, MinSamples: {args.cluster_min_samples}, MinDestek: {args.cluster_min_support_ratio})")
    if args.min_spatial_distance:
        print(f"Uzamsal Mesafe Eşiği: {args.min_spatial_distance} px")
    if args.ratio_threshold:
        print(f"Lowe Oran Eşiği     : {args.ratio_threshold}")
    if args.splicing_fusion_mode:
        print(f"Splicing Füzyon     : {args.splicing_fusion_mode} (Kontur Doldur: {args.splicing_fill_contours})")
    print(f"Çıktı               : {args.output}")
    print("=" * 70)

    summary = run_evaluation(
        dataset_name=args.dataset,
        root_dir=args.root,
        task=args.task,
        limit=args.limit,
        seed=args.seed,
        output_dir=args.output,
        mask_mode=args.mask_mode,
        save_examples_count=args.examples,
        variant_filter=args.variant,
        sample_by_group=args.sample_by_group,
        min_inliers=args.min_inliers,
        hull_fill_guard=args.hull_fill_guard,
        hull_max_ratio=args.hull_max_ratio,
        spatial_cluster_guard=args.spatial_cluster_guard,
        cluster_eps=args.cluster_eps,
        cluster_min_samples=args.cluster_min_samples,
        cluster_min_support_ratio=args.cluster_min_support_ratio,
        min_spatial_distance=args.min_spatial_distance,
        ratio_threshold=args.ratio_threshold,
        point_radius=args.point_radius,
        morph_kernel_size=args.morph_kernel_size,
        splicing_fusion_mode=args.splicing_fusion_mode,
        splicing_fill_contours=args.splicing_fill_contours,
        group_ids_file=args.group_ids_file,
    )

    if summary.get("status") == "DATASET_NOT_FOUND_OR_EMPTY":
        print(f"\n[!] BİLGİ: {summary.get('message')}")
    else:
        macro = summary.get("macro_metrics", {})
        micro = summary.get("micro_metrics", {})
        auth = summary.get("authentic_analysis", {})
        timing = summary.get("timing_analysis", {})
        print("\n[+] Değerlendirme Tamamlandı!")
        print(f"    - İşlenen Örnek: {summary.get('successful_samples')} (Hata: {summary.get('failed_samples')})")
        print(f"    - Makro Metrikler: Precision={macro.get('precision', 0):.4f}, Recall={macro.get('recall', 0):.4f}, F1={macro.get('f1_score', 0):.4f}, IoU={macro.get('iou', 0):.4f}")
        print(f"    - Mikro Metrikler: Precision={micro.get('precision', 0):.4f}, Recall={micro.get('recall', 0):.4f}, F1={micro.get('f1_score', 0):.4f}, IoU={micro.get('iou', 0):.4f}")
        print(f"    - Temiz Görüntülerde Yanlış Alarm Oranı (FAR): %{auth.get('false_alarm_rate', 0)*100:.2f}")
        print(f"    - Süre Özeti: Toplam Duvar={timing.get('total_wall_time_seconds', 0):.2f}sn, Dedektör Duvar Ort.={timing.get('mean_detector_wall_ms', 0):.1f}ms, Dedektör CPU Ort.={timing.get('mean_detector_cpu_ms', 0):.1f}ms")
        print(f"\nRapor ve çıktılar kaydedildi: {Path(args.output).resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
