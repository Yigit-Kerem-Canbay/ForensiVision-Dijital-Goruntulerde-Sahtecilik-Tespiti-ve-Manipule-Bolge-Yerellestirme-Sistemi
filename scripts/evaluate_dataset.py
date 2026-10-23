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
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from forensivision.copy_move import CopyMoveConfig, detect_copy_move
from forensivision.datasets import BaseDataset, CoMoFoDDataset, ColumbiaDataset, DatasetSample
from forensivision.evaluation import calculate_metrics
from forensivision.splicing import SplicingConfig, detect_splicing
from forensivision.visualization import create_mask_overlay


def get_dataset_reader(dataset_name: str, root_dir: str | Path) -> BaseDataset:
    """Belirtilen veri seti adı için uygun okuyucu sınıfını döndürür."""
    name_clean = dataset_name.strip().lower()
    if name_clean == "comofod":
        return CoMoFoDDataset(root_dir)
    elif name_clean == "columbia":
        return ColumbiaDataset(root_dir)
    else:
        raise ValueError(
            f"Bilinmeyen veri seti: '{dataset_name}'. Desteklenenler: 'comofod', 'columbia'"
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
) -> dict[str, Any]:
    """Veri seti üzerinde değerlendirme deneyini çalıştırır ve çıktıları kaydeder."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    examples_dir = out_path / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Veri setini yükle
    dataset = custom_dataset or get_dataset_reader(dataset_name, root_dir)
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
    rng = np.random.RandomState(seed)
    if limit > 0 and len(samples) > limit:
        indices = rng.choice(len(samples), size=limit, replace=False)
        selected_samples = [samples[i] for i in indices]
    else:
        selected_samples = samples

    # 3. Örnekleri Değerlendir
    metrics_records: list[dict[str, Any]] = []
    failures_records: list[dict[str, Any]] = []

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

    start_all = time.time()
    examples_saved = 0

    for idx, sample in enumerate(selected_samples, 1):
        t0 = time.time()
        try:
            # Görüntüyü oku
            img = dataset.load_image(sample)
            h, w = img.shape[:2]

            # Dedektörü çalıştır
            if effective_task == "copy_move":
                res = detect_copy_move(img)
                pred_mask = res.mask
                is_tampered_decision = res.is_tampered
            elif effective_task == "splicing":
                res = detect_splicing(img)
                pred_mask = res.mask
                is_tampered_decision = res.is_suspicious
            else:
                raise ValueError(f"Desteklenmeyen görev türü: {effective_task}")

            # Doğruluk maskesini oku
            gt_mask = dataset.load_mask(sample, mask_mode=mask_mode, target_shape=(h, w))
            elapsed_ms = (time.time() - t0) * 1000.0

            # Metrikleri hesapla
            eval_res = calculate_metrics(pred_mask, gt_mask)

            pred_positive = int(np.count_nonzero(pred_mask))
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
                "marked_pixel_ratio": marked_ratio,
                "is_detected": is_tampered_decision,
                "time_ms": elapsed_ms,
                "status": "SUCCESS",
            }
            metrics_records.append(record)

            # Temiz vs Manipüle ayrımı
            if sample.is_authentic():
                auth_total += 1
                if is_tampered_decision or pred_positive > 0:
                    auth_false_alarms += 1
                auth_marked_ratios.append(marked_ratio)
            else:
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

            # Örnek görsel kaydı (ilk N örnek için)
            if examples_saved < save_examples_count:
                overlay = create_mask_overlay(img, pred_mask, color=(0, 0, 255), alpha=0.45)
                gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
                pred_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

                grid = np.hstack([img, gt_bgr, pred_bgr, overlay])
                ex_path = examples_dir / f"{sample.sample_id}_eval.png"
                cv2.imwrite(str(ex_path), grid)
                examples_saved += 1

        except Exception as err:
            failures_records.append({
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "error_type": type(err).__name__,
                "error_message": str(err),
            })

    total_time_s = time.time() - start_all
    manip_count = len(manip_precisions)

    # 4. Toplu İstatistikler (Macro & Micro)
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
        "mask_mode": mask_mode,
        "seed": seed,
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
        "total_time_seconds": round(total_time_s, 2),
        "average_time_per_sample_ms": round((total_time_s / len(selected_samples)) * 1000, 2) if selected_samples else 0.0,
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
        writer.writerows(failures_records)

    # C. summary.json
    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # D. report.md
    report_md = _build_markdown_report(summary)
    with open(out_path / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    return summary


def _build_markdown_report(s: dict[str, Any]) -> str:
    """Deney özetinden akademik GitHub markdown raporu üretir."""
    macro = s.get("macro_metrics", {})
    micro = s.get("micro_metrics", {})
    auth = s.get("authentic_analysis", {})
    var_dict = s.get("variant_analysis", {})

    lines = [
        f"# ForensiVision Deney Değerlendirme Raporu",
        f"**Veri Seti:** {s.get('dataset_name')} | **Görev:** {s.get('task_type')} | **Tohum (Seed):** {s.get('seed')}",
        "",
        "---",
        "",
        "## 1. Deney Özeti",
        f"- **Toplam Değerlendirilen:** {s.get('total_evaluated_samples')} örnek",
        f"- **Başarılı:** {s.get('successful_samples')} | **Hata/Atlanan:** {s.get('failed_samples')}",
        f"- **Manipüle Edilmiş Örnek:** {s.get('manipulated_samples_count')}",
        f"- **Temiz/Orijinal Örnek:** {s.get('authentic_samples_count')}",
        f"- **Toplam Süre:** {s.get('total_time_seconds')} sn (Örnek başına ort. {s.get('average_time_per_sample_ms')} ms)",
        "",
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
        f"- **Yanlış Alarm Sayısı:** {auth.get('false_alarms', 0)}",
        f"- **Yanlış Alarm Oranı (FAR):** %{auth.get('false_alarm_rate', 0)*100:.2f}",
        f"- **Ortalama İşaretlenen Piksel Oranı:** %{auth.get('mean_marked_pixel_ratio', 0)*100:.4f}",
        "",
    ]

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
        "## 5. Çıktı Dosyaları",
        "- `metrics.csv`: Bütün örneklerin satır bazında metrikleri.",
        "- `summary.json`: Makine tarafından okunabilir deney özeti.",
        "- `failures.csv`: Hata veren örnekler ve istisnalar.",
        "- `examples/`: İlk örneklerin görsel bindirme (overlay) karşılaştırmaları.",
        "",
        "---",
        "*Rapor ForensiVision Automated Evaluation Engine tarafından oluşturulmuştur.*",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ForensiVision Veri Seti Değerlendirme Betiği")
    parser.add_argument("--dataset", type=str, default="comofod", help="Veri seti adı ('comofod', 'columbia')")
    parser.add_argument("--root", type=str, default="data/raw/comofod", help="Veri seti kök dizini")
    parser.add_argument("--task", type=str, default="auto", help="Görev türü ('copy_move', 'splicing', 'auto')")
    parser.add_argument("--limit", type=int, default=30, help="Değerlendirilecek maksimum örnek sayısı (CPU güvenliği için)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministik örnekleme için rastgelelik tohumu")
    parser.add_argument("--output", type=str, default="data/output/experiments/default_run", help="Sonuç çıktı dizini")
    parser.add_argument("--mask-mode", type=str, default="both", help="Maske modu ('both', 'target_only')")
    parser.add_argument("--examples", type=int, default=5, help="Kaydedilecek görsel örnek sayısı")

    args = parser.parse_args()

    print("=" * 70)
    print("ForensiVision — Veri Seti Değerlendirme Başlatılıyor")
    print(f"Veri Seti : {args.dataset}")
    print(f"Kök Dizin : {args.root}")
    print(f"Örnek Sınırı: {args.limit} (Tohum: {args.seed})")
    print(f"Çıktı     : {args.output}")
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
    )

    if summary.get("status") == "DATASET_NOT_FOUND_OR_EMPTY":
        print(f"\n[!] BİLGİ: {summary.get('message')}")
    else:
        macro = summary.get("macro_metrics", {})
        micro = summary.get("micro_metrics", {})
        auth = summary.get("authentic_analysis", {})
        print("\n[+] Değerlendirme Tamamlandı!")
        print(f"    - İşlenen Örnek: {summary.get('successful_samples')} (Hata: {summary.get('failed_samples')})")
        print(f"    - Makro Metrikler: Precision={macro.get('precision', 0):.4f}, Recall={macro.get('recall', 0):.4f}, F1={macro.get('f1_score', 0):.4f}, IoU={macro.get('iou', 0):.4f}")
        print(f"    - Mikro Metrikler: Precision={micro.get('precision', 0):.4f}, Recall={micro.get('recall', 0):.4f}, F1={micro.get('f1_score', 0):.4f}, IoU={micro.get('iou', 0):.4f}")
        print(f"    - Temiz Görüntülerde Yanlış Alarm Oranı (FAR): %{auth.get('false_alarm_rate', 0)*100:.2f}")
        print(f"\nRapor ve çıktılar kaydedildi: {Path(args.output).resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
