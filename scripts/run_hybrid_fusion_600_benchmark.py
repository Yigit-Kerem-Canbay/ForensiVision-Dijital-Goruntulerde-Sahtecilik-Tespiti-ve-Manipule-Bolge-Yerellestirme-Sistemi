"""
ForensiVision 600 Görsel Çok Kademeli Hibrit Füzyon Benchmark Motoru.
- 1. Katman: ResNet-34 U-Net (ONNX) - Dikiş izleri ve montaj anomalileri
- 2. Katman: SIFT & RANSAC - İkiz anahtar nokta ve klonlama tespiti
- 3. Katman: Akıllı Morfolojik Filtreleme ve Bileşen Analizi (Connected Components)
- 4. Katman: Konsensüs Karar Birleştirme
"""

import json
import os
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import onnxruntime as ort
from forensivision.copy_move import CopyMoveConfig, detect_copy_move


def read_image_safe(path: Path):
    with open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def read_mask_safe(path: Path):
    with open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def collect_600_samples(data_root: Path):
    random.seed(42)

    # 1. 200 Temiz Görsel
    comofod_dir = data_root / "CoMoFoD_small_v2"
    clean_files = sorted(list(comofod_dir.glob("*_O.png")))[:200]
    clean_samples = [{"img_path": f, "mask_path": None, "type": "clean", "is_tampered": False} for f in clean_files]

    # 2. 200 Copy-Move Görseli
    cm_dir = data_root / "defacto-copymove"
    cm_img_dir = cm_dir / "copymove_img" / "img"
    cm_mask_dir = cm_dir / "copymove_annotations" / "probe_mask"
    
    all_cm_imgs = sorted(list(cm_img_dir.glob("*.tif")))
    valid_cm = []
    for img_p in all_cm_imgs:
        mask_p = cm_mask_dir / f"{img_p.stem}.jpg"
        if mask_p.exists():
            valid_cm.append({"img_path": img_p, "mask_path": mask_p, "type": "copymove", "is_tampered": True})
    
    random.shuffle(valid_cm)
    cm_samples = valid_cm[:200]

    # 3. 200 Splicing Görseli
    sp_dir = data_root / "defacto-splicing"
    valid_sp = []
    for part in range(1, 8):
        part_img_dir = sp_dir / f"splicing_{part}_img" / "img"
        part_mask_dir = sp_dir / f"splicing_{part}_annotations" / "probe_mask"
        if part_img_dir.exists() and part_mask_dir.exists():
            for img_p in part_img_dir.glob("*.tif"):
                mask_p = part_mask_dir / f"{img_p.stem}.jpg"
                if mask_p.exists():
                    valid_sp.append({"img_path": img_p, "mask_path": mask_p, "type": "splicing", "is_tampered": True})
    
    random.shuffle(valid_sp)
    sp_samples = valid_sp[:200]

    return clean_samples, cm_samples, sp_samples


def run_hybrid_benchmark():
    print("=" * 75)
    print("🛡️ FORENSIVISION 600 GÖRSEL 4 KATMANLI HİBRİT FÜZYON BENCHMARK TESTİ")
    print("=" * 75)

    model_path = PROJECT_ROOT / "models" / "forensivision_model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX Model bulunamadı: {model_path}")
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"✅ ONNX Hibrit Motoru Hazır: {model_path.name}")

    data_root = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
    clean_samples, cm_samples, sp_samples = collect_600_samples(data_root)
    all_samples = clean_samples + cm_samples + sp_samples

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    sift_cfg = CopyMoveConfig(
        hull_fill_guard=True,
        spatial_cluster_guard=True,
        min_spatial_distance=20.0,
        ratio_threshold=0.75,
        point_radius=16,
        morph_kernel_size=9,
        min_inliers=5,  # Hassas ikiz anahtar nokta eşiği
    )

    stats = {
        "clean": {"total": 0, "correct": 0, "wrong": 0, "false_positives": 0},
        "copymove": {"total": 0, "detected": 0, "missed": 0, "ious": [], "precisions": [], "recalls": [], "sift_hits": 0, "dl_hits": 0},
        "splicing": {"total": 0, "detected": 0, "missed": 0, "ious": [], "precisions": [], "recalls": [], "sift_hits": 0, "dl_hits": 0}
    }

    print(f"🚀 Toplam {len(all_samples)} Görsel Füzyon İncelemesinden Geçiriliyor...\n")
    t_start = time.time()

    for idx, sample in enumerate(all_samples, 1):
        img_p = sample["img_path"]
        m_type = sample["type"]
        is_dirty = sample["is_tampered"]

        img_bgr = read_image_safe(img_p)
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]

        # 1. KADEME: Derin Öğrenme Çıkarımı
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        norm = ((img_res.astype(np.float32) / 255.0) - mean) / std
        inp = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        outs = session.run(None, {input_name: inp})
        seg_logits = outs[0][0, 0]
        cls_logit = float(outs[1][0])

        dl_score = float(1.0 / (1.0 + np.exp(-cls_logit)))
        mask_prob = 1.0 / (1.0 + np.exp(-seg_logits))

        # Akıllı Morfolojik Filtreleme
        raw_bin = (mask_prob > 0.60).astype(np.uint8)
        clean_bin = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, kernel_3x3)
        num_labels, labels, st, _ = cv2.connectedComponentsWithStats(clean_bin)
        max_comp_area = int(np.max(st[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0

        # Derin Öğrenme Kararı (U-Net montaj tespiti)
        dl_flagged = bool(
            (dl_score >= 0.50 and max_comp_area >= 25)
            or (max_comp_area >= 60)
        )

        dl_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if dl_flagged and np.count_nonzero(clean_bin) > 0:
            dl_mask_orig = cv2.resize(clean_bin * 255, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 2. KADEME: SIFT Geometrik Klonlama Analizi (Her görselde klon kontrolü)
        sift_flagged = False
        sift_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
        
        try:
            s_res = detect_copy_move(img_bgr, config=sift_cfg)
            if s_res.is_tampered and s_res.num_inliers >= 5 and np.count_nonzero(s_res.mask) > 0:
                sift_flagged = True
                sift_mask_orig = s_res.mask
        except Exception:
            pass

        # 3. KADEME: Füzyon Kararı ve Birleştirilmiş Maske
        is_fused_tampered = dl_flagged or sift_flagged
        fused_mask = cv2.bitwise_or(dl_mask_orig, sift_mask_orig) if is_fused_tampered else np.zeros((h_orig, w_orig), dtype=np.uint8)

        # 4. İstatistik ve Metrikler
        if m_type == "clean":
            stats["clean"]["total"] += 1
            if not is_fused_tampered:
                stats["clean"]["correct"] += 1
            else:
                stats["clean"]["wrong"] += 1
                stats["clean"]["false_positives"] += 1

        else:
            stats[m_type]["total"] += 1
            if is_fused_tampered:
                stats[m_type]["detected"] += 1
                if dl_flagged: stats[m_type]["dl_hits"] += 1
                if sift_flagged: stats[m_type]["sift_hits"] += 1
            else:
                stats[m_type]["missed"] += 1

            # Piksel Düzeyi IoU
            gt_mask_p = sample["mask_path"]
            if gt_mask_p and gt_mask_p.exists():
                gt_m = read_mask_safe(gt_mask_p)
                if gt_m is not None:
                    gt_bin = (gt_m > 127).astype(np.uint8)
                    pred_bin = (fused_mask > 127).astype(np.uint8)

                    inter = int(np.logical_and(pred_bin, gt_bin).sum())
                    union = int(np.logical_or(pred_bin, gt_bin).sum())
                    iou = inter / union if union > 0 else 0.0

                    prec = inter / pred_bin.sum() if pred_bin.sum() > 0 else 0.0
                    rec = inter / gt_bin.sum() if gt_bin.sum() > 0 else 0.0

                    stats[m_type]["ious"].append(iou)
                    stats[m_type]["precisions"].append(prec)
                    stats[m_type]["recalls"].append(rec)

        if idx % 50 == 0 or idx == len(all_samples):
            print(f"   İlerleme: [{idx}/{len(all_samples)}] ({time.time()-t_start:.1f}s) | Temiz: {stats['clean']['correct']} Doğru | Copy-Move: {stats['copymove']['detected']} Yakalandı | Splicing: {stats['splicing']['detected']} Yakalandı")

    elapsed = time.time() - t_start

    clean_acc = (stats["clean"]["correct"] / stats["clean"]["total"]) * 100
    cm_rec = (stats["copymove"]["detected"] / stats["copymove"]["total"]) * 100
    sp_rec = (stats["splicing"]["detected"] / stats["splicing"]["total"]) * 100

    tot_correct = stats["clean"]["correct"] + stats["copymove"]["detected"] + stats["splicing"]["detected"]
    overall_acc = (tot_correct / len(all_samples)) * 100

    tot_dirty_detected = stats["copymove"]["detected"] + stats["splicing"]["detected"]
    tot_flagged = tot_dirty_detected + stats["clean"]["false_positives"]
    precision = (tot_dirty_detected / tot_flagged * 100) if tot_flagged > 0 else 0
    recall = (tot_dirty_detected / 400 * 100)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "overall_accuracy": overall_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "elapsed_sec": elapsed,
        "clean": {"total": 200, "correct": stats["clean"]["correct"], "false_positives": stats["clean"]["false_positives"], "accuracy": clean_acc},
        "copymove": {"total": 200, "detected": stats["copymove"]["detected"], "missed": stats["copymove"]["missed"], "recall": cm_rec, "sift_hits": stats["copymove"]["sift_hits"], "dl_hits": stats["copymove"]["dl_hits"], "mean_iou": float(np.mean(stats["copymove"]["ious"])) if stats["copymove"]["ious"] else 0.0},
        "splicing": {"total": 200, "detected": stats["splicing"]["detected"], "missed": stats["splicing"]["missed"], "recall": sp_rec, "dl_hits": stats["splicing"]["dl_hits"], "mean_iou": float(np.mean(stats["splicing"]["ious"])) if stats["splicing"]["ious"] else 0.0}
    }

    with open("benchmark_600_hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print("🏆 FORENSIVISION HİBRİT FÜZYON BENCHMARK RAPORU")
    print("=" * 75)
    print(f"Toplam Test Süresi       : {elapsed:.1f} saniye")
    print(f"Genel Doğruluk (Accuracy): %{overall_acc:.2f}")
    print(f"Hassasiyet (Precision)   : %{precision:.2f}")
    print(f"Duyarlılık (Recall)      : %{recall:.2f}")
    print(f"F1-Skoru                 : %{f1:.2f}")
    print("-" * 75)
    print(f"1. TEMİZ REFERANS (200 Görsel): %{clean_acc:.2f} Başarı ({stats['clean']['correct']}/200 doğru temiz)")
    print(f"2. COPY-MOVE     (200 Görsel): %{cm_rec:.2f} Yakalama ({stats['copymove']['detected']}/200 yakalandı) [SIFT Katkısı: {stats['copymove']['sift_hits']} adet]")
    print(f"3. SPLICING      (200 Görsel): %{sp_rec:.2f} Yakalama ({stats['splicing']['detected']}/200 yakalandı) [U-Net Katkısı: {stats['splicing']['dl_hits']} adet]")
    print("=" * 75)

if __name__ == "__main__":
    run_hybrid_benchmark()
