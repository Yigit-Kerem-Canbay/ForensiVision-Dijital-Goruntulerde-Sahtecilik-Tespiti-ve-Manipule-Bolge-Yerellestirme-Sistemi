"""
ForensiVision 600 Görsel Adil ve Bağımsız Doğrulama Benchmarkı.
Havuz:
- 200 Gerçek Doğal Temiz Görsel (MS-COCO - DEFACTO Orijinal Temiz Kaynakları)
- 200 Copy-Move Sahte Görsel (DEFACTO Copy-Move)
- 200 Splicing Sahte Görsel (DEFACTO Splicing)
Toplam: 600 Bağımsız Görsel

Karşılaştırma:
1. Saf Model (Pür ResNet-34 U-Net)
2. 4 Katmanlı Hibrit Füzyon (U-Net + SIFT + Morfolojik CC Filtresi)
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


def collect_fair_600_samples():
    random.seed(42)

    # 1. 200 Gerçek Temiz Görsel (COCO Orijinalleri)
    clean_dir = PROJECT_ROOT / "data" / "clean_coco_200"
    all_clean = sorted(list(clean_dir.glob("*.jpg")))
    if len(all_clean) < 200:
        raise ValueError(f"Yetersiz temiz görsel: {len(all_clean)} adet bulundu, 200 gerekli.")
    clean_samples = [{"img_path": p, "mask_path": None, "type": "clean", "is_tampered": False} for p in all_clean[:200]]

    # 2. 200 Copy-Move Görseli (DEFACTO Copy-Move)
    data_root = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
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

    # 3. 200 Splicing Görseli (DEFACTO Splicing)
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


def run_fair_benchmark():
    print("=" * 80)
    print("🔬 FORENSIVISION 600 GÖRSEL ADİL KARŞILAŞTIRMA BENCHMARK TESTİ")
    print("=" * 80)

    model_path = PROJECT_ROOT / "models" / "forensivision_model.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    clean_samples, cm_samples, sp_samples = collect_fair_600_samples()
    all_samples = clean_samples + cm_samples + sp_samples

    print(f"✅ Havuz Hazırlandı: {len(clean_samples)} Temiz | {len(cm_samples)} Copy-Move | {len(sp_samples)} Splicing")
    print(f"Toplam: {len(all_samples)} Görsel. Her görsel hem SAF MODEL hem HİBRİT MODEL ile inceleniyor...\n")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    sift_cfg = CopyMoveConfig(
        hull_fill_guard=True,
        spatial_cluster_guard=True,
        min_spatial_distance=20.0,
        ratio_threshold=0.75,
        min_inliers=5,
    )

    stats_pure = {
        "clean": {"correct": 0, "total": 0},
        "copymove": {"detected": 0, "total": 0},
        "splicing": {"detected": 0, "total": 0},
    }

    stats_hybrid = {
        "clean": {"correct": 0, "total": 0},
        "copymove": {"detected": 0, "sift_hits": 0, "dl_hits": 0, "total": 0},
        "splicing": {"detected": 0, "sift_hits": 0, "dl_hits": 0, "total": 0},
    }

    t_start = time.time()

    for idx, sample in enumerate(all_samples, 1):
        img_p = sample["img_path"]
        m_type = sample["type"]

        img_bgr = read_image_safe(img_p)
        if img_bgr is None:
            continue

        # Model Girişini Hazırla
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        norm = ((img_res.astype(np.float32) / 255.0) - mean) / std
        inp = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        outs = session.run(None, {input_name: inp})
        seg_logits = outs[0][0, 0]
        cls_logit = float(outs[1][0])

        cls_score = 1.0 / (1.0 + np.exp(-cls_logit))
        mask_prob = 1.0 / (1.0 + np.exp(-seg_logits))
        pos_pixels_pure = int(np.sum(mask_prob > 0.50))

        # ---------------------------------------------
        # 1. SAF MODEL KARARI (Pür CNN)
        # ---------------------------------------------
        pure_tampered = (cls_score > 0.50) or (pos_pixels_pure >= 80)

        stats_pure[m_type]["total"] += 1
        if m_type == "clean":
            if not pure_tampered:
                stats_pure["clean"]["correct"] += 1
        else:
            if pure_tampered:
                stats_pure[m_type]["detected"] += 1

        # ---------------------------------------------
        # 2. 4 KATMANLI HİBRİT MODEL KARARI
        # ---------------------------------------------
        # Katman 1 & 3: U-Net + Morfolojik Açma
        clean_bin = cv2.morphologyEx((mask_prob > 0.50).astype(np.uint8), cv2.MORPH_OPEN, kernel_3x3)
        num_labels, labels, st, _ = cv2.connectedComponentsWithStats(clean_bin)
        max_comp_area = int(np.max(st[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0

        # Derin öğrenme montaj tespiti (Hassas tespit: cls>0.50 & comp>=20 veya comp>=60)
        dl_flagged = bool((cls_score > 0.50 and max_comp_area >= 20) or (max_comp_area >= 60))

        # Katman 2: SIFT Klon Tespiti
        sift_flagged = False
        try:
            s_res = detect_copy_move(img_bgr, config=sift_cfg)
            if s_res.is_tampered and s_res.num_inliers >= 5 and np.count_nonzero(s_res.mask) > 0:
                sift_flagged = True
        except Exception:
            pass

        hybrid_tampered = dl_flagged or sift_flagged

        stats_hybrid[m_type]["total"] += 1
        if m_type == "clean":
            if not hybrid_tampered:
                stats_hybrid["clean"]["correct"] += 1
        else:
            if hybrid_tampered:
                stats_hybrid[m_type]["detected"] += 1
                if dl_flagged: stats_hybrid[m_type]["dl_hits"] += 1
                if sift_flagged: stats_hybrid[m_type]["sift_hits"] += 1

        if idx % 50 == 0 or idx == len(all_samples):
            elapsed = time.time() - t_start
            print(f"   [{idx:3d}/600] ({elapsed:5.1f}s) | "
                  f"Saf Temiz: {stats_pure['clean']['correct']:2d} / Hibrit Temiz: {stats_hybrid['clean']['correct']:2d} | "
                  f"Saf CM: {stats_pure['copymove']['detected']:2d} / Hibrit CM: {stats_hybrid['copymove']['detected']:2d} | "
                  f"Saf SP: {stats_pure['splicing']['detected']:2d} / Hibrit SP: {stats_hybrid['splicing']['detected']:2d}", flush=True)

    elapsed_tot = time.time() - t_start

    # Metrikleri Hesapla
    tot_pure_correct = stats_pure["clean"]["correct"] + stats_pure["copymove"]["detected"] + stats_pure["splicing"]["detected"]
    pure_acc = (tot_pure_correct / 600) * 100
    pure_clean_acc = (stats_pure["clean"]["correct"] / 200) * 100
    pure_cm_rec = (stats_pure["copymove"]["detected"] / 200) * 100
    pure_sp_rec = (stats_pure["splicing"]["detected"] / 200) * 100

    tot_hyb_correct = stats_hybrid["clean"]["correct"] + stats_hybrid["copymove"]["detected"] + stats_hybrid["splicing"]["detected"]
    hyb_acc = (tot_hyb_correct / 600) * 100
    hyb_clean_acc = (stats_hybrid["clean"]["correct"] / 200) * 100
    hyb_cm_rec = (stats_hybrid["copymove"]["detected"] / 200) * 100
    hyb_sp_rec = (stats_hybrid["splicing"]["detected"] / 200) * 100

    results = {
        "elapsed_sec": elapsed_tot,
        "pure_model": {
            "overall_accuracy": pure_acc,
            "clean_accuracy": pure_clean_acc,
            "clean_correct": stats_pure["clean"]["correct"],
            "copymove_recall": pure_cm_rec,
            "copymove_detected": stats_pure["copymove"]["detected"],
            "splicing_recall": pure_sp_rec,
            "splicing_detected": stats_pure["splicing"]["detected"],
        },
        "hybrid_model": {
            "overall_accuracy": hyb_acc,
            "clean_accuracy": hyb_clean_acc,
            "clean_correct": stats_hybrid["clean"]["correct"],
            "copymove_recall": hyb_cm_rec,
            "copymove_detected": stats_hybrid["copymove"]["detected"],
            "copymove_sift_hits": stats_hybrid["copymove"]["sift_hits"],
            "splicing_recall": hyb_sp_rec,
            "splicing_detected": stats_hybrid["splicing"]["detected"],
            "splicing_dl_hits": stats_hybrid["splicing"]["dl_hits"],
        }
    }

    with open("benchmark_600_fair_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("🏆 FORENSIVISION 600 GÖRSEL ADİL BAŞARI RAPORU")
    print("=" * 80)
    print(f"Toplam Test Süresi       : {elapsed_tot:.1f} saniye")
    print("-" * 80)
    print(f"Kategori                 | Saf Model (Pür U-Net)      | 4 Katmanlı Hibrit Model")
    print("-" * 80)
    print(f"1. TEMİZ FOTOĞRAFLAR     | %{pure_clean_acc:5.2f} ({stats_pure['clean']['correct']:3d}/200)         | %{hyb_clean_acc:5.2f} ({stats_hybrid['clean']['correct']:3d}/200)")
    print(f"2. COPY-MOVE (KLONLAMA)  | %{pure_cm_rec:5.2f} ({stats_pure['copymove']['detected']:3d}/200)         | %{hyb_cm_rec:5.2f} ({stats_hybrid['copymove']['detected']:3d}/200)")
    print(f"3. SPLICING (MONTAJ)     | %{pure_sp_rec:5.2f} ({stats_pure['splicing']['detected']:3d}/200)         | %{hyb_sp_rec:5.2f} ({stats_hybrid['splicing']['detected']:3d}/200)")
    print("-" * 80)
    print(f"⭐ GENEL DOĞRULUK (ACC)   | %{pure_acc:5.2f} ({tot_pure_correct:3d}/600)         | %{hyb_acc:5.2f} ({tot_hyb_correct:3d}/600)")
    print("=" * 80)


if __name__ == "__main__":
    run_fair_benchmark()
