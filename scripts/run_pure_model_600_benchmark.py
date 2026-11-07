"""
ForensiVision - Saf Model 600 Görsel Bağımsız Akademik Benchmark Testi.
- 200 Orijinal Temiz Görsel (CoMoFoD)
- 200 Copy-Move Klonlama Görseli (DEFACTO Copy-Move)
- 200 Splicing Montaj Görseli (DEFACTO Splicing)
Yalnızca ResNet-34 U-Net Saf Derin Öğrenme Modeli Test Edilir.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class ForensicUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base = models.resnet34(weights=None)
        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))

        self.final_up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 1, 1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x0 = self.in_block(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        pooled = self.avg_pool(x4).view(x.size(0), -1)
        cls_out = self.cls_head(pooled).squeeze(1)

        d4 = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))
        seg_out = self.seg_head(self.final_up(d1))

        return seg_out, cls_out


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


def load_model(weights_path: Path):
    model = ForensicUNet(pretrained=False)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def collect_samples(data_root: Path):
    random.seed(42)

    # 1. 200 Temiz Görsel (CoMoFoD Orijinalleri)
    comofod_dir = data_root / "CoMoFoD_small_v2"
    clean_files = sorted(list(comofod_dir.glob("*_O.png")))
    if len(clean_files) < 200:
        raise ValueError(f"Yeterli CoMoFoD temiz görsel bulunamadı: {len(clean_files)}")
    clean_samples = [{"img_path": f, "mask_path": None, "type": "clean", "is_tampered": False} for f in clean_files[:200]]

    # 2. 200 Copy-Move Görseli (DEFACTO Copy-Move)
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

    print("✅ Test Veri Havuzu Eksiksiz Doğrulandı:")
    print(f"   • 1. Temiz Görseller   : {len(clean_samples)} adet (CoMoFoD Orijinalleri)")
    print(f"   • 2. Copy-Move Sahteler: {len(cm_samples)} adet (DEFACTO Copy-Move)")
    print(f"   • 3. Splicing Sahteler : {len(sp_samples)} adet (DEFACTO Splicing)")
    print(f"   ⭐ TOPLAM TEST HAVUZU  : {len(clean_samples) + len(cm_samples) + len(sp_samples)} adet\n")

    return clean_samples + cm_samples + sp_samples


def run_benchmark():
    print("=" * 70)
    print("🔬 FORENSIVISION 600 GÖRSEL SAF MODEL BENCHMARK TESTİ")
    print("=" * 70)

    weights_path = Path("models/forensivision_model.pth")
    if not weights_path.exists():
        weights_path = Path("models/forensic_model_epoch6_final.pth")
    print(f"Yüklenen Model: {weights_path}")

    model = load_model(weights_path)
    data_root = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
    samples = collect_samples(data_root)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    stats = {
        "clean": {"total": 0, "correct": 0, "wrong": 0, "false_positives": 0, "cls_scores": [], "pixel_counts": []},
        "copymove": {"total": 0, "detected": 0, "missed": 0, "ious": [], "precisions": [], "recalls": [], "dices": [], "cls_scores": [], "pixel_counts": [], "fp_pixels": [], "fn_pixels": []},
        "splicing": {"total": 0, "detected": 0, "missed": 0, "ious": [], "precisions": [], "recalls": [], "dices": [], "cls_scores": [], "pixel_counts": [], "fp_pixels": [], "fn_pixels": []}
    }

    total_time = 0.0

    print("🚀 600 Görsel İçin Çıkarım Başlatıldı... Lütfen bekleyin...")
    t_start = time.time()

    for idx, sample in enumerate(samples, 1):
        img_p = sample["img_path"]
        m_type = sample["type"]
        is_dirty = sample["is_tampered"]

        # Görseli Güvenli Oku
        img = read_image_safe(img_p)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        norm = ((img_resized.astype(np.float32) / 255.0) - mean) / std
        t_input = torch.from_numpy(np.transpose(norm, (2, 0, 1)).astype(np.float32)).unsqueeze(0)

        # Saf Model Çıkarımı
        t0 = time.time()
        with torch.no_grad():
            seg_out, cls_out = model(t_input)
            cls_prob = torch.sigmoid(cls_out).item()
            mask_prob = torch.sigmoid(seg_out)[0, 0].numpy()
        total_time += (time.time() - t0)

        pred_mask_bin = (mask_prob > 0.50).astype(np.uint8)
        pred_pixel_count = int(np.sum(pred_mask_bin))

        # Modelin Adli Kararı:
        # Sınıflandırıcı skoru > 0.50 VEYA piksel maskesinde en az 100 şüpheli piksel öbeği varsa sahte kabul edilir
        flagged_as_tampered = (cls_prob > 0.50) or (pred_pixel_count >= 100)

        stats[m_type]["total"] += 1
        stats[m_type]["cls_scores"].append(cls_prob)
        stats[m_type]["pixel_counts"].append(pred_pixel_count)

        if m_type == "clean":
            if not flagged_as_tampered:
                stats["clean"]["correct"] += 1
            else:
                stats["clean"]["wrong"] += 1
                stats["clean"]["false_positives"] += 1

        else:
            if flagged_as_tampered:
                stats[m_type]["detected"] += 1
            else:
                stats[m_type]["missed"] += 1

            # Piksel Düzeyinde Adli Maske Doğrulaması
            gt_mask_p = sample["mask_path"]
            if gt_mask_p and gt_mask_p.exists():
                gt_m = read_mask_safe(gt_mask_p)
                if gt_m is not None:
                    gt_resized = cv2.resize(gt_m, (256, 256), interpolation=cv2.INTER_NEAREST)
                    gt_bin = (gt_resized > 127).astype(np.uint8)

                    intersection = int(np.logical_and(pred_mask_bin, gt_bin).sum())
                    union = int(np.logical_or(pred_mask_bin, gt_bin).sum())
                    iou = intersection / union if union > 0 else 0.0

                    pred_pos = int(pred_mask_bin.sum())
                    gt_pos = int(gt_bin.sum())

                    # Piksel bazında hassasiyet, yakalama ve hata analizleri
                    prec = intersection / pred_pos if pred_pos > 0 else 0.0
                    rec = intersection / gt_pos if gt_pos > 0 else 0.0
                    dice = (2.0 * intersection) / (pred_pos + gt_pos) if (pred_pos + gt_pos) > 0 else 0.0

                    # Gereksiz boyanan (False Positive) ve Kaçırılan (False Negative) pikseller
                    fp_pix = max(0, pred_pos - intersection)
                    fn_pix = max(0, gt_pos - intersection)

                    stats[m_type]["ious"].append(iou)
                    stats[m_type]["precisions"].append(prec)
                    stats[m_type]["recalls"].append(rec)
                    stats[m_type]["dices"].append(dice)
                    stats[m_type]["fp_pixels"].append(fp_pix)
                    stats[m_type]["fn_pixels"].append(fn_pix)

        if idx % 100 == 0 or idx == len(samples):
            print(f"   İlerleyiş: [{idx}/{len(samples)}] görsel tamamlandı... (Geçen süre: {time.time()-t_start:.1f}s)")

    elapsed_all = time.time() - t_start
    avg_inference_ms = (total_time / len(samples)) * 1000

    # Metrikleri Hesapla
    clean_acc = (stats["clean"]["correct"] / stats["clean"]["total"]) * 100 if stats["clean"]["total"] > 0 else 0
    cm_recall = (stats["copymove"]["detected"] / stats["copymove"]["total"]) * 100 if stats["copymove"]["total"] > 0 else 0
    sp_recall = (stats["splicing"]["detected"] / stats["splicing"]["total"]) * 100 if stats["splicing"]["total"] > 0 else 0

    total_correct = stats["clean"]["correct"] + stats["copymove"]["detected"] + stats["splicing"]["detected"]
    overall_acc = (total_correct / len(samples)) * 100

    total_dirty_detected = stats["copymove"]["detected"] + stats["splicing"]["detected"]
    total_flagged = total_dirty_detected + stats["clean"]["false_positives"]
    precision = (total_dirty_detected / total_flagged * 100) if total_flagged > 0 else 0
    recall = (total_dirty_detected / 400 * 100)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "total_images": len(samples),
        "total_elapsed_sec": elapsed_all,
        "avg_inference_ms": avg_inference_ms,
        "overall_accuracy": overall_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clean": {
            "total": stats["clean"]["total"],
            "correct_clean": stats["clean"]["correct"],
            "false_alarms": stats["clean"]["false_positives"],
            "accuracy": clean_acc,
            "mean_cls_score": float(np.mean(stats["clean"]["cls_scores"])),
            "mean_pixel_count": float(np.mean(stats["clean"]["pixel_counts"]))
        },
        "copymove": {
            "total": stats["copymove"]["total"],
            "detected": stats["copymove"]["detected"],
            "missed": stats["copymove"]["missed"],
            "recall": cm_recall,
            "mean_cls_score": float(np.mean(stats["copymove"]["cls_scores"])),
            "mean_iou": float(np.mean(stats["copymove"]["ious"])) if stats["copymove"]["ious"] else 0.0,
            "mean_pixel_precision": float(np.mean(stats["copymove"]["precisions"])) if stats["copymove"]["precisions"] else 0.0,
            "mean_pixel_recall": float(np.mean(stats["copymove"]["recalls"])) if stats["copymove"]["recalls"] else 0.0,
            "mean_fp_pixels": float(np.mean(stats["copymove"]["fp_pixels"])) if stats["copymove"]["fp_pixels"] else 0.0,
            "mean_fn_pixels": float(np.mean(stats["copymove"]["fn_pixels"])) if stats["copymove"]["fn_pixels"] else 0.0
        },
        "splicing": {
            "total": stats["splicing"]["total"],
            "detected": stats["splicing"]["detected"],
            "missed": stats["splicing"]["missed"],
            "recall": sp_recall,
            "mean_cls_score": float(np.mean(stats["splicing"]["cls_scores"])),
            "mean_iou": float(np.mean(stats["splicing"]["ious"])) if stats["splicing"]["ious"] else 0.0,
            "mean_pixel_precision": float(np.mean(stats["splicing"]["precisions"])) if stats["splicing"]["precisions"] else 0.0,
            "mean_pixel_recall": float(np.mean(stats["splicing"]["recalls"])) if stats["splicing"]["recalls"] else 0.0,
            "mean_fp_pixels": float(np.mean(stats["splicing"]["fp_pixels"])) if stats["splicing"]["fp_pixels"] else 0.0,
            "mean_fn_pixels": float(np.mean(stats["splicing"]["fn_pixels"])) if stats["splicing"]["fn_pixels"] else 0.0
        }
    }

    out_file = Path("benchmark_600_pure_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("🎯 AKADEMİK ADLİ BİLİŞİM BENCHMARK RAPORU (600 GÖRSEL)")
    print("=" * 70)
    print(f"Toplam Test Süresi       : {elapsed_all:.1f} saniye")
    print(f"Görsel Başına Çıkarım    : {avg_inference_ms:.1f} ms (Yerel CPU)")
    print(f"Genel Doğruluk (Accuracy): %{overall_acc:.2f}")
    print(f"Hassasiyet (Precision)   : %{precision:.2f}")
    print(f"Duyarlılık (Recall)      : %{recall:.2f} (400 sahtenin yakalanma oranı)")
    print(f"F1-Skoru                 : %{f1:.2f}")
    print("-" * 70)
    print(f"1. TEMİZ REFERANS (200 Görsel) : %{clean_acc:.2f} Doğruluk ({stats['clean']['correct']}/200 doğru temiz, {stats['clean']['false_positives']} yanlış alarm)")
    print(f"2. COPY-MOVE     (200 Görsel) : %{cm_recall:.2f} Yakalama ({stats['copymove']['detected']}/200 yakalandı, {stats['copymove']['missed']} kaçtı)")
    if stats["copymove"]["ious"]:
        print(f"   • Ortalama Piksel IoU      : %{np.mean(stats['copymove']['ious'])*100:.2f}")
        print(f"   • Piksel Yakalama İsabeti  : %{np.mean(stats['copymove']['recalls'])*100:.2f}")
        print(f"   • Ortalama Yanlış Boyama   : {np.mean(stats['copymove']['fp_pixels']):.0f} piksel")
        print(f"   • Ortalama Kaçırılan Piksel: {np.mean(stats['copymove']['fn_pixels']):.0f} piksel")
    print(f"3. SPLICING      (200 Görsel) : %{sp_recall:.2f} Yakalama ({stats['splicing']['detected']}/200 yakalandı, {stats['splicing']['missed']} kaçtı)")
    if stats["splicing"]["ious"]:
        print(f"   • Ortalama Piksel IoU      : %{np.mean(stats['splicing']['ious'])*100:.2f}")
        print(f"   • Piksel Yakalama İsabeti  : %{np.mean(stats['splicing']['recalls'])*100:.2f}")
        print(f"   • Ortalama Yanlış Boyama   : {np.mean(stats['splicing']['fp_pixels']):.0f} piksel")
        print(f"   • Ortalama Kaçırılan Piksel: {np.mean(stats['splicing']['fn_pixels']):.0f} piksel")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
