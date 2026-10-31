"""
ForensiVision Hafif Derin Öğrenme Modeli (ForensicLiteNet) Eğitimi ve Test Sistemi.

Bu betik:
- İşlemciyi zorlamadan (160x160 çözünürlük, 4 kanallı RGB + SRM gürültü artık kanalı),
- ~350k parametreli kompakt bir U-Net mimarisini yerel CPU üzerinde 5 epoch eğitir (~2 dakika).
- Eğitim bittikten sonra tamamen AYRI bir test kümesinde (50 Temiz, 50 CoMoFoD CM, 50 DEFACTO CM, 50 DEFACTO Splicing)
  doğruluk, yanlış alarm ve piksel IoU/hassasiyet oranlarını ölçer.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forensivision.datasets import CoMoFoDDataset, DefactoDataset
from forensivision.datasets.base import DatasetSample, _safe_read_image

# 5x5 SRM Yüksek Frekans Adli Çekirdek
SRM_KERNEL = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1]
], dtype=np.float32) / 12.0


def extract_srm_residual(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = cv2.filter2D(gray, -1, SRM_KERNEL)
    res_norm = np.clip(np.abs(residual) / 32.0, 0.0, 1.0)
    return res_norm


class ForensicDataset(Dataset):
    def __init__(self, samples: list[DatasetSample], target_size=(160, 160)):
        self.samples = samples
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = _safe_read_image(s.image_path)
        if img is None:
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # 1-3. Kanallar: RGB [-1, 1]
        rgb = img_resized.astype(np.float32) / 127.5 - 1.0
        # 4. Kanal: SRM Gürültü Artığı [0, 1]
        srm = extract_srm_residual(img_resized)

        # 4 Kanallı Tensor (4, H, W)
        ch_rgb = np.transpose(rgb, (2, 0, 1))
        ch_srm = np.expand_dims(srm, axis=0)
        x_tensor = np.concatenate([ch_rgb, ch_srm], axis=0).astype(np.float32)

        # Hedef Maske (1, H, W)
        mask_tensor = np.zeros((1, self.target_size[1], self.target_size[0]), dtype=np.float32)
        is_tampered = not s.is_authentic()

        if is_tampered and s.mask_path and Path(s.mask_path).is_file():
            gt = _safe_read_image(s.mask_path)
            if gt is not None:
                if len(gt.shape) == 3:
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                gt_resized = cv2.resize(gt, self.target_size, interpolation=cv2.INTER_NEAREST)
                mask_tensor[0] = (gt_resized > 127).astype(np.float32)

        return (
            torch.from_numpy(x_tensor),
            torch.from_numpy(mask_tensor),
            torch.tensor(1.0 if is_tampered else 0.0, dtype=torch.float32),
            str(s.image_path),
            (h_orig, w_orig),
        )


class ForensicLiteNet(nn.Module):
    """~380k Parametreli Kompakt Adli Segmentasyon U-Net Ağı."""
    def __init__(self, in_channels=4):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 80x80

        self.enc2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # 40x40

        self.enc3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)  # 20x20

        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(48 + 96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(24 + 48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.out = nn.Sequential(
            nn.Conv2d(16 + in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        out = self.out(torch.cat([self.up1(d2), x], dim=1))
        return out


def dice_loss(pred_logits, targets, smooth=1.0):
    probs = torch.sigmoid(pred_logits)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (probs_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    return 1.0 - dice


def main():
    print("=" * 80)
    print("FORENSIVISION YEREL HAFİF DERİN ÖĞRENME MODELİ (ForensicLiteNet)")
    print("=" * 80)

    # 1. Veri Kümelerini Oluştur
    print("\n[1/4] Veri Kümeleri Hazırlanıyor...")
    rng = np.random.RandomState(42)

    cmfd = CoMoFoDDataset("data/raw/comofod")
    cm_clean = [s for s in cmfd if s.is_authentic()]
    cm_dirty = [s for s in cmfd if not s.is_authentic()]
    rng.shuffle(cm_clean)
    rng.shuffle(cm_dirty)

    df_cm = DefactoDataset("data/raw/defacto/copymove", task_type="copy_move")
    df_cm_list = list(df_cm.samples)
    rng.shuffle(df_cm_list)

    df_spl = DefactoDataset("data/raw/defacto/splicing", task_type="splicing")
    df_spl_list = list(df_spl.samples)
    rng.shuffle(df_spl_list)

    # Eğitim Kümesi (160 Görsel)
    train_samples = (
        cm_clean[:50] +
        cm_dirty[:35] +
        df_cm_list[:35] +
        df_spl_list[:40]
    )
    rng.shuffle(train_samples)

    # Test Kümeleri (Tamamen Ayrı Holdout: 50'şer Görsel)
    test_clean = cm_clean[50:100]
    test_cm_dirty = cm_dirty[35:85]
    test_df_cm = df_cm_list[35:85]
    test_df_spl = df_spl_list[40:90]

    print(f"  -> Eğitim Kümesi : {len(train_samples)} Görsel (50 Temiz, 110 Kirli)")
    print(f"  -> Test Temiz    : {len(test_clean)} Görsel")
    print(f"  -> Test CoMoFoD  : {len(test_cm_dirty)} Görsel")
    print(f"  -> Test DEFACTO CM: {len(test_df_cm)} Görsel")
    print(f"  -> Test DEFACTO SP: {len(test_df_spl)} Görsel")

    train_ds = ForensicDataset(train_samples, target_size=(160, 160))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)

    # 2. Modeli Başlat
    print("\n[2/4] Model Başlatılıyor...")
    model = ForensicLiteNet(in_channels=4)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Model Parametre Sayısı: {param_count:,} ({param_count * 4 / (1024 * 1024):.2f} MB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # 3. Yerel Hızlı Eğitim (5 Epoch)
    print("\n[3/4] Yerel CPU Eğitimi Başlıyor (5 Epoch)...")
    model.train()
    t_start = time.perf_counter()

    for epoch in range(1, 6):
        ep_loss = 0.0
        t_ep0 = time.perf_counter()
        for x, y_mask, y_label, _, _ in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss_bce = bce_loss_fn(pred, y_mask)
            loss_dice = dice_loss(pred, y_mask)
            loss = loss_bce + 0.5 * loss_dice
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        dt_ep = time.perf_counter() - t_ep0
        avg_loss = ep_loss / len(train_loader)
        print(f"  Epoch {epoch}/5 | Ortalama Kayıp (Loss): {avg_loss:.4f} | Süre: {dt_ep:.1f} saniye", flush=True)

    total_train_time = time.perf_counter() - t_start
    print(f"  -> Eğitim Tamamlandı! Toplam Süre: {total_train_time:.1f} saniye.", flush=True)

    # Modeli Kaydet
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "forensic_lite.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  -> Model kaydedildi: {model_path}")

    # 4. Test Kümelerinde Değerlendirme
    print("\n[4/4] Bağımsız Test Kümelerinde Değerlendirme Yapılıyor...")
    model.eval()

    def evaluate_suite(name: str, samples: list[DatasetSample], is_clean: bool = False):
        ds = ForensicDataset(samples, target_size=(160, 160))
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        flagged = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            for x, y_mask, y_label, img_path, orig_shape in loader:
                pred_logits = model(x)
                probs = torch.sigmoid(pred_logits)[0, 0].numpy()

                # Şüpheli piksel tespiti (Eşik: 0.50)
                pred_binary = (probs > 0.50).astype(np.uint8)
                gt_binary = (y_mask[0, 0].numpy() > 0.50).astype(np.uint8)

                # Anlamlı alan filtresi (> 40 piksel)
                detected_pixels = int(np.sum(pred_binary))
                is_flagged = detected_pixels >= 40

                if is_flagged:
                    flagged += 1

                if not is_clean:
                    tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
                    fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
                    fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        dt = time.perf_counter() - t0
        total = len(samples)

        if is_clean:
            correct_clean = total - flagged
            clean_acc = (correct_clean / total) * 100.0
            fa_rate = (flagged / total) * 100.0
            print(f"\n>>> {name.upper()}")
            print(f"    Doğru Temiz : {correct_clean} / {total} (%{clean_acc:.1f})")
            print(f"    Yanlış Alarm: {flagged} / {total} (%{fa_rate:.1f})")
            print(f"    Süre        : {dt:.2f} sn ({dt/total*1000:.1f} ms/görsel)")
            return {"acc": clean_acc, "fa": fa_rate}
        else:
            det_rate = (flagged / total) * 100.0
            prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
            rec = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 0.0
            iou = (total_tp / (total_tp + total_fp + total_fn) * 100.0) if (total_tp + total_fp + total_fn) > 0 else 0.0
            print(f"\n>>> {name.upper()}")
            print(f"    Yakalanan       : {flagged} / {total} (%{det_rate:.1f})")
            print(f"    Piksel Hassasiyet: %{prec:.2f}")
            print(f"    Piksel Duyarlılık: %{rec:.2f}")
            print(f"    Piksel IoU       : %{iou:.2f}")
            print(f"    Süre            : {dt:.2f} sn ({dt/total*1000:.1f} ms/görsel)")
            return {"det": det_rate, "prec": prec, "rec": rec, "iou": iou}

    res_clean = evaluate_suite("Test 1: Temiz Görseller (Kontrol)", test_clean, is_clean=True)
    res_comofod = evaluate_suite("Test 2: CoMoFoD Copy-Move (50 Kirli)", test_cm_dirty)
    res_df_cm = evaluate_suite("Test 3: DEFACTO Copy-Move (50 Kirli)", test_df_cm)
    res_df_spl = evaluate_suite("Test 4: DEFACTO Splicing (50 Kirli)", test_df_spl)

    print("\n" + "=" * 80)
    print("ÖZET KARŞILAŞTIRMA TABLOSU")
    print("=" * 80)
    print(f"Temiz Görsel Başarısı (Doğru Temiz) : %{res_clean['acc']:.1f}  (Hedef: >= %90)")
    print(f"CoMoFoD Copy-Move Yakalama Oranı     : %{res_comofod['det']:.1f}")
    print(f"DEFACTO Copy-Move Yakalama Oranı     : %{res_df_cm['det']:.1f}")
    print(f"DEFACTO Splicing Yakalama Oranı      : %{res_df_spl['det']:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
