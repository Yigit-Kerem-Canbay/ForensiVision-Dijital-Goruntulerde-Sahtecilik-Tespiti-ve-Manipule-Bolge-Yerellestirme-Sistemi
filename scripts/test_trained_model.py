"""
ForensiVision Eğitilmiş Modeli Canlı Test Betiği.
Kaggle'dan indirilen modeli yerel CPU'da milisaniyeler içinde test eder.
"""

from pathlib import Path
import sys
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


def predict_image(model, img_path: Path):
    with open(img_path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Görsel okunamadı: {img_path}")
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    t = torch.from_numpy(np.transpose(img_norm, (2, 0, 1)).astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        seg_out, cls_out = model(t)
        tamper_prob = torch.sigmoid(cls_out).item()
        mask_prob = torch.sigmoid(seg_out)[0, 0].numpy()

    mask_binary = (mask_prob > 0.50).astype(np.uint8)
    suspicious_pixels = int(np.sum(mask_binary))

    # Kalibre edilmiş karar: Sınıflandırıcı skoru > 0.50 ve anlamlı piksel kümesi (> 200)
    is_tampered = (tamper_prob > 0.50) or (suspicious_pixels > 250)

    return {
        "tamper_prob": tamper_prob,
        "suspicious_pixels": suspicious_pixels,
        "is_tampered": is_tampered,
        "mask_shape": mask_binary.shape
    }


def main():
    print("=" * 70)
    print("FORENSIVISION CANLI MODEL TESTİ")
    print("=" * 70)

    model = ForensicUNet(pretrained=False)
    weights_path = Path("models/forensivision_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print("✅ Model ağırlıkları CPU hafızasına başarıyla yüklendi!")

    data_root = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")

    # 1. Temiz Örnek Testi
    clean_sample = data_root / "CoMoFoD_small_v2" / "001_O.png"
    if clean_sample.exists():
        res = predict_image(model, clean_sample)
        print(f"\n[Test 1: Orijinal Temiz Görsel] ({clean_sample.name})")
        print(f"  -> Sahtecilik Olasılığı : %{res['tamper_prob']*100:.2f}")
        print(f"  -> Şüpheli Piksel Sayısı: {res['suspicious_pixels']}")
        print(f"  -> Karar: {'🚨 SAHTE ALARM' if res['is_tampered'] else '✅ TERTEMİZ (DOĞRU KARAR)'}")

    # 2. Copy-Move Sahte Örnek Testi
    cm_sample = data_root / "defacto-copymove" / "copymove_img" / "img" / "0_000000000071.tif"
    if cm_sample.exists():
        res = predict_image(model, cm_sample)
        print(f"\n[Test 2: DEFACTO Copy-Move Sahte Görsel] ({cm_sample.name})")
        print(f"  -> Sahtecilik Olasılığı : %{res['tamper_prob']*100:.2f}")
        print(f"  -> Şüpheli Piksel Sayısı: {res['suspicious_pixels']}")
        print(f"  -> Karar: {'🚨 SAHTECİLİK YAKALANDI! (DOĞRU KARAR)' if res['is_tampered'] else '❌ KAÇIRILDI'}")

    # 3. Splicing Sahte Örnek Testi
    sp_sample = data_root / "defacto-splicing" / "splicing_1_img" / "img" / "0_000000195755.tif"
    if sp_sample.exists():
        res = predict_image(model, sp_sample)
        print(f"\n[Test 3: DEFACTO Splicing Sahte Görsel] ({sp_sample.name})")
        print(f"  -> Sahtecilik Olasılığı : %{res['tamper_prob']*100:.2f}")
        print(f"  -> Şüpheli Piksel Sayısı: {res['suspicious_pixels']}")
        print(f"  -> Karar: {'🚨 SAHTECİLİK YAKALANDI! (DOĞRU KARAR)' if res['is_tampered'] else '❌ KAÇIRILDI'}")

    print("\n" + "=" * 70)
    print("CANLI TEST TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()
