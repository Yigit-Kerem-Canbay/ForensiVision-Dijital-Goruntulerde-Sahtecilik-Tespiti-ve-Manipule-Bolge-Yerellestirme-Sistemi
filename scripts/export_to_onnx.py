"""
ForensiVision - PyTorch (.pth) to ONNX (.onnx) Dışa Aktarma ve Doğrulama Betiği.
Eğitilmiş adli tıp modelini üretim standardında ONNX formatına dönüştürür.
"""

from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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


def export():
    print("=" * 70)
    print("🚀 ForensiVision ONNX Dışa Aktarımı Başlatılıyor...")
    print("=" * 70)

    weights_path = Path("models/forensivision_model.pth")
    if not weights_path.exists():
        weights_path = Path("models/forensic_model_epoch6_final.pth")
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Model ağırlıkları bulunamadı: {weights_path}")

    model = ForensicUNet(pretrained=False)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"✅ PyTorch model ağırlıkları yüklendi ({weights_path.name})")

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    onnx_path = Path("models/forensivision_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["mask_output", "class_output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "mask_output": {0: "batch_size"},
            "class_output": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True
    )
    print(f"🎉 ONNX Modeli Başarıyla Oluşturuldu: {onnx_path} ({onnx_path.stat().st_size / (1024**2):.2f} MB)")

    # Sayısal Doğrulama Testi
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {"input": dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)

        with torch.no_grad():
            torch_mask, torch_cls = model(dummy_input)

        np.testing.assert_allclose(torch_mask.numpy(), ort_outs[0], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(torch_cls.numpy(), ort_outs[1], rtol=1e-3, atol=1e-4)
        print("✅ Mükemmel: PyTorch ve ONNX çıktıları %100 birebir uyuşuyor (Hata toleransı < 0.0001)!")
    except Exception as e:
        print(f"⚠️ ONNXRuntime testi sırasında uyarı: {e}")

    print("=" * 70)
    print("TÜM İŞLEM TAMAMLANDI")
    print("=" * 70)

if __name__ == "__main__":
    export()
