import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from forensivision.copy_move import CopyMoveConfig, detect_copy_move

DATA_ROOT = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
com_dir = DATA_ROOT / "CoMoFoD_small_v2"
def safe_read(p):
    with open(p, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

clean_pngs = [p for p in com_dir.glob("*.png") if "_o" in p.name.lower() and "_b" not in p.name.lower()][:50]
dirty_pngs = [p for p in com_dir.glob("*.png") if "_o" not in p.name.lower() and "_b" not in p.name.lower() and "_m" not in p.name.lower()][:50]

clean_imgs = [safe_read(p) for p in clean_pngs if safe_read(p) is not None]
dirty_imgs = [safe_read(p) for p in dirty_pngs if safe_read(p) is not None]

print("--- SIFT OPTİMİZASYON DENEYİ (50 Temiz + 50 Kirli) ---")
for ratio in [0.68, 0.70, 0.72, 0.75]:
    for dist in [25.0, 30.0, 35.0]:
        for min_inl in [6, 8, 10]:
            cfg = CopyMoveConfig(
                hull_fill_guard=True,
                spatial_cluster_guard=True,
                min_spatial_distance=dist,
                ratio_threshold=ratio,
                min_inliers=min_inl,
            )
            # Clean
            clean_ok = sum(1 for img in clean_imgs if not detect_copy_move(img, config=cfg).is_tampered)
            c_acc = (clean_ok / len(clean_imgs)) * 100.0

            if c_acc >= 88.0:
                dirty_ok = sum(1 for img in dirty_imgs if detect_copy_move(img, config=cfg).is_tampered)
                d_acc = (dirty_ok / len(dirty_imgs)) * 100.0
                print(f"Ratio={ratio:.2f}, Dist={dist:4.1f}, MinInl={min_inl:2d} -> TEMİZ: %{c_acc:.1f} | KİRLİ: %{d_acc:.1f}")
