import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from forensivision.copy_move import CopyMoveConfig, detect_copy_move

DATA_ROOT = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
MODEL_PATH = PROJECT_ROOT / "models" / "forensivision_model.onnx"

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
session = ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])

sift_cfg = CopyMoveConfig(
    hull_fill_guard=True,
    spatial_cluster_guard=True,
    min_spatial_distance=20.0,
    ratio_threshold=0.78,
    point_radius=16,
    morph_kernel_size=9,
)

com_dir = DATA_ROOT / "CoMoFoD_small_v2"
all_pngs = list(com_dir.glob("*.png"))
clean_pngs = [p for p in all_pngs if "_o" in p.name.lower() and "_b" not in p.name.lower()][:100]

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("--- 100 TEMİZ GÖRSELİN DEĞER DAĞILIMI ---")
dl_scores = []
max_areas = []
sift_inlier_counts = []

for idx, p in enumerate(clean_pngs):
    with open(p, "rb") as f:
        buf = f.read()
    img_bgr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    norm = ((img_res.astype(np.float32) / 255.0) - mean) / std
    inp = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    outs = session.run(None, {"input_image": inp})
    seg_logits = outs[0][0, 0]
    cls_logit = float(outs[1][0])

    dl_score = 1.0 / (1.0 + np.exp(-cls_logit))
    mask_prob = 1.0 / (1.0 + np.exp(-seg_logits))

    raw_bin = (mask_prob > 0.60).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_bin = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bin)
    max_area = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0

    s_res = detect_copy_move(img_bgr, config=sift_cfg)
    inliers = s_res.num_inliers if s_res.is_tampered else 0

    dl_scores.append(dl_score)
    max_areas.append(max_area)
    sift_inlier_counts.append(inliers)

print(f"DL Score:   Min={np.min(dl_scores):.3f}, Medyan={np.median(dl_scores):.3f}, Max={np.max(dl_scores):.3f}")
print(f"Max Area:   Min={np.min(max_areas)}, Medyan={np.median(max_areas)}, Max={np.max(max_areas)}")
print(f"SIFT Inlier: Min={np.min(sift_inlier_counts)}, Medyan={np.median(sift_inlier_counts)}, Max={np.max(sift_inlier_counts)}")

print("\n--- ÇEŞİTLİ EŞİKLERDE TEMİZ BAŞARISI (100 Temiz Görsel) ---")
for area_th in [50, 100, 150, 200, 300]:
    for score_th in [0.70, 0.80, 0.85, 0.90, 0.95]:
        fa = sum(1 for s, a in zip(dl_scores, max_areas) if (s >= score_th and a >= area_th) or a >= 400)
        acc = 100 - fa
        if acc >= 88:
            print(f"  DL Eşik: score >= {score_th:.2f} & area >= {area_th:3d} -> Temiz Başarı: %{acc}")

print("\nSIFT Yanlış Alarm:")
for inlier_th in [4, 6, 8, 10, 12]:
    sift_fa = sum(1 for inl in sift_inlier_counts if inl >= inlier_th)
    print(f"  SIFT inlier >= {inlier_th:2d} -> Yanlış Alarm Sayısı: {sift_fa}/100 (%{100 - sift_fa} Temiz)")
