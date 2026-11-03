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
opts.intra_op_num_threads = 8
session = ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])

sift_cfg = CopyMoveConfig(
    hull_fill_guard=True,
    spatial_cluster_guard=True,
    min_spatial_distance=20.0,
    ratio_threshold=0.78,
    point_radius=16,
    morph_kernel_size=9,
    min_inliers=6,
)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict(img_bgr):
    # 1. DL Inference
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

    # DL Kararı
    dl_tampered = (dl_score >= 0.60 and max_area >= 60) or (max_area >= 250) or (dl_score >= 0.90)

    # 2. SIFT Doğrulaması (Eğer DL temiz dediyse klonlama kontrolü)
    sift_tampered = False
    sift_inliers = 0
    if not dl_tampered:
        try:
            s_res = detect_copy_move(img_bgr, config=sift_cfg)
            sift_inliers = s_res.num_inliers
            # Güçlü geometrik eşleşme (en az 8 inlier)
            sift_tampered = s_res.is_tampered and (sift_inliers >= 8) and (np.count_nonzero(s_res.mask) > 0)
        except Exception:
            pass

    final_verdict = dl_tampered or sift_tampered
    return final_verdict, dl_tampered, sift_tampered


def evaluate_set(name, paths, is_clean):
    correct = 0
    for p in paths:
        with open(p, "rb") as f:
            buf = f.read()
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        v, _, _ = predict(img)
        if is_clean and not v:
            correct += 1
        elif not is_clean and v:
            correct += 1
    acc = (correct / len(paths)) * 100.0
    print(f"{name:<35}: %{acc:.2f} ({correct}/{len(paths)})", flush=True)
    return acc


def main():
    rng = np.random.RandomState(42)
    com_dir = DATA_ROOT / "CoMoFoD_small_v2"
    all_pngs = list(com_dir.glob("*.png"))
    clean_pngs = [p for p in all_pngs if "_o" in p.name.lower() and "_b" not in p.name.lower()]
    dirty_pngs = [p for p in all_pngs if "_o" not in p.name.lower() and "_b" not in p.name.lower() and "_m" not in p.name.lower()]
    rng.shuffle(clean_pngs)
    rng.shuffle(dirty_pngs)

    cm_dir = DATA_ROOT / "defacto-copymove" / "copymove_img" / "img"
    df_cm = list(cm_dir.glob("*.tif"))
    rng.shuffle(df_cm)

    sp_dir = DATA_ROOT / "defacto-splicing" / "splicing_1_img" / "img"
    df_sp = list(sp_dir.glob("*.tif"))
    rng.shuffle(df_sp)

    print("=" * 60)
    print("MÜKEMMEL FÜZYON KALİBRASYON TESTİ (100'er Örnek)")
    print("=" * 60)
    evaluate_set("1. CoMoFoD Temiz (Yanlış Alarm)", clean_pngs[:100], is_clean=True)
    evaluate_set("2. CoMoFoD Kopyala-Kaydır (Kirli)", dirty_pngs[:100], is_clean=False)
    evaluate_set("3. DEFACTO Kopyala-Kaydır (Kirli)", df_cm[:100], is_clean=False)
    evaluate_set("4. DEFACTO Splicing / Montaj (Kirli)", df_sp[:100], is_clean=False)
    print("=" * 60)


if __name__ == "__main__":
    main()
