"""
ForensiVision Hibrit Füzyon (Derin Öğrenme + SIFT + Akıllı Eşik) Doğrulama Testi.

Bu betik:
1. Yol 1: Akıllı Parazit Temizleme (Morphology & Connected Components) ile derin öğrenme eşiğini kalibre eder.
2. Yol 2: Derin Öğrenme ile SIFT Geometrik Eşleştirme modülünü birleştirir.
3. 200'er adet temsili görsel üzerinde:
   - Eski Kural (Ham DL)
   - Yol 1 (Kalibre Edilmiş Akıllı DL)
   - Yol 1 + Yol 2 (ForensiVision Hibrit Füzyon)
   karşılaştırmasını çıkarır.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forensivision.copy_move import CopyMoveConfig, detect_copy_move

DATA_ROOT = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
MODEL_PATH = PROJECT_ROOT / "models" / "forensivision_model.onnx"

sift_cfg = CopyMoveConfig(
    hull_fill_guard=True,
    spatial_cluster_guard=True,
    min_spatial_distance=20.0,
    ratio_threshold=0.78,
    point_radius=16,
    morph_kernel_size=9,
)


def safe_read_image(path: Path) -> np.ndarray | None:
    try:
        with open(path, "rb") as f:
            buf = f.read()
        return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_read_mask(path: Path) -> np.ndarray | None:
    try:
        with open(path, "rb") as f:
            buf = f.read()
        return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def run_hybrid_inference(session, img_bgr: np.ndarray):
    """
    Hibrit Karar Motoru:
    - DL Segmentasyon ve Sınıflandırma
    - Akıllı Parazit Filtresi (Morphology + Connected Components)
    - Gerektiğinde veya Klonlama durumunda SIFT Geometrik Eşleştirme
    """
    h_orig, w_orig = img_bgr.shape[:2]

    # --- 1. Derin Öğrenme Ön İşleme ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = ((img_res.astype(np.float32) / 255.0) - mean) / std
    inp = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    # ONNX Çıkarımı
    outs = session.run(None, {"input_image": inp})
    seg_logits = outs[0][0, 0] # (256, 256)
    cls_logit = float(outs[1][0])

    dl_score = 1.0 / (1.0 + np.exp(-cls_logit))
    dl_mask_prob = 1.0 / (1.0 + np.exp(-seg_logits))

    # --- ESKİ BASİT KURAL ---
    old_raw_bin = (dl_mask_prob > 0.50).astype(np.uint8)
    old_flag = (dl_score > 0.50) or (int(np.sum(old_raw_bin)) > 200)

    # --- YOL 1: AKILLI PARAZİT FİLTRESİ (Morphology & Connected Components) ---
    raw_bin = (dl_mask_prob > 0.55).astype(np.uint8)
    # 3x3 eliptik çekirdekle izole tekil pikselleri (gürültüyü) sil
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_bin = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, kernel)

    # Bağlantılı bileşenler analizi (Connected Components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bin)
    
    max_comp_area = 0
    meaningful_pixels = 0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_comp_area = int(np.max(areas))
        meaningful_pixels = int(np.sum(areas[areas >= 25]))

    # Akıllı DL Karar Mantığı:
    # 1. Ya model sahtecilikten çok emin (dl_score >= 0.70) ve ortada minik de olsa bir leke var (max_comp_area >= 40)
    # 2. Ya da ekranda en az 200 piksellik birbirine kenetli sağlam bir leke var (max_comp_area >= 180)
    # 3. Veya sınıflandırıcı aşırı yüksek (dl_score >= 0.90)
    dl_smart_flag = (
        (dl_score >= 0.70 and max_comp_area >= 40) or
        (max_comp_area >= 180) or
        (dl_score >= 0.90)
    )

    # --- YOL 2: SIFT GEOMETRİK EŞLEŞTİRME (Klonlama Dedektörü) ---
    # Eğer DL sahtecilik yakalayamadıysa veya DL skoru belirsizse (0.20 - 0.75 arası)
    # veya ikiz klonlamadan şüpheleniliyorsa SIFT devreye girer.
    sift_flag = False
    sift_inliers = 0
    sift_mask = None

    # Akıllı Tetikleme: SIFT'i her zaman veya DL'in kaçırabileceği durumlarda çalıştır
    if not dl_smart_flag or dl_score < 0.85:
        try:
            sift_res = detect_copy_move(img_bgr, config=sift_cfg)
            sift_inliers = sift_res.num_inliers
            sift_flag = sift_res.is_tampered and (np.count_nonzero(sift_res.mask) > 0)
            if sift_flag:
                sift_mask = cv2.resize(sift_res.mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        except Exception:
            pass

    # --- HİBRİT FÜZYON KARARI (Kalibre Edilmiş Karar Motoru) ---
    # 1. DL güçlü bir anomali yakaladıysa: SAHTEDİR.
    # 2. DL temiz diyorsa, SIFT'in bunu geçersiz kılması için sağlam bir geometrik kanıt
    #    (en az 6 inlier ve anlamlı kümelenme) sunması gerekir.
    #    Böylece temiz görsellerdeki şans eseri 4 inlier'lık gürültüler elenir!
    if dl_smart_flag:
        hybrid_flag = True
    elif sift_flag and (sift_inliers >= 6):
        hybrid_flag = True
    else:
        hybrid_flag = False

    # Füzyon Maskesi: İki yöntemin bulduğu şüpheli pikselleri birleştir
    fused_mask = clean_bin.copy()
    if sift_mask is not None and hybrid_flag:
        fused_mask = cv2.bitwise_or(fused_mask, (sift_mask > 0).astype(np.uint8))

    return {
        "old_flag": old_flag,
        "dl_smart_flag": dl_smart_flag,
        "sift_flag": sift_flag,
        "hybrid_flag": hybrid_flag,
        "dl_score": dl_score,
        "max_comp_area": max_comp_area,
        "sift_inliers": sift_inliers,
        "fused_mask": fused_mask,
    }


def test_samples(name: str, samples: list, is_clean: bool, session, limit: int = 200):
    print(f"\n---> Test Ediliyor: {name} ({min(limit, len(samples))} Görsel) ...", flush=True)
    
    old_correct = 0
    dl_smart_correct = 0
    hybrid_correct = 0

    evaluated = 0
    t0 = time.perf_counter()

    for idx, item in enumerate(samples[:limit]):
        img = safe_read_image(item["path"])
        if img is None:
            continue

        res = run_hybrid_inference(session, img)
        evaluated += 1

        if is_clean:
            # Temiz görsellerde hedef: Flag VERMEMEK (False Alarm olmamak)
            if not res["old_flag"]:
                old_correct += 1
            if not res["dl_smart_flag"]:
                dl_smart_correct += 1
            if not res["hybrid_flag"]:
                hybrid_correct += 1
        else:
            # Kirli görsellerde hedef: Flag VERMEK (Tespit etmek)
            if res["old_flag"]:
                old_correct += 1
            if res["dl_smart_flag"]:
                dl_smart_correct += 1
            if res["hybrid_flag"]:
                hybrid_correct += 1

    elapsed = time.perf_counter() - t0
    old_acc = (old_correct / evaluated) * 100.0 if evaluated > 0 else 0
    smart_acc = (dl_smart_correct / evaluated) * 100.0 if evaluated > 0 else 0
    hybrid_acc = (hybrid_correct / evaluated) * 100.0 if evaluated > 0 else 0

    print(f"     [Sonuç - {name}]")
    print(f"     * Eski Kural (Ham DL):          %{old_acc:.2f}")
    print(f"     * Yol 1 (Akıllı Eşik & Parazit): %{smart_acc:.2f}")
    print(f"     * Yol 1 + 2 (HİBRİT FÜZYON):     %{hybrid_acc:.2f}  (Süre: {elapsed:.1f}s)")

    return {
        "name": name,
        "is_clean": is_clean,
        "old_acc": old_acc,
        "smart_acc": smart_acc,
        "hybrid_acc": hybrid_acc,
    }


def main():
    print("=" * 70)
    print("FORENSIVISION: HİBRİT FÜZYON DOĞRULAMA DENEYİ")
    print("=" * 70)

    # ONNX Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 12
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(MODEL_PATH), opts, providers=["CPUExecutionProvider"])

    rng = np.random.RandomState(42)

    # 1. CoMoFoD Hazırla
    com_dir = DATA_ROOT / "CoMoFoD_small_v2"
    all_pngs = list(com_dir.glob("*.png"))
    com_clean = [{"path": p} for p in all_pngs if "_o" in p.name.lower() and "_b" not in p.name.lower()]
    com_dirty = [{"path": p} for p in all_pngs if "_o" not in p.name.lower() and "_b" not in p.name.lower() and "_m" not in p.name.lower()]
    rng.shuffle(com_clean)
    rng.shuffle(com_dirty)

    # 2. DEFACTO Copy-Move Hazırla
    cm_dir = DATA_ROOT / "defacto-copymove" / "copymove_img" / "img"
    defacto_cm = [{"path": p} for p in cm_dir.glob("*.tif")] if cm_dir.exists() else []
    rng.shuffle(defacto_cm)

    # 3. DEFACTO Splicing Hazırla
    sp_dir = DATA_ROOT / "defacto-splicing" / "splicing_1_img" / "img"
    defacto_sp = [{"path": p} for p in sp_dir.glob("*.tif")] if sp_dir.exists() else []
    rng.shuffle(defacto_sp)

    # Deneyleri Çalıştır (200'er örnek)
    results = []
    results.append(test_samples("1. CoMoFoD Temiz (Yanlış Alarm)", com_clean, is_clean=True, session=session, limit=200))
    results.append(test_samples("2. CoMoFoD Kopyala-Kaydır (Kirli)", com_dirty, is_clean=False, session=session, limit=200))
    results.append(test_samples("3. DEFACTO Kopyala-Kaydır (Kirli)", defacto_cm, is_clean=False, session=session, limit=200))
    results.append(test_samples("4. DEFACTO Splicing / Montaj (Kirli)", defacto_sp, is_clean=False, session=session, limit=200))

    print("\n" + "=" * 70)
    print("ÖZET KARŞILAŞTIRMA TABLOSU")
    print("=" * 70)
    print(f"{'Test Kümesi':<32} | {'Eski Ham DL':<12} | {'Yol 1 (Akıllı)':<14} | {'HİBRİT FÜZYON':<14}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<32} | %{r['old_acc']:<10.2f} | %{r['smart_acc']:<12.2f} | %{r['hybrid_acc']:<12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
