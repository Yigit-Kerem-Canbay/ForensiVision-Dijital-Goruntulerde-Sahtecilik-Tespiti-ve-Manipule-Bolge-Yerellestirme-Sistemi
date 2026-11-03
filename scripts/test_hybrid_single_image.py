"""
ForensiVision Hibrit Füzyon Tekil Görsel Analiz ve Görselleştirme Aracı.

Kullanım:
    python scripts/test_hybrid_single_image.py [GORSEL_YOLU]
    (Eğer argüman verilmezse, otomatik olarak CoMoFoD ve DEFACTO'dan örnek test görselleri seçip analiz eder.)
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forensivision import detect_forgery_hybrid, FusionConfig

DATA_ROOT = Path("C:/Users/yigit/Desktop/Sayısal Görünntü İşleme Veri Seti")
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "demo_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_and_visualize(image_path: Path):
    print("\n" + "=" * 70)
    print(f"İNCELEME BAŞLATILIYOR: {image_path.name}")
    print("=" * 70)

    # 1. Hibrit Analiz
    result = detect_forgery_hybrid(image_path)

    # 2. Raporlama
    status_str = "🔴 SAHTE / MANİPÜLE EDİLMİŞ" if result.is_tampered else "🟢 TERTEMİZ / ORİJİNAL"
    print(f"Nihai Adli Karar     : {status_str}")
    print(f"Manipülasyon Türü    : {result.tamper_type}")
    print(f"Genel Güven Skoru    : %{result.confidence_score * 100:.2f}")
    print(f"Birincil Dedektör    : {result.primary_detector}")
    print("-" * 70)
    print(f"Derin Öğrenme U-Net  : Skor=%{result.dl_score * 100:.1f} | Leke={result.dl_max_component_area} px | Flag={result.dl_flagged}")
    print(f"Geometrik SIFT       : Inlier={result.sift_inliers} | Flag={result.sift_flagged}")
    print(f"Etkilenen Piksel     : {result.dl_tampered_pixels} piksel")
    print("=" * 70)

    # 3. Görselleştirme Paneli Üretimi
    with open(image_path, "rb") as f:
        orig = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

    h, w = orig.shape[:2]
    # Standartlaştırma boyutu (görselleştirme için)
    disp_h, disp_w = 400, 400
    p1 = cv2.resize(orig, (disp_w, disp_h))

    # Panel 2: Maske
    mask_disp = cv2.resize(result.mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    p2 = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2BGR)

    # Panel 3: Isı Haritası Giydirmesi
    overlay = result.generate_visualization(orig)
    p3 = cv2.resize(overlay, (disp_w, disp_h))

    # Başlık Şeritleri Ekle
    def add_banner(img, text, color=(40, 40, 40)):
        canvas = np.full((disp_h + 40, disp_w, 3), color, dtype=np.uint8)
        canvas[40:, :] = img
        cv2.putText(canvas, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    b1 = add_banner(p1, "1. Orijinal Goruntu")
    b2 = add_banner(p2, f"2. Tahmin Maskesi ({np.count_nonzero(result.mask)} px)")
    verdict_color = (0, 0, 160) if result.is_tampered else (0, 140, 0)
    b3 = add_banner(p3, f"3. Adli Analiz: {result.tamper_type}", color=verdict_color)

    combined = np.hstack([b1, b2, b3])

    out_file = OUTPUT_DIR / f"analiz_{image_path.stem}.jpg"
    is_success, buf = cv2.imencode(".jpg", combined)
    if is_success:
        with open(out_file, "wb") as f:
            f.write(buf)
        print(f"💾 Adli Analiz Paneli Kaydedildi -> {out_file}\n")
    else:
        print(f"UYARI: Görsel kodlanamadı -> {out_file}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="ForensiVision Hibrit Füzyon Tekil Görsel Analizi")
    parser.add_argument("image_path", nargs="?", default=None, help="İncelenecek görselin dosya yolu")
    args = parser.parse_args()

    if args.image_path:
        p = Path(args.image_path)
        if not p.exists():
            print(f"HATA: Dosya bulunamadı -> {p}")
            sys.exit(1)
        analyze_and_visualize(p)
    else:
        print("Parametre belirtilmedi. 4 temsili kategoriden örnek görseller otomatik analiz ediliyor...")
        test_samples = []

        # 1. CoMoFoD Temiz
        com_dir = DATA_ROOT / "CoMoFoD_small_v2"
        if com_dir.exists():
            clean = [p for p in com_dir.glob("*.png") if "_o" in p.name.lower() and "_b" not in p.name.lower()]
            if clean:
                test_samples.append(clean[0])

            # 2. CoMoFoD Kopyala-Kaydır
            dirty = [p for p in com_dir.glob("*.png") if "_o" not in p.name.lower() and "_b" not in p.name.lower() and "_m" not in p.name.lower()]
            if dirty:
                test_samples.append(dirty[0])

        # 3. DEFACTO Copy-Move
        df_cm_dir = DATA_ROOT / "defacto-copymove" / "copymove_img" / "img"
        if df_cm_dir.exists():
            df_cm = list(df_cm_dir.glob("*.tif"))
            if df_cm:
                test_samples.append(df_cm[0])

        # 4. DEFACTO Splicing
        df_sp_dir = DATA_ROOT / "defacto-splicing" / "splicing_1_img" / "img"
        if df_sp_dir.exists():
            df_sp = list(df_sp_dir.glob("*.tif"))
            if df_sp:
                test_samples.append(df_sp[0])

        for s in test_samples:
            analyze_and_visualize(s)


if __name__ == "__main__":
    main()
