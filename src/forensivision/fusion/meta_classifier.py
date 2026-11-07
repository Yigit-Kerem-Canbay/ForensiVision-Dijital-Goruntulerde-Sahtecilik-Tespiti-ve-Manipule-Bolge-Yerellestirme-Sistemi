"""
ForensiVision v2 — Çoklu Uzman Adli Sinyal Çıkarıcı ve Stacking Meta-Sınıflandırıcı.

Neden v1'deki `dl_flagged or sift_flagged` (Basit VEYA Kapısı) Başarısız Oldu?
- Basit VEYA (OR) kapısında 4 katmanın her birinin %8 yanlış pozitif (FP) üretmesi,
  toplam yanlış alarm oranını %28'e çıkarır ve temiz fotoğrafların başarısını düşürür.

ForensiVision v2 Çözümü (Öğrenilmiş Stacking Meta-Sınıflandırıcı + Kalibrasyon):
Görüntüden 4 bağımsız adli uzmandan 16 boyutlu sürekli (continuous) özellik vektörü çıkarılır:
1. Derin Öğrenme Uzmanı (`ForensiVisionV2Net` - Çift Akışlı Global + Yerel Çıkarım):
   - Global geçiş (512x512): Görüntünün tamamında Öz-Korelasyon (Copy-Move) ve Splicing maskesi üretir.
   - Yerel tam çözünürlüklü ROI geçişi: Orijinal pikseller küçültülmeden en şüpheli 512x512 bölge
     kesilip SRM/Bayar gürültü ve sınır (edge) doğrulamasına sokulur.
2. Geometrik + Yoğun Copy-Move Uzmanı (`SIFT+RANSAC` + `DenseFieldCopyMoveDetector`):
   - Anahtar nokta inlier sayısı, küme kompaktlığı ve yoğun alan (Zernike/DCT) öteleme tutarlılığı.
3. Sıkıştırma ve Gürültü Tutarlılığı Uzmanı (ELA + Yerel Gürültü Varyansı Uyumsuzluğu):
   - Şüpheli maske içi ile maske dışı arka plan arasındaki ELA (Error Level Analysis) ve
     yüksek frekanslı Laplasyen gürültü varyansı oranı (Splicing'de çok güçlü ayrım sağlar).
4. Format ve Çözünürlük Bağlamı (Görüntü entropisi, sıkıştırma blokluluk oranı):
   - Düz/düşük entropili görsellerde veya ağır WhatsApp sıkıştırmasında eşiği otomatik dengeler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from forensivision.copy_move import (
    CopyMoveConfig,
    DenseFieldCopyMoveDetector,
    detect_copy_move,
)
from forensivision.splicing.ela import analyze_ela
from forensivision.splicing.noise import analyze_noise_inconsistency


FEATURE_NAMES: list[str] = [
    "dl_cls_prob_global",         # 0: Global 512x512 modelin görüntü seviyesi sahtecilik olasılığı
    "dl_cls_prob_local_roi",      # 1: Orijinal çözünürlüklü şüpheli ROI kırpmasının sahtecilik olasılığı
    "dl_weighted_anom_mean",      # 2: Güvenilirlik ağırlıklı anomali haritası ortalaması
    "dl_weighted_anom_max",       # 3: Güvenilirlik ağırlıklı anomali haritası zirve (peak) değeri
    "dl_weighted_anom_top5pct",   # 4: En şüpheli %5 pikselin ortalama güvenilirlikli olasılığı
    "dl_edge_peak",               # 5: Manipülasyon sınırı (boundary) başlığı zirve değeri
    "dl_self_corr_peak",          # 6: Derin Öz-Korelasyon (Copy-Move) ısı haritası zirvesi
    "dl_self_corr_area_ratio",    # 7: Öz-Korelasyonu > 0.70 olan piksel alanı oranı
    "dl_mask_area_ratio",         # 8: Tahmin edilen maskenin toplam görüntü alanına oranı
    "dl_largest_cc_ratio",        # 9: En büyük bağlantılı bileşenin (connected component) alan oranı
    "sift_ransac_inliers_norm",   # 10: SIFT + RANSAC geometrik inlier skoru (normalize)
    "dense_cm_score",             # 11: Yoğun alan (Dense-Field) Copy-Move skoru
    "ela_inside_outside_ratio",   # 12: Şüpheli bölge içi / dışı ELA enerji oranı (Splicing kanıtı)
    "noise_inside_outside_diff",  # 13: Şüpheli bölge içi / dışı yerel gürültü farkı (PRNU/sensör kanıtı)
    "jpeg_grid_blockiness",       # 14: 8x8 JPEG blok sınırı enerjisi (format/sıkıştırma bağlamı)
    "image_texture_entropy",      # 15: Görüntü gradyan entropisi (düz alan yanlış alarm koruyucusu)
]


@dataclass(frozen=True)
class ForensicEvidenceVector:
    """Tek bir görüntüden çıkarılan 16 adli kanıt özelliği ve birleşik piksel maskesi."""

    features: np.ndarray          # (16,) float32 vektör
    combined_mask: np.ndarray     # (H, W) uint8 ikili maske (0 veya 255)
    probability_map: np.ndarray   # (H, W) float32 [0.0, 1.0] olasılık haritası
    diagnostics: dict[str, Any]

    def to_feature_dict(self) -> dict[str, float]:
        return {name: float(self.features[i]) for i, name in enumerate(FEATURE_NAMES)}


def compute_jpeg_blockiness(gray: np.ndarray) -> float:
    """Görüntüdeki 8x8 JPEG blok sınırlarındaki periyodik sıçrama enerjisini ölçer."""
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0
    g = gray.astype(np.float32)
    # 8'in katı olan sütun/satır farkları ile iç sütun/satır farklarını oranla
    dh = np.abs(g[:, 1:] - g[:, :-1])
    dv = np.abs(g[1:, :] - g[:-1, :])

    grid_x = dh[:, 7::8].mean() if dh.shape[1] > 8 else 0.0
    non_grid_x = dh.mean() + 1e-5
    grid_y = dv[7::8, :].mean() if dv.shape[0] > 8 else 0.0
    non_grid_y = dv.mean() + 1e-5

    ratio = 0.5 * ((grid_x / non_grid_x) + (grid_y / non_grid_y))
    return float(np.clip(ratio - 1.0, -0.5, 2.0))


def compute_region_discrepancy(
    image_bgr: np.ndarray,
    prob_map: np.ndarray,
) -> tuple[float, float]:
    """Şüpheli bölge (iç) ile arka plan (dış) arasındaki ELA ve Gürültü uyumsuzluğunu hesaplar."""
    h, w = image_bgr.shape[:2]
    work_w, work_h = min(w, 512), min(h, 512)
    img_s = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    prob_s = cv2.resize(prob_map, (work_w, work_h), interpolation=cv2.INTER_LINEAR)

    inside = prob_s >= 0.45
    outside = prob_s < 0.25

    if np.count_nonzero(inside) < 25 or np.count_nonzero(outside) < 100:
        # Maske çok küçükse en yüksek %5'lik dilim ile en düşük %50'lik dilimi karşılaştır
        q95 = float(np.quantile(prob_s, 0.95))
        inside = prob_s >= max(q95, 1e-4)
        outside = prob_s <= float(np.quantile(prob_s, 0.50))

    if np.count_nonzero(inside) == 0 or np.count_nonzero(outside) == 0:
        return 1.0, 0.0

    ela_res = analyze_ela(img_s, quality=90, is_jpeg=True)
    noise_res = analyze_noise_inconsistency(img_s, window_size=9)
    ela_map = ela_res.raw_difference
    noise_map = noise_res.local_variance_map

    ela_in = float(ela_map[inside].mean())
    ela_out = float(ela_map[outside].mean()) + 1e-3
    ela_ratio = float(np.clip(ela_in / ela_out, 0.0, 10.0))

    noise_in = float(noise_map[inside].mean())
    noise_out = float(noise_map[outside].mean())
    noise_diff = float(np.clip(abs(noise_in - noise_out) / (noise_out + 1.0), 0.0, 5.0))
    return ela_ratio, noise_diff


class StackingForensicClassifier:
    """4 Uzmanlı Kanıt Vektörünü Birleştiren Kalibre Edilmiş Meta-Sınıflandırıcı.

    - Eğitim/Kalibrasyon verisi verildiğinde `HistGradientBoostingClassifier` + `LogisticRegression`
      topluluğunu (ensemble) eğitir ve **Temiz (Authentic) Özgüllük >= %93** kısıtını sağlayan
      optimal karar eşiğini (`decision_threshold`) otomatik bulur.
    - Henüz kalibre edilmemişse doğrulanmış ağırlıklı lojistik kuralla çalışır.
    """

    def __init__(self, target_specificity: float = 0.93) -> None:
        self.target_specificity = target_specificity
        self.gbdt = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            learning_rate=0.06,
            min_samples_leaf=15,
            random_state=42,
        )
        self.logreg = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.is_fitted: bool = False
        self.decision_threshold: float = 0.50
        self.feature_means: np.ndarray = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.feature_stds: np.ndarray = np.ones(len(FEATURE_NAMES), dtype=np.float32)

    def fit_and_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Meta-sınıflandırıcıyı eğitir ve kalibrasyon kümesinde Temiz >= hedef özgüllük eşiğini seçer."""
        self.feature_means = X_train.mean(axis=0).astype(np.float32)
        self.feature_stds = np.maximum(X_train.std(axis=0), 1e-5).astype(np.float32)

        X_tr_norm = (X_train - self.feature_means) / self.feature_stds
        self.gbdt.fit(X_tr_norm, y_train)
        self.logreg.fit(X_tr_norm, y_train)
        self.is_fitted = True

        X_eval = X_cal if X_cal is not None else X_train
        y_eval = y_cal if y_cal is not None else y_train

        probs = self.predict_proba(X_eval)
        clean_probs = probs[y_eval == 0]
        tampered_probs = probs[y_eval == 1]

        if len(clean_probs) > 0:
            # Temiz görsellerin en az `target_specificity` kadarının eşiğin altında kalmasını sağla
            cand_thresh = float(np.quantile(clean_probs, self.target_specificity))
            self.decision_threshold = float(np.clip(cand_thresh, 0.30, 0.85))
        else:
            self.decision_threshold = 0.50

        preds = (probs >= self.decision_threshold).astype(np.int32)
        spec = float((preds[y_eval == 0] == 0).mean()) if len(clean_probs) > 0 else 0.0
        rec = float((preds[y_eval == 1] == 1).mean()) if len(tampered_probs) > 0 else 0.0
        acc = float((preds == y_eval).mean())

        return {
            "decision_threshold": round(self.decision_threshold, 4),
            "clean_specificity": round(spec, 4),
            "tampered_recall": round(rec, 4),
            "overall_accuracy": round(acc, 4),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Girdi (N, 16) veya (16,) için [0.0, 1.0] sahtecilik olasılığı döndürür."""
        single = X.ndim == 1
        X_2d = X.reshape(1, -1) if single else X

        if self.is_fitted:
            X_norm = (X_2d - self.feature_means) / self.feature_stds
            p_gbdt = self.gbdt.predict_proba(X_norm)[:, 1]
            p_lr = self.logreg.predict_proba(X_norm)[:, 1]
            probs = 0.65 * p_gbdt + 0.35 * p_lr
        else:
            # Kalibrasyon öncesi dengeli analitik birleştirme (Naive OR yerine ağırlıklı kanıt skoru)
            dl_core = 0.45 * X_2d[:, 0] + 0.25 * X_2d[:, 1] + 0.30 * X_2d[:, 4]
            cm_core = np.maximum(X_2d[:, 6], np.maximum(X_2d[:, 10], X_2d[:, 11] * 0.85))
            sp_core = np.clip((X_2d[:, 12] - 1.15) * 0.35 + X_2d[:, 13] * 0.40, 0.0, 1.0)
            probs = np.clip(0.55 * dl_core + 0.30 * cm_core + 0.15 * sp_core, 0.0, 1.0)

        return float(probs[0]) if single else probs

    def save_calibration(self, path: str | Path) -> None:
        """Lojistik ağırlıkları, normalizasyon sabitlerini ve eşiği JSON olarak kaydeder."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "is_fitted": self.is_fitted,
            "decision_threshold": self.decision_threshold,
            "target_specificity": self.target_specificity,
            "feature_names": FEATURE_NAMES,
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "logreg_coef": self.logreg.coef_.tolist() if self.is_fitted else None,
            "logreg_intercept": self.logreg.intercept_.tolist() if self.is_fitted else None,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
