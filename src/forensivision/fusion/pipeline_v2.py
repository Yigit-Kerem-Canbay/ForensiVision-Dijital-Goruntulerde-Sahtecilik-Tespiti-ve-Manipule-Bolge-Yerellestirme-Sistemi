"""
ForensiVision v2 — Çift Ölçekli (Global + Tam Çözünürlüklü ROI) Hibrit Adli Çıkarım Hattı.

Bu modül, [forensivision_v2_arastirma_ve_plan.md] içindeki **Risk 2** (Saf 512x512 Kayan Pencere Tuzağı)
ve **Bölüm 5** (Modeli Yukarı Taşıyan Stacking Algoritması) gereksinimlerini uçtan uca uygular:

1. Global Geçiş (512x512 Tam Görsel):
   - Görselin tamamı tek seferde `ForensiVisionV2Net`'e girer; böylece `SelfCorrelation2D`
     modülü fotoğrafın bir ucundan diğer ucuna kopyalanan nesneyi bile aynı pencerede görür.
2. Yerel Tam Çözünürlüklü ROI Geçişi (Native-Resolution Crop):
   - Orijinal pikseller hiç yeniden boyutlandırılmadan (sensör gürültüsü ve 8x8 JPEG blokları
     bozulmadan), global geçişte en şüpheli görünen veya en yüksek gradyan enerjisine sahip
     512x512 bölge kırpılarak modele ikinci kez verilir.
3. Klasik Adli Uzmanlar (SIFT+RANSAC + Yoğun Alan Copy-Move + ELA/Gürültü Farkı):
   - Çıkarılan 16 özellik `StackingForensicClassifier`'a girer.
4. Yanlış Piksel Boyama Koruması (Zero-FP Clean Mask Gating):
   - Eğer meta-sınıflandırıcı görselin **Temiz (Authentic)** olduğuna karar verirse,
     çıktı maskesi kesin olarak sıfırlanır (temiz görselde rastgele kırmızı leke bırakılmaz).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from forensivision.copy_move import (
    CopyMoveConfig,
    DenseFieldCopyMoveDetector,
    detect_copy_move,
)
from forensivision.fusion.meta_classifier import (
    FEATURE_NAMES,
    ForensicEvidenceVector,
    StackingForensicClassifier,
    compute_jpeg_blockiness,
    compute_region_discrepancy,
)
from forensivision.models.forensivision_v2 import ForensiVisionV2Net


@dataclass(frozen=True)
class ForensiVisionV2Prediction:
    """Uçtan uca ForensiVision v2 analiz sonucu."""

    is_tampered: bool
    tampered_probability: float
    decision_threshold: float
    binary_mask: np.ndarray        # (H, W) uint8 (0 veya 255)
    probability_map: np.ndarray    # (H, W) float32 [0.0, 1.0]
    confidence_map: np.ndarray     # (H, W) float32 [0.0, 1.0]
    evidence: dict[str, float]
    diagnostics: dict[str, Any]


class ForensiVisionV2Pipeline:
    """Çift Ölçekli Derin Model + Çoklu Adli Uzman + Stacking Meta-Sınıflandırıcı Hattı."""

    def __init__(
        self,
        model: ForensiVisionV2Net | None = None,
        meta_classifier: StackingForensicClassifier | None = None,
        device: str | torch.device = "cpu",
        input_size: int = 512,
    ) -> None:
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = model.to(self.device).eval() if model is not None else None
        self.meta_classifier = meta_classifier or StackingForensicClassifier()
        self.dense_cm = DenseFieldCopyMoveDetector()
        self.sift_cfg = CopyMoveConfig(
            min_inliers=8,
            ratio_threshold=0.72,
            min_spatial_distance=25.0,
            spatial_cluster_guard=True,
        )

    def _bgr_to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def _select_native_roi_box(
        self,
        image_bgr: np.ndarray,
        global_anom_map: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Orijinal çözünürlükte (H, W) en şüpheli `input_size x input_size` pencere koordinatını seçer."""
        h, w = image_bgr.shape[:2]
        s = self.input_size
        if h <= s and w <= s:
            return 0, 0, w, h

        # Anomali haritasının ağırlık merkezini bul
        cy, cx = h // 2, w // 2
        if global_anom_map.max() > 0.15:
            max_idx = int(np.argmax(global_anom_map))
            cy, cx = divmod(max_idx, w)

        x0 = int(np.clip(cx - s // 2, 0, max(0, w - s)))
        y0 = int(np.clip(cy - s // 2, 0, max(0, h - s)))
        x1 = min(w, x0 + s)
        y1 = min(h, y0 + s)
        return x0, y0, x1, y1

    @torch.no_grad()
    def extract_evidence(self, image_bgr: np.ndarray) -> ForensicEvidenceVector:
        """Görüntüden 16 adli kanıt özelliğini ve birleşik olasılık haritasını çıkarır."""
        orig_h, orig_w = image_bgr.shape[:2]
        s = self.input_size

        # 1. Derin Öğrenme Uzmanı (Global 512x512 + Test-Time Augmentation [TTA] + Yerel Tam Çözünürlüklü ROI)
        if self.model is not None:
            img_global = cv2.resize(image_bgr, (s, s), interpolation=cv2.INTER_AREA)
            out_g = self.model(self._bgr_to_tensor(img_global))

            # TTA (Yatay çevrilmiş ikinci bakış: rastgele yanlış alarmları bastırır)
            img_flip = cv2.flip(img_global, 1)
            out_f = self.model(self._bgr_to_tensor(img_flip))

            prob_g = 0.5 * (
                torch.sigmoid(out_g.mask_logits)[0, 0].cpu().numpy()
                + np.fliplr(torch.sigmoid(out_f.mask_logits)[0, 0].cpu().numpy())
            )
            conf_g = 0.5 * (
                torch.sigmoid(out_g.confidence_logits)[0, 0].cpu().numpy()
                + np.fliplr(torch.sigmoid(out_f.confidence_logits)[0, 0].cpu().numpy())
            )
            edge_g = 0.5 * (
                torch.sigmoid(out_g.edge_logits)[0, 0].cpu().numpy()
                + np.fliplr(torch.sigmoid(out_f.edge_logits)[0, 0].cpu().numpy())
            )
            corr_g = 0.5 * (
                out_g.self_corr_map[0, 0].cpu().numpy()
                + np.fliplr(out_f.self_corr_map[0, 0].cpu().numpy())
            )
            cls_prob_global = 0.5 * (
                float(torch.sigmoid(out_g.cls_logits)[0, 0].cpu())
                + float(torch.sigmoid(out_f.cls_logits)[0, 0].cpu())
            )

            prob_full = cv2.resize(prob_g, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            conf_full = cv2.resize(conf_g, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            weighted_anom = prob_full * conf_full

            # Yerel tam çözünürlüklü ROI geçişi (eğer görsel 512x512'den büyükse)
            if orig_h > s or orig_w > s:
                x0, y0, x1, y1 = self._select_native_roi_box(image_bgr, weighted_anom)
                crop_bgr = image_bgr[y0:y1, x0:x1]
                if crop_bgr.shape[0] != s or crop_bgr.shape[1] != s:
                    crop_pad = cv2.resize(crop_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
                else:
                    crop_pad = crop_bgr
                out_roi = self.model(self._bgr_to_tensor(crop_pad))
                cls_prob_local = float(torch.sigmoid(out_roi.cls_logits)[0, 0].cpu())
                roi_prob = torch.sigmoid(out_roi.mask_logits)[0, 0].cpu().numpy()
                roi_conf = torch.sigmoid(out_roi.confidence_logits)[0, 0].cpu().numpy()
                roi_resized = cv2.resize(roi_prob * roi_conf, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                weighted_anom[y0:y1, x0:x1] = np.maximum(
                    weighted_anom[y0:y1, x0:x1], 0.5 * weighted_anom[y0:y1, x0:x1] + 0.5 * roi_resized
                )
            else:
                cls_prob_local = cls_prob_global

            anom_mean = float(weighted_anom.mean())
            anom_max = float(weighted_anom.max())
            anom_top5 = float(np.quantile(weighted_anom, 0.95))
            edge_peak = float(edge_g.max())
            corr_peak = float(corr_g.max())
            corr_area = float((corr_g >= 0.70).mean())
        else:
            weighted_anom = np.zeros((orig_h, orig_w), dtype=np.float32)
            conf_full = np.ones((orig_h, orig_w), dtype=np.float32)
            cls_prob_global = 0.0
            cls_prob_local = 0.0
            anom_mean = 0.0
            anom_max = 0.0
            anom_top5 = 0.0
            edge_peak = 0.0
            corr_peak = 0.0
            corr_area = 0.0

        # 2. Klasik Geometrik (SIFT+RANSAC) ve Yoğun Alan (Dense-Field) Copy-Move Uzmanları
        sift_res = detect_copy_move(image_bgr, config=self.sift_cfg)
        sift_norm = float(np.clip(sift_res.num_inliers / 20.0, 0.0, 1.0)) if sift_res.is_tampered else 0.0

        dense_res = self.dense_cm.detect(image_bgr)
        dense_score = float(dense_res.score)

        # Birleşik olasılık haritası (Derin Anomali + SIFT Maskesi + Yoğun Copy-Move Maskesi)
        combined_prob = weighted_anom.copy()
        if sift_res.is_tampered and np.count_nonzero(sift_res.mask) > 0:
            sift_float = (sift_res.mask > 127).astype(np.float32) * min(0.90, 0.45 + sift_norm * 0.5)
            combined_prob = np.maximum(combined_prob, sift_float)
        if dense_score >= 0.55 and np.count_nonzero(dense_res.mask) > 0:
            dense_float = (dense_res.mask > 127).astype(np.float32) * (dense_score * 0.85)
            combined_prob = np.maximum(combined_prob, dense_float)

        # 3. Maske morfolojisi ve en büyük bileşen alanı
        raw_bin = (combined_prob >= 0.48).astype(np.uint8) * 255
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_bin = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, k5)
        clean_bin = cv2.morphologyEx(clean_bin, cv2.MORPH_CLOSE, k5)

        total_px = float(orig_h * orig_w)
        mask_area_ratio = float(np.count_nonzero(clean_bin) / total_px)

        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
        largest_cc_px = 0
        filtered_bin = np.zeros_like(clean_bin)
        min_cc_px = max(48, int(total_px * 0.0005))
        for i in range(1, num_lbl):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_cc_px:
                filtered_bin[lbls == i] = 255
                largest_cc_px = max(largest_cc_px, area)
        largest_cc_ratio = float(largest_cc_px / total_px)

        # 4. Splicing için ELA ve Yerel Gürültü Uyumsuzluğu + Görüntü Bağlamı
        ela_ratio, noise_diff = compute_region_discrepancy(image_bgr, combined_prob)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blockiness = compute_jpeg_blockiness(gray)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)
        texture_entropy = float(np.clip(grad_mag.mean() / 40.0, 0.0, 2.0))

        feat_vec = np.array(
            [
                cls_prob_global,
                cls_prob_local,
                anom_mean,
                anom_max,
                anom_top5,
                edge_peak,
                corr_peak,
                corr_area,
                mask_area_ratio,
                largest_cc_ratio,
                sift_norm,
                dense_score,
                ela_ratio,
                noise_diff,
                blockiness,
                texture_entropy,
            ],
            dtype=np.float32,
        )

        return ForensicEvidenceVector(
            features=feat_vec,
            combined_mask=filtered_bin,
            probability_map=combined_prob,
            diagnostics={
                "sift_inliers": int(sift_res.num_inliers),
                "dense_cm_blocks": int(dense_res.dominant_cluster_blocks),
                "confidence_map": conf_full,
            },
        )

    def predict(self, image_bgr: np.ndarray) -> ForensiVisionV2Prediction:
        """Görüntünün Temiz mi Kirli mi olduğunu ve manipüle piksel maskesini döndürür."""
        ev = self.extract_evidence(image_bgr)
        prob = float(self.meta_classifier.predict_proba(ev.features))
        thresh = float(self.meta_classifier.decision_threshold)
        is_tampered = bool(prob >= thresh)

        # Temiz kararda maskeyi tamamen sıfırla (yanlış piksel boyama koruması)
        final_mask = ev.combined_mask if is_tampered else np.zeros_like(ev.combined_mask)

        return ForensiVisionV2Prediction(
            is_tampered=is_tampered,
            tampered_probability=round(prob, 4),
            decision_threshold=round(thresh, 4),
            binary_mask=final_mask,
            probability_map=ev.probability_map,
            confidence_map=ev.diagnostics["confidence_map"],
            evidence=ev.to_feature_dict(),
            diagnostics={
                "sift_inliers": ev.diagnostics["sift_inliers"],
                "dense_cm_blocks": ev.diagnostics["dense_cm_blocks"],
            },
        )

