"""
ForensiVision Çok Kademeli Hibrit Füzyon (Hybrid Fusion) Sahtecilik Tespit Modülü.

Bu modül; dijital görüntülerdeki sahtecilikleri (Copy-Move, Splicing ve Karma Manipülasyonlar)
iki farklı adli inceleme katmanını birleştirerek tespit eder:
    1. Kademe (Derin Öğrenme / U-Net): ResNet-34 tabanlı konvolüsyonel sinir ağı ile
       piksel düzeyindeki gürültü tutarsızlıkları, dikiş izleri ve montaj sınırlarını tespit eder.
       Akıllı morfolojik filtreleme ve bağlantılı bileşen analizi (connected components) ile
       yanlış alarmlar elenir.
    2. Kademe (Geometrik Eşleştirme / SIFT): Eğer görsel derin öğrenme katmanında temiz görünüyorsa,
       kendi içinden kopyalanıp yapıştırılan (ikiz klon) desenleri yakalamak üzere SIFT ve RANSAC
       geometrik tutarlılık motorunu devreye sokar.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import onnxruntime as ort

from forensivision.copy_move import CopyMoveConfig, CopyMoveResult, detect_copy_move


@dataclass
class FusionConfig:
    """Hibrit Füzyon Tespit Motoru Yapılandırması."""

    # Model Yolu
    model_path: Path | str | None = None

    # Derin Öğrenme Eşikleri
    dl_score_threshold: float = 0.65
    dl_pixel_threshold: float = 0.55
    dl_min_component_area: int = 60
    dl_high_confidence_area: int = 250
    dl_auto_tamper_score: float = 0.90
    dl_morph_kernel_size: int = 3

    # SIFT Geometrik Eşleştirme Eşikleri
    sift_min_inliers: int = 8
    sift_ratio_threshold: float = 0.75
    sift_spatial_distance: float = 25.0
    sift_point_radius: int = 16
    sift_morph_kernel_size: int = 9

    # Donanım ve Çalışma Ayarları
    onnx_num_threads: int = 4


@dataclass(frozen=True)
class FusionResult:
    """Hibrit füzyon analizi sonuç nesnesi."""

    is_tampered: bool
    tamper_type: str  # "clean", "splicing_or_boundary", "copy_move_cloning", "hybrid_composite"
    confidence_score: float  # 0.0 - 1.0 arası genel güven skoru
    mask: np.ndarray  # Orijinal boyutlu ikili tahmin maskesi (uint8, 0 ve 255)
    
    # Alt Modül Çıktıları
    dl_flagged: bool
    dl_score: float
    dl_tampered_pixels: int
    dl_max_component_area: int

    sift_flagged: bool
    sift_inliers: int

    # Teşhis ve Raporlama
    primary_detector: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON serileştirilebilir özet sözlük."""
        return {
            "is_tampered": self.is_tampered,
            "tamper_type": self.tamper_type,
            "confidence_score": round(self.confidence_score, 4),
            "mask_shape": list(self.mask.shape),
            "mask_positive_pixels": int(np.count_nonzero(self.mask)),
            "dl_flagged": self.dl_flagged,
            "dl_score": round(self.dl_score, 4),
            "dl_tampered_pixels": self.dl_tampered_pixels,
            "dl_max_component_area": self.dl_max_component_area,
            "sift_flagged": self.sift_flagged,
            "sift_inliers": self.sift_inliers,
            "primary_detector": self.primary_detector,
            "diagnostics": self.diagnostics,
        }

    def generate_visualization(self, original_bgr: np.ndarray) -> np.ndarray:
        """Orijinal görsel üzerine şüpheli bölgeleri yarı şeffaf kırmızı ısı haritasıyla giydirir."""
        h, w = original_bgr.shape[:2]
        vis = original_bgr.copy()

        if self.is_tampered and np.count_nonzero(self.mask) > 0:
            mask_resized = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            heatmap = np.zeros_like(original_bgr)
            heatmap[:, :, 2] = 255  # Kırmızı kanal

            mask_bool = mask_resized > 127
            vis[mask_bool] = cv2.addWeighted(original_bgr[mask_bool], 0.45, heatmap[mask_bool], 0.55, 0.0)

            # Sınır konturlarını çiz
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        return vis


class ForensiVisionFusionDetector:
    """ForensiVision Hibrit Füzyon Tespit Sınıfı."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self.session = None
        self._init_model()

    def _init_model(self):
        """ONNX modelini başlatır."""
        model_path = self.config.model_path
        if model_path is None:
            # Varsayılan model yolunu tespit et
            proj_root = Path(__file__).resolve().parent.parent.parent.parent
            model_path = proj_root / "models" / "forensivision_model.onnx"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX Model dosyası bulunamadı: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.config.onnx_num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def detect(self, image: Union[np.ndarray, Path, str]) -> FusionResult:
        """Görseli hem derin öğrenme hem de geometrik eşleştirme ile hibrit adli analize tabi tutar."""
        img_bgr = self._load_image(image)
        h_orig, w_orig = img_bgr.shape[:2]

        # -------------------------------------------------------------
        # 1. KADEME: DERİN ÖĞRENME (ResNet-34 U-Net) ÇIKARIMI
        # -------------------------------------------------------------
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        norm = ((img_res.astype(np.float32) / 255.0) - self.mean) / self.std
        inp = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        outs = self.session.run(None, {"input_image": inp})
        seg_logits = outs[0][0, 0]  # (256, 256)
        cls_logit = float(outs[1][0])

        dl_score = float(1.0 / (1.0 + np.exp(-cls_logit)))
        mask_prob = 1.0 / (1.0 + np.exp(-seg_logits))

        # Akıllı Parazit Filtreleme (Morphology & Connected Components)
        raw_bin = (mask_prob > self.config.dl_pixel_threshold).astype(np.uint8)
        k_size = self.config.dl_morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dl_mask_256 = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dl_mask_256)
        max_comp_area = 0
        total_coherent_pixels = 0
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_comp_area = int(np.max(areas))
            total_coherent_pixels = int(np.sum(areas[areas >= 20]))

        # Derin Öğrenme Karar Kuralı
        dl_flagged = bool(
            (dl_score >= self.config.dl_score_threshold and max_comp_area >= self.config.dl_min_component_area)
            or (max_comp_area >= self.config.dl_high_confidence_area)
            or (dl_score >= self.config.dl_auto_tamper_score)
        )

        dl_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if dl_flagged and np.count_nonzero(dl_mask_256) > 0:
            dl_mask_orig = cv2.resize(dl_mask_256 * 255, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # -------------------------------------------------------------
        # 2. KADEME: GEOMETRİK SIFT KLONLAMA KONTROLÜ
        # -------------------------------------------------------------
        sift_flagged = False
        sift_inliers = 0
        sift_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # DL sahtecilik yakalayamadıysa veya skor orta banttaysa SIFT klon kontrolü yapar
        if not dl_flagged or dl_score < 0.85:
            try:
                sift_cfg = CopyMoveConfig(
                    hull_fill_guard=True,
                    spatial_cluster_guard=True,
                    min_spatial_distance=self.config.sift_spatial_distance,
                    ratio_threshold=self.config.sift_ratio_threshold,
                    point_radius=self.config.sift_point_radius,
                    morph_kernel_size=self.config.sift_morph_kernel_size,
                    min_inliers=self.config.sift_min_inliers,
                )
                s_res = detect_copy_move(img_bgr, config=sift_cfg)
                sift_inliers = s_res.num_inliers
                sift_flagged = bool(
                    s_res.is_tampered
                    and (sift_inliers >= self.config.sift_min_inliers)
                    and (np.count_nonzero(s_res.mask) > 0)
                )
                if sift_flagged:
                    sift_mask_orig = s_res.mask
            except Exception:
                pass

        # -------------------------------------------------------------
        # 3. KADEME: FÜZYON BİRLEŞTİRME VE NİHAİ KARAR
        # -------------------------------------------------------------
        is_tampered = dl_flagged or sift_flagged

        # Maskeleri birleştir (Bitwise OR)
        fused_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        if is_tampered:
            if dl_flagged and sift_flagged:
                fused_mask = cv2.bitwise_or(dl_mask_orig, sift_mask_orig)
                tamper_type = "hybrid_composite"
                primary_detector = "Ensemble (Deep Learning + SIFT)"
                confidence = max(dl_score, min(1.0, sift_inliers / 20.0))
            elif dl_flagged:
                fused_mask = dl_mask_orig
                tamper_type = "splicing_or_boundary"
                primary_detector = "Deep Learning U-Net"
                confidence = dl_score
            else:
                fused_mask = sift_mask_orig
                tamper_type = "copy_move_cloning"
                primary_detector = "Geometric SIFT Cloner"
                confidence = min(1.0, sift_inliers / 15.0)
        else:
            tamper_type = "clean"
            primary_detector = "Consensus Clean"
            confidence = 1.0 - max(dl_score, min(0.5, sift_inliers / 20.0))

        diagnostics = {
            "image_dimensions": [w_orig, h_orig],
            "dl_raw_score": dl_score,
            "dl_max_connected_area_256": max_comp_area,
            "dl_coherent_pixels_256": total_coherent_pixels,
            "sift_inliers_count": sift_inliers,
            "config_used": asdict(self.config),
        }

        return FusionResult(
            is_tampered=is_tampered,
            tamper_type=tamper_type,
            confidence_score=confidence,
            mask=fused_mask,
            dl_flagged=dl_flagged,
            dl_score=dl_score,
            dl_tampered_pixels=int(np.count_nonzero(dl_mask_orig)),
            dl_max_component_area=max_comp_area,
            sift_flagged=sift_flagged,
            sift_inliers=sift_inliers,
            primary_detector=primary_detector,
            diagnostics=diagnostics,
        )

    def _load_image(self, image: Union[np.ndarray, Path, str]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image

        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Görsel dosyası bulunamadı: {p}")

        with open(p, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Görsel açılamadı veya desteklenmeyen format: {p}")
        return img


# Tek satırda çağrılabilir yardımcı fonksiyon
_global_detector: ForensiVisionFusionDetector | None = None


def detect_forgery_hybrid(
    image: Union[np.ndarray, Path, str],
    config: FusionConfig | None = None,
) -> FusionResult:
    """Tek fonksiyonla hibrit sahtecilik analizi yapar (önbellekli dedektör kullanır)."""
    global _global_detector
    if config is not None:
        detector = ForensiVisionFusionDetector(config=config)
        return detector.detect(image)

    if _global_detector is None:
        _global_detector = ForensiVisionFusionDetector()
    return _global_detector.detect(image)
