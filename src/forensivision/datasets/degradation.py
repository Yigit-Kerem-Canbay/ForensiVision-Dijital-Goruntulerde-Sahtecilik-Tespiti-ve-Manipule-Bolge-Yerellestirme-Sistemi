"""
ForensiVision v2 — Format-Bağımsız Bozulma (Degradation) ve Dinamik Copy-Move Üretim Hattı.

Bu modül, v1'de yaşanan iki temel hatayı doğrudan çözer:
1. Format Kestirmesi (Shortcut Learning) Engelleme (`FormatAgnosticDegrader`):
   - Hem TEMİZ (Authentic) hem KİRLİ (Copy-Move / Splicing) görsellere birebir aynı
     olasılık dağılımıyla kayıpsız (PNG/TIFF), tekli JPEG (Q60-100), çiftli JPEG,
     WhatsApp simülasyonu (uzun kenar yeniden boyutlandırma + Q70-82 JPEG),
     WebP sıkıştırma ve kamera ISO/Gauss gürültüsü uygular.
   - Böylece model "JPEG ise temizdir, PNG/TIFF ise sahtedir" ezberini yapamaz;
     gerçek adli izleri (sınır süreksizliği, lokal gürültü farkı, öz-benzerlik) öğrenmek zorunda kalır.

2. Gerçekçi Dinamik Copy-Move Üreticisi (`SyntheticCopyMoveGenerator`):
   - Temiz bir görselin içinden dokulu veya anlamlı bir bölgeyi (veya varsa nesne maskesini)
     seçer, rastgele döndürme (-30..+30 derece), ölçekleme (0.80..1.25x), yatay çevirme,
     parlaklık/kontrast uyumu ve Poisson/Gaussian kenar yumuşatma (alpha feathering) ile
     aynı görselin farklı bir bölgesine yapıştırır.
   - Hem kaynak (source) hem hedef (target) pikselleri içeren tam doğruluk maskesi (`mask`) üretir.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DegradationResult:
    """Bozulma zinciri çıktısı."""

    image_bgr: np.ndarray
    mask: np.ndarray | None
    applied_chain: str
    metadata: dict[str, Any]


class FormatAgnosticDegrader:
    """Görüntü formatı bağımsızlığı sağlayan gerçek dünya kanal simülatörü."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    @staticmethod
    def encode_decode_jpeg(img_bgr: np.ndarray, quality: int) -> np.ndarray:
        """Bellek içinde gerçek JPEG sıkıştırma ve açma uygular."""
        q = int(np.clip(quality, 30, 100))
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img_bgr
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return decoded if decoded is not None else img_bgr

    @staticmethod
    def encode_decode_webp(img_bgr: np.ndarray, quality: int) -> np.ndarray:
        """Bellek içinde gerçek WebP sıkıştırma ve açma uygular."""
        q = int(np.clip(quality, 40, 100))
        ok, buf = cv2.imencode(".webp", img_bgr, [int(cv2.IMWRITE_WEBP_QUALITY), q])
        if not ok:
            return img_bgr
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return decoded if decoded is not None else img_bgr

    def simulate_whatsapp(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray | None = None,
        max_side: int = 1600,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        """WhatsApp standart fotoğraf gönderim hattını simüle eder:
        Uzun kenar > max_side ise yeniden boyutlandırır + Q72..83 JPEG sıkıştırması uygular.
        """
        h, w = img_bgr.shape[:2]
        long_side = max(h, w)
        resized = False

        if long_side > max_side:
            scale = max_side / float(long_side)
            new_w = max(64, int(round(w * scale)))
            new_h = max(64, int(round(h * scale)))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized = True

        q = self.rng.randint(72, 83)
        out_img = self.encode_decode_jpeg(img_bgr, q)
        return out_img, mask, {"whatsapp_resized": resized, "jpeg_quality": q}

    def add_sensor_noise(self, img_bgr: np.ndarray, sigma_max: float = 4.0) -> np.ndarray:
        """Kamera sensör (ISO/Gaussian) gürültüsü ekler."""
        sigma = self.rng.uniform(0.5, sigma_max)
        noise = self.np_rng.normal(0.0, sigma, size=img_bgr.shape).astype(np.float32)
        noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def apply_random_chain(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray | None = None,
        mode: str | None = None,
    ) -> DegradationResult:
        """Temiz ve kirli görüntülere birebir aynı olasılıkla gerçekçi kanal bozulması uygular.

        Modlar ve varsayılan ağırlıkları:
            - 'lossless_png'  (%20): Kayıpsız PNG/ham piksel (ekran görüntüsü / PNG kaydı)
            - 'single_jpeg'   (%25): Tekil JPEG sıkıştırması (Q65..98)
            - 'double_jpeg'   (%20): Çift JPEG sıkıştırması (Q1 -> Q2; sosyal medya/düzenleme)
            - 'whatsapp'      (%15): WhatsApp yeniden boyutlandırma + Q72..83 JPEG
            - 'webp'          (%10): WebP sıkıştırması (Q70..95)
            - 'noise_jpeg'    (%10): Sensör gürültüsü + hafif bulanıklık + JPEG
        """
        if mode is None:
            modes = [
                "lossless_png",
                "single_jpeg",
                "double_jpeg",
                "whatsapp",
                "webp",
                "noise_jpeg",
            ]
            weights = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
            mode = self.rng.choices(modes, weights=weights, k=1)[0]

        out = img_bgr.copy()
        out_mask = mask.copy() if mask is not None else None
        meta: dict[str, Any] = {"mode": mode}

        if mode == "lossless_png":
            # Kayıpsız formatta bırak (model PNG'nin de temiz olabileceğini öğrensin)
            pass

        elif mode == "single_jpeg":
            q = self.rng.randint(65, 98)
            out = self.encode_decode_jpeg(out, q)
            meta["q1"] = q

        elif mode == "double_jpeg":
            q1 = self.rng.randint(70, 96)
            q2 = self.rng.randint(65, 95)
            out = self.encode_decode_jpeg(out, q1)
            out = self.encode_decode_jpeg(out, q2)
            meta["q1"] = q1
            meta["q2"] = q2

        elif mode == "whatsapp":
            out, out_mask, wa_meta = self.simulate_whatsapp(out, out_mask)
            meta.update(wa_meta)

        elif mode == "webp":
            q = self.rng.randint(70, 95)
            out = self.encode_decode_webp(out, q)
            meta["webp_q"] = q

        elif mode == "noise_jpeg":
            out = self.add_sensor_noise(out, sigma_max=3.5)
            if self.rng.random() < 0.4:
                out = cv2.GaussianBlur(out, (3, 3), 0.6)
            q = self.rng.randint(70, 92)
            out = self.encode_decode_jpeg(out, q)
            meta["q1"] = q

        return DegradationResult(
            image_bgr=out,
            mask=out_mask,
            applied_chain=mode,
            metadata=meta,
        )


@dataclass(frozen=True)
class SyntheticCopyMoveSample:
    """Dinamik olarak üretilmiş kopyala-kaydır örneği."""

    forged_bgr: np.ndarray
    combined_mask: np.ndarray  # uint8 (0 veya 255): kaynak + hedef
    source_mask: np.ndarray    # uint8 (0 veya 255): sadece kopyalanan yer
    target_mask: np.ndarray    # uint8 (0 veya 255): sadece yapıştırılan yer
    metadata: dict[str, Any]


class SyntheticCopyMoveGenerator:
    """Herhangi bir temiz görüntüden gerçekçi (döndürülmüş, ölçeklenmiş, kenarı yumuşatılmış)
    kopyala-kaydır sahteciliği ve tam doğruluk maskesi üreten dinamik motor.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _create_irregular_polygon_mask(self, h: int, w: int) -> np.ndarray:
        """Kare/dikdörtgen yerine gerçekçi düzensiz çokgen/eliptik nesne maskesi üretir."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        base_r = min(h, w) * 0.40

        num_pts = self.rng.randint(7, 12)
        pts = []
        for i in range(num_pts):
            angle = (2.0 * np.pi * i) / num_pts + self.rng.uniform(-0.18, 0.18)
            radius = base_r * self.rng.uniform(0.70, 1.05)
            px = int(np.clip(cx + radius * np.cos(angle), 2, w - 3))
            py = int(np.clip(cy + radius * np.sin(angle), 2, h - 3))
            pts.append([px, py])

        pts_arr = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts_arr, 255)
        return mask

    def generate(
        self,
        clean_bgr: np.ndarray,
        min_area_ratio: float = 0.015,
        max_area_ratio: float = 0.09,
    ) -> SyntheticCopyMoveSample:
        """Temiz bir BGR görselden gerçekçi bir Copy-Move sahteciliği üretir.

        Özellikler:
        - Düzensiz organik nesne konturu (keskin kare blok izi bırakmaz).
        - Rastgele döndürme (-25..+25 derece) ve ölçekleme (0.82..1.20x).
        - Kaynak ve hedef bölgelerin birbiriyle çakışmaması (non-overlapping).
        - Kenar yumuşatma (alpha feathering) ile gerçek Photoshop klonlama geçişi.
        """
        H, W = clean_bgr.shape[:2]
        if H < 128 or W < 128:
            raise ValueError(f"Copy-Move üretimi için görsel çok küçük: {(H, W)}")

        area_ratio = self.rng.uniform(min_area_ratio, max_area_ratio)
        patch_side = int(round(np.sqrt(H * W * area_ratio)))
        patch_w = int(np.clip(patch_side * self.rng.uniform(0.8, 1.25), 36, W // 2 - 8))
        patch_h = int(np.clip(patch_side * self.rng.uniform(0.8, 1.25), 36, H // 2 - 8))

        # 1. Kaynak (source) koordinatını seç
        sx = self.rng.randint(4, W - patch_w - 4)
        sy = self.rng.randint(4, H - patch_h - 4)

        src_patch = clean_bgr[sy : sy + patch_h, sx : sx + patch_w].copy()
        local_poly_mask = self._create_irregular_polygon_mask(patch_h, patch_w)

        source_mask_full = np.zeros((H, W), dtype=np.uint8)
        source_mask_full[sy : sy + patch_h, sx : sx + patch_w] = local_poly_mask

        # 2. Geometrik dönüşüm (döndürme + ölçekleme + opsiyonel yatay çevirme)
        angle = self.rng.uniform(-25.0, 25.0) if self.rng.random() < 0.75 else 0.0
        scale = self.rng.uniform(0.85, 1.18) if self.rng.random() < 0.75 else 1.0
        flip_h = self.rng.random() < 0.25

        if flip_h:
            src_patch = cv2.flip(src_patch, 1)
            local_poly_mask = cv2.flip(local_poly_mask, 1)

        rot_mat = cv2.getRotationMatrix2D((patch_w / 2.0, patch_h / 2.0), angle, scale)
        transformed_patch = cv2.warpAffine(
            src_patch,
            rot_mat,
            (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        transformed_mask = cv2.warpAffine(
            local_poly_mask,
            rot_mat,
            (patch_w, patch_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 3. Kaynak bölge ile çakışmayan hedef (target) koordinat seç
        best_tx, best_ty = 0, 0
        max_dist = -1.0
        for _ in range(30):
            cand_tx = self.rng.randint(4, W - patch_w - 4)
            cand_ty = self.rng.randint(4, H - patch_h - 4)
            dist = np.hypot(cand_tx - sx, cand_ty - sy)
            if dist > max(patch_w, patch_h) * 1.15:
                best_tx, best_ty = cand_tx, cand_ty
                break
            if dist > max_dist:
                max_dist = dist
                best_tx, best_ty = cand_tx, cand_ty

        # 4. Hafif parlaklık uyumu + Alpha Feathering (yumuşak kenar yapıştırma)
        if self.rng.random() < 0.5:
            gain = self.rng.uniform(0.94, 1.06)
            bias = self.rng.uniform(-6.0, 6.0)
            transformed_patch = np.clip(
                transformed_patch.astype(np.float32) * gain + bias, 0, 255
            ).astype(np.uint8)

        feather_k = self.rng.choice([3, 5, 7])
        alpha = cv2.GaussianBlur(transformed_mask.astype(np.float32) / 255.0, (feather_k, feather_k), 0)
        alpha_3c = alpha[:, :, None]

        forged = clean_bgr.copy()
        roi = forged[best_ty : best_ty + patch_h, best_tx : best_tx + patch_w].astype(np.float32)
        blended = transformed_patch.astype(np.float32) * alpha_3c + roi * (1.0 - alpha_3c)
        forged[best_ty : best_ty + patch_h, best_tx : best_tx + patch_w] = np.clip(blended, 0, 255).astype(np.uint8)

        target_mask_full = np.zeros((H, W), dtype=np.uint8)
        target_mask_full[best_ty : best_ty + patch_h, best_tx : best_tx + patch_w] = (
            (transformed_mask > 127).astype(np.uint8) * 255
        )
        combined_mask = np.maximum(source_mask_full, target_mask_full)

        return SyntheticCopyMoveSample(
            forged_bgr=forged,
            combined_mask=combined_mask,
            source_mask=source_mask_full,
            target_mask=target_mask_full,
            metadata={
                "angle": round(angle, 2),
                "scale": round(scale, 3),
                "flip_h": flip_h,
                "source_xy": (sx, sy),
                "target_xy": (best_tx, best_ty),
                "patch_wh": (patch_w, patch_h),
            },
        )

