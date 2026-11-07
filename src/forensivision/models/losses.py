"""
ForensiVision v2 — Çok Görevli (Multi-Task) Adli Kayıp Fonksiyonları (Loss Functions).

Bu modül, v1'deki piksel boyama ve yanlış alarm hatalarını önleyen 4 bileşenli kaybı tanımlar:
1. Piksel Segmentasyon Kaybı (BCE + Yumuşak Dice):
   - Manipüle bölge küçük olsa bile (%1-%5 alan) sınıf dengesizliğinden etkilenmez.
   - ÖNEMLİ: Temiz (is_tampered=0) görüntülerde `mask_target` tamamen sıfırdır;
     BCE + yanlış pozitif (FP) cezası temiz görüntülerde tek bir pikselin bile
     boyanmasını sert şekilde cezalandırır.
2. Manipülasyon Sınırı (Edge / Boundary) Kaybı:
   - Ground-truth maskeden morfolojik gradyan (dilate - erode) ile kesim sınırını
     otomatik çıkarır ve `edge_logits` başlığını eğitir.
3. Güvenilirlik (Confidence / Reliability) Kaybı (TruFor yaklaşımı):
   - Hedef güvenilirlik = `1.0 - |sigmoid(mask_logits).detach() - mask_target|`.
   - Modelin doğru tahmin ettiği bölgelerde yüksek güven, yanıldığı düz/karanlık
     bölgelerde düşük güven üretmesini öğretir.
4. Görüntü Seviyesi Sınıflandırma Kaybı (Focal / Label-Smoothed BCE):
   - Görüntünün Temiz (0) mi yoksa Kirli (1) mi olduğunu doğrudan denetler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forensivision.models.forensivision_v2 import ForensiVisionV2Output


def extract_boundary_target(mask_target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """İkili doğruluk maskesinden (B, 1, H, W) morfolojik sınır bandını çıkarır."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask_target, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask_target, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return boundary


class ForensiVisionV2Loss(nn.Module):
    """ForensiVision v2 Çok Görevli Birleşik Adli Kayıp Fonksiyonu."""

    def __init__(
        self,
        w_seg: float = 1.0,
        w_edge: float = 0.5,
        w_conf: float = 0.3,
        w_cls: float = 0.8,
        clean_fp_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_seg = w_seg
        self.w_edge = w_edge
        self.w_conf = w_conf
        self.w_cls = w_cls
        self.clean_fp_penalty = clean_fp_penalty

    @staticmethod
    def _soft_dice_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        eps: float = 1.0,
    ) -> torch.Tensor:
        """Yalnızca kirli (tampered) görüntüler üzerinde yumuşak Dice kaybı hesaplar;
        temiz görüntülerde ise yanlış pozitif (FP) enerjisini cezalandırır.
        """
        probs = torch.sigmoid(logits)
        b = probs.shape[0]
        probs_flat = probs.view(b, -1)
        targets_flat = targets.view(b, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_per_sample = 1.0 - (2.0 * intersection + eps) / (union + eps)

        if valid_mask.sum() > 0:
            return (dice_per_sample * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        return dice_per_sample.mean()

    def forward(
        self,
        outputs: ForensiVisionV2Output,
        mask_target: torch.Tensor,
        cls_target: torch.Tensor,
        has_valid_mask: torch.Tensor | None = None,
        pixel_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Toplam kaybı ve alt bileşen sözlüğünü hesaplar.

        Args:
            outputs: `ForensiVisionV2Net` çıktı nesnesi.
            mask_target: (B, 1, H, W) [0.0, 1.0] aralığında ikili maske.
            cls_target: (B, 1) veya (B,) görüntü seviyesi etiket (0: Temiz, 1: Kirli).
            has_valid_mask: (B,) bool/float tensör. Maskesi olan tüm örnekler için 1.0.
            pixel_valid_mask: (B, 1, H, W) [0.0, 1.0] letterbox dolgu olmayan gerçek görüntü alanı.
        """
        b = mask_target.shape[0]
        cls_target_2d = cls_target.view(b, 1).float()
        if has_valid_mask is None:
            valid_vec = torch.ones(b, device=mask_target.device, dtype=torch.float32)
        else:
            valid_vec = has_valid_mask.view(b).float()

        if pixel_valid_mask is None:
            p_mask = torch.ones_like(mask_target, dtype=torch.float32)
        else:
            p_mask = (pixel_valid_mask >= 0.5).float()
        p_denom = p_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)

        # 1. Piksel BCE + Dice Kaybı (yalnızca geçerli görüntü pikselleri üzerinde)
        bce_px = F.binary_cross_entropy_with_logits(
            outputs.mask_logits, mask_target, reduction="none"
        ) * p_mask
        bce_map = bce_px.sum(dim=(1, 2, 3)) / p_denom
        loss_bce = (bce_map * valid_vec).sum() / valid_vec.sum().clamp(min=1.0)

        tampered_vec = (cls_target_2d.view(b) > 0.5).float() * valid_vec
        loss_dice = self._soft_dice_loss(outputs.mask_logits, mask_target * p_mask, tampered_vec)

        # Temiz (cls_target == 0) görüntülerde odaklanmış (Focal) yanlış piksel boyama (FP) cezası:
        clean_vec = (cls_target_2d.view(b) < 0.5).float() * valid_vec
        probs = torch.sigmoid(outputs.mask_logits) * p_mask
        probs_flat = probs.view(b, -1)
        k_hard = max(1, probs_flat.shape[1] // 50)
        hard_fp_peaks = torch.topk(probs_flat, k=k_hard, dim=1)[0]
        focal_fp_per_sample = (probs.sum(dim=(1, 2, 3)) / p_denom) + 1.5 * ((hard_fp_peaks ** 2).mean(dim=1))
        if clean_vec.sum() > 0:
            loss_clean_fp = (focal_fp_per_sample * clean_vec).sum() / clean_vec.sum()
        else:
            loss_clean_fp = torch.tensor(0.0, device=mask_target.device)

        loss_seg = loss_bce + loss_dice + self.clean_fp_penalty * loss_clean_fp

        # 2. Manipülasyon Sınırı (Edge) Kaybı (yalnızca geçerli görüntü pikselleri üzerinde)
        edge_target = extract_boundary_target(mask_target) * p_mask
        edge_bce_px = F.binary_cross_entropy_with_logits(
            outputs.edge_logits, edge_target, pos_weight=torch.tensor(4.0, device=mask_target.device), reduction="none"
        ) * p_mask
        edge_bce = edge_bce_px.sum(dim=(1, 2, 3)) / p_denom
        loss_edge = (edge_bce * valid_vec).sum() / valid_vec.sum().clamp(min=1.0)

        # 3. Güvenilirlik (Confidence) Kaybı (yalnızca geçerli görüntü pikselleri üzerinde)
        with torch.no_grad():
            conf_target = (1.0 - (probs.detach() - mask_target).abs()).clamp(0.0, 1.0) * p_mask
        conf_bce_px = F.binary_cross_entropy_with_logits(
            outputs.confidence_logits, conf_target, reduction="none"
        ) * p_mask
        conf_bce = conf_bce_px.sum(dim=(1, 2, 3)) / p_denom
        loss_conf = (conf_bce * valid_vec).sum() / valid_vec.sum().clamp(min=1.0)

        # 4. Görüntü Seviyesi Sınıflandırma Kaybı (Temiz vs Kirli)
        loss_cls = F.binary_cross_entropy_with_logits(outputs.cls_logits, cls_target_2d)

        total_loss = (
            self.w_seg * loss_seg
            + self.w_edge * loss_edge
            + self.w_conf * loss_conf
            + self.w_cls * loss_cls
        )

        metrics = {
            "loss_total": float(total_loss.detach().cpu()),
            "loss_seg": float(loss_seg.detach().cpu()),
            "loss_edge": float(loss_edge.detach().cpu()),
            "loss_conf": float(loss_conf.detach().cpu()),
            "loss_cls": float(loss_cls.detach().cpu()),
            "loss_clean_fp": float(loss_clean_fp.detach().cpu()),
        }
        return total_loss, metrics

