"""
ForensiVision v2 — Üç Akışlı (RGB SwinV2 Transformer + SRM/Bayar Gürültü + 8x8 DCT Frekans) +
Öteleme Vektörlü (Displacement-Field) Öz-Korelasyon ve Güvenilirlik Haritalı Adli Ağ Mimarisi.

Mimari Tasarım (2026 SOTA Sentezi: TruFor + CAT-Net + MVSS-Net + Dense Displacement CMFD):
1. Akış 1 (Anlamsal ve Bağlamsal Akış - SwinV2-Tiny Hiyerarşik Transformer):
   - 4 ölçekte özellik üretir: 1/4 (128x128, 96c), 1/8 (64x64, 192c),
     1/16 (32x32, 384c), 1/32 (16x16, 768c).
   - Ön-eğitimli ağırlıklarla (IMAGENET1K_V1) başlar; sıfırdan eğitim tuzağına düşmez.

2. Akış 2 (Adli Gürültü + 8x8 DCT Frekans Akışı - SRM + Softmax-Bayar + Spektral DCT):
   - Görüntü içeriğini (manzara, kedi, gökyüzü) matematiksel olarak bastırır;
     piksel komşuluk farklarını (SRM + Bayar) ve 8x8 JPEG DCT frekans enerjisini çıkarır.
   - 1/4 ve 1/8 ölçeklerinde gürültü ve sıkıştırma özellik haritaları üretir.

3. Öteleme Vektörlü Öz-Korelasyon Modülü (`SelfCorrelation2D` + Displacement Field):
   - Copy-Move (Kopyala-Kaydır) sahteciliğinde kaynak ve hedef bölge aynı kameradan
     geldiği için gürültüleri aynıdır; standart CNN bunu göremez.
   - Bu modül, 1/16 (32x32) ve 1/32 (16x16) özellik haritalarında her yamanın
     görüntüdeki TÜM diğer yamalarla L2-normalize kosinüs benzerliğini hesaplar.
   - Sadece benzerlik skorunu değil, en iyi eşleşen ikiz yamanın **normalize öteleme vektörünü
     (dx, dy, mesafe)** de kanal olarak çıkarır! Böylece model, yan yana duran rastgele tuğla/pencere
     dokusu ile tutarlı bir (dx, dy) vektörüyle taşınmış gerçek klon nesneyi birbirinden ayırır.

4. Çok Görevli Çıktı Başlığı (Multi-Task Decoder + TruFor Reliability Map):
   - `mask_logits` (B, 1, H, W): Piksel bazlı sahtecilik maskesi.
   - `confidence_logits` (B, 1, H, W): Her piksel için modelin ne kadar emin olduğunu gösteren
     Güvenilirlik (Reliability) haritası (düz gökyüzü/karanlık bölgelerde yanlış alarmı bastırır).
   - `edge_logits` (B, 1, H, W): Manipüle bölgenin kesim/yapıştırma sınırı (Boundary/Edge).
   - `cls_logits`  (B, 1): Görüntü bazlı Temiz (0) vs Kirli (1) kararı.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_V2_T_Weights, swin_v2_t


class SRMFilterBank(nn.Module):
    """Fridrich Spatial Rich Models (SRM) yüksek geçiren adli filtre bankası +
    Öğrenilebilir Softmax-Bayar kısıtlı evrişim katmanı (merkez ağırlık = -1, çevre toplamı = +1).
    """

    def __init__(self, learnable_channels: int = 9) -> None:
        super().__init__()
        srm_kernels = self._build_srm_kernels()
        self.register_buffer("srm_weight", srm_kernels)

        self.bayar_conv = nn.Conv2d(3, learnable_channels, kernel_size=5, padding=2, bias=False)
        nn.init.kaiming_normal_(self.bayar_conv.weight)

    @staticmethod
    def _build_srm_kernels() -> torch.Tensor:
        q1 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 4.0

        q2 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        ) / 12.0

        q3 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 2.0

        base = torch.stack([q1, q2, q3], dim=0)
        kernels = torch.zeros((9, 3, 5, 5), dtype=torch.float32)
        for c in range(3):
            kernels[c * 3 : (c + 1) * 3, c, :, :] = base
        return kernels

    def _get_constrained_bayar_weights(self) -> torch.Tensor:
        """Bayar & Stamm (2018) kısıtını türevlenebilir (autograd-safe) biçimde uygular:
        Her (out_c, in_c) 5x5 çekirdeğinde merkez piksel = -1.0, çevre 24 pikselin toplamı = +1.0.
        """
        w = self.bayar_conv.weight
        b_out, b_in, kh, kw = w.shape
        cy, cx = kh // 2, kw // 2

        mask = torch.ones_like(w)
        mask[:, :, cy, cx] = 0.0

        w_masked = w.masked_fill(mask == 0.0, float("-inf"))
        w_flat = F.softmax(w_masked.view(b_out, b_in, -1), dim=-1).view(b_out, b_in, kh, kw)
        w_norm = w_flat * mask

        center_patch = torch.zeros_like(w)
        center_patch[:, :, cy, cx] = -1.0
        return (w_norm + center_patch) / float(b_in)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        srm_out = torch.clamp(F.conv2d(x_rgb, self.srm_weight, padding=2), -2.0, 2.0)
        bayar_w = self._get_constrained_bayar_weights()
        bayar_out = F.conv2d(x_rgb, bayar_w, padding=2)
        return torch.cat([srm_out, bayar_out], dim=1)


class DCTSpectralExtractor(nn.Module):
    """CAT-Net / F3-Net esinli 8x8 2B Ayrık Kosinüs Dönüşümü (DCT) Frekans Enerjisi Çıkarıcı.

    JPEG ve WhatsApp sıkıştırmaları 8x8 bloklarda yapıldığı için montajlanan (spliced) veya
    klonlanan bölgelerde 8x8 blok ızgarası uyumsuzluğu ve yüksek/orta frekans enerji farkı oluşur.
    Sabit ortonormal 8x8 DCT bazlarıyla TAM 8x8 BLOK HİZALI (`kernel_size=8, stride=8, padding=0`)
    1/8 çözünürlükte spektral enerji haritası üretir.
    """

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        dct_bases = self._build_8x8_dct_bases()  # (15, 1, 8, 8) - DC hariç 15 AC frekans bandı
        self.register_buffer("dct_weight", dct_bases)
        self.proj = nn.Sequential(
            nn.Conv2d(15, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _build_8x8_dct_bases() -> torch.Tensor:
        bs = 8
        bases = []
        y, x = torch.meshgrid(torch.arange(bs, dtype=torch.float32), torch.arange(bs, dtype=torch.float32), indexing="ij")
        # Zigzag benzeri orta ve yüksek AC frekans çiftleri (DC (0,0) hariç)
        freq_pairs = [
            (0, 1), (1, 0), (1, 1), (0, 2), (2, 0),
            (1, 2), (2, 1), (2, 2), (0, 3), (3, 0),
            (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
        ]
        for u, v in freq_pairs:
            cu = torch.cos(math.pi * (2.0 * x + 1.0) * u / (2.0 * bs))
            cv = torch.cos(math.pi * (2.0 * y + 1.0) * v / (2.0 * bs))
            b_uv = cu * cv
            b_uv = b_uv - b_uv.mean()
            norm = torch.linalg.norm(b_uv).clamp(min=1e-6)
            bases.append(b_uv / norm)
        return torch.stack(bases, dim=0).unsqueeze(1)  # (15, 1, 8, 8)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """Girdi: [0, 1] RGB (B, 3, H, W). Çıktı: (B, out_channels, H/8, W/8) 8x8 blok hizalı spektral özellik."""
        luma = 0.299 * x_rgb[:, 0:1] + 0.587 * x_rgb[:, 1:2] + 0.114 * x_rgb[:, 2:3]
        # Tam 8x8 JPEG makro-blok sınırlarına hizalı (stride=8, padding=0)
        dct_resp = F.conv2d(luma, self.dct_weight, stride=8, padding=0)
        dct_mag = torch.log1p(dct_resp.abs())
        return self.proj(dct_mag)


class NoiseResidualStem(nn.Module):
    """SRM + Bayar uzamsal gürültü kalıntısı ile tam 8x8 hizalı DCT frekans enerjisini birleştiren adli kol."""

    def __init__(self, out_c4: int = 64, out_c8: int = 128) -> None:
        super().__init__()
        self.srm_bayar = SRMFilterBank(learnable_channels=9)  # 18 kanal
        self.dct_branch = DCTSpectralExtractor(out_channels=32)
        self.stage1 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c4),
            nn.GELU(),
        )
        self.stage2_srm = nn.Sequential(
            nn.Conv2d(out_c4, out_c8 - 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c8 - 32),
            nn.GELU(),
        )
        self.fuse_c8 = nn.Sequential(
            nn.Conv2d(out_c8, out_c8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c8),
            nn.GELU(),
        )

    def forward(self, x_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self.srm_bayar(x_rgb)
        n4 = self.stage1(noise)              # (B, out_c4, H/4, W/4)
        s8 = self.stage2_srm(n4)             # (B, out_c8 - 32, H/8, W/8)
        f8 = self.dct_branch(x_rgb)          # (B, 32, H/8, W/8) - Tam 8x8 JPEG blok hizalı!
        if f8.shape[2:] != s8.shape[2:]:
            f8 = F.interpolate(f8, size=s8.shape[2:], mode="bilinear", align_corners=False)
        n8 = self.fuse_c8(torch.cat([s8, f8], dim=1))  # (B, out_c8, H/8, W/8)
        return n4, n8


class SelfCorrelation2D(nn.Module):
    """Copy-Move (Kopyala-Kaydır) tespiti için Anlamsal + Adli Gürültü Birleşik Öteleme Vektörlü 2B Öz-Korelasyon.

    Özellik haritasındaki her (y, x) konumunun, kendi yakın komşuluğu (`exclusion_radius`)
    dışındaki tüm diğer konumlarla olan kosinüs benzerliğini VE en güçlü ikizinin
    **uzamsal öteleme vektörünü (dx, dy, mesafe, Lowe 1./2. komşu farkı)** hesaplar.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        proj_channels: int = 128,
        top_k: int = 8,
        exclusion_radius: int = 3,
    ) -> None:
        super().__init__()
        out_c = out_channels if out_channels is not None else (in_channels // 2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.GELU(),
        )
        self.top_k = top_k
        self.exclusion_radius = exclusion_radius
        # top_k + 7 kanal: [topk_vals (K), max, mean, std, lowe_gap, norm_dx, norm_dy, norm_dist]
        self.out_conv = nn.Sequential(
            nn.Conv2d(top_k + 7, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
        self._cached_grid: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_spatial_buffers(
        self, h: int, w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h, w, str(device))
        if key in self._cached_grid:
            return self._cached_grid[key]

        ys = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1).repeat(1, w).reshape(-1)
        xs = torch.arange(w, device=device, dtype=torch.float32).view(1, -1).repeat(h, 1).reshape(-1)

        dy = (ys.unsqueeze(1) - ys.unsqueeze(0)).abs()
        dx = (xs.unsqueeze(1) - xs.unsqueeze(0)).abs()
        invalid = (dy <= self.exclusion_radius) & (dx <= self.exclusion_radius)

        norm_ys = ys / max(float(h - 1), 1.0)
        norm_xs = xs / max(float(w - 1), 1.0)
        self._cached_grid[key] = (invalid, norm_xs, norm_ys)
        return invalid, norm_xs, norm_ys

    def forward(
        self, feat: torch.Tensor, valid_mask_2d: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = feat.shape
        n = h * w
        z = self.proj(feat)
        z = F.normalize(z, p=2, dim=1)

        flat = z.view(b, z.shape[1], n)
        sim = torch.bmm(flat.transpose(1, 2), flat)  # (B, N, N)

        invalid_mask, norm_xs, norm_ys = self._get_spatial_buffers(h, w, feat.device)
        sim = sim.masked_fill(invalid_mask.unsqueeze(0), -1.0)

        if valid_mask_2d is not None:
            # Sabit gri letterbox dolgu (padding) bölgelerinin birbirleriyle veya görüntüyle eşleşmesini engelle
            tok_valid = (valid_mask_2d.view(b, n) >= 0.5)  # (B, N)
            pair_valid = tok_valid.unsqueeze(2) & tok_valid.unsqueeze(1)  # (B, N, N)
            sim = sim.masked_fill(~pair_valid, -1.0)

        k = min(self.top_k, max(2, n - 1))
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=-1)  # (B, N, K)
        topk_vals = topk_vals.clamp(min=0.0)

        max_val = topk_vals[:, :, 0:1]
        second_val = topk_vals[:, :, -1:]  # K'ıncı komşuya göre belirginlik farkı (Lowe-benzeri)
        lowe_gap = (max_val - second_val).clamp(min=0.0)
        mean_val = topk_vals.mean(dim=-1, keepdim=True)
        std_val = topk_vals.std(dim=-1, keepdim=True, unbiased=False)

        # En güçlü eşleşmenin (top-1) uzamsal öteleme vektörü (dx, dy, dist)
        best_idx = topk_idx[:, :, 0]  # (B, N)
        matched_x = norm_xs[best_idx]
        matched_y = norm_ys[best_idx]
        dx_vec = (matched_x - norm_xs.unsqueeze(0)).abs().unsqueeze(-1)
        dy_vec = (matched_y - norm_ys.unsqueeze(0)).abs().unsqueeze(-1)
        dist_vec = torch.hypot(dx_vec, dy_vec)

        stats = torch.cat(
            [topk_vals, max_val, mean_val, std_val, lowe_gap, dx_vec, dy_vec, dist_vec],
            dim=-1,
        )  # (B, N, K+7)
        stats_2d = stats.transpose(1, 2).reshape(b, k + 7, h, w)
        if valid_mask_2d is not None:
            stats_2d = stats_2d * valid_mask_2d

        corr_feat = self.out_conv(stats_2d)
        # Lowe belirginliği ile güçlendirilmiş öz-korelasyon haritası
        effective_sim = torch.clamp(max_val * (0.7 + 0.6 * lowe_gap), 0.0, 1.0)
        max_sim_map = effective_sim.transpose(1, 2).reshape(b, 1, h, w)
        if valid_mask_2d is not None:
            corr_feat = corr_feat * valid_mask_2d
            max_sim_map = max_sim_map * valid_mask_2d
        return corr_feat, max_sim_map


class ForensiVisionV2Output(NamedTuple):
    """Modelin tek ileri geçişte (forward pass) ürettiği tüm çıktılar (NamedTuple: 2x GPU DataParallel ile %100 uyumlu)."""

    mask_logits: torch.Tensor       # (B, 1, H, W)
    confidence_logits: torch.Tensor # (B, 1, H, W) - TruFor tarzı güvenilirlik haritası
    edge_logits: torch.Tensor       # (B, 1, H, W) - Manipülasyon sınır haritası
    cls_logits: torch.Tensor        # (B, 1) - Görüntü seviyesi Temiz(0) / Kirli(1)
    self_corr_map: torch.Tensor     # (B, 1, H, W) - Copy-Move öz-benzerlik ısı haritası


class ForensiVisionV2Net(nn.Module):
    """ForensiVision v2 Üç Akışlı (RGB SwinV2 + SRM/Bayar + 8x8 DCT Frekans) +
    Gürültü Destekli Öteleme Vektörlü Öz-Korelasyon Adli Görüntü Sahtecilik Ağı.
    """

    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()
        if pretrained_backbone:
            try:
                swin = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
            except Exception as exc:
                raise RuntimeError(
                    "❌ HATA: SwinV2-Tiny ImageNet ön-eğitimli ağırlıkları indirilemedi! "
                    "Rastgele ağırlıklarla saatlerce GPU kotası harcamamak için eğitim durduruldu. "
                    "Lütfen Kaggle sağ panelden (Settings -> Internet -> ON) interneti açıp tekrar başlatın."
                ) from exc
        else:
            swin = swin_v2_t(weights=None)
        self.rgb_features = swin.features

        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

        # Akış 2: SRM + Bayar Uzamsal Gürültü + 8x8 DCT Spektral Frekans Kolu
        self.noise_stem = NoiseResidualStem(out_c4=64, out_c8=128)

        # Akış 3: Hem Anlamsal (SwinV2) hem Adli Gürültü/DCT (n8->n16/n32) birleşik Öz-Korelasyon (Copy-Move)
        self.corr_s16 = SelfCorrelation2D(
            in_channels=384 + 128, out_channels=192, proj_channels=128, top_k=8, exclusion_radius=2
        )
        self.corr_s32 = SelfCorrelation2D(
            in_channels=768 + 128, out_channels=384, proj_channels=128, top_k=6, exclusion_radius=1
        )

        embed_dim = 192
        self.lat4 = nn.Sequential(
            nn.Conv2d(96 + 64, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.lat8 = nn.Sequential(
            nn.Conv2d(192 + 128, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.lat16 = nn.Sequential(
            nn.Conv2d(384 + 192, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.lat32 = nn.Sequential(
            nn.Conv2d(768 + 384, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.fusion_decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 4 + 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.mask_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.conf_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim + 6, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def _extract_swin_pyramid(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x_norm
        h = self.rgb_features[0](h)
        h = self.rgb_features[1](h)
        f4 = h.permute(0, 3, 1, 2).contiguous()

        h = self.rgb_features[2](h)
        h = self.rgb_features[3](h)
        f8 = h.permute(0, 3, 1, 2).contiguous()

        h = self.rgb_features[4](h)
        h = self.rgb_features[5](h)
        f16 = h.permute(0, 3, 1, 2).contiguous()

        h = self.rgb_features[6](h)
        h = self.rgb_features[7](h)
        f32 = h.permute(0, 3, 1, 2).contiguous()
        return f4, f8, f16, f32

    def forward(
        self, x_rgb: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> ForensiVisionV2Output:
        orig_h, orig_w = x_rgb.shape[2], x_rgb.shape[3]

        # 1. Uzamsal Gürültü (SRM+Bayar) + 8x8 DCT Spektral Frekans akışı
        n4, n8 = self.noise_stem(x_rgb)

        # 2. RGB SwinV2 Transformer akışı
        x_norm = (x_rgb - self.rgb_mean) / self.rgb_std
        f4, f8, f16, f32 = self._extract_swin_pyramid(x_norm)

        # 3. Gürültü Destekli Öteleme Vektörlü Öz-Korelasyon (Copy-Move)
        # Hem anlamsal SwinV2 (f16, f32) hem de SRM/Bayar/DCT adli gürültü (n16, n32) özelliklerini
        # birlikte eşleştirerek doğal tekrarlı dokularda (pencere, tuğla) yanlış alarmı önler.
        n16 = F.adaptive_avg_pool2d(n8, output_size=(f16.shape[2], f16.shape[3]))
        n32 = F.adaptive_avg_pool2d(n8, output_size=(f32.shape[2], f32.shape[3]))

        v4: torch.Tensor | None = None
        v16: torch.Tensor | None = None
        v32: torch.Tensor | None = None
        if valid_mask is not None:
            v4 = (F.adaptive_avg_pool2d(valid_mask, output_size=(f4.shape[2], f4.shape[3])) >= 0.90).float()
            v16 = (F.adaptive_avg_pool2d(valid_mask, output_size=(f16.shape[2], f16.shape[3])) >= 0.85).float()
            v32 = (F.adaptive_avg_pool2d(valid_mask, output_size=(f32.shape[2], f32.shape[3])) >= 0.85).float()

        c16, sim16 = self.corr_s16(torch.cat([f16, n16], dim=1), valid_mask_2d=v16)
        c32, sim32 = self.corr_s32(torch.cat([f32, n32], dim=1), valid_mask_2d=v32)

        # 4. Çok ölçekli birleştirme
        target_h4, target_w4 = f4.shape[2], f4.shape[3]
        p4 = self.lat4(torch.cat([f4, n4], dim=1))
        p8 = F.interpolate(
            self.lat8(torch.cat([f8, n8], dim=1)),
            size=(target_h4, target_w4),
            mode="bilinear",
            align_corners=False,
        )
        p16 = F.interpolate(
            self.lat16(torch.cat([f16, c16], dim=1)),
            size=(target_h4, target_w4),
            mode="bilinear",
            align_corners=False,
        )
        p32 = F.interpolate(
            self.lat32(torch.cat([f32, c32], dim=1)),
            size=(target_h4, target_w4),
            mode="bilinear",
            align_corners=False,
        )

        sim16_up = F.interpolate(sim16, size=(target_h4, target_w4), mode="bilinear", align_corners=False)
        sim32_up = F.interpolate(sim32, size=(target_h4, target_w4), mode="bilinear", align_corners=False)

        fused = self.fusion_decoder(torch.cat([p4, p8, p16, p32, sim16_up, sim32_up], dim=1))
        if v4 is not None:
            fused = fused * v4

        mask_logits_4 = self.mask_head(fused)
        conf_logits_4 = self.conf_head(fused)
        edge_logits_4 = self.edge_head(fused)
        corr_map_4 = torch.maximum(sim16_up, sim32_up)

        # 5. Güvenilirlik Ağırlıklı Görüntü Seviyesi Sınıflandırma
        prob_4 = torch.sigmoid(mask_logits_4)
        conf_4 = torch.sigmoid(conf_logits_4)
        edge_4 = torch.sigmoid(edge_logits_4)
        if v4 is not None:
            prob_4 = prob_4 * v4
            conf_4 = conf_4 * v4
            edge_4 = edge_4 * v4
            corr_map_4 = corr_map_4 * v4

        weighted_anomaly = prob_4 * conf_4
        flat_anom = weighted_anomaly.view(x_rgb.shape[0], -1)
        k_top = max(1, flat_anom.shape[1] // 25)
        topk_anom = torch.topk(flat_anom, k=k_top, dim=1)[0].mean(dim=1, keepdim=True)

        valid_area_4 = v4.sum(dim=(2, 3)).clamp(min=1.0) if v4 is not None else float(target_h4 * target_w4)
        anom_mean = weighted_anomaly.sum(dim=(2, 3)) / valid_area_4

        anom_stats = torch.cat(
            [
                anom_mean,
                weighted_anomaly.amax(dim=(2, 3)),
                topk_anom,
                edge_4.amax(dim=(2, 3)),
                sim16.amax(dim=(2, 3)),
                sim32.amax(dim=(2, 3)),
            ],
            dim=1,
        )

        global_feat = (fused * conf_4).sum(dim=(2, 3)) / (conf_4.sum(dim=(2, 3)) + 1e-6)
        cls_logits = self.cls_head(torch.cat([global_feat, anom_stats], dim=1))

        mask_logits = F.interpolate(mask_logits_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        conf_logits = F.interpolate(conf_logits_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_logits_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        self_corr_map = F.interpolate(corr_map_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        if valid_mask is not None:
            mask_logits = mask_logits.masked_fill(valid_mask < 0.5, -10.0)
            self_corr_map = self_corr_map * valid_mask

        return ForensiVisionV2Output(
            mask_logits=mask_logits,
            confidence_logits=conf_logits,
            edge_logits=edge_logits,
            cls_logits=cls_logits,
            self_corr_map=self_corr_map,
        )
