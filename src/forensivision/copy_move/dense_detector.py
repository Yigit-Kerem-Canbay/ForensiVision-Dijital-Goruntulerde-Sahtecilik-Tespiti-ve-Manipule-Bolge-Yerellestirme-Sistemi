"""
ForensiVision v2 — Tam Çözünürlüklü (Native-Resolution) Vektörize Yoğun Alan Copy-Move Dedektörü.

Neden `cv2.resize` Yapılmamalı?
- Görüntüyü 384x384'e küçültmek, kopyalanan bölgenin ızgarasını kaydırır (interpolation phase shift)
  ve 40x40 piksellik küçük klonları yok eder.
- Orijinal çözünürlükte (veya sadece >1024px ise tam sayı katıyla) çalışıldığında,
  kopyalanan bölge ile yapıştırıldığı yer arasındaki pikseller neredeyse birebir örtüşür!

Neden Doğal Tekrarlayan Dokular (Tuğla, Pencere, Çimen) Yanlış Alarm Verir ve Nasıl Önlenir?
- Doğal bir fotoğrafta benzer iki taş/pencere arasında piksel-piksel ortalama mutlak fark (RGB MAD)
  genellikle 4.0 - 15.0 seviyesindedir ve kenarlarında keskin bir "fark sıçraması" yoktur.
- Gerçek bir Copy-Move işleminde ise (JPEG sıkıştırması sonrası bile) kopyalanan iç bölgenin
  kaynak bölgeyle olan RGB MAD değeri çok düşüktür (< 2.8) ve Lowe 2. komşu oranı çok keskindi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DenseCopyMoveResult:
    """Yoğun alan (Dense-Field) Copy-Move analiz sonucu."""

    mask: np.ndarray               # (H, W) uint8 (0 veya 255): kaynak + hedef
    score: float                   # [0.0, 1.0] normalize yoğun eşleşme güven skoru
    dominant_cluster_blocks: int   # Aynı öteleme vektörünü paylaşan tutarlı blok sayısı
    mean_similarity: float         # Tutarlı kümedeki ortalama benzerlik
    shift_vector_xy: tuple[int, int]
    diagnostics: dict[str, Any]


class DenseFieldCopyMoveDetector:
    """Tam Çözünürlüklü DCT/Radyal + Lowe Oran Testi + Doğrudan RGB Doğrulamalı Yoğun Copy-Move Dedektörü."""

    def __init__(
        self,
        max_work_side: int = 640,
        block_size: int = 8,
        stride: int = 4,
        min_block_std: float = 2.0,
        min_spatial_dist_grid: int = 5,
        ratio_threshold: float = 0.72,
        max_rgb_mad: float = 3.2,
        min_consistent_blocks: int = 9,
    ) -> None:
        self.max_work_side = max_work_side
        self.block_size = block_size
        self.stride = stride
        self.min_block_std = min_block_std
        self.min_spatial_dist_grid = min_spatial_dist_grid
        self.ratio_threshold = ratio_threshold
        self.max_rgb_mad = max_rgb_mad
        self.min_consistent_blocks = min_consistent_blocks
        self._basis = self._build_orthogonal_basis(block_size)

    @staticmethod
    def _build_orthogonal_basis(bs: int) -> np.ndarray:
        basis_list = []
        y, x = np.mgrid[0:bs, 0:bs].astype(np.float32)
        for u in range(4):
            for v in range(4):
                if u == 0 and v == 0:
                    continue
                cu = np.cos(np.pi * (2 * x + 1) * u / (2.0 * bs))
                cv = np.cos(np.pi * (2 * y + 1) * v / (2.0 * bs))
                b_uv = cu * cv
                norm = np.linalg.norm(b_uv)
                if norm > 1e-6:
                    basis_list.append(b_uv / norm)
        return np.stack(basis_list, axis=0).astype(np.float32)

    def detect(self, image_bgr: np.ndarray) -> DenseCopyMoveResult:
        """Görüntü üzerinde tam çözünürlüklü yoğun alan kopyala-kaydır analizi çalıştırır."""
        orig_h, orig_w = image_bgr.shape[:2]
        long_side = max(orig_h, orig_w)

        # Yalnızca çok büyük fotoğrafları küçült; 512x512 veya 640x480 görsellere ASLA dokunma!
        if long_side > self.max_work_side:
            scale = self.max_work_side / float(long_side)
            work_w = max(64, int(round(orig_w * scale)))
            work_h = max(64, int(round(orig_h * scale)))
            img_work = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            work_h, work_w = orig_h, orig_w
            img_work = image_bgr

        bs = self.block_size
        st = self.stride
        grid_h = (work_h - bs) // st + 1
        grid_w = (work_w - bs) // st + 1
        n_blocks = grid_h * grid_w

        from numpy.lib.stride_tricks import sliding_window_view

        bgr_f = img_work.astype(np.float32)
        y_chan = 0.114 * bgr_f[:, :, 0] + 0.587 * bgr_f[:, :, 1] + 0.299 * bgr_f[:, :, 2]

        y_blocks = sliding_window_view(y_chan, (bs, bs))[::st, ::st].reshape(n_blocks, bs * bs)
        b_blocks = sliding_window_view(bgr_f[:, :, 0], (bs, bs))[::st, ::st].reshape(n_blocks, bs * bs)
        g_blocks = sliding_window_view(bgr_f[:, :, 1], (bs, bs))[::st, ::st].reshape(n_blocks, bs * bs)
        r_blocks = sliding_window_view(bgr_f[:, :, 2], (bs, bs))[::st, ::st].reshape(n_blocks, bs * bs)

        y_std = y_blocks.std(axis=1)
        valid_mask = y_std >= self.min_block_std
        valid_indices = np.nonzero(valid_mask)[0]

        if len(valid_indices) < self.min_consistent_blocks * 2:
            return DenseCopyMoveResult(
                mask=np.zeros((orig_h, orig_w), dtype=np.uint8),
                score=0.0,
                dominant_cluster_blocks=0,
                mean_similarity=0.0,
                shift_vector_xy=(0, 0),
                diagnostics={"valid_blocks": int(len(valid_indices)), "reason": "flat_image"},
            )

        # Eğer >5500 blok varsa homojen/orta-yüksek varyanslıları dengeli örnekle
        if len(valid_indices) > 5500:
            step = max(1, len(valid_indices) // 5500)
            valid_indices = valid_indices[::step][:5500]

        yv = y_blocks[valid_indices]
        bv = b_blocks[valid_indices]
        gv = g_blocks[valid_indices]
        rv = r_blocks[valid_indices]

        yv_zero = yv - yv.mean(axis=1, keepdims=True)
        basis_flat = self._basis.reshape(self._basis.shape[0], -1)
        struct_feat = (yv_zero @ basis_flat.T) / float(bs)

        color_feat = np.stack(
            [
                bv.mean(axis=1) * 0.5,
                gv.mean(axis=1) * 0.5,
                rv.mean(axis=1) * 0.5,
                bv.std(axis=1) * 0.4,
                gv.std(axis=1) * 0.4,
                rv.std(axis=1) * 0.4,
            ],
            axis=1,
        )
        feats = np.concatenate([struct_feat, color_feat], axis=1).astype(np.float32)

        gy = valid_indices // grid_w
        gx = valid_indices % grid_w

        sq_norms = np.sum(feats ** 2, axis=1, keepdims=True)
        dist_sq = np.maximum(sq_norms + sq_norms.T - 2.0 * (feats @ feats.T), 0.0)
        dist_mat = np.sqrt(dist_sq)

        dy_mat = np.abs(gy[:, None] - gy[None, :])
        dx_mat = np.abs(gx[:, None] - gx[None, :])
        spatial_close = (dy_mat < self.min_spatial_dist_grid) & (dx_mat < self.min_spatial_dist_grid)
        dist_mat[spatial_close] = 1e9

        m = len(valid_indices)
        arange_m = np.arange(m)
        best1_j = np.argmin(dist_mat, axis=1)
        d1 = dist_mat[arange_m, best1_j]

        # 2. komşu için 1. komşunun 4x4 çevresini maskele
        b1_gy = gy[best1_j]
        b1_gx = gx[best1_j]
        near_best1 = (np.abs(gy[None, :] - b1_gy[:, None]) <= 4) & (np.abs(gx[None, :] - b1_gx[:, None]) <= 4)
        dist_mat_2 = dist_mat.copy()
        dist_mat_2[near_best1] = 1e9
        d2 = np.min(dist_mat_2, axis=1)

        ratios = d1 / np.maximum(d2, 1e-5)

        # Doğrudan 3-kanallı RGB Piksel Farkı (MAD) kontrolü:
        # Doğal benzer dokularda (iki farklı tuğla/pencere) RGB farkı > 4.0 iken,
        # gerçek kopyala-kaydırda RGB farkı < 3.2'dir!
        rgb_mad = (
            np.mean(np.abs(bv - bv[best1_j]), axis=1)
            + np.mean(np.abs(gv - gv[best1_j]), axis=1)
            + np.mean(np.abs(rv - rv[best1_j]), axis=1)
        ) / 3.0

        cand_mask = (ratios <= self.ratio_threshold) & (rgb_mad <= self.max_rgb_mad)
        cand_i = np.nonzero(cand_mask)[0]

        if len(cand_i) < self.min_consistent_blocks:
            return DenseCopyMoveResult(
                mask=np.zeros((orig_h, orig_w), dtype=np.uint8),
                score=0.0,
                dominant_cluster_blocks=0,
                mean_similarity=0.0,
                shift_vector_xy=(0, 0),
                diagnostics={"valid_blocks": m, "candidates_after_rgb_gate": int(len(cand_i))},
            )

        cand_j = best1_j[cand_i]
        cand_mad = rgb_mad[cand_i]

        raw_dy = gy[cand_j] - gy[cand_i]
        raw_dx = gx[cand_j] - gx[cand_i]
        flip_sign = (raw_dx < 0) | ((raw_dx == 0) & (raw_dy < 0))
        std_dx = np.where(flip_sign, -raw_dx, raw_dx)
        std_dy = np.where(flip_sign, -raw_dy, raw_dy)

        bins_2d = np.stack([std_dx, std_dy], axis=1)
        unique_bins, counts = np.unique(bins_2d, axis=0, return_counts=True)

        best_bin_idx = int(np.argmax(counts))
        dom_bin = unique_bins[best_bin_idx]

        # Öteleme vektörü tam (veya +-1 grid komşusu) olan blokları al
        inlier_sel = (np.abs(std_dx - dom_bin[0]) <= 1) & (np.abs(std_dy - dom_bin[1]) <= 1)
        inlier_i = cand_i[inlier_sel]
        inlier_j = cand_j[inlier_sel]
        inlier_count = int(len(inlier_i))

        if inlier_count < self.min_consistent_blocks:
            return DenseCopyMoveResult(
                mask=np.zeros((orig_h, orig_w), dtype=np.uint8),
                score=0.0,
                dominant_cluster_blocks=inlier_count,
                mean_similarity=0.0,
                shift_vector_xy=(int(dom_bin[0] * st), int(dom_bin[1] * st)),
                diagnostics={"valid_blocks": m, "dominant_inliers": inlier_count},
            )

        grid_src = np.zeros((grid_h, grid_w), dtype=np.uint8)
        grid_dst = np.zeros((grid_h, grid_w), dtype=np.uint8)
        grid_src[gy[inlier_i], gx[inlier_i]] = 255
        grid_dst[gy[inlier_j], gx[inlier_j]] = 255

        k3 = np.ones((3, 3), dtype=np.uint8)
        grid_combined = cv2.morphologyEx(np.maximum(grid_src, grid_dst), cv2.MORPH_CLOSE, k3)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid_combined, connectivity=8)
        valid_comp_mask = np.zeros_like(grid_combined)
        max_comp_cells = 0
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= max(5, self.min_consistent_blocks // 2):
                valid_comp_mask[labels == lbl] = 255
                max_comp_cells = max(max_comp_cells, area)

        active_cells = int(np.count_nonzero(valid_comp_mask))
        if active_cells < self.min_consistent_blocks:
            return DenseCopyMoveResult(
                mask=np.zeros((orig_h, orig_w), dtype=np.uint8),
                score=0.0,
                dominant_cluster_blocks=active_cells,
                mean_similarity=0.0,
                shift_vector_xy=(int(dom_bin[0] * st), int(dom_bin[1] * st)),
                diagnostics={"reason": "no_compact_cluster", "active_cells": active_cells},
            )

        work_mask = np.zeros((work_h, work_w), dtype=np.uint8)
        ay, ax = np.nonzero(valid_comp_mask)
        for r_idx, c_idx in zip(ay, ax):
            y0, x0 = r_idx * st, c_idx * st
            work_mask[y0 : y0 + bs, x0 : x0 + bs] = 255

        full_mask = (
            cv2.resize(work_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if (work_h != orig_h or work_w != orig_w)
            else work_mask
        )
        mean_sim = float(np.clip(1.0 - (cand_mad[inlier_sel].mean() / 15.0), 0.0, 1.0))
        score = float(np.clip(inlier_count / 22.0, 0.40, 0.99) * mean_sim)

        return DenseCopyMoveResult(
            mask=full_mask,
            score=score,
            dominant_cluster_blocks=inlier_count,
            mean_similarity=mean_sim,
            shift_vector_xy=(int(dom_bin[0] * st), int(dom_bin[1] * st)),
            diagnostics={
                "valid_blocks": m,
                "candidates_after_rgb_gate": int(len(cand_i)),
                "dominant_inliers": inlier_count,
                "active_grid_cells": active_cells,
                "max_comp_cells": max_comp_cells,
            },
        )
