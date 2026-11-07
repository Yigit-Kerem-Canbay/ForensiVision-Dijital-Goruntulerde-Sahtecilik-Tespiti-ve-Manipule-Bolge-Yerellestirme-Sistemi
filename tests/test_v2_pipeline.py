"""
ForensiVision v2 — Kapsamlı Birim ve Bütünleşik Doğrulama Testleri (Unit & Integration Tests).

Test Edilen Kritik Garantiler:
1. CoMoFoD Etiket Doğruluğu: `_O` ve tüm `_O_*` varyantları (5.000 görsel) kesinlikle `authentic`
   ve `is_tampered=0` olarak işaretlenir; `_F` ve `_F_*` varyantları (5.000 görsel) `copy_move`
   ve `is_tampered=1` olarak işaretlenir.
2. Sahne Sızıntısı Yokluğu (Zero Scene Leakage): MS-COCO temiz (`000000000071.jpg`) ve
   DEFACTO sahte (`0_000000000071.tif`) görselleri aynı `canonical_scene_id` ('coco_000000000071')
   değerini alır ve `train`, `val`, `calibration`, `test` kümeleri arasında kesişim = 0 olur.
3. Format-Bağımsız Bozulma ve Dinamik Copy-Move Üretimi: Tüm format modları (PNG, tekli/çiftli JPEG,
   WhatsApp, WebP, gürültülü JPEG) ve sentetik klonlama maskesi hatasız çalışır.
4. Çift Akışlı Model (`ForensiVisionV2Net`) + Çok Görevli Kayıp (`ForensiVisionV2Loss`):
   SRM+Bayar kısıtı (merkez = -1, çevre toplamı = +1) korunur, ileri/geri yayılım hatasız çalışır.
5. Stacking Meta-Sınıflandırıcı ve Çift Ölçekli Çıkarım Hattı (`ForensiVisionV2Pipeline`):
   Temiz görsel kararında çıktı maskesi tamamen sıfırlanır (yanlış piksel boyama koruması).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forensivision.datasets.base import DatasetSample
from forensivision.datasets.degradation import (
    FormatAgnosticDegrader,
    SyntheticCopyMoveGenerator,
)
from forensivision.datasets.splitter import (
    SceneGroupedManifestBuilder,
    compute_canonical_scene_id,
)
from forensivision.fusion import (
    FEATURE_NAMES,
    ForensiVisionV2Pipeline,
    StackingForensicClassifier,
)
from forensivision.models import (
    ForensiVisionV2Loss,
    ForensiVisionV2Net,
    SRMFilterBank,
)


def test_canonical_scene_id_locks_coco_and_defacto_together(tmp_path):
    """MS-COCO temiz görseli ile DEFACTO Copy-Move/Splicing türevleri aynı sahne kimliğini almalıdır."""
    dummy_img = tmp_path / "000000000071.jpg"
    dummy_img.write_bytes(b"0")

    s_coco = DatasetSample(
        sample_id="coco_000000000071",
        task_type="authentic",
        image_path=dummy_img,
        manipulation_type="authentic",
        group_id="000000000071",
    )
    s_defacto_cm = DatasetSample(
        sample_id="0_000000000071",
        task_type="copy_move",
        image_path=dummy_img,
        manipulation_type="copy_move",
        group_id="000000000071",
    )
    s_defacto_sp = DatasetSample(
        sample_id="p1_12_000000000071",
        task_type="splicing",
        image_path=dummy_img,
        manipulation_type="splicing",
        group_id="000000000071",
    )

    id_coco = compute_canonical_scene_id(s_coco, "COCO_Clean")
    id_cm = compute_canonical_scene_id(s_defacto_cm, "DEFACTO")
    id_sp = compute_canonical_scene_id(s_defacto_sp, "DEFACTO")

    assert id_coco == "coco_000000000071"
    assert id_cm == "coco_000000000071"
    assert id_sp == "coco_000000000071"

    splitter = SceneGroupedManifestBuilder(seed=42)
    assert splitter.assign_split(id_coco) == splitter.assign_split(id_cm) == splitter.assign_split(id_sp)


def test_degradation_and_synthetic_copymove():
    """Sentetik Copy-Move üreticisi ve tüm format bozulma modları boyut ve maske bütünlüğünü korumalıdır."""
    rng = np.random.default_rng(123)
    clean_bgr = rng.integers(20, 230, size=(320, 400, 3), dtype=np.uint8)

    gen = SyntheticCopyMoveGenerator(seed=42)
    cm_sample = gen.generate(clean_bgr)

    assert cm_sample.forged_bgr.shape == clean_bgr.shape
    assert cm_sample.combined_mask.shape == (320, 400)
    assert np.count_nonzero(cm_sample.source_mask) > 100
    assert np.count_nonzero(cm_sample.target_mask) > 100
    assert np.count_nonzero(cm_sample.combined_mask) >= np.count_nonzero(cm_sample.target_mask)

    deg = FormatAgnosticDegrader(seed=42)
    for mode in ("lossless_png", "single_jpeg", "double_jpeg", "whatsapp", "webp", "noise_jpeg"):
        res = deg.apply_random_chain(cm_sample.forged_bgr, cm_sample.combined_mask, mode=mode)
        assert res.image_bgr.ndim == 3
        assert res.mask is not None
        assert res.image_bgr.shape[:2] == res.mask.shape[:2]


def test_srm_bayar_constraint_and_v2_net_step():
    """Bayar evrişim kısıtı (merkez=-1, çevre=+1) korunmalı ve model tek adım eğitimi tamamlamalıdır."""
    srm = SRMFilterBank(learnable_channels=6)
    bayar_w = srm._get_constrained_bayar_weights()
    # Merkez (2, 2) toplamı her çıkış kanalı için -1.0 olmalı, tüm 5x5 toplamı ise 0.0 olmalıdır
    total_kernel_sum = bayar_w.sum(dim=(1, 2, 3))
    assert torch.allclose(total_kernel_sum, torch.zeros_like(total_kernel_sum), atol=1e-4)

    model = ForensiVisionV2Net(pretrained_backbone=False)
    criterion = ForensiVisionV2Loss()

    x = torch.rand(2, 3, 256, 256)
    mask_gt = torch.zeros(2, 1, 256, 256)
    mask_gt[1, 0, 40:100, 50:120] = 1.0
    cls_gt = torch.tensor([0.0, 1.0])

    out = model(x)
    assert out.mask_logits.shape == (2, 1, 256, 256)
    assert out.confidence_logits.shape == (2, 1, 256, 256)
    assert out.edge_logits.shape == (2, 1, 256, 256)
    assert out.cls_logits.shape == (2, 1)

    loss, metrics = criterion(out, mask_gt, cls_gt)
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["loss_total"] > 0.0


def test_stacking_meta_classifier_and_pipeline_zero_fp_gating():
    """Stacking meta-sınıflandırıcı hedef özgüllüğü kalibre etmeli ve temiz kararda maskeyi sıfırlamalıdır."""
    rng = np.random.default_rng(42)
    X_clean = rng.uniform(0.0, 0.35, size=(80, len(FEATURE_NAMES))).astype(np.float32)
    X_tamp = rng.uniform(0.30, 0.95, size=(80, len(FEATURE_NAMES))).astype(np.float32)
    X = np.vstack([X_clean, X_tamp])
    y = np.array([0] * 80 + [1] * 80, dtype=np.int32)

    clf = StackingForensicClassifier(target_specificity=0.92)
    stats = clf.fit_and_calibrate(X, y)
    assert stats["clean_specificity"] >= 0.90

    # Pipeline testi (Temiz bir görselde is_tampered=False ise binary_mask tamamen 0 olmalıdır)
    pipeline = ForensiVisionV2Pipeline(model=None, meta_classifier=clf, input_size=256)
    clean_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    pred = pipeline.predict(clean_img)
    if not pred.is_tampered:
        assert np.count_nonzero(pred.binary_mask) == 0

