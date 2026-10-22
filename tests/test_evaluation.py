"""
ForensiVision Değerlendirme Motoru (Evaluation Metrics) Birim Testleri.

Bu test modülü, piksel seviyesinde manipüle bölge yerelleştirme doğruluğunu ölçen
Precision, Recall, F1-Score ve IoU metriklerinin matematiksel doğruluğunu,
sınır durumlarını ve girdi doğrulama kurallarını test eder.
"""

import math
import numpy as np
import pytest

from forensivision.evaluation import EvaluationResult, calculate_metrics


def test_perfect_match():
    """1. Senaryo: İki maske tamamen aynı olduğunda bütün metrikler 1.0 olmalıdır."""
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 5:15] = 1  # 10x10 = 100 piksel manipülasyon
    pred = gt.copy()

    res = calculate_metrics(pred, gt)

    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.iou == 1.0
    assert res.tp == 100
    assert res.fp == 0
    assert res.fn == 0


def test_prediction_wider_than_ground_truth():
    """2. Senaryo: Tahmin maskesi gerçek maskeden daha geniş (FP > 0, FN = 0)."""
    gt = np.zeros((30, 30), dtype=np.uint8)
    gt[10:20, 10:20] = 1  # 100 piksel gerçek

    pred = np.zeros((30, 30), dtype=np.uint8)
    pred[10:20, 10:20] = 1  # 100 piksel TP
    pred[20:25, 10:20] = 1  # 5x10 = 50 piksel FP

    res = calculate_metrics(pred, gt)

    assert res.tp == 100
    assert res.fp == 50
    assert res.fn == 0
    # Recall = 100 / (100 + 0) = 1.0
    assert res.recall == 1.0
    # Precision = 100 / (100 + 50) = 100 / 150 = 2/3
    assert res.precision == pytest.approx(2 / 3)
    # F1 = 2*100 / (2*100 + 50 + 0) = 200 / 250 = 0.8
    assert res.f1_score == pytest.approx(0.8)
    # IoU = 100 / (100 + 50 + 0) = 100 / 150 = 2/3
    assert res.iou == pytest.approx(2 / 3)


def test_prediction_subset_of_ground_truth():
    """3. Senaryo: Tahmin maskesi gerçek maskenin yalnızca bir alt kümesi (FP = 0, FN > 0)."""
    gt = np.zeros((30, 30), dtype=np.uint8)
    gt[10:20, 10:20] = 1  # 100 piksel gerçek

    pred = np.zeros((30, 30), dtype=np.uint8)
    pred[10:15, 10:20] = 1  # 50 piksel tahmin (yalnızca yarısı yakalandı)

    res = calculate_metrics(pred, gt)

    assert res.tp == 50
    assert res.fp == 0
    assert res.fn == 50
    # Precision = 50 / (50 + 0) = 1.0
    assert res.precision == 1.0
    # Recall = 50 / (50 + 50) = 0.5
    assert res.recall == 0.5
    # F1 = 2*50 / (2*50 + 0 + 50) = 100 / 150 = 2/3
    assert res.f1_score == pytest.approx(2 / 3)
    # IoU = 50 / (50 + 0 + 50) = 50 / 100 = 0.5
    assert res.iou == 0.5


def test_mixed_tp_fp_fn_sample():
    """4. Senaryo: TP, FP ve FN değerlerinin birlikte bulunduğu karma örnek."""
    # 10x10 = 100 piksellik ızgara
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)

    # TP = 20 piksel: [0:2, 0:10] (20 piksel)
    pred[0:2, :] = 1
    gt[0:2, :] = 1

    # FP = 10 piksel: [2:3, 0:10] (sadece tahminde 1)
    pred[2:3, :] = 1

    # FN = 5 piksel: [3:4, 0:5] (sadece gerçekte 1)
    gt[3:4, 0:5] = 1

    res = calculate_metrics(pred, gt)

    expected_tp = 20
    expected_fp = 10
    expected_fn = 5

    assert res.tp == expected_tp
    assert res.fp == expected_fp
    assert res.fn == expected_fn

    expected_precision = 20 / (20 + 10)  # 2/3 ≈ 0.6667
    expected_recall = 20 / (20 + 5)      # 0.8
    expected_f1 = (2 * 20) / ((2 * 20) + 10 + 5)  # 40/55 ≈ 0.7273
    expected_iou = 20 / (20 + 10 + 5)    # 20/35 = 4/7 ≈ 0.5714

    assert res.precision == pytest.approx(expected_precision)
    assert res.recall == pytest.approx(expected_recall)
    assert res.f1_score == pytest.approx(expected_f1)
    assert res.iou == pytest.approx(expected_iou)


def test_both_masks_empty_convention():
    """5. Senaryo: İki maske de boş olduğunda (orijinal ve tahrifatsız görüntü) konvansiyon gereği tüm metrikler 1.0 olmalıdır."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)

    res = calculate_metrics(pred, gt)

    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.iou == 1.0
    assert res.tp == 0
    assert res.fp == 0
    assert res.fn == 0


def test_prediction_empty_ground_truth_full():
    """6. Senaryo: Tahmin boş ama gerçekte manipülasyon var (Kaçırılan manipülasyon) -> Tüm metrikler 0.0 olmalıdır."""
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:10, 5:10] = 1  # 25 piksel
    pred = np.zeros((20, 20), dtype=np.uint8)

    res = calculate_metrics(pred, gt)

    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1_score == 0.0
    assert res.iou == 0.0
    assert res.tp == 0
    assert res.fp == 0
    assert res.fn == 25


def test_ground_truth_empty_prediction_full():
    """7. Senaryo: Gerçekte manipülasyon yok ama tahmin dolu (Yanlış alarm) -> Tüm metrikler 0.0 olmalıdır."""
    gt = np.zeros((20, 20), dtype=np.uint8)
    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5:10, 5:10] = 1  # 25 piksel yanlış alarm

    res = calculate_metrics(pred, gt)

    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1_score == 0.0
    assert res.iou == 0.0
    assert res.tp == 0
    assert res.fp == 25
    assert res.fn == 0


def test_boolean_mask_inputs():
    """8. Senaryo: Boolean tipindeki diziler doğrudan desteklenmelidir."""
    gt = np.array([[True, False], [False, True]])
    pred = np.array([[True, True], [False, False]])

    res = calculate_metrics(pred, gt)

    assert res.tp == 1
    assert res.fp == 1
    assert res.fn == 1
    assert res.precision == 0.5
    assert res.recall == 0.5
    assert res.f1_score == 0.5
    assert res.iou == pytest.approx(1 / 3)


def test_zero_and_255_masks():
    """9. Senaryo: Standart görüntü işleme formatı olan 0 ve 255 piksel değerleri doğru binarize edilmelidir."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:5, 0:5] = 255  # 25 piksel ön plan

    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[0:5, 0:5] = 255

    res = calculate_metrics(pred, gt)

    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.iou == 1.0
    assert res.tp == 25


def test_shape_mismatch_raises_value_error():
    """10. Senaryo: Boyut uyuşmazlığında sessizce yeniden boyutlandırmak yerine ValueError fırlatılmalıdır."""
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt = np.zeros((50, 60), dtype=np.uint8)

    with pytest.raises(ValueError, match="boyutları uyuşmuyor"):
        calculate_metrics(pred, gt)


def test_empty_array_raises_value_error():
    """11. Senaryo: Boş dizi (size=0) verildiğinde ValueError fırlatılmalıdır."""
    empty_arr = np.array([], dtype=np.uint8)
    valid_arr = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="boş bir dizi olamaz"):
        calculate_metrics(empty_arr, valid_arr)

    with pytest.raises(ValueError, match="boş bir dizi olamaz"):
        calculate_metrics(valid_arr, empty_arr)


def test_non_2d_inputs_raise_value_error():
    """12. Senaryo: 1D veya 3D gibi iki boyutlu olmayan dizilerde ValueError fırlatılmalıdır."""
    one_d = np.zeros((100,), dtype=np.uint8)
    three_d = np.zeros((10, 10, 3), dtype=np.uint8)
    valid_2d = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="iki boyutlu"):
        calculate_metrics(one_d, valid_2d)

    with pytest.raises(ValueError, match="iki boyutlu"):
        calculate_metrics(three_d, valid_2d)


def test_invalid_types_raise_error():
    """13. Senaryo: None veya metin gibi geçersiz tiplerde uygun istisna fırlatılmalıdır."""
    valid_2d = np.zeros((10, 10), dtype=np.uint8)

    # None girdisi
    with pytest.raises(ValueError, match="None olamaz"):
        calculate_metrics(None, valid_2d)

    with pytest.raises(ValueError, match="None olamaz"):
        calculate_metrics(valid_2d, None)

    # String veya dönüştürülemeyen tip
    with pytest.raises(TypeError, match="sayısal veya mantıksal"):
        calculate_metrics("invalid_string", valid_2d)


def test_no_nan_or_inf_in_non_overlapping_scenario():
    """14. Senaryo: Hiçbir pozitif pikselin kesişmediği (TP=0, FP>0, FN>0) durumda NaN veya inf oluşmamalıdır."""
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[0:5, 0:5] = 1  # 25 piksel sol üst

    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[15:20, 15:20] = 1  # 25 piksel sağ alt (hiç kesişim yok)

    res = calculate_metrics(pred, gt)

    assert res.tp == 0
    assert res.fp == 25
    assert res.fn == 25

    for val in [res.precision, res.recall, res.f1_score, res.iou]:
        assert not math.isnan(val), "Metrik değeri NaN olamaz."
        assert not math.isinf(val), "Metrik değeri sonsuz (inf) olamaz."
        assert 0.0 <= val <= 1.0, f"Metrik değeri [0.0, 1.0] aralığında olmalıdır; alınan: {val}"


def test_input_immutability():
    """15. Senaryo: Fonksiyon girdileri yerinde değiştirmemeli (immutability korunmalıdır)."""
    gt_original = np.array([[0, 255], [128, 0]], dtype=np.uint8)
    pred_original = np.array([[255, 0], [0, 64]], dtype=np.uint8)

    gt_copy = gt_original.copy()
    pred_copy = pred_original.copy()

    _ = calculate_metrics(pred_original, gt_original)

    np.testing.assert_array_equal(gt_original, gt_copy, err_msg="Gerçek maske yerinde değiştirilmemelidir.")
    np.testing.assert_array_equal(pred_original, pred_copy, err_msg="Tahmin maskesi yerinde değiştirilmemelidir.")


def test_evaluation_result_dataclass_methods():
    """EvaluationResult nesnesinin sözlüğe dönüştürme ve değiştirilemezlik (frozen) özelliklerinin testi."""
    res = EvaluationResult(
        precision=0.8,
        recall=0.75,
        f1_score=0.774,
        iou=0.632,
        tp=100,
        fp=25,
        fn=33,
    )

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["precision"] == 0.8
    assert d["tp"] == 100

    # Frozen kontrolü: Doğrudan alan değiştirilememelidir
    with pytest.raises(Exception):
        res.precision = 0.9  # type: ignore
