"""
ForensiVision Değerlendirme Motoru (Evaluation Metrics).

Bu modül; adli görüntü analizi sonucunda üretilen tahmin maskesi (prediction) ile
gerçek manipülasyon maskesi (ground-truth) arasındaki piksel düzeyindeki yerelleştirme
başarımını Precision, Recall, F1-Score ve IoU metrikleri ile hesaplar.

Standart İkili Maske Kuralı:
    - 0 veya 0'dan küçük değerler: Arka plan (orijinal/değiştirilmemiş alan).
    - 0'dan büyük değerler (> 0): Ön plan (manipüle edilmiş/şüpheli alan).

Boş Maske Konvansiyonu:
    - İki maske de boşsa (TP=0, FP=0, FN=0):
      Her iki maske de sahtecilik içermediğinde tam uyuşma kabul edilerek
      Precision=1.0, Recall=1.0, F1-Score=1.0, IoU=1.0 döndürülür.
      (Bu proje için açıkça tercih edilmiş adli bilişim konvansiyonudur.)
    - Yalnızca tahmin maskesi boşsa (FP=0, FN>0):
      Metriklerin tümü 0.0 döndürülür.
    - Yalnızca gerçek maske boşsa (FP>0, FN=0):
      Metriklerin tümü 0.0 döndürülür.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Piksel seviyesinde manipülasyon yerelleştirme değerlendirme sonuçları.

    Attributes:
        precision: Kesinlik [0.0, 1.0] -> TP / (TP + FP)
        recall: Duyarlılık [0.0, 1.0] -> TP / (TP + FN)
        f1_score: F1 Başarım Skoru [0.0, 1.0] -> 2*TP / (2*TP + FP + FN)
        iou: Kesişim / Birleşim Oranı [0.0, 1.0] -> TP / (TP + FP + FN)
        tp: Doğru pozitif piksel sayısı (True Positives)
        fp: Yanlış pozitif piksel sayısı (False Positives)
        fn: Yanlış negatif piksel sayısı (False Negatives)
    """

    precision: float
    recall: float
    f1_score: float
    iou: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        """Sonuçları sözlük yapısına dönüştürür."""
        return asdict(self)


def _validate_and_binarize_mask(mask: Any, name: str) -> np.ndarray:
    """Maske dizisini doğrular, kopyasını korur ve ikili (boolean) forma dönüştürür.

    Args:
        mask: İncelenecek maske dizisi (numpy array, liste vb.).
        name: Hata mesajlarında kullanılacak değişken adı.

    Returns:
        np.ndarray: İki boyutlu boolean NumPy dizisi.

    Raises:
        TypeError: Girdi None ise veya sayısal/boolean bir diziye dönüştürülemezse.
        ValueError: Dizi boşsa veya tam olarak 2 boyutlu değilse.
    """
    if mask is None:
        raise ValueError(f"'{name}' değeri None olamaz.")

    if not isinstance(mask, np.ndarray):
        try:
            arr = np.asarray(mask)
        except Exception as err:
            raise TypeError(f"'{name}' geçerli bir NumPy dizisine dönüştürülemedi: {err}") from err
    else:
        # Girdiyi yerinde değiştirmemek için referans kontrolü
        arr = mask

    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError(
            f"'{name}' sayısal veya mantıksal (boolean) veri tipinde olmalıdır; alınan tip: {arr.dtype}"
        )

    if arr.size == 0:
        raise ValueError(f"'{name}' boş bir dizi olamaz.")

    if arr.ndim != 2:
        raise ValueError(
            f"'{name}' iki boyutlu (2D) bir maske olmalıdır; alınan boyut sayısı: {arr.ndim} (şekil: {arr.shape})"
        )

    # 0'dan büyük değerler ön plan (manipüle), diğerleri arka plan
    return arr > 0


def calculate_metrics(
    prediction_mask: Any,
    ground_truth_mask: Any,
) -> EvaluationResult:
    """Tahmin maskesi ile gerçek maske arasındaki adli yerelleştirme metriklerini hesaplar.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1-Score = 2*TP / (2*TP + FP + FN)
    IoU = TP / (TP + FP + FN)

    Args:
        prediction_mask: Model/algoritma tarafından üretilen tahmin maskesi (2D array).
        ground_truth_mask: Gerçek referans manipülasyon maskesi (2D array).

    Returns:
        EvaluationResult: Precision, Recall, F1, IoU, TP, FP ve FN değerlerini içeren nesne.

    Raises:
        TypeError: Girdiler uygun veri tipinde değilse.
        ValueError: Girdiler None, boş, 2D değilse veya boyutları birbiriyle uyuşmuyorsa.
    """
    pred_bin = _validate_and_binarize_mask(prediction_mask, "prediction_mask")
    gt_bin = _validate_and_binarize_mask(ground_truth_mask, "ground_truth_mask")

    if pred_bin.shape != gt_bin.shape:
        raise ValueError(
            f"Tahmin maskesi ile gerçek maskenin boyutları uyuşmuyor: "
            f"tahmin={pred_bin.shape}, gerçek={gt_bin.shape}. "
            "Boyut uyuşmazlığında maskeler sessizce yeniden boyutlandırılamaz."
        )

    pred_positives = int(np.count_nonzero(pred_bin))
    gt_positives = int(np.count_nonzero(gt_bin))

    # Durum 1: İki maske de tamamen boş (Manipülasyon yok ve tespit de yok)
    if pred_positives == 0 and gt_positives == 0:
        return EvaluationResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            iou=1.0,
            tp=0,
            fp=0,
            fn=0,
        )

    # Durum 2: Tahmin maskesi boş ama gerçekte manipülasyon var (Kaçırılan manipülasyon)
    if pred_positives == 0 and gt_positives > 0:
        return EvaluationResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            iou=0.0,
            tp=0,
            fp=0,
            fn=gt_positives,
        )

    # Durum 3: Gerçekte manipülasyon yok ama sistem sahte tespit üretti (Tamamen yanlış alarm)
    if pred_positives > 0 and gt_positives == 0:
        return EvaluationResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            iou=0.0,
            tp=0,
            fp=pred_positives,
            fn=0,
        )

    # Durum 4: Her iki maskede de pozitif pikseller mevcut
    tp = int(np.count_nonzero(pred_bin & gt_bin))
    fp = int(np.count_nonzero(pred_bin & (~gt_bin)))
    fn = int(np.count_nonzero((~pred_bin) & gt_bin))

    if tp == 0:
        # Pozitif pikseller var fakat hiçbiri birbiriyle örtüşmüyor (tam uyumsuzluk)
        return EvaluationResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            iou=0.0,
            tp=0,
            fp=fp,
            fn=fn,
        )

    # Standart metrik hesaplamaları
    precision = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    f1_score = float((2.0 * tp) / ((2.0 * tp) + fp + fn))
    iou = float(tp / (tp + fp + fn))

    # Sayısal sınır kontrolü: Değerlerin [0.0, 1.0] aralığında kalmasını güvence altına al
    precision = min(max(precision, 0.0), 1.0)
    recall = min(max(recall, 0.0), 1.0)
    f1_score = min(max(f1_score, 0.0), 1.0)
    iou = min(max(iou, 0.0), 1.0)

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        iou=iou,
        tp=tp,
        fp=fp,
        fn=fn,
    )
