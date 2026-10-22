"""
Performans ve Metrik Değerlendirme Modülü.

Piksel ve bölge seviyesinde üretilen tahmin maskeleri ile ground-truth maskeleri
arasındaki Precision, Recall, F1-Score ve IoU (Intersection over Union) metriklerinin
akademik standartlara uygun doğrulanabilir hesaplamaları.
"""

from forensivision.evaluation.metrics import EvaluationResult, calculate_metrics

__all__ = ["calculate_metrics", "EvaluationResult"]
