"""
ForensiVision Hibrit Füzyon (Deep Learning + Geometrik Eşleştirme) Paketi.
"""

from forensivision.fusion.detector import (
    FusionConfig,
    FusionResult,
    ForensiVisionFusionDetector,
    detect_forgery_hybrid,
)
from forensivision.fusion.meta_classifier import (
    FEATURE_NAMES,
    ForensicEvidenceVector,
    StackingForensicClassifier,
    compute_jpeg_blockiness,
    compute_region_discrepancy,
)
from forensivision.fusion.pipeline_v2 import (
    ForensiVisionV2Pipeline,
    ForensiVisionV2Prediction,
)

__all__ = [
    "FusionConfig",
    "FusionResult",
    "ForensiVisionFusionDetector",
    "detect_forgery_hybrid",
    "FEATURE_NAMES",
    "ForensicEvidenceVector",
    "StackingForensicClassifier",
    "compute_jpeg_blockiness",
    "compute_region_discrepancy",
    "ForensiVisionV2Pipeline",
    "ForensiVisionV2Prediction",
]
