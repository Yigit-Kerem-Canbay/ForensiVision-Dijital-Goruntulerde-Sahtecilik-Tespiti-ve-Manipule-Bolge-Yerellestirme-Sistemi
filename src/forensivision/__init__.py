"""
ForensiVision: Dijital Görüntülerde Sahtecilik Tespiti ve Manipüle Bölge Yerelleştirme Sistemi.

Bu paket; dijital görüntülerdeki sahtecilik ve manipülasyonların tespiti,
Copy-Move ve Image Splicing şüphesi analizi, piksel düzeyinde yerelleştirme (maskeleme)
ve başarım ölçümü modüllerini içerir.
"""

__version__ = "0.1.0"
__author__ = "Yiğit Kerem Canbay"

from forensivision.fusion import (
    FusionConfig,
    FusionResult,
    ForensiVisionFusionDetector,
    detect_forgery_hybrid,
)

__all__ = [
    "__version__",
    "__author__",
    "FusionConfig",
    "FusionResult",
    "ForensiVisionFusionDetector",
    "detect_forgery_hybrid",
]

