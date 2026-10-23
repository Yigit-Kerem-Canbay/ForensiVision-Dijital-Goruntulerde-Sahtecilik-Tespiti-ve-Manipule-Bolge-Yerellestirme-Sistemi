"""
Image Splicing Tespiti ve Yerelleştirme Modülü.

Farklı bir görüntüden eklenen yabancı nesne ve bölgelerin;
- Error Level Analysis (ELA) ve
- Yerel gürültü tutarsızlığı (Noise Inconsistency)
analizleriyle tespiti ve şüpheli bölge yerelleştirmesi.
"""

from forensivision.splicing.detector import (
    SplicingConfig,
    SplicingResult,
    detect_splicing,
)
from forensivision.splicing.ela import ELAResult, analyze_ela
from forensivision.splicing.noise import (
    NoiseAnalysisResult,
    analyze_noise_inconsistency,
)

__all__ = [
    "analyze_ela",
    "ELAResult",
    "analyze_noise_inconsistency",
    "NoiseAnalysisResult",
    "detect_splicing",
    "SplicingConfig",
    "SplicingResult",
]
