"""
ForensiVision Web Arayüzü ve Yardımcı Araçları (UI Module).

Bu paket; Streamlit tabanlı kullanıcı arayüzü için veri yükleme, renk dönüşümleri,
analiz motoru çağrıları, doğruluk maskesi denetimi ve dışa aktarım işlemlerini
arayüz kodundan bağımsız, modüler ve birim testlerle sınanabilir biçimde sunar.
"""

from __future__ import annotations

from forensivision.ui.helpers import (
    compute_optional_metrics,
    generate_export_payload,
    image_to_png_bytes,
    load_uploaded_image,
    mask_to_png_bytes,
    run_copy_move_analysis,
    run_splicing_analysis,
    validate_and_load_ground_truth,
)
from forensivision.ui.synthetic import get_synthetic_demo_sample

__all__ = [
    "load_uploaded_image",
    "validate_and_load_ground_truth",
    "run_copy_move_analysis",
    "run_splicing_analysis",
    "compute_optional_metrics",
    "generate_export_payload",
    "mask_to_png_bytes",
    "image_to_png_bytes",
    "get_synthetic_demo_sample",
]
