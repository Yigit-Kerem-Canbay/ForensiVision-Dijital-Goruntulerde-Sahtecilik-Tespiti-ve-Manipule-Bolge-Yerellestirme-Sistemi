"""
Copy-Move Forgery Tespiti ve Yerelleştirme Modülü.

Görüntünün kendi içerisinden kopyalanıp başka bir bölgeye taşınmasıyla
oluşturulan tahrifatların SIFT öznitelik çıkarımı, öz-eşleme filtreleme,
RANSAC geometrik tutarlılık doğrulaması ve morfolojik bölge yerelleştirmesi ile analizi.
"""

from forensivision.copy_move.detector import (
    CopyMoveConfig,
    CopyMoveResult,
    detect_copy_move,
)

__all__ = [
    "detect_copy_move",
    "CopyMoveConfig",
    "CopyMoveResult",
]
