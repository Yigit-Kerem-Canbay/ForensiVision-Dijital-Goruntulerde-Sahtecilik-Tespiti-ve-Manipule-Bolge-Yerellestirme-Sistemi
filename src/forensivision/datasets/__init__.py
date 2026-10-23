"""
ForensiVision Adli Veri Setleri Paketi.

Bu paket; CoMoFoD, Columbia ve benzeri adli görüntü manipülasyon veri setlerinin
standart veri modelleri (`DatasetSample`, `BaseDataset`) ile taranmasını,
indekslenmesini ve manifest üretimini sağlar.
"""

from forensivision.datasets.base import BaseDataset, DatasetSample
from forensivision.datasets.columbia import ColumbiaDataset
from forensivision.datasets.comofod import COMOFOD_ATTACK_MAP, CoMoFoDDataset
from forensivision.datasets.manifest import (
    build_manifest,
    load_manifest,
    save_manifest,
)

__all__ = [
    "DatasetSample",
    "BaseDataset",
    "CoMoFoDDataset",
    "COMOFOD_ATTACK_MAP",
    "ColumbiaDataset",
    "build_manifest",
    "save_manifest",
    "load_manifest",
]
