"""
ForensiVision Adli Veri Setleri Paketi.

Bu paket; CoMoFoD, Columbia ve benzeri adli görüntü manipülasyon veri setlerinin
standart veri modelleri (`DatasetSample`, `BaseDataset`) ile taranmasını,
indekslenmesini ve manifest üretimini sağlar.
"""

from forensivision.datasets.base import BaseDataset, DatasetSample
from forensivision.datasets.casia_clean import CasiaV2Dataset, CleanImageDataset
from forensivision.datasets.columbia import ColumbiaDataset
from forensivision.datasets.comofod import COMOFOD_ATTACK_MAP, CoMoFoDDataset
from forensivision.datasets.defacto import DefactoDataset
from forensivision.datasets.degradation import (
    DegradationResult,
    FormatAgnosticDegrader,
    SyntheticCopyMoveGenerator,
    SyntheticCopyMoveSample,
)
from forensivision.datasets.manifest import (
    build_manifest,
    load_manifest,
    save_manifest,
)
from forensivision.datasets.splitter import (
    IndexedSampleRecord,
    SceneGroupedManifestBuilder,
    compute_canonical_scene_id,
)

__all__ = [
    "DatasetSample",
    "BaseDataset",
    "CasiaV2Dataset",
    "CleanImageDataset",
    "CoMoFoDDataset",
    "COMOFOD_ATTACK_MAP",
    "ColumbiaDataset",
    "DefactoDataset",
    "DegradationResult",
    "FormatAgnosticDegrader",
    "SyntheticCopyMoveGenerator",
    "SyntheticCopyMoveSample",
    "IndexedSampleRecord",
    "SceneGroupedManifestBuilder",
    "compute_canonical_scene_id",
    "build_manifest",
    "save_manifest",
    "load_manifest",
]
