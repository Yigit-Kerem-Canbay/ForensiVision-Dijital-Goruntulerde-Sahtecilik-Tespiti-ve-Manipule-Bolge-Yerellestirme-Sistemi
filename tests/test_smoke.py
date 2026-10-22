"""
ForensiVision Temel Kurulum ve İçe Aktarma (Smoke) Testleri.

Bu test modülü; Python ortamının, paket mimarisinin ve temel bağımlılıkların
(OpenCV, NumPy, PIL, Matplotlib, scikit-learn) doğru kurulduğunu ve çalıştığını doğrular.
"""

import pytest


def test_package_metadata():
    """Paketin ana modülünün ve versiyon bilgisinin içe aktarılabilirlik testi."""
    import forensivision
    assert hasattr(forensivision, "__version__")
    assert forensivision.__version__ == "0.1.0"


def test_submodules_importable():
    """Tüm alt modüllerin (copy_move, splicing, evaluation, visualization) içe aktarılabilirlik testi."""
    from forensivision import copy_move, splicing, evaluation, visualization
    assert copy_move is not None
    assert splicing is not None
    assert evaluation is not None
    assert visualization is not None


def test_third_party_dependencies():
    """Temel bilimsel hesaplama ve görüntü işleme kütüphanelerinin içe aktarım testi."""
    import cv2
    import numpy as np
    import PIL
    import matplotlib
    import sklearn

    # Temel kütüphane varlık kontrolleri
    assert cv2.__version__ is not None
    assert np.__version__ is not None
    assert PIL.__version__ is not None
    assert matplotlib.__version__ is not None
    assert sklearn.__version__ is not None


def test_opencv_feature_detectors_available():
    """Copy-Move için kritik olan SIFT ve ORB öznitelik çıkarıcıların OpenCV içinde aktif olduğunu test eder."""
    import cv2
    import numpy as np

    # SIFT kontrolü (Patent süresi dolduğu için modern OpenCV sürümlerinde yerleşik olmalıdır)
    sift = cv2.SIFT_create()
    assert sift is not None

    # ORB kontrolü
    orb = cv2.ORB_create()
    assert orb is not None

    # Sentetik kontrastlı ve şekilli test görüntüsü (SIFT ve ORB'nin köşe/doku yakalayabilmesi için)
    synthetic_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(synthetic_img, (100, 100), 40, 255, -1)
    cv2.rectangle(synthetic_img, (20, 20), (60, 60), 255, -1)
    cv2.putText(synthetic_img, "FV", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

    kp_sift, des_sift = sift.detectAndCompute(synthetic_img, None)
    assert len(kp_sift) > 0, "SIFT sentetik görüntü üzerinde anahtar nokta tespit edebilmelidir."
    assert des_sift is not None and des_sift.shape[0] == len(kp_sift)

    kp_orb, des_orb = orb.detectAndCompute(synthetic_img, None)
    assert len(kp_orb) > 0, "ORB sentetik görüntü üzerinde anahtar nokta tespit edebilmelidir."
    assert des_orb is not None and des_orb.shape[0] == len(kp_orb)

