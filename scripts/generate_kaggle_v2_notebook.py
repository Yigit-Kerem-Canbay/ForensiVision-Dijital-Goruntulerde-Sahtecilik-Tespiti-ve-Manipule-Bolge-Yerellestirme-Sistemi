"""
ForensiVision v2 — Kaggle Çift GPU (2x T4) / AMP / Checkpoint-Resume Eğitim ve
Otomatik Stacking Meta-Sınıflandırıcı Kalibrasyon Notebook Üreticisi.

Bu betik çalıştırıldığında `notebooks/forensivision_v2_kaggle_training.ipynb` dosyasını üretir.
Hem Kaggle'da (`/kaggle/input`) hem de yerel bilgisayarda (DRY_RUN / CPU testi olarak)
birebir aynı kodla çalışacak şekilde tasarlanmıştır.

Giderilen Tüm v1 Hataları ve Eklenen v2 Teknolojileri:
1. CoMoFoD Etiket Hatası Düzeltmesi: `_O` ve tüm `_O_*` varyantları (5.000 fotoğraf) kesinlikle
   TEMİZ (`is_tampered=0`); `_F` ve `_F_*` varyantları (5.000 fotoğraf) KOPYALA-KAYDIR (`is_tampered=1`).
2. CASIA v2 Maske Koruması: Yalnızca doğruluk maskesi (`_gt`) eşleşen `Tp` görselleri piksel
   kaybına girer; maskesiz kirli görseller sıfır maskeyle modeli kör etmez.
3. Sıfır Sahne Sızıntısı (`canonical_scene_id`): MS-COCO temiz (`000000000071.jpg`) ile
   DEFACTO Copy-Move/Splicing (`0_000000000071.tif`) aynı `coco_000000000071` sahnesine kilitlenir.
   SHA-256 deterministik bölme ile `train (%75)`, `val (%10)`, `calibration (%5)`, `test (%10)`
   arasında ortak sahne sayısı = 0 olur.
4. Format-Bağımsız Bozulma + Dinamik Copy-Move: Temiz ve kirli görsellere aynı olasılıkla
   PNG (%20), tekli JPEG (%25), çiftli JPEG (%20), WhatsApp (%15), WebP (%10), Gürültülü JPEG (%10)
   uygulanır. Ayrıca temiz görsellerin bir kısmından anlık döndürülmüş/ölçeklenmiş Copy-Move üretilir.
5. `ForensiVisionV2Net`: SwinV2-Tiny (IMAGENET1K_V1 ön-eğitimli) + SRM & Softmax-Bayar Gürültü Kolu +
   `SelfCorrelation2D` (Copy-Move öz-korelasyon modülü) + Güvenilirlik (Confidence) ve Sınır (Edge) başlıkları.
6. Otomatik Stacking Kalibrasyonu: Eğitim bitince `calibration` kümesi üzerinde 16 adli özellik
   çıkarılır, `HistGradientBoosting + LogisticRegression` eğitilir ve Temiz Özgüllük >= %93
   sağlayan optimal eşik (`forensivision_v2_calibration.json`) kaydedilir.
"""

from __future__ import annotations

import json
from pathlib import Path


CELL_1_MD = """# 🛡️ ForensiVision v2: Üç Akışlı (SwinV2 + SRM/Bayar + 8x8 DCT) + Hibrit İkili Doğrulama (DL + Klasik SIFT/RANSAC & Yoğun Alan) Adli Eğitim Motoru

Bu notebook, **ForensiVision v2** mimarisini tüm bağımsız teknik denetim kriterlerini karşılayacak şekilde eğitir ve doğrular:
1. **Sıfır Sahne Sızıntısı + Bağımsız Çapraz Veri Seti (Cross-Dataset) Testi:** CoMoFoD (`comofod_{gid}`) ve MS-COCO/DEFACTO (`coco_{12_haneli_id}`) sahneleri `%0` kesişimle ayrılır; tüm havuzlar (`train/val/cal/test`) deterministik karıştırılır. **CASIA v2** eğitimden %100 hariç tutularak (`cross_test`) yalnızca son aşamada **Zero-Shot Çapraz Veri Seti Testi** için kullanılır.
2. **Adli İz Dostu Ön-İşleme (Önce Boyutlandırma, En Son Sıkıştırma + Tam 8x8 DCT Hizalaması):** Görseller önce maske-korumalı kırpma/ölçekleme ile `512x512` boyuta getirilir (CoMoFoD dahil tüm veri setlerine hafif ölçek/kırpma uygulanarak veri seti parmak izi engellenir), **sıkıştırma zinciri (PNG, Tek/Çift JPEG, WhatsApp, WebP) EN SON adımda** uygulanır. `DCTSpectralExtractor` tam `8x8` blok hizalıdır (`stride=8, padding=0`).
3. **%70 Gerçek + %30 Organik Dinamik Copy-Move Dengesi & Gürültü Destekli Öz-Korelasyon:** Copy-Move eğitim örneklerinin %70'i gerçek (`DEFACTO` + `CoMoFoD`), %30'u dinamik üretilen örneklerden oluşur. `SelfCorrelation2D` hem anlamsal `SwinV2` hem de `SRM/Bayar/DCT` gürültü özelliklerini birlikte eşleştirir.
4. **Tür Bazlı Ayrı Doğrulama Metrikleri + Diferansiyel LR + Warmup + Erken Durdurma:** Her epoch sonunda **Temiz (Spec %)**, **Copy-Move (Recall % & IoU)** ve **Splicing (Recall % & IoU)** ayrı ayrı raporlanır. Ön-eğitimli `SwinV2` omurgası `3e-5`, yeni adli başlıklar `1.5e-4` öğrenme oranıyla, `Linear Warmup + Cosine Decay` ve `Grad Clipping` ile eğitilir.
5. **Gerçek 16-Sinyal Hibrit İkili Doğrulama (DL + Klasik SIFT+RANSAC + Yoğun DCT Copy-Move + ELA/Gürültü) & Sızıntısız Eşik Kalibrasyonu:** Kalibrasyon kümesi iki ayrık parçaya (`CAL_FIT %60` ve `CAL_THRESH %40`) bölünerek aşırı uyum (overfitting) olmadan eşik seçilir; Klasik, Saf DL ve Hibrit sistemin başarıları ayrı ayrı karşılaştırılır."""


CELL_2_SETUP = """# 1. Donanım, Ortam ve Kütüphane Kurulumu
import os
import re
import io
import csv
import json
import math
import time
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IS_KAGGLE = Path('/kaggle/input').exists()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = (DEVICE.type == 'cuda')
NUM_GPUS = torch.cuda.device_count() if USE_AMP else 0

OUTPUT_DIR = Path('/kaggle/working/forensivision_v2_outputs') if IS_KAGGLE else Path('models/v2_pilot')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'🚀 Ortam        : {"Kaggle Cloud" if IS_KAGGLE else "Yerel Bilgisayar (Pilot/Test)"}', flush=True)
print(f'🚀 Cihaz        : {DEVICE} (GPU Sayısı: {NUM_GPUS})', flush=True)
if USE_AMP:
    for g_idx in range(NUM_GPUS):
        print(f'   GPU [{g_idx}]      : {torch.cuda.get_device_name(g_idx)}', flush=True)
    torch.backends.cudnn.benchmark = True
"""


CELL_3_INDEXER = """# 2. Sızıntısız (Zero-Leakage) Çoklu Veri Seti Tarayıcısı ve Sahne Ayırıcı
@dataclass
class V2Sample:
    image_path: str
    mask_path: Optional[str]
    source_mask_path: Optional[str]
    is_tampered: int          # 0: Temiz (Authentic), 1: Kirli (Copy-Move veya Splicing)
    manip_type: str           # 'authentic' | 'copy_move' | 'splicing'
    dataset_name: str
    canonical_scene_id: str
    split: str                # 'train' | 'val' | 'calibration' | 'test'

COCO_12_RE = re.compile(r'(\\d{12})')
COMOFOD_MASK_RE = re.compile(r'^(\\d+)_([BM])$', re.IGNORECASE)
COMOFOD_IMG_RE = re.compile(r'^(\\d+)_([OF])(?:_([A-Z]{2}\\d*))?$', re.IGNORECASE)

def deterministic_split(scene_id: str, seed: int = 42) -> str:
    digest = hashlib.sha256(f'{seed}:{scene_id}'.encode('utf-8')).hexdigest()
    u = int(digest[:12], 16) / float(0xFFFFFFFFFFFF + 1)
    if u < 0.75:
        return 'train'
    elif u < 0.85:
        return 'val'
    elif u < 0.90:
        return 'calibration'
    return 'test'

def scan_all_datasets(search_roots: List[Path]) -> List[V2Sample]:
    t0 = time.time()
    samples: List[V2Sample] = []

    # 1. Doğrudan klasör hedefleri (os.walk ile 700.000 dosyayı 2 saat taramak yerine doğrudan klasör listeleme!)
    target_dirs = []
    for base_root in search_roots:
        if not base_root.exists():
            continue
        for child in base_root.iterdir():
            target_dirs.append(child)

    casia_gt_map: Dict[str, str] = {}
    
    # Hızlı tarama: os.scandir veya sığ os.walk
    for root_dir in target_dirs:
        for root, dirs, files in os.walk(str(root_dir)):
            if not files:
                continue
            rl = root.lower()

            # CASIA GT Maskeleri
            if 'casia' in rl and ('gt' in rl or 'mask' in rl or 'groundtruth' in rl):
                for fn in files:
                    stem_l = os.path.splitext(fn)[0].lower()
                    clean_k = re.sub(r'_gt$', '', stem_l)
                    casia_gt_map[clean_k] = os.path.join(root, fn)
                continue

            # A. CoMoFoD Taraması (5.000 Temiz _O/_O_* + 5.000 Copy-Move _F/_F_*)
            if 'comofod' in rl and any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) for f in files[:10]):
                b_masks = {}
                for fn in files:
                    st = os.path.splitext(fn)[0]
                    mm = COMOFOD_MASK_RE.match(st)
                    if mm and mm.group(2).upper() == 'B':
                        b_masks[mm.group(1)] = os.path.join(root, fn)

                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in ('.png', '.jpg', '.jpeg', '.tif', '.bmp'):
                        continue
                    st = os.path.splitext(fn)[0]
                    if COMOFOD_MASK_RE.match(st):
                        continue
                    im_m = COMOFOD_IMG_RE.match(st)
                    if not im_m:
                        continue
                    gid = im_m.group(1)
                    tcode = im_m.group(2).upper()
                    scene_id = f'comofod_{gid}'
                    sp = deterministic_split(scene_id)
                    img_p = os.path.join(root, fn)

                    if tcode == 'O':
                        samples.append(V2Sample(img_p, None, None, 0, 'authentic', 'CoMoFoD', scene_id, sp))
                    else:
                        mp = b_masks.get(gid)
                        if mp:
                            samples.append(V2Sample(img_p, mp, None, 1, 'copy_move', 'CoMoFoD', scene_id, sp))

            # B. DEFACTO Copy-Move & Splicing Taraması (Doğrudan _img klasörlerinde çalış)
            elif ('copymove_img' in rl or 'splicing_' in rl) and rl.endswith('img'):
                is_cm = 'copymove' in rl
                m_type = 'copy_move' if is_cm else 'splicing'
                parent_dir = Path(root).parent.parent
                if is_cm:
                    annot_dir = parent_dir / 'copymove_annotations'
                else:
                    part_prefix = Path(root).parent.name.replace('_img', '')
                    annot_dir = parent_dir / f'{part_prefix}_annotations'

                probe_dir = annot_dir / 'probe_mask'
                donor_dir = annot_dir / 'donor_mask'
                if not probe_dir.exists():
                    continue

                for fn in files:
                    if not fn.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                        continue
                    st = os.path.splitext(fn)[0]
                    parts = st.split('_', 1)
                    coco_id = parts[1] if len(parts) > 1 else st
                    scene_id = f'coco_{coco_id}'
                    sp = deterministic_split(scene_id)

                    pm_path = None
                    for m_ext in ('.jpg', '.png', '.tif'):
                        cand = probe_dir / f'{st}{m_ext}'
                        if cand.exists():
                            pm_path = str(cand)
                            break
                    if not pm_path:
                        continue

                    dm_path = None
                    if is_cm and donor_dir.exists():
                        for m_ext in ('.tif', '.png', '.jpg'):
                            cand_d = donor_dir / f'{st}{m_ext}'
                            if cand_d.exists():
                                dm_path = str(cand_d)
                                break

                    samples.append(V2Sample(os.path.join(root, fn), pm_path, dm_path, 1, m_type, 'DEFACTO', scene_id, sp))

            # C. CASIA v2 Taraması (cross_test)
            elif 'casia' in rl and ('au' in Path(root).name.lower() or 'tp' in Path(root).name.lower()):
                folder_l = Path(root).name.lower()
                for fn in files:
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                        continue
                    st_l = os.path.splitext(fn)[0].lower()
                    img_p = os.path.join(root, fn)
                    if folder_l == 'au' or st_l.startswith('au_'):
                        scene_id = f'casia_{st_l}'
                        samples.append(V2Sample(img_p, None, None, 0, 'authentic', 'CASIA_v2', scene_id, 'cross_test'))
                    elif folder_l == 'tp' or st_l.startswith('tp_'):
                        gt_p = casia_gt_map.get(st_l)
                        if not gt_p:
                            continue
                        m_type = 'splicing' if '_d_' in st_l else 'copy_move'
                        scene_id = f'casia_{st_l}'
                        samples.append(V2Sample(img_p, gt_p, None, 1, m_type, 'CASIA_v2', scene_id, 'cross_test'))

            # D. MS-COCO Temiz Havuz (train2017, val2017, test2017 klasörleri)
            elif ('train2017' in rl or 'val2017' in rl or 'test2017' in rl) and 'defacto' not in rl and 'annot' not in rl:
                for fn in files:
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    st = os.path.splitext(fn)[0]
                    cm = COCO_12_RE.search(st)
                    scene_id = f'coco_{cm.group(1)}' if cm else f'clean_{st}'
                    sp = deterministic_split(scene_id)
                    samples.append(V2Sample(os.path.join(root, fn), None, None, 0, 'authentic', 'COCO_Clean', scene_id, sp))

    # Klasör sırası yanlılığını önlemek için tüm örnek listesini sabit tohumla karıştır
    global_shuffler = random.Random(SEED + 101)
    global_shuffler.shuffle(samples)

    # Sıfır Sahne Sızıntısı Doğrulaması (train, val, calibration, test, cross_test)
    scenes = {k: set() for k in ('train', 'val', 'calibration', 'test', 'cross_test')}
    ds_counts: Dict[str, int] = {}
    coco_scenes = set()
    defacto_scenes = set()
    for s in samples:
        scenes[s.split].add(s.canonical_scene_id)
        ds_counts[s.dataset_name] = ds_counts.get(s.dataset_name, 0) + 1
        if s.dataset_name == 'COCO_Clean':
            coco_scenes.add(s.canonical_scene_id)
        elif s.dataset_name == 'DEFACTO':
            defacto_scenes.add(s.canonical_scene_id)

    for k1 in scenes:
        for k2 in scenes:
            if k1 < k2:
                inter = len(scenes[k1].intersection(scenes[k2]))
                assert inter == 0, f'SIZINTI TESPİT EDİLDİ: {k1} vs {k2} = {inter}'

    shared_coco_defacto = len(coco_scenes.intersection(defacto_scenes))
    print(f'✅ Tarama tamamlandı ({time.time() - t0:.2f} sn) — Toplam: {len(samples)} örnek, Bölmeler Arası Sahne Sızıntısı: 0', flush=True)
    print(f'📊 Veri Seti Dağılımı: {ds_counts}', flush=True)
    print(
        f'🔒 COCO-DEFACTO Ortak Sahne Kilit Kontrolü: COCO ({len(coco_scenes)} sahne) ∩ DEFACTO ({len(defacto_scenes)} sahne) = '
        f'{shared_coco_defacto} ortak kaynak sahne aynı bölmeye kilitlendi (Bölmeler arası sızıntı = 0).',
        flush=True,
    )
    return samples

if IS_KAGGLE:
    SEARCH_ROOTS = [Path('/kaggle/input')]
else:
    SEARCH_ROOTS = [
        Path(r'C:\\Users\\yigit\\Desktop\\Sayısal Görünntü İşleme Veri Seti'),
        Path('data/clean_coco_200'),
    ]

ALL_SAMPLES = scan_all_datasets(SEARCH_ROOTS)
for sp_name in ('train', 'val', 'calibration', 'test', 'cross_test'):
    sub = [s for s in ALL_SAMPLES if s.split == sp_name]
    auth_s = [s for s in sub if s.is_tampered == 0]
    cm_s = [s for s in sub if s.manip_type == 'copy_move']
    sp_s = [s for s in sub if s.manip_type == 'splicing']
    u_auth = len({s.canonical_scene_id for s in auth_s})
    u_cm = len({s.canonical_scene_id for s in cm_s})
    u_sp = len({s.canonical_scene_id for s in sp_s})
    print(
        f'  [{sp_name.upper():11s}] Toplam: {len(sub):6d} | '
        f'Temiz: {len(auth_s):5d} ({u_auth:5d} sahne) | '
        f'Copy-Move: {len(cm_s):5d} ({u_cm:5d} sahne) | '
        f'Splicing: {len(sp_s):5d} ({u_sp:5d} sahne)',
        flush=True,
    )
total_coco_clean = sum(1 for s in ALL_SAMPLES if s.dataset_name == 'COCO_Clean')
if total_coco_clean < 5000:
    print(
        f'⚠️ UYARI: Ortamda yalnızca {total_coco_clean} adet COCO_Clean görseli bulundu! '
        'CoMoFoD temiz havuzu ~200 ana sahnenin varyantlarından oluştuğu için, Kaggle eğitiminde '
        'sağ panelden (+ Add Input) mutlaka "coco-2017-dataset" (train2017/val2017) eklenmelidir!',
        flush=True,
    )
"""


CELL_4_AUGMENT = """# 3. Sabit Gri Letterbox + Geçerli Alan Maskesi (valid_mask) + Tekrarsız Değerlendirme Havuzu + Veri Yükleyici
def safe_imread_color(path_str: str) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(path_str, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path_str, cv2.IMREAD_COLOR)

def safe_imread_gray(path_str: str) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(path_str, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)

def generate_dynamic_copymove(clean_bgr: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Temiz bir görüntüden döndürülmüş, ölçeklenmiş ve kenarı yumuşatılmış Copy-Move + tam maske üretir.\"\"\"
    H, W = clean_bgr.shape[:2]
    area_ratio = rng.uniform(0.02, 0.085)
    side = int(round(math.sqrt(H * W * area_ratio)))
    pw = int(np.clip(side * rng.uniform(0.8, 1.25), 32, max(36, W // 2 - 8)))
    ph = int(np.clip(side * rng.uniform(0.8, 1.25), 32, max(36, H // 2 - 8)))

    sx = rng.randint(4, max(5, W - pw - 4))
    sy = rng.randint(4, max(5, H - ph - 4))
    src_patch = clean_bgr[sy:sy+ph, sx:sx+pw].copy()

    # Düzensiz çokgen maske
    poly_mask = np.zeros((ph, pw), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    base_r = min(ph, pw) * 0.42
    pts = []
    for i in range(9):
        ang = (2.0 * math.pi * i) / 9.0 + rng.uniform(-0.18, 0.18)
        rad = base_r * rng.uniform(0.72, 1.04)
        pts.append([int(np.clip(cx + rad * math.cos(ang), 2, pw - 3)), int(np.clip(cy + rad * math.sin(ang), 2, ph - 3))])
    cv2.fillPoly(poly_mask, [np.array(pts, dtype=np.int32)], 255)

    src_full = np.zeros((H, W), dtype=np.uint8)
    src_full[sy:sy+ph, sx:sx+pw] = poly_mask

    angle = rng.uniform(-22.0, 22.0) if rng.random() < 0.75 else 0.0
    scale = rng.uniform(0.85, 1.18) if rng.random() < 0.75 else 1.0
    rot_m = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, scale)
    warped_patch = cv2.warpAffine(src_patch, rot_m, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.warpAffine(poly_mask, rot_m, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    best_tx, best_ty = 4, 4
    for _ in range(25):
        tx = rng.randint(4, max(5, W - pw - 4))
        ty = rng.randint(4, max(5, H - ph - 4))
        if math.hypot(tx - sx, ty - sy) > max(pw, ph) * 1.1:
            best_tx, best_ty = tx, ty
            break

    alpha = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (5, 5), 0)[:, :, None]
    forged = clean_bgr.copy()
    roi = forged[best_ty:best_ty+ph, best_tx:best_tx+pw].astype(np.float32)
    blended = warped_patch.astype(np.float32) * alpha + roi * (1.0 - alpha)
    forged[best_ty:best_ty+ph, best_tx:best_tx+pw] = np.clip(blended, 0, 255).astype(np.uint8)

    tgt_full = np.zeros((H, W), dtype=np.uint8)
    tgt_full[best_ty:best_ty+ph, best_tx:best_tx+pw] = (warped_mask > 127).astype(np.uint8) * 255
    return forged, np.maximum(src_full, tgt_full)

def letterbox_to_square(
    img: np.ndarray, mask: np.ndarray, target_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"En-boy oranını (aspect ratio) %100 koruyarak görüntüyü SABİT NÖTR GRİ (128) dolgu (BORDER_CONSTANT) ile
    target_size x target_size yapar ve gerçek görüntü bölgesini gösteren 'valid_mask' (255=gerçek, 0=dolgu) döndürür.
    NOT: BORDER_REFLECT_101 (aynalı dolgu) öz-korelasyon modülünde sahte Copy-Move ikizi ürettiği için KESİNLİKLE kullanılmaz!
    \"\"\"
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        valid_full = np.full((target_size, target_size), 255, dtype=np.uint8)
        return img, mask, valid_full
    scale = float(target_size) / float(max(h, w))
    new_w = max(8, min(target_size, int(round(w * scale))))
    new_h = max(8, min(target_size, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img_r = cv2.resize(img, (new_w, new_h), interpolation=interp)
    mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    valid_r = np.full((new_h, new_w), 255, dtype=np.uint8)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    # 8x8 JPEG/DCT blok ızgarasını kaydırmamak için üst/sol dolguyu 8'in katına hizala
    top = ((pad_h // 2) // 8) * 8
    bottom = pad_h - top
    left = ((pad_w // 2) // 8) * 8
    right = pad_w - left

    img_sq = cv2.copyMakeBorder(img_r, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    mask_sq = cv2.copyMakeBorder(mask_r, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    valid_sq = cv2.copyMakeBorder(valid_r, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return img_sq, mask_sq, valid_sq

def standardize_geometry_to_square(
    img: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    rng: random.Random,
    is_train: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"Sıfır Maske Sızıntılı (Zero-Mask-Leakage) ve Sabit Gri Letterbox + valid_mask Boyutlandırma:
    1) Değerlendirmede (is_train=False): Maskeye KESİNLİKLE bakılmaz (kör/blind) ve sabit gri dolgulu
       'letterbox_to_square' + 'valid_mask' kullanılır.
    2) Eğitimde (is_train=True): Kare CoMoFoD görsellerinde 'dolgu yok = CoMoFoD' kısayolu oluşmaması için
       kare görsellerin %35'inden rastgele 4:3 dilim alınarak letterbox uygulanır; ayrıca %50 letterbox,
       %50 rastgele kare kırpma yapılır.
    \"\"\"
    h, w = img.shape[:2]
    if is_train:
        if rng.random() < 0.35:
            rot_k = rng.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, rot_k).copy()
                mask = np.rot90(mask, rot_k).copy()
                h, w = img.shape[:2]

        # Kare (1:1 CoMoFoD) görsellerde 'dolgu yok' veri seti izini kırmak için %35 olasılıkla 4:3 pencere al
        if h == w and h >= 160 and rng.random() < 0.35:
            sub_h = max(128, int(round(h * rng.uniform(0.74, 0.84))))
            y0 = rng.randint(0, h - sub_h)
            cand_m = mask[y0:y0+sub_h, :]
            fg_tot = int((mask > 127).sum())
            if fg_tot == 0 or int((cand_m > 127).sum()) >= max(16, int(0.35 * fg_tot)):
                img = img[y0:y0+sub_h, :]
                mask = cand_m
                h, w = img.shape[:2]

        min_dim = min(h, w)
        if rng.random() < 0.45 and min_dim >= 128:
            crop_side = max(128, min(int(min_dim * rng.uniform(0.85, 1.0)), h, w))
            y0 = rng.randint(0, max(0, h - crop_side))
            x0 = rng.randint(0, max(0, w - crop_side))
            cand_mask = mask[y0:y0+crop_side, x0:x0+crop_side]
            fg_total = int((mask > 127).sum())
            if fg_total == 0 or int((cand_mask > 127).sum()) >= max(16, int(0.30 * fg_total)):
                img_c = img[y0:y0+crop_side, x0:x0+crop_side]
                interp = cv2.INTER_AREA if crop_side >= target_size else cv2.INTER_LINEAR
                valid_full = np.full((target_size, target_size), 255, dtype=np.uint8)
                return (
                    cv2.resize(img_c, (target_size, target_size), interpolation=interp),
                    cv2.resize(cand_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST),
                    valid_full,
                )

    return letterbox_to_square(img, mask, target_size)

def normalize_eval_format(img_bgr: np.ndarray, mode: str = 'jpeg_q90') -> np.ndarray:
    \"\"\"Değerlendirme ve klasik özellik çıkarımında veri seti dosya biçimi kısayolunu
    (COCO=.jpg vs DEFACTO=.tif vs CoMoFoD=.png) sıfırlamak için tüm görselleri aynı standart kanaldan geçirir.
    Ayrıca çoklu format dayanıklılık testinde ('png', 'jpeg_q75', 'whatsapp') kullanılır.
    \"\"\"
    out = img_bgr
    h, w = out.shape[:2]
    if mode == 'png':
        ok, buf = cv2.imencode('.png', out)
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'jpeg_q90':
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'jpeg_q75':
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'whatsapp':
        wa_w, wa_h = max(64, int(w * 0.80)), max(64, int(h * 0.80))
        small = cv2.resize(out, (wa_w, wa_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out

def apply_format_agnostic_chain(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    \"\"\"DİKKAT: Bu fonksiyon görüntü 512x512 boyuta getirildikten SONRA (en son adımda) çağrılır!
    Böylece JPEG 8x8 blok sınırları ve gürültü dokusu modelin SRM/Bayar ve 8x8 DCT kollarına %100 bozulmadan ulaşır.
    \"\"\"
    mode = rng.choices(
        ['lossless_png', 'single_jpeg', 'double_jpeg', 'whatsapp', 'webp', 'noise_jpeg'],
        weights=[0.20, 0.25, 0.20, 0.15, 0.10, 0.10],
        k=1
    )[0]
    out = img_bgr
    h, w = out.shape[:2]
    if mode == 'single_jpeg':
        q = rng.randint(65, 98)
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'double_jpeg':
        for q in (rng.randint(70, 96), rng.randint(65, 95)):
            ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'whatsapp':
        scale_wa = rng.uniform(0.75, 0.90)
        wa_w, wa_h = max(64, int(w * scale_wa)), max(64, int(h * scale_wa))
        small = cv2.resize(out, (wa_w, wa_h), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        q = rng.randint(71, 83)
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'webp':
        q = rng.randint(70, 95)
        ok, buf = cv2.imencode('.webp', out, [int(cv2.IMWRITE_WEBP_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode == 'noise_jpeg':
        sigma = rng.uniform(0.8, 3.0)
        noise = np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        q = rng.randint(70, 92)
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out

class BalancedForensicDataset(Dataset):
    \"\"\"İadesiz Dönemsel Karıştırma (Epoch-Aware Non-Repeating Deck) ile %34 Temiz, %33 Copy-Move ve %33 Splicing örnekleyen veri seti.
    - Değerlendirmede (is_train=False) HAVUZ TEKRARI (wrap-around / slot % len) KESİNLİKLE YOKTUR!
      Her sınıftan en fazla mevcut benzersiz örnek sayısı kadar alınır; hiçbir fotoğraf iki kez değerlendirilmez.
    - Her örnek için (x_tensor, m_tensor, v_tensor, y_tensor, manip_code, is_square_orig) döndürür.
    \"\"\"
    def __init__(
        self,
        samples: List[V2Sample],
        input_size: int = 512,
        is_train: bool = True,
        epoch_size: int = 60000,
        eval_format_mode: str = 'jpeg_q90',
    ):
        self.input_size = input_size
        self.is_train = is_train
        self.eval_format_mode = eval_format_mode
        self.clean_pool = [s for s in samples if s.is_tampered == 0]
        self.cm_pool = [s for s in samples if s.manip_type == 'copy_move']
        self.sp_pool = [s for s in samples if s.manip_type == 'splicing']
        self.all_samples = list(samples)
        self.current_epoch = 0

        # TÜM bölmelerde (train, val, calibration, test) havuzları deterministik karıştır!
        deck_rng = random.Random(SEED + (777 if is_train else 999) + len(samples))
        deck_rng.shuffle(self.clean_pool)
        deck_rng.shuffle(self.cm_pool)
        deck_rng.shuffle(self.sp_pool)
        deck_rng.shuffle(self.all_samples)

        self.eval_deck: List[V2Sample] = []
        if not self.is_train:
            # Değerlendirmede (val/cal/test/cross_test) wrap-around (tekrarlama) OLMADAN dengeli liste oluştur!
            non_empty = [len(p) for p in (self.clean_pool, self.cm_pool, self.sp_pool) if len(p) > 0]
            if len(non_empty) == 3:
                per_b = min(max(1, epoch_size // 3), min(non_empty))
                for i in range(per_b):
                    self.eval_deck.append(self.clean_pool[i])
                    self.eval_deck.append(self.cm_pool[i])
                    self.eval_deck.append(self.sp_pool[i])
            else:
                self.eval_deck = self.all_samples[:min(epoch_size, len(self.all_samples))]
            self.epoch_size = len(self.eval_deck)
        else:
            self.epoch_size = epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def _pick_sample(self, idx: int, rng: random.Random) -> Tuple[V2Sample, bool]:
        if not self.is_train:
            return self.eval_deck[idx], False

        bucket = idx % 3
        slot = idx // 3
        per_bucket_epoch = max(1, self.epoch_size // 3)
        global_cursor = self.current_epoch * per_bucket_epoch + slot

        if bucket == 0 and self.clean_pool:
            return self.clean_pool[global_cursor % len(self.clean_pool)], False
        elif bucket == 1:
            if self.cm_pool and ((global_cursor % 10) < 7):
                real_cm_cursor = (global_cursor // 10) * 7 + (global_cursor % 10)
                return self.cm_pool[real_cm_cursor % len(self.cm_pool)], False
            elif self.clean_pool:
                synth_cursor = (global_cursor // 10) * 3 + max(0, (global_cursor % 10) - 7)
                synth_idx = (synth_cursor + len(self.clean_pool) // 2) % len(self.clean_pool)
                return self.clean_pool[synth_idx], True
            return self.all_samples[global_cursor % len(self.all_samples)], False
        else:
            if self.sp_pool:
                return self.sp_pool[global_cursor % len(self.sp_pool)], False
            return self.all_samples[global_cursor % len(self.all_samples)], False

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(SEED + self.current_epoch * 100003 + idx)
        sample, make_synth_cm = self._pick_sample(idx, rng)

        img = safe_imread_color(sample.image_path)
        if img is None:
            img = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
            mask = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
            is_tamp = 0.0
            manip_code = 0  # 0: Temiz, 1: Copy-Move, 2: Splicing
            is_square_orig = 1
        else:
            is_square_orig = 1 if (img.shape[0] == img.shape[1]) else 0
            if make_synth_cm and img.shape[0] >= 128 and img.shape[1] >= 128:
                img, mask = generate_dynamic_copymove(img, rng)
                is_tamp = 1.0
                manip_code = 1
            else:
                is_tamp = float(sample.is_tampered)
                if sample.is_tampered == 0:
                    manip_code = 0
                elif sample.manip_type == 'copy_move':
                    manip_code = 1
                else:
                    manip_code = 2

                if sample.is_tampered == 0 or not sample.mask_path:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                else:
                    m_tgt = safe_imread_gray(sample.mask_path)
                    if m_tgt is None:
                        mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    else:
                        bin_t = (m_tgt > 127) if m_tgt.max() > 127 else (m_tgt > 0)
                        if sample.source_mask_path:
                            m_src = safe_imread_gray(sample.source_mask_path)
                            if m_src is not None and m_src.shape == m_tgt.shape:
                                bin_s = (m_src > 127) if m_src.max() > 127 else (m_src > 0)
                                bin_t = bin_t | bin_s
                        mask = bin_t.astype(np.uint8) * 255
                        if mask.shape[:2] != img.shape[:2]:
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ADIM 1: Sabit gri letterbox (+ valid_mask) veya rastgele kırpma ile 512x512 kare boyuta getir
        img_resized, mask_resized, valid_resized = standardize_geometry_to_square(
            img, mask, target_size=self.input_size, rng=rng, is_train=self.is_train
        )

        # ADIM 2: Simetrik aynalama (Flip)
        if self.is_train:
            if rng.random() < 0.5:
                img_resized = cv2.flip(img_resized, 1)
                mask_resized = cv2.flip(mask_resized, 1)
                valid_resized = cv2.flip(valid_resized, 1)
            if rng.random() < 0.25:
                img_resized = cv2.flip(img_resized, 0)
                mask_resized = cv2.flip(mask_resized, 0)
                valid_resized = cv2.flip(valid_resized, 0)

        # ADIM 3: En son sıkıştırma zincirini uygula ve gri dolgu bölgesini (valid_resized == 0) tekrar sabit 128 yap
        if self.is_train:
            img_resized = apply_format_agnostic_chain(img_resized, rng)
        elif self.eval_format_mode:
            img_resized = normalize_eval_format(img_resized, mode=self.eval_format_mode)

        if (valid_resized == 0).any():
            img_resized[valid_resized == 0] = 128

        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        m_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32)).unsqueeze(0).contiguous()
        v_tensor = torch.from_numpy((valid_resized > 127).astype(np.float32)).unsqueeze(0).contiguous()
        y_tensor = torch.tensor(is_tamp, dtype=torch.float32)
        manip_code_tensor = torch.tensor(manip_code, dtype=torch.long)
        is_sq_tensor = torch.tensor(is_square_orig, dtype=torch.long)
        return x_tensor, m_tensor, v_tensor, y_tensor, manip_code_tensor, is_sq_tensor
"""


CELL_5_MODEL = """# 4. ForensiVisionV2Net Mimarisi (SwinV2 + SRM/Bayar Gürültü + Öz-Korelasyon) ve Çok Görevli Kayıp
from forensivision.models import ForensiVisionV2Net, ForensiVisionV2Loss, ForensiVisionV2Output
print('✅ ForensiVisionV2Net ve ForensiVisionV2Loss başarıyla yüklendi.', flush=True)
"""


def _read_model_and_loss_source() -> str:
    """Kaggle'da `src/forensivision` paketi yüklü olmasa bile çalışması için model ve kayıp kodunu gömer."""
    m_path = Path("src/forensivision/models/forensivision_v2.py")
    l_path = Path("src/forensivision/models/losses.py")
    m_code = m_path.read_text(encoding="utf-8")
    l_code = l_path.read_text(encoding="utf-8")
    # İç importları temizle
    m_code = m_code.replace("from __future__ import annotations", "")
    l_code = l_code.replace("from __future__ import annotations", "")
    l_code = l_code.replace("from forensivision.models.forensivision_v2 import ForensiVisionV2Output", "")
    return f"# 4. ForensiVisionV2Net ve ForensiVisionV2Loss (Kendi Kendine Yeten Tam Mimari)\n{m_code}\n\n{l_code}"


CELL_6_TRAIN = """# 5. Eğitim Döngüsü (Diferansiyel LR + Linear Warmup + Grad Clip + Tür Bazlı Metrikler + Erken Durdurma + 11.5 Saat Sigortası)
def run_training(
    train_samples: List[V2Sample],
    val_samples: List[V2Sample],
    input_size: int = 512,
    epochs: int = 20,
    batch_size: int = 12,
    train_epoch_size: int = 60000,
    val_max_samples: int = 2400,
    lr: float = 1.5e-4,
    pretrained: bool = True,
    max_session_hours: float = 11.5,
    early_stopping_patience: int = 4,
) -> nn.Module:
    session_start_time = time.time()
    max_session_seconds = max_session_hours * 3600.0

    train_ds = BalancedForensicDataset(train_samples, input_size=input_size, is_train=True, epoch_size=train_epoch_size)
    val_ds = BalancedForensicDataset(val_samples, input_size=input_size, is_train=False, epoch_size=min(val_max_samples, len(val_samples)))

    effective_batch_size = batch_size
    # Kaggle Python 3.13 / PyTorch AMP DataParallel CUDA misaligned address çökmesini önlemek için:
    # Kararlı tek-GPU veya sıfır-kilitli num_workers=2
    num_workers = 2 if IS_KAGGLE else 0
    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    model = ForensiVisionV2Net(pretrained_backbone=pretrained).to(DEVICE)
    base_model = model

    criterion = ForensiVisionV2Loss().to(DEVICE)

    # Diferansiyel Öğrenme Oranı: Ön-eğitimli SwinV2 omurgası için düşük LR (lr * 0.2 = 3e-5),
    # sıfırdan öğrenen SRM/Bayar/DCT, Öz-Korelasyon ve Kod Çözücü başlıklar için tam LR (1.5e-4)
    backbone_params = list(base_model.rgb_features.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in base_model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': lr * 0.2},
            {'params': head_params, 'lr': lr},
        ],
        weight_decay=1e-4,
    )

    # 1 Epoch Doğrusal Isınma (Linear Warmup) + Kosinüs Soğutma (Cosine Annealing)
    if epochs > 1:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.15, end_factor=1.0, total_iters=1)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[1])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    ckpt_path = OUTPUT_DIR / 'forensivision_v2_checkpoint.pt'
    best_path = OUTPUT_DIR / 'forensivision_v2_best.pt'
    start_epoch = 0
    best_val_score = -1.0
    epochs_without_improvement = 0

    # Önceki oturumdan veya Kaggle input'a yüklenen checkpoint'ten devam etme kontrolü
    resume_candidates = [ckpt_path]
    if IS_KAGGLE:
        for r_root, _, r_files in os.walk('/kaggle/input'):
            if 'forensivision_v2_checkpoint.pt' in r_files:
                resume_candidates.append(Path(r_root) / 'forensivision_v2_checkpoint.pt')

    for cand_ckpt in resume_candidates:
        if cand_ckpt.exists():
            print(f'🔄 Mevcut checkpoint bulundu, kaldığı yerden devam ediliyor: {cand_ckpt}', flush=True)
            ckpt = torch.load(cand_ckpt, map_location=DEVICE)
            base_model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_val_score = ckpt.get('best_val_score', -1.0)
            epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
            break

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        elapsed_sec = time.time() - session_start_time
        if elapsed_sec >= max_session_seconds:
            print(f'⏱️ 11.5 Saatlik Güvenlik Sigortası devreye girdi ({elapsed_sec/3600:.2f} saat). Oturum güvenle kapatılıyor!', flush=True)
            break

        t_ep = time.time()
        model.train()
        running_loss = 0.0
        steps = 0
        early_session_stop = False

        for step_idx, (x_b, m_b, v_b, y_b, _, _) in enumerate(train_loader):
            x_b = x_b.to(DEVICE, non_blocking=True)
            m_b = m_b.to(DEVICE, non_blocking=True)
            v_b = v_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(x_b, valid_mask=v_b)
                loss, metrics = criterion(out, m_b, y_b, pixel_valid_mask=v_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())
            steps += 1
            if (step_idx + 1) % max(1, len(train_loader) // 5) == 0:
                el_h = (time.time() - session_start_time) / 3600.0
                cur_lr = optimizer.param_groups[-1]['lr']
                print(
                    f'  Epoch [{epoch+1}/{epochs}] Adım [{step_idx+1}/{len(train_loader)}] '
                    f'Ort. Kayıp: {running_loss/steps:.4f} | LR: {cur_lr:.2e} | Süre: {el_h:.2f} sa',
                    flush=True,
                )

            if (time.time() - session_start_time) >= max_session_seconds:
                print('⏱️ Adım ortasında oturum süre sınırına ulaşıldı, doğrulama ve kayıt aşamasına geçiliyor...', flush=True)
                early_session_stop = True
                break

        scheduler.step()

        # Tür Bazlı Doğrulama (Validation: Temiz, Copy-Move, Splicing + Kare/Kare-Olmayan Temiz Yanlış Alarm Oranı)
        model.eval()
        val_preds, val_targets, val_mtypes, val_is_sq = [], [], [], []
        cm_ious, sp_ious = [], []
        with torch.no_grad():
            for x_b, m_b, v_b, y_b, mc_b, sq_b in val_loader:
                x_b = x_b.to(DEVICE, non_blocking=True)
                m_b = m_b.to(DEVICE, non_blocking=True)
                v_b = v_b.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    out = model(x_b, valid_mask=v_b)
                probs_cls = torch.sigmoid(out.cls_logits).view(-1).cpu().numpy()
                val_preds.extend(probs_cls.tolist())
                val_targets.extend(y_b.numpy().tolist())
                val_mtypes.extend(mc_b.numpy().tolist())
                val_is_sq.extend(sq_b.numpy().tolist())

                probs_m = (torch.sigmoid(out.mask_logits) >= 0.5).float() * v_b
                inter = (probs_m * m_b).sum(dim=(1, 2, 3))
                union = (probs_m + m_b).clamp(0, 1).sum(dim=(1, 2, 3))
                for u_val, i_val, mc_val in zip(union.cpu().numpy(), inter.cpu().numpy(), mc_b.numpy()):
                    if mc_val == 1 and u_val > 0:
                        cm_ious.append(float(i_val / (u_val + 1e-6)))
                    elif mc_val == 2 and u_val > 0:
                        sp_ious.append(float(i_val / (u_val + 1e-6)))

        val_preds_arr = np.array(val_preds)
        val_mtypes_arr = np.array(val_mtypes)
        val_sq_arr = np.array(val_is_sq)

        clean_mask = (val_mtypes_arr == 0)
        cm_mask = (val_mtypes_arr == 1)
        sp_mask = (val_mtypes_arr == 2)
        tamp_mask = (val_mtypes_arr > 0)

        clean_spec = float((val_preds_arr[clean_mask] < 0.5).mean()) if clean_mask.any() else 0.0
        cm_rec = float((val_preds_arr[cm_mask] >= 0.5).mean()) if cm_mask.any() else 0.0
        sp_rec = float((val_preds_arr[sp_mask] >= 0.5).mean()) if sp_mask.any() else 0.0
        tamp_rec = float((val_preds_arr[tamp_mask] >= 0.5).mean()) if tamp_mask.any() else 0.0

        # Kare (1:1) vs Kare-Olmayan (4:3 vb. sabit gri dolgulu) temiz görsellerde yanlış alarm (False Positive) kontrolü
        sq_clean_mask = clean_mask & (val_sq_arr == 1)
        nonsq_clean_mask = clean_mask & (val_sq_arr == 0)
        fp_sq = float((val_preds_arr[sq_clean_mask] >= 0.5).mean()) if sq_clean_mask.any() else 0.0
        fp_nonsq = float((val_preds_arr[nonsq_clean_mask] >= 0.5).mean()) if nonsq_clean_mask.any() else 0.0

        cm_iou = float(np.mean(cm_ious)) if cm_ious else 0.0
        sp_iou = float(np.mean(sp_ious)) if sp_ious else 0.0
        mean_iou = float(np.mean(cm_ious + sp_ious)) if (cm_ious or sp_ious) else 0.0

        min_class_acc = min(clean_spec, cm_rec, sp_rec)
        composite_score = (
            0.30 * clean_spec
            + 0.25 * cm_rec
            + 0.20 * sp_rec
            + 0.15 * mean_iou
            + 0.10 * min_class_acc
        )

        print(
            f'🏁 Epoch [{epoch+1}/{epochs}] ({time.time()-t_ep:.1f}s) | '
            f'Temiz (Spec): %{clean_spec*100:.1f} [Yanlış Alarm Kare:%{fp_sq*100:.1f} / Dolgulu:%{fp_nonsq*100:.1f}] | '
            f'Copy-Move (Rec): %{cm_rec*100:.1f} (IoU:{cm_iou:.3f}) | '
            f'Splicing (Rec): %{sp_rec*100:.1f} (IoU:{sp_iou:.3f}) | '
            f'Skor: {composite_score:.4f}',
            flush=True,
        )

        # 3. Epoch Pilot Sağlık Kontrolü
        if epoch == 2:
            print(
                f'🩺 [3. EPOCH PİLOT KONTROLÜ] Temiz=%{clean_spec*100:.1f}, '
                f'Copy-Move=%{cm_rec*100:.1f}, Splicing=%{sp_rec*100:.1f} — '
                + ('Tüm sınıflar dengeli ilerliyor!' if min_class_acc >= 0.55 else 'Dikkat: En düşük sınıf yakından izleniyor.'),
                flush=True,
            )

        if composite_score > best_val_score + 1e-4:
            best_val_score = composite_score
            epochs_without_improvement = 0
            torch.save(base_model.state_dict(), best_path)
            print(f'   ⭐ Yeni en iyi model kaydedildi: {best_path} (Skor: {best_val_score:.4f})', flush=True)
        else:
            epochs_without_improvement += 1
            print(f'   ℹ️ İyileşme olmayan ardışık epoch: {epochs_without_improvement}/{early_stopping_patience}', flush=True)

        state = {
            'epoch': epoch,
            'model_state': base_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_score': best_val_score,
            'epochs_without_improvement': epochs_without_improvement,
        }
        torch.save(state, ckpt_path)

        if epoch >= 5 and epochs_without_improvement >= early_stopping_patience:
            print(
                f'🛑 Erken Durdurma (Early Stopping): {early_stopping_patience} epoch boyunca gelişme olmadı. '
                f'En iyi ağırlıklar ({best_path}) yükleniyor...',
                flush=True,
            )
            break

        if early_session_stop:
            break

    if best_path.exists():
        base_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return base_model
"""


CELL_7_RUN = """# 6. Eğitimi Başlat (PILOT_MODE = True Varsayılan: Önce 3 Epoch Hızlı Pilot, Sonra İstenirse 12 Epoch Tam Eğitim)
# İnceleme önerisi doğrultusunda: Varsayılan olarak PILOT_MODE = True (3 Epoch) açıktır; böylece tüm GPU kotasını
# harcamadan önce gerçek Kaggle verisinde tür bazlı skorları ve kısayol teşhisini doğrularsınız. Tam eğitim için False yapın.
PILOT_MODE = True
TARGET_EPOCHS = 3 if PILOT_MODE else 12

TRAIN_SAMPLES = [s for s in ALL_SAMPLES if s.split == 'train']
VAL_SAMPLES = [s for s in ALL_SAMPLES if s.split == 'val']
CAL_SAMPLES = [s for s in ALL_SAMPLES if s.split == 'calibration']
TEST_SAMPLES = [s for s in ALL_SAMPLES if s.split == 'test']
CROSS_TEST_SAMPLES = [s for s in ALL_SAMPLES if s.split == 'cross_test']

if IS_KAGGLE:
    trained_model = run_training(
        TRAIN_SAMPLES,
        VAL_SAMPLES,
        input_size=512,
        epochs=TARGET_EPOCHS,
        batch_size=12,
        train_epoch_size=60000,
        val_max_samples=2400,
        lr=1.5e-4,
        pretrained=True,
        max_session_hours=5.5 if PILOT_MODE else 11.0,
        early_stopping_patience=4,
    )
else:
    # Yerel CPU'da yalnızca kodun uçtan uca hatasız çalıştığını doğrulayan teknik duman testi (istatistiksel kanıt değildir)
    trained_model = run_training(
        TRAIN_SAMPLES,
        VAL_SAMPLES,
        input_size=256,
        epochs=1,
        batch_size=4,
        train_epoch_size=16,
        val_max_samples=12,
        lr=1e-4,
        pretrained=False,
        max_session_hours=1.0,
        early_stopping_patience=4,
    )
""".lstrip()


CELL_8_CALIBRATE = """# 7. Gerçek 16-Sinyal Hibrit İkili Doğrulama + Kısayol Teşhis Testi + Çoklu Format (PNG/JPEG75/WhatsApp) Dayanıklılık Raporu
# 1) Klasik sinyaller (ELA, Gürültü, Blokluluk, SIFT, DCT) çıkarılmadan önce TÜM görseller sabit bir 'normalize_eval_format'
#    (varsayılan: JPEG Q=90) zincirinden geçirilir; böylece COCO (.jpg) vs DEFACTO (.tif) dosya uzantısı kısayolu sıfırlanır!
# 2) GBDT yalnızca 14-16. özelliklerle (ELA, Gürültü, Blokluluk) eğitilerek 'Kısayol Teşhis Testi' (Shortcut Diagnostic) raporlanır.
# 3) Klasik, Sadece DL ve Hibrit yöntemlerin ÜÇÜ DE X_cal_thresh üzerinde %93 Temiz Özgüllük hedefine göre adil kalibre edilir.
# 4) Test kümesi PNG, JPEG Q=75 ve WhatsApp varyantlarıyla yeniden kodlanarak format dayanıklılığı tür bazlı ölçülür.

def load_best_model_if_needed() -> nn.Module:
    if 'trained_model' in globals() and isinstance(globals()['trained_model'], nn.Module):
        return globals()['trained_model']
    print('🔄 Bellek içi model bulunamadı, kaydedilmiş en iyi ağırlıklar (forensivision_v2_best.pt) aranıyor...', flush=True)
    m = ForensiVisionV2Net(pretrained_backbone=False).to(DEVICE)
    candidates = [OUTPUT_DIR / 'forensivision_v2_best.pt', OUTPUT_DIR / 'forensivision_v2_checkpoint.pt']
    if IS_KAGGLE:
        for r_root, _, r_files in os.walk('/kaggle/input'):
            for fn in ('forensivision_v2_best.pt', 'forensivision_v2_checkpoint.pt'):
                if fn in r_files:
                    candidates.append(Path(r_root) / fn)
    for c_path in candidates:
        if c_path.exists():
            obj = torch.load(c_path, map_location=DEVICE)
            state_d = obj['model_state'] if (isinstance(obj, dict) and 'model_state' in obj) else obj
            m.load_state_dict(state_d)
            print(f'✅ Ağırlıklar başarıyla yüklendi: {c_path}', flush=True)
            return m
    raise FileNotFoundError('Hiçbir eğitilmiş model ağırlığı (forensivision_v2_best.pt) bulunamadı!')

def classical_sift_ransac_inliers(bgr_u8: np.ndarray) -> float:
    \"\"\"Klasik Bağımsız Algoritma 1: Sabit kanaldan geçirilmiş görsel üzerinde SIFT + Lowe Oran Testi + RANSAC Affine Inliers.\"\"\"
    h, w = bgr_u8.shape[:2]
    if max(h, w) > 640:
        sc = 640.0 / float(max(h, w))
        bgr_u8 = cv2.resize(bgr_u8, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(nfeatures=650, contrastThreshold=0.025, edgeThreshold=12)
        kps, descs = sift.detectAndCompute(gray, None)
    except Exception:
        return 0.0
    if descs is None or kps is None or len(kps) < 8:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(descs, descs, k=3)
    src_pts, dst_pts = [], []
    for m_list in raw_matches:
        if len(m_list) < 3:
            continue
        q_idx = m_list[0].queryIdx
        p_q = np.array(kps[q_idx].pt, dtype=np.float32)
        valid = []
        for cand in m_list:
            if cand.trainIdx == q_idx:
                continue
            p_t = np.array(kps[cand.trainIdx].pt, dtype=np.float32)
            if float(np.linalg.norm(p_q - p_t)) >= 20.0:
                valid.append((cand, p_t))
        if len(valid) >= 2 and valid[0][0].distance < 0.78 * valid[1][0].distance:
            src_pts.append(p_q)
            dst_pts.append(valid[0][1])

    if len(src_pts) < 4:
        return 0.0
    src_arr = np.array(src_pts, dtype=np.float32)
    dst_arr = np.array(dst_pts, dtype=np.float32)
    try:
        _, inliers = cv2.estimateAffinePartial2D(src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=600)
        if inliers is None:
            return 0.0
        n_in = int(inliers.sum())
        return float(min(n_in / 14.0, 1.0)) if n_in >= 4 else 0.0
    except Exception:
        return 0.0

# Vektörize 8x8 DCT bazlarını bir kez oluştur (Python for-döngüsü olmadan <3 ms/görsel hız için!)
_FAST_DCT_BASES = DCTSpectralExtractor._build_8x8_dct_bases().to(DEVICE)

@torch.no_grad()
def classical_dense_dct_copymove(bgr_u8: np.ndarray) -> float:
    \"\"\"Klasik Bağımsız Algoritma 2: Tam Vektörize (F.conv2d) Yoğun 8x8 DCT Blok Eşleştirme + 2B Hough Öteleme Oylaması.\"\"\"
    h, w = bgr_u8.shape[:2]
    if max(h, w) > 384:
        sc = 384.0 / float(max(h, w))
        bgr_u8 = cv2.resize(bgr_u8, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    stride = 6
    t_gray = torch.from_numpy(gray).to(DEVICE).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    dct_feat = F.conv2d(t_gray, _FAST_DCT_BASES, stride=stride)
    _, c_dim, gh, gw = dct_feat.shape
    if gh * gw < 32:
        return 0.0

    energy = (dct_feat ** 2).sum(dim=1).view(-1)  # (N,)
    valid_mask = (energy >= 18.0)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    if valid_idx.numel() < 24:
        return 0.0
    if valid_idx.numel() > 900:
        _, top_e_idx = torch.topk(energy[valid_idx], k=900)
        valid_idx = valid_idx[top_e_idx]

    flat_feat = dct_feat.view(c_dim, -1).transpose(0, 1)[valid_idx]  # (M, 15)
    flat_feat = F.normalize(flat_feat, p=2, dim=1)

    ys = (valid_idx // gw).float() * stride
    xs = (valid_idx % gw).float() * stride
    dy = (ys.unsqueeze(1) - ys.unsqueeze(0)).abs()
    dx = (xs.unsqueeze(1) - xs.unsqueeze(0)).abs()
    dist_ok = (dy > 18.0) | (dx > 18.0)

    sim_mat = torch.mm(flat_feat, flat_feat.transpose(0, 1))
    sim_mat = sim_mat.masked_fill(~dist_ok, -1.0)

    top2_vals, top2_idx = torch.topk(sim_mat, k=2, dim=1)
    s1 = top2_vals[:, 0]
    s2 = top2_vals[:, 1]
    best_j = top2_idx[:, 0]

    good = (s1 >= 0.965) & ((s1 - s2) >= 0.015)
    if int(good.sum().item()) < 8:
        return 0.0

    g_idx = torch.nonzero(good, as_tuple=False).view(-1)
    m_idx = best_j[g_idx]
    shift_y = torch.round((ys[m_idx] - ys[g_idx]) / float(stride)).to(torch.int32)
    shift_x = torch.round((xs[m_idx] - xs[g_idx]) / float(stride)).to(torch.int32)
    flip = (shift_y < 0) | ((shift_y == 0) & (shift_x < 0))
    shift_y = torch.where(flip, -shift_y, shift_y)
    shift_x = torch.where(flip, -shift_x, shift_x)

    shifts_np = torch.stack([shift_y, shift_x], dim=1).cpu().numpy()
    _, counts = np.unique(shifts_np, axis=0, return_counts=True)
    max_cluster = int(counts.max()) if len(counts) > 0 else 0
    return float(min(max_cluster / 16.0, 1.0)) if max_cluster >= 7 else 0.0

def classical_ela_and_noise_signals(norm_bgr_u8: np.ndarray, mask_prob_2d: np.ndarray) -> Tuple[float, float, float, float]:
    \"\"\"Klasik Bağımsız Algoritma 3: Sabit JPEG Q=90 zincirinden geçirilmiş görsel üzerinde
    bölgesel (iç/dış) ELA oranı, Gürültü Farkı, 8x8 Blokluluk ve Entropi.
    \"\"\"
    nh, nw = norm_bgr_u8.shape[:2]
    if mask_prob_2d.shape[:2] != (nh, nw):
        mask_prob_2d = cv2.resize(mask_prob_2d, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # 1. ELA (Error Level Analysis) Q=85 yeniden sıkıştırma farkı
    ok, buf = cv2.imencode('.jpg', norm_bgr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        recomp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ela_map = np.abs(norm_bgr_u8.astype(np.float32) - recomp.astype(np.float32)).mean(axis=2)
    else:
        ela_map = np.zeros((nh, nw), dtype=np.float32)

    # 2. Yüksek geçiren gürültü kalıntısı (Laplacian)
    gray = cv2.cvtColor(norm_bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    fg = (mask_prob_2d >= 0.35)
    bg = ~fg
    if int(fg.sum()) >= 24 and int(bg.sum()) >= 24:
        ela_ratio = float((ela_map[fg].mean() + 0.5) / (ela_map[bg].mean() + 0.5))
        noise_diff = float(abs(lap[fg].mean() - lap[bg].mean()) / (lap.mean() + 1.0))
    else:
        ela_ratio = float(np.percentile(ela_map, 95) / (ela_map.mean() + 0.5))
        noise_diff = float(np.std(lap) / (lap.mean() + 1.0))

    # 3. 8x8 JPEG ızgara blokluluğu
    dh = np.abs(gray[7:-1:8, :] - gray[8::8, :]).mean() if gray.shape[0] > 16 else 0.0
    in_h = np.abs(gray[3:-5:8, :] - gray[4:-4:8, :]).mean() if gray.shape[0] > 16 else 1.0
    blockiness = float(dh / (in_h + 1e-4))

    # 4. Doku entropisi
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [64], [0, 256]).ravel()
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)) / 6.0)
    return ela_ratio, noise_diff, blockiness, entropy

@torch.no_grad()
def extract_16_hybrid_features(
    model: nn.Module,
    samples: List[V2Sample],
    max_count: int = 2400,
    input_size: int = 512,
    eval_format_mode: str = 'jpeg_q90',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"Hem Derin Öğrenme (10 sinyal, TTA + valid_mask dahil) hem de Sabit Kanaldan Geçirilmiş Klasik Algoritmalar (6 sinyal)
    olmak üzere tam 16 adli özelliği, manipülasyon türünü (0=Temiz, 1=Copy-Move, 2=Splicing) ve kare/dolgulu bayrağını çıkarır.
    \"\"\"
    model.eval()
    ds = BalancedForensicDataset(
        samples,
        input_size=input_size,
        is_train=False,
        epoch_size=min(max_count, len(samples)),
        eval_format_mode=eval_format_mode,
    )
    loader = DataLoader(ds, batch_size=12 if IS_KAGGLE else 4, shuffle=False)
    feats_list, labels_list, mtypes_list, sq_list = [], [], [], []
    global_sample_idx = 0

    for x_b, _, v_b, y_b, mc_b, sq_b in loader:
        x_b = x_b.to(DEVICE)
        v_b_dev = v_b.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            out = model(x_b, valid_mask=v_b_dev)
            out_flip = model(torch.flip(x_b, dims=[3]), valid_mask=torch.flip(v_b_dev, dims=[3]))

        prob_m = 0.5 * (torch.sigmoid(out.mask_logits) + torch.flip(torch.sigmoid(out_flip.mask_logits), dims=[3])) * v_b_dev
        conf_m = 0.5 * (torch.sigmoid(out.confidence_logits) + torch.flip(torch.sigmoid(out_flip.confidence_logits), dims=[3])) * v_b_dev
        edge_m = 0.5 * (torch.sigmoid(out.edge_logits) + torch.flip(torch.sigmoid(out_flip.edge_logits), dims=[3])) * v_b_dev
        corr_m = 0.5 * (out.self_corr_map + torch.flip(out_flip.self_corr_map, dims=[3])) * v_b_dev

        cls_p_global = torch.sigmoid(out.cls_logits).view(-1)
        cls_p_tta = 0.5 * (cls_p_global + torch.sigmoid(out_flip.cls_logits).view(-1))

        w_anom = prob_m * conf_m
        b_sz = x_b.shape[0]
        flat_a = w_anom.view(b_sz, -1)
        k5 = max(1, flat_a.shape[1] // 20)
        top5 = torch.topk(flat_a, k=k5, dim=1)[0].mean(dim=1)

        x_np = (x_b.cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        w_anom_np = w_anom.squeeze(1).cpu().numpy()
        v_np = v_b.squeeze(1).numpy()

        for i in range(b_sz):
            picked_sample, _ = ds._pick_sample(global_sample_idx, random.Random(SEED))
            global_sample_idx += 1
            raw_bgr = safe_imread_color(picked_sample.image_path)
            if raw_bgr is None:
                raw_bgr = cv2.cvtColor(x_np[i], cv2.COLOR_RGB2BGR)

            norm_native_bgr = normalize_eval_format(raw_bgr, mode=eval_format_mode)

            m_i = w_anom_np[i]
            v_i = (v_np[i] >= 0.5)
            valid_denom = max(float(v_i.sum()), 1.0)
            bin_m = ((m_i >= 0.45) & v_i).astype(np.uint8)
            num_l, _, stats, _ = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
            largest_cc = float(stats[1:, cv2.CC_STAT_AREA].max()) / valid_denom if num_l > 1 else 0.0

            sift_score = classical_sift_ransac_inliers(norm_native_bgr)
            dense_cm_score = classical_dense_dct_copymove(norm_native_bgr)
            classical_cm_max = max(sift_score, dense_cm_score)
            ela_r, noise_d, blk_s, ent_s = classical_ela_and_noise_signals(norm_native_bgr, m_i)

            row_16 = np.array([
                float(cls_p_global[i].cpu()),                       # 1. dl_cls_prob_global
                float(cls_p_tta[i].cpu()),                          # 2. dl_cls_prob_tta
                float(m_i[v_i].mean()) if v_i.any() else 0.0,       # 3. dl_weighted_anom_mean (valid_mask içinde)
                float(m_i.max()),                                   # 4. dl_weighted_anom_max
                float(top5[i].cpu()),                               # 5. dl_weighted_anom_top5pct
                float(edge_m[i].amax().cpu()),                      # 6. dl_edge_peak
                float(corr_m[i].amax().cpu()),                      # 7. dl_self_corr_peak
                float((corr_m[i] >= 0.70).float().sum().cpu() / valid_denom), # 8. dl_self_corr_area_ratio
                float(bin_m.sum() / valid_denom),                   # 9. dl_mask_area_ratio
                float(largest_cc),                                  # 10. dl_largest_cc_ratio
                float(sift_score),                                  # 11. classical_sift_ransac_inliers
                float(dense_cm_score),                              # 12. classical_dense_cm_score
                float(classical_cm_max),                            # 13. classical_cm_combined
                float(ela_r),                                       # 14. ela_inside_outside_ratio (format-normalized)
                float(noise_d),                                     # 15. noise_inside_outside_diff (format-normalized)
                float(blk_s * ent_s),                               # 16. jpeg_blockiness_x_entropy (format-normalized)
            ], dtype=np.float32)
            feats_list.append(row_16)

        labels_list.append(y_b.numpy())
        mtypes_list.append(mc_b.numpy())
        sq_list.append(sq_b.numpy())

    return np.vstack(feats_list), np.concatenate(labels_list), np.concatenate(mtypes_list), np.concatenate(sq_list)

def split_calibration_samples_by_scene(cal_samples: List[V2Sample], fit_ratio: float = 0.60) -> Tuple[List[V2Sample], List[V2Sample]]:
    \"\"\"CAL_SAMPLES kümesini özellik çıkarımından ÖNCE canonical_scene_id bazında %60 CAL_FIT ve %40 CAL_THRESH
    olarak iki ayrık sahne havuzuna böler; sıfır sahne kesişimini garanti eder.
    \"\"\"
    scene_to_samples: Dict[str, List[V2Sample]] = {}
    for s in cal_samples:
        scene_to_samples.setdefault(s.canonical_scene_id, []).append(s)

    buckets: Dict[str, List[str]] = {'authentic': [], 'copy_move': [], 'splicing': []}
    for sc_id, s_list in sorted(scene_to_samples.items()):
        mtypes_in_sc = {x.manip_type for x in s_list}
        if 'copy_move' in mtypes_in_sc:
            buckets['copy_move'].append(sc_id)
        elif 'splicing' in mtypes_in_sc:
            buckets['splicing'].append(sc_id)
        else:
            buckets['authentic'].append(sc_id)

    rng = random.Random(SEED + 99)
    fit_scenes, thresh_scenes = set(), set()
    for m_key, sc_list in buckets.items():
        sc_copy = list(sc_list)
        rng.shuffle(sc_copy)
        n_fit = max(1, int(round(len(sc_copy) * fit_ratio))) if len(sc_copy) >= 2 else len(sc_copy)
        fit_scenes.update(sc_copy[:n_fit])
        thresh_scenes.update(sc_copy[n_fit:])

    assert len(fit_scenes & thresh_scenes) == 0, 'HATA: CAL_FIT ve CAL_THRESH arasında sahne sızıntısı!'
    fit_pool = [s for s in cal_samples if s.canonical_scene_id in fit_scenes]
    thresh_pool = [s for s in cal_samples if s.canonical_scene_id in thresh_scenes]
    if len(thresh_pool) == 0:
        thresh_pool = fit_pool
    print(
        f'🔒 Sahne Bazlı Kalibrasyon Bölmesi -> CAL_FIT: {len(fit_pool)} görsel ({len(fit_scenes)} sahne) | '
        f'CAL_THRESH: {len(thresh_pool)} görsel ({len(thresh_scenes)} sahne) | Ortak Sahne: {len(fit_scenes & thresh_scenes)}',
        flush=True,
    )
    return fit_pool, thresh_pool

from sklearn.metrics import roc_auc_score

eval_model = load_best_model_if_needed()
cal_fit_max = 1440 if IS_KAGGLE else 12
cal_thresh_max = 960 if IS_KAGGLE else 12
test_count = 2400 if IS_KAGGLE else 18
inp_sz = 512 if IS_KAGGLE else 256

CAL_FIT_SAMPLES, CAL_THRESH_SAMPLES = split_calibration_samples_by_scene(CAL_SAMPLES, fit_ratio=0.60)

t_cal0 = time.time()
print('🔍 CAL_FIT, CAL_THRESH ve TEST kümeleri için 16 Adli Sinyal (Sabit JPEG Q=90 + Valid Mask) çıkarılıyor...', flush=True)
X_cal_fit, y_cal_fit, mt_cal_fit, sq_cal_fit = extract_16_hybrid_features(
    eval_model, CAL_FIT_SAMPLES, max_count=cal_fit_max, input_size=inp_sz, eval_format_mode='jpeg_q90'
)
X_cal_thresh, y_cal_thresh, mt_cal_thresh, sq_cal_thresh = extract_16_hybrid_features(
    eval_model, CAL_THRESH_SAMPLES, max_count=cal_thresh_max, input_size=inp_sz, eval_format_mode='jpeg_q90'
)
X_test, y_test, mt_test, sq_test = extract_16_hybrid_features(
    eval_model, TEST_SAMPLES, max_count=test_count, input_size=inp_sz, eval_format_mode='jpeg_q90'
)
print(f'⚡ 16-Sinyal çıkarımı tamamlandı ({time.time() - t_cal0:.1f} sn).', flush=True)

# 3A: Hibrit-13 (14-16. özellikler olan ELA, Gürültü, Blokluluk HARİÇ: Yalnızca 10 DL + 3 SIFT/DCT özelliği)
meta_gbdt_13 = HistGradientBoostingClassifier(max_iter=100, max_depth=3, min_samples_leaf=8, l2_regularization=1.0, random_state=SEED)
meta_lr_13 = LogisticRegression(C=0.8, max_iter=300, random_state=SEED)
meta_gbdt_13.fit(X_cal_fit[:, :13], y_cal_fit)
meta_lr_13.fit(X_cal_fit[:, :13], y_cal_fit)

# 3B: Hibrit-16 (Tüm 16 özellik DAHİL)
meta_gbdt = HistGradientBoostingClassifier(max_iter=100, max_depth=3, min_samples_leaf=8, l2_regularization=1.0, random_state=SEED)
meta_lr = LogisticRegression(C=0.8, max_iter=300, random_state=SEED)
meta_gbdt.fit(X_cal_fit, y_cal_fit)
meta_lr.fit(X_cal_fit, y_cal_fit)

import pickle
meta_pkl_path = OUTPUT_DIR / 'forensivision_v2_meta_models.pkl'
with open(meta_pkl_path, 'wb') as f_pkl:
    pickle.dump({'gbdt_16': meta_gbdt, 'logreg_16': meta_lr, 'gbdt_13': meta_gbdt_13, 'logreg_13': meta_lr_13}, f_pkl)

clean_mask_thresh = (y_cal_thresh < 0.5)

thresh_probs_13 = 0.60 * meta_gbdt_13.predict_proba(X_cal_thresh[:, :13])[:, 1] + 0.40 * meta_lr_13.predict_proba(X_cal_thresh[:, :13])[:, 1]
opt_threshold_13 = float(np.clip(np.quantile(thresh_probs_13[clean_mask_thresh], 0.93) + 1e-4, 0.25, 0.90)) if clean_mask_thresh.any() else 0.50

thresh_probs = 0.60 * meta_gbdt.predict_proba(X_cal_thresh)[:, 1] + 0.40 * meta_lr.predict_proba(X_cal_thresh)[:, 1]
opt_threshold = float(np.clip(np.quantile(thresh_probs[clean_mask_thresh], 0.93) + 1e-4, 0.25, 0.90)) if clean_mask_thresh.any() else 0.50

dl_cal_scores = 0.55 * X_cal_thresh[:, 1] + 0.45 * X_cal_thresh[:, 4]
dl_threshold = float(np.clip(np.quantile(dl_cal_scores[clean_mask_thresh], 0.93) + 1e-4, 0.15, 0.90)) if clean_mask_thresh.any() else 0.50

cal_classical_raw = np.maximum(X_cal_thresh[:, 12], np.clip((X_cal_thresh[:, 13] - 1.15) * 0.4 + X_cal_thresh[:, 14] * 0.3, 0.0, 1.0))
classical_threshold = float(np.clip(np.quantile(cal_classical_raw[clean_mask_thresh], 0.93) + 1e-4, 0.15, 0.90)) if clean_mask_thresh.any() else 0.45

def evaluate_predictions(preds_bin: np.ndarray, mtypes: np.ndarray, is_sq: np.ndarray, title: str, probs_opt: Optional[np.ndarray] = None) -> Dict[str, float]:
    c_mask = (mtypes == 0)
    cm_mask = (mtypes == 1)
    sp_mask = (mtypes == 2)
    t_mask = (mtypes > 0)
    c_sq = c_mask & (is_sq == 1)
    c_nsq = c_mask & (is_sq == 0)

    spec = float((preds_bin[c_mask] == 0).mean()) if c_mask.any() else 0.0
    fp_sq = float((preds_bin[c_sq] == 1).mean()) if c_sq.any() else 0.0
    fp_nsq = float((preds_bin[c_nsq] == 1).mean()) if c_nsq.any() else 0.0
    cm_r = float((preds_bin[cm_mask] == 1).mean()) if cm_mask.any() else 0.0
    sp_r = float((preds_bin[sp_mask] == 1).mean()) if sp_mask.any() else 0.0
    t_r = float((preds_bin[t_mask] == 1).mean()) if t_mask.any() else 0.0
    acc = float(((preds_bin == 0) & c_mask | (preds_bin == 1) & t_mask).mean())

    auc_val = 0.5
    if probs_opt is not None and c_mask.any() and t_mask.any():
        try:
            auc_val = float(roc_auc_score(t_mask.astype(np.int32), probs_opt))
        except Exception:
            auc_val = 0.5

    print(
        f'  📌 {title:42s} -> Temiz(Spec): %{spec*100:5.1f} (FP Kare:%{fp_sq*100:4.1f}/Dolgulu:%{fp_nsq*100:4.1f}) | '
        f'CM(Rec): %{cm_r*100:5.1f} | Splice(Rec): %{sp_r*100:5.1f} | ROC-AUC: {auc_val:.3f}',
        flush=True,
    )
    return {
        'clean_specificity': round(spec, 4),
        'clean_fp_square': round(fp_sq, 4),
        'clean_fp_padded': round(fp_nsq, 4),
        'copy_move_recall': round(cm_r, 4),
        'splicing_recall': round(sp_r, 4),
        'tampered_recall': round(t_r, 4),
        'overall_accuracy': round(acc, 4),
        'roc_auc': round(auc_val, 4),
    }

print('\\n🔬 [KISAYOL TEŞHİS TESTİ — SADECE 14-16. ÖZELLİKLER (ELA / GÜRÜLTÜ / BLOKLULUK) — EŞİKTEN BAĞIMSIZ ROC-AUC]', flush=True)
diag_gbdt = HistGradientBoostingClassifier(max_iter=80, max_depth=3, min_samples_leaf=8, random_state=SEED)
diag_gbdt.fit(X_cal_fit[:, 13:16], y_cal_fit)
diag_cal_p = diag_gbdt.predict_proba(X_cal_thresh[:, 13:16])[:, 1]
diag_th = float(np.clip(np.quantile(diag_cal_p[clean_mask_thresh], 0.93) + 1e-4, 0.25, 0.90)) if clean_mask_thresh.any() else 0.50
diag_test_probs = diag_gbdt.predict_proba(X_test[:, 13:16])[:, 1]
diag_test_preds = (diag_test_probs > diag_th).astype(np.int32)
res_shortcut_diag = evaluate_predictions(diag_test_preds, mt_test, sq_test, 'Teşhis: Sadece ELA+Gürültü+Blokluluk', probs_opt=diag_test_probs)

print('\\n📊 [BAĞIMSIZ TEST KÜMESİ — ADİL KALİBRE EDİLMİŞ KARŞILAŞTIRMA (3A: HİBRİT-13 vs 3B: HİBRİT-16)]', flush=True)
test_classical_raw = np.maximum(X_test[:, 12], np.clip((X_test[:, 13] - 1.15) * 0.4 + X_test[:, 14] * 0.3, 0.0, 1.0))
classical_preds = (test_classical_raw > classical_threshold).astype(np.int32)
res_classical = evaluate_predictions(classical_preds, mt_test, sq_test, '1. Sadece Klasik (SIFT+DenseCM+ELA)', probs_opt=test_classical_raw)

dl_test_scores = 0.55 * X_test[:, 1] + 0.45 * X_test[:, 4]
dl_only_preds = (dl_test_scores > dl_threshold).astype(np.int32)
res_dl_only = evaluate_predictions(dl_only_preds, mt_test, sq_test, '2. Sadece Derin Öğrenme (Kalibreli DL)', probs_opt=dl_test_scores)

hybrid_13_probs = 0.60 * meta_gbdt_13.predict_proba(X_test[:, :13])[:, 1] + 0.40 * meta_lr_13.predict_proba(X_test[:, :13])[:, 1]
hybrid_13_preds = (hybrid_13_probs > opt_threshold_13).astype(np.int32)
res_hybrid_13 = evaluate_predictions(hybrid_13_preds, mt_test, sq_test, '3A. Hibrit-13 (DL + SIFT/DCT, ELA Hariç)', probs_opt=hybrid_13_probs)

hybrid_probs = 0.60 * meta_gbdt.predict_proba(X_test)[:, 1] + 0.40 * meta_lr.predict_proba(X_test)[:, 1]
hybrid_preds = (hybrid_probs > opt_threshold).astype(np.int32)
res_hybrid = evaluate_predictions(hybrid_preds, mt_test, sq_test, '3B. Hibrit-16 (Tüm 16 Sinyal Dahil)', probs_opt=hybrid_probs)

print('\\n🛡️ [ÇOKLU FORMAT VE KANAL DAYANIKLILIK TESTİ — PNG vs JPEG Q=75 vs WHATSAPP (Hibrit-13 ile)]', flush=True)
format_robustness_results = {'jpeg_q90_standard': res_hybrid_13}
fmt_test_count = 600 if IS_KAGGLE else 12
for fmt_mode, fmt_label in [
    ('png', 'Format Testi: Kayıpsız PNG'),
    ('jpeg_q75', 'Format Testi: Sıkıştırılmış JPEG Q=75'),
    ('whatsapp', 'Format Testi: WhatsApp (0.8x + Q=75)'),
]:
    X_fmt, _, mt_fmt, sq_fmt = extract_16_hybrid_features(
        eval_model, TEST_SAMPLES, max_count=fmt_test_count, input_size=inp_sz, eval_format_mode=fmt_mode
    )
    fmt_probs = 0.60 * meta_gbdt_13.predict_proba(X_fmt[:, :13])[:, 1] + 0.40 * meta_lr_13.predict_proba(X_fmt[:, :13])[:, 1]
    fmt_preds = (fmt_probs > opt_threshold_13).astype(np.int32)
    format_robustness_results[fmt_mode] = evaluate_predictions(fmt_preds, mt_fmt, sq_fmt, fmt_label, probs_opt=fmt_probs)

res_cross_casia = None
if len(CROSS_TEST_SAMPLES) > 0:
    print(f'\\n🌍 [ZERO-SHOT ÇAPRAZ VERİ SETİ TESTİ — EĞİTİMDE HİÇ GÖRÜLMEYEN CASIA v2 ({len(CROSS_TEST_SAMPLES)} görsel)]', flush=True)
    X_cr, y_cr, mt_cr, sq_cr = extract_16_hybrid_features(
        eval_model, CROSS_TEST_SAMPLES, max_count=min(test_count, len(CROSS_TEST_SAMPLES)), input_size=inp_sz, eval_format_mode='jpeg_q90'
    )
    cr_probs = 0.60 * meta_gbdt_13.predict_proba(X_cr[:, :13])[:, 1] + 0.40 * meta_lr_13.predict_proba(X_cr[:, :13])[:, 1]
    cr_preds = (cr_probs > opt_threshold_13).astype(np.int32)
    res_cross_casia = evaluate_predictions(cr_preds, mt_cr, sq_cr, 'CASIA v2 (Zero-Shot Hibrit-13)', probs_opt=cr_probs)

cal_json_path = OUTPUT_DIR / 'forensivision_v2_calibration.json'
with open(cal_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'optimal_threshold_16': round(opt_threshold, 4),
        'optimal_threshold_13': round(opt_threshold_13, 4),
        'dl_only_threshold': round(dl_threshold, 4),
        'classical_threshold': round(classical_threshold, 4),
        'calibration_fit_samples': int(len(y_cal_fit)),
        'calibration_thresh_samples': int(len(y_cal_thresh)),
        'shortcut_diagnostic_ela_noise_blockiness': res_shortcut_diag,
        'test_classical_only': res_classical,
        'test_dl_only': res_dl_only,
        'test_hybrid_13_no_ela': res_hybrid_13,
        'test_hybrid_16_all': res_hybrid,
        'multi_format_robustness': format_robustness_results,
        'cross_dataset_casia_v2': res_cross_casia,
        'logreg_13_coef': meta_lr_13.coef_.tolist(),
        'logreg_16_coef': meta_lr.coef_.tolist(),
    }, f, indent=2)

print(f'\\n🎯 Sahne-Ayrık Kalibrasyon Eşiği (Hibrit-13: {opt_threshold_13:.4f}, Hibrit-16: {opt_threshold:.4f}) | Rapor: {cal_json_path}', flush=True)
"""


def build_notebook_file(out_path: str | Path = "notebooks/forensivision_v2_kaggle_training.ipynb") -> Path:
    cell_5_full = _read_model_and_loss_source()

    def make_code_cell(src: str) -> dict:
        lines = [line + "\n" for line in src.strip().splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }

    def make_md_cell(src: str) -> dict:
        lines = [line + "\n" for line in src.strip().splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        }

    nb = {
        "cells": [
            make_md_cell(CELL_1_MD),
            make_code_cell(CELL_2_SETUP),
            make_code_cell(CELL_3_INDEXER),
            make_code_cell(CELL_4_AUGMENT),
            make_code_cell(cell_5_full),
            make_code_cell(CELL_6_TRAIN),
            make_code_cell(CELL_7_RUN),
            make_code_cell(CELL_8_CALIBRATE),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    return p


if __name__ == "__main__":
    saved = build_notebook_file()
    print(f"Created Kaggle v2 notebook at: {saved}")
