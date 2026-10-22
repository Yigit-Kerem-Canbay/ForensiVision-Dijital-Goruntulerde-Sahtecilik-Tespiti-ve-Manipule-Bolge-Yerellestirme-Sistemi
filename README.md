# 🔍 ForensiVision
### Dijital Görüntülerde Sahtecilik Tespiti ve Manipüle Bölge Yerelleştirme Sistemi
*(Digital Image Forgery Detection & Manipulated Region Localization System)*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)](#)
[![Field](https://img.shields.io/badge/Domain-Digital%20Forensics-red)](#)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

---

## 📌 Proje Hakkında

**ForensiVision**, dijital görüntüler üzerinde gerçekleştirilen kasıtlı sahtecilik ve tahrifatların (image tampering/forgery) otomatik olarak tespit edilmesi, doğrulanması ve tahrif edilen bölgelerin piksel/bölge düzeyinde yerelleştirilmesi (localization) amacıyla geliştirilmekte olan bir adli görüntü analiz sistemidir.

Günümüzde gelişmiş görüntü işleme yazılımları ve yapay zeka araçları ile manipüle edilen dijital içerikler; adli vakalarda, haber doğrulama süreçlerinde, sigorta ekspertizlerinde ve telif hakkı korumasında ciddi güvenilirlik sorunları yaratmaktadır. Bu sistem; insan gözüyle fark edilemeyen piksel anomalilerini, öznitelik tutarsızlıklarını ve geometrik izleri tespit ederek güvenilir, doğrulanabilir ve ölçülebilir bir kanıt çıktısı üretmeyi hedefler.

---

## 🎯 Kapsam ve Tespit Edilen Manipülasyon Türleri

Sistem öncelikli olarak adli bilişim literatüründeki iki ana tahrifat kategorisine odaklanmaktadır:

| Manipülasyon Türü | Açıklama | Analiz Yaklaşımı |
| :--- | :--- | :--- |
| **Copy-Move (Klonlama)** | Görüntünün kendi içindeki bir alanın kopyalanıp başka bir bölgeye yapıştırılması (çoğunlukla nesne çoğaltma veya silme amacıyla). | Anahtar nokta (keypoint) eşleştirme, blok tabanlı öznitelik analizi, RANSAC ile afinite dönüşüm tutarlılığı. |
| **Image Splicing (Ekleme / Montaj)** | Başka bir bağımsız kaynak görüntüden alınan bir parçanın hedef görüntüye entegre edilmesi. | Kenar/sınır tutarsızlıkları, bölgesel gürültü analizleri (Noise Inconsistency), parlaklık ve aydınlatma anomalileri. |

---

## ⚙️ Yöntem ve Teknik Yaklaşım

ForensiVision, klasik görüntü işleme ve modern bilgisayarla görme tekniklerini çok katmanlı bir analiz boru hattında birleştirmektedir:

```mermaid
flowchart TD
    A[Girdi Görüntüsü] --> B[Ön İşleme & Normalizasyon]
    B --> C[Öznitelik Çıkarımı: SIFT / ORB]
    B --> D[Bölgesel Benzerlik & Tutarsızlık Analizi]
    C --> E[Öznitelik Eşleme & Kümeleme]
    E --> F[RANSAC ile Geometrik Tutarlılık]
    D --> G[Gürültü ve Sıkıştırma Analizi]
    F --> H[Segmentasyon & Morfolojik İşleme]
    G --> H
    H --> I[Manipüle Bölge Maskesi ve Görselleştirme]
    I --> J[Performans ve Dayanıklılık Raporu]
```

1. **Öznitelik Çıkarımı ve Eşleştirme (Feature Extraction & Matching):**
   - Ölçek ve rotasyondan bağımsız öznitelik çıkarımı için **SIFT** (*Scale-Invariant Feature Transform*) ve yüksek işlem hızı gerektiren adımlar için **ORB** (*Oriented FAST and Rotated BRIEF*) algoritmaları.
   - Benzer anahtar noktaların eşlenmesi ve yalancı pozitiflerin filtrelenmesi.

2. **Geometrik Tutarlılık Doğrulaması:**
   - **RANSAC** (*Random Sample Consensus*) algoritması ile eşleşen bölgeler arasındaki geometrik dönüşümlerin (öteleme, rotasyon, ölçekleme) tutarlılığı denetlenir; rastgele/gürültülü eşleşmeler ayıklanır.

3. **Görüntü Tutarsızlığı ve Bölgesel Analiz:**
   - Farklı kaynaklardan birleştirilen parçaların yarattığı yerel gürültü tutarsızlıkları, hata seviyesi analizleri (ELA) ve bölgesel doku farkları incelenir.

4. **Segmentasyon ve Morfolojik İşleme:**
   - Tespit edilen şüpheli noktaların etrafı morfolojik operatörler (genişletme, aşındırma, kapanış) ve segmentasyon teknikleri ile taranarak tahrif edilen bölgenin kesintisiz sınırları belirlenir.

5. **Görsel Çıktı ve Yerelleştirme (Localization):**
   - Şüpheli bölgeler **ikili maske (binary mask)**, **ısı haritası (heat-map)** ve **çerçeveleme (bounding-box/overlay)** görselleştirmeleri ile adli raporlama formatında sunulur.

---

## 📊 Başarım Değerlendirme ve Akademik Metrikler

Sistemin doğruluğu ve başarımı, ground-truth (gerçek manipülasyon maskeleri) bulunan akademik referans veri setleri üzerinde piksel ve nesne bazında test edilecektir:

- **IoU (Intersection over Union / Jaccard Index):** Tespit edilen bölge ile gerçek manipüle bölgenin örtüşme kalitesi.
- **Precision (Kesinlik):** Sistemce sahte olarak işaretlenen alanların gerçekte sahte olma oranı.
- **Recall (Duyarlılık):** Görüntüdeki gerçek manipülasyonların sistem tarafından yakalanma oranı.
- **F1-Score:** Precision ve Recall arasındaki harmonik denge.

---

## 🛡️ Dayanıklılık (Robustness) Testleri

Saldırganlar veya sosyal medya platformları manipülasyon izlerini gizlemek amacıyla ikincil işlemler uygulayabilmektedir. ForensiVision, aşağıdaki deformasyonlar altında test edilecektir:

- 🔄 **Döndürme (Rotation)** (Farklı açılarda döndürme etkileri)
- 🔍 **Ölçekleme (Scaling)** (Büyütme/küçültme operasyonları)
- 🗜️ **JPEG Sıkıştırması (Compression)** (Düşük kalite faktörlerinde sıkıştırma)
- 🌫️ **Bulanıklaştırma (Blurring)** (Gaussian, Median, Motion blur)
- ⚡ **Gürültü Ekleme (Noise Injection)** (Gaussian, Salt & Pepper gürültüleri)

---

## 📁 Proje Dizin Yapısı

```text
ForensiVision/
├── docs/                      # Raporlar, mimari dökümanlar ve analizler
├── src/
│   └── forensivision/         # Ana Python paketi
│       ├── copy_move/         # Copy-Move tespiti ve yerelleştirme (SIFT/ORB + RANSAC)
│       ├── splicing/          # Image Splicing tespiti ve analizi (Gürültü/ELA)
│       ├── evaluation/        # Değerlendirme metrikleri (Precision, Recall, F1, IoU)
│       └── visualization/     # Maske ve adli tespit görselleştirmeleri
├── tests/                     # Birim ve doğrulama testleri (pytest)
├── scripts/                   # Yardımcı veri hazırlama ve toplu çalıştırma betikleri
├── data/
│   ├── raw/                   # Ham test görüntüleri (Git dışı)
│   ├── interim/               # Ara işlenmiş görüntüler/öznitelikler (Git dışı)
│   └── processed/             # Sonuç maskeleri ve test çıktıları (Git dışı)
├── pyproject.toml             # Paketleme ve pytest yapılandırması
├── requirements.txt           # Temel bağımlılıklar
├── .gitignore                 # Git dışlama kuralları
└── README.md                  # Proje ana tanıtım dökümanı
```

---

## 💻 Geliştirme Ortamı ve Kurulum (Windows)

### 1. Sanal Ortamı Etkinleştirme
Windows PowerShell üzerinde:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Komut İstemi / CMD için: `.\.venv\Scripts\activate.bat`)*

> [!TIP]
> Eğer PowerShell üzerinde betik çalıştırma yetki uyarısı (*execution policy*) alırsanız, geçici olarak yetki vermek için:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` komutunu çalıştırabilirsiniz.

### 2. Bağımlılıkların Kurulumu ve Paket Kaydı
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Testlerin Çalıştırılması
Altyapı doğrulama ve smoke testlerini çalıştırmak için:
```bash
pytest
```

---

## 🚀 Geliştirme Süreci ve Yol Haritası

- [x] **Aşama 1 — Proje Altyapısı:** Modüler paket mimarisi, sanal ortam, `.gitignore`, bağımlılık yönetimi ve smoke testleri tamamlandı.
- [x] **Aşama 2 — Değerlendirme Motoru:** Precision, Recall, F1-Score ve IoU piksel maske karşılaştırma testleri tamamlandı.
- [x] **Aşama 3 — Copy-Move Modülü:** SIFT öznitelik çıkarımı, öz-eşleme filtreleme, RANSAC afin doğrulama ve morfolojik bölge maskeleme prototipi tamamlandı.
- [ ] **Aşama 4 — Splicing Modülü:** Gürültü tutarsızlığı (Noise Discrepancy) ve ELA analizleri.
- [ ] **Aşama 5 — Veri Seti ve Kapsamlı Deneyler:** DEFACTO / CASIA üzerinde toplu deneyler ve dayanıklılık analizleri.
- [ ] **Aşama 6 — Arayüz ve Sunum:** Canlı adli analiz ve görsel maskeleme Streamlit arayüzü.


---

## 👨‍💻 Geliştirici & İletişim

**Yiğit Kerem Canbay**  
- Proje: *ForensiVision - Dijital Görüntülerde Sahtecilik Tespiti ve Manipüle Bölge Yerelleştirme Sistemi*  
- GitHub: [@Yigit-Kerem-Canbay](https://github.com/Yigit-Kerem-Canbay)
