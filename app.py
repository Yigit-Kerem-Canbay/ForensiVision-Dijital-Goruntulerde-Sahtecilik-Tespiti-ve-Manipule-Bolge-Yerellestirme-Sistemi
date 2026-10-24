"""
ForensiVision — Dijital Görüntülerde Sahtecilik Tespiti ve Manipüle Bölge Yerelleştirme Sistemi
Streamlit Kullanıcı Arayüzü (Web UI).

Bu uygulama; adli bilişim uzmanları ve araştırmacılar için dijital görüntüler üzerinde
Copy-Move (Klonlama) ve Image Splicing (Ekleme/Montaj) manipülasyonlarını tespit eden,
şüpheli bölgeleri yerelleştiren ve dürüst metrik değerlendirmesi sunan modern bir arayüzdür.

Çalıştırma:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st

from forensivision.copy_move import CopyMoveConfig, CopyMoveResult
from forensivision.splicing import SplicingConfig, SplicingResult
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
from forensivision.visualization import create_mask_overlay, draw_copy_move_matches

# Sayfa Yapılandırması
st.set_page_config(
    page_title="ForensiVision — Adli Görüntü Analizi",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    """Streamlit oturum değişkenlerini başlatır."""
    defaults: dict[str, Any] = {
        "analysis_executed": False,
        "copy_move_result": None,
        "splicing_result": None,
        "copy_move_time": 0.0,
        "splicing_time": 0.0,
        "metrics_copy_move": None,
        "metrics_splicing": None,
        "loaded_image_bgr": None,
        "loaded_image_rgb": None,
        "detected_format": None,
        "loaded_gt_mask": None,
        "last_file_id": None,
        "analysis_type": "both",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    _init_session_state()

    # =========================================================================
    # 1. Üst Başlık ve Akademik Bilgilendirme
    # =========================================================================
    st.title("🔍 ForensiVision")
    st.subheader("Dijital Görüntülerde Sahtecilik Tespiti ve Manipüle Bölge Yerelleştirme")

    st.markdown(
        """
        > ⚖️ **Akademik & Adli Uyarı:** Bu sistem, bir yazılım mühendisliği bitirme projesi kapsamında geliştirilmiş 
        > **deneysel bir adli analiz prototipidir**. Elde edilen haritalar ve maskeler kesin bir sahtecilik hükmü 
        > oluşturmaz; uzman incelemesine yardımcı olan piksel, doku ve frekans tutarsızlıklarıdır.
        """
    )

    # =========================================================================
    # 2. Kenar Çubuğu (Sidebar) — Mod ve Yapılandırma
    # =========================================================================
    st.sidebar.header("⚙️ Analiz Ayarları")

    mode = st.sidebar.radio(
        "Çalışma Modu:",
        options=["Canlı Görsel Yükle", "Sentetik Demo Modu"],
        help="Kendi görselinizi yükleyebilir veya veri setleri olmadan sistemi sınamak için sentetik örnekleri kullanabilirsiniz.",
    )

    analysis_selection = st.sidebar.selectbox(
        "Uygulanacak Analiz Yöntemi:",
        options=[
            "Her İki Analiz (Copy-Move + Splicing)",
            "Yalnızca Copy-Move Analizi (SIFT + RANSAC)",
            "Yalnızca Image Splicing Analizi (ELA + Gürültü)",
        ],
        index=0,
    )

    # Analiz Türü Kodu
    if "Her İki" in analysis_selection:
        chosen_task = "both"
    elif "Copy-Move" in analysis_selection:
        chosen_task = "copy_move"
    else:
        chosen_task = "splicing"

    # Gelişmiş Parametre Ayarları (Açılır Kutu)
    with st.sidebar.expander("🛠️ Gelişmiş Dedektör Parametreleri"):
        st.markdown("**Copy-Move (SIFT/RANSAC)**")
        cm_ratio = st.slider("Lowe Oran Eşiği", 0.50, 0.90, 0.75, 0.05)
        cm_dist = st.slider("Min. Uzamsal Mesafe (px)", 10.0, 60.0, 25.0, 5.0)
        cm_inliers = st.slider("Min. RANSAC Inlier Sayısı", 3, 15, 4, 1)

        st.markdown("---")
        st.markdown("**Splicing (ELA & Gürültü)**")
        ela_q = st.slider("ELA JPEG Kalite Faktörü", 60, 98, 90, 2)
        fusion_mode = st.selectbox(
            "Füzyon Modu",
            options=["union", "intersection", "ela_only", "noise_only"],
            index=0,
            format_func=lambda x: {
                "union": "Birleşim (Union - Hassas)",
                "intersection": "Kesişim (Intersection - Katı)",
                "ela_only": "Yalnızca ELA",
                "noise_only": "Yalnızca Gürültü Analizi",
            }.get(x, x),
        )

    cm_config = CopyMoveConfig(
        ratio_threshold=cm_ratio,
        min_spatial_distance=cm_dist,
        min_inliers=cm_inliers,
    )
    sp_config = SplicingConfig(
        ela_quality=ela_q,
        fusion_mode=fusion_mode,
    )

    # =========================================================================
    # 3. Görsel Yükleme ve Hazırlık
    # =========================================================================
    image_bgr: np.ndarray | None = None
    image_rgb: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    detected_format: str = "UNKNOWN"
    info_message: str | None = None

    if mode == "Canlı Görsel Yükle":
        col_up1, col_up2 = st.columns([1, 1])

        with col_up1:
            uploaded_file = st.file_uploader(
                "📁 İncelenecek Görüntüyü Yükleyiniz (JPEG, PNG, BMP):",
                type=["jpg", "jpeg", "png", "bmp"],
                key="main_image_uploader",
            )

        with col_up2:
            uploaded_mask = st.file_uploader(
                "🎯 İsteğe Bağlı: Gerçek Doğruluk Maskesi (Ground Truth PNG):",
                type=["png", "bmp", "jpg"],
                key="mask_image_uploader",
                help="Görüntünün gerçek tahrifat maskesini yüklerseniz Precision, Recall, F1 ve IoU hesaplanır.",
            )

        if uploaded_file is not None:
            try:
                image_bgr, image_rgb, detected_format = load_uploaded_image(
                    uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )
            except Exception as e:
                st.error(f"❌ Görüntü yükleme hatası: {e}")
                return

            if uploaded_mask is not None:
                try:
                    gt_mask = validate_and_load_ground_truth(
                        uploaded_mask.getvalue(),
                        expected_shape=image_bgr.shape[:2],
                    )
                    st.success("✅ Doğruluk maskesi başarıyla doğrulandı ve yüklendi (Metrik hesabı aktif).")
                except ValueError as ve:
                    st.error(f"❌ Doğruluk Maskesi Boyut Hatası: {ve}")
                    gt_mask = None

    else:
        # Sentetik Demo Modu
        st.info("🧪 **Sentetik Demo Modu Aktif:** Sistem, yerel ham veri seti dosyalarına ihtiyaç duymadan kontrollü sentetik senaryolar üretir.")
        demo_scenario = st.radio(
            "Sentetik Senaryo Seçiniz:",
            options=["Copy-Move Senaryosu", "Splicing Senaryosu"],
            horizontal=True,
        )
        scenario_key = "splicing" if "Splicing" in demo_scenario else "copy_move"
        image_bgr, image_rgb, gt_mask, info_message = get_synthetic_demo_sample(scenario_key)
        detected_format = "SYNTHETIC_PNG"
        st.caption(f"ℹ️ {info_message}")

    if image_bgr is None or image_rgb is None:
        st.markdown("---")
        st.info("👈 Lütfen başlamak için sol menüden veya yukarıdaki yükleyiciden bir görüntü seçiniz.")
        return

    # Görsel Bilgi Şeridi
    h, w = image_bgr.shape[:2]
    st.markdown(
        f"**Görüntü Boyutları:** `{w} x {h}` px | **Format:** `{detected_format}` | "
        f"**Doğruluk Maskesi Durumu:** {'✅ Sağlandı' if gt_mask is not None else '⚠️ Sağlanmadı (Yalnızca görsel tespit)'}"
    )

    # Analiz Başlatma Butonu
    if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
        cm_res = None
        sp_res = None
        cm_time = 0.0
        sp_time = 0.0

        with st.spinner("Adli analiz motorları çalıştırılıyor, lütfen bekleyiniz..."):
            # 1. Copy-Move Analizi
            if chosen_task in ("both", "copy_move"):
                try:
                    cm_res, cm_time = run_copy_move_analysis(image_bgr, config=cm_config)
                except Exception as e:
                    st.error(f"Copy-Move analizi sırasında hata: {e}")

            # 2. Splicing Analizi
            if chosen_task in ("both", "splicing"):
                try:
                    sp_res, sp_time = run_splicing_analysis(image_bgr, config=sp_config)
                except Exception as e:
                    st.error(f"Splicing analizi sırasında hata: {e}")

        # Sonuçları oturuma kaydet
        st.session_state["analysis_executed"] = True
        st.session_state["copy_move_result"] = cm_res
        st.session_state["splicing_result"] = sp_res
        st.session_state["copy_move_time"] = cm_time
        st.session_state["splicing_time"] = sp_time
        st.session_state["loaded_image_bgr"] = image_bgr
        st.session_state["loaded_image_rgb"] = image_rgb
        st.session_state["detected_format"] = detected_format
        st.session_state["loaded_gt_mask"] = gt_mask
        st.session_state["analysis_type"] = chosen_task

    # =========================================================================
    # 4. Sonuç Ekranı ve Görsel Paneller
    # =========================================================================
    if not st.session_state.get("analysis_executed", False):
        return

    cm_result: CopyMoveResult | None = st.session_state["copy_move_result"]
    sp_result: SplicingResult | None = st.session_state["splicing_result"]
    cm_time = st.session_state["copy_move_time"]
    sp_time = st.session_state["splicing_time"]
    active_gt_mask: np.ndarray | None = st.session_state["loaded_gt_mask"]

    st.markdown("---")
    st.header("📊 Adli Analiz Bulguları ve Görselleştirme")

    # Sekmeli Sonuç Gösterimi
    tab_list = []
    if chosen_task in ("both", "copy_move"):
        tab_list.append("Copy-Move Analizi")
    if chosen_task in ("both", "splicing"):
        tab_list.append("Image Splicing Analizi")
    tab_list.append("📁 İndirilebilir Çıktılar")

    tabs = st.tabs(tab_list)
    tab_idx = 0

    # -------------------------------------------------------------------------
    # TAB 1: Copy-Move Sonuçları
    # -------------------------------------------------------------------------
    if chosen_task in ("both", "copy_move"):
        with tabs[tab_idx]:
            tab_idx += 1
            if cm_result is not None:
                st.subheader("Copy-Move Sahteciliği Tespiti (SIFT + RANSAC)")

                # Durum Rozeti
                if cm_result.is_tampered:
                    st.error(
                        f"🚨 **Şüpheli Klonlama Tespit Edildi!** "
                        f"({cm_result.num_inliers} adet geometrik olarak tutarlı inlier eşleşmesi doğrulandı. İşlem süresi: {cm_time:.3f} sn)"
                    )
                else:
                    st.success(
                        f"✅ **Belirgin Copy-Move İzi Bulunamadı.** "
                        f"(Anahtar nokta: {cm_result.num_keypoints}, Aday eşleşme: {cm_result.num_candidate_matches}, RANSAC inlier: {cm_result.num_inliers}. İşlem süresi: {cm_time:.3f} sn)"
                    )

                col_cm1, col_cm2, col_cm3 = st.columns(3)

                with col_cm1:
                    st.image(image_rgb, caption="Orijinal Görüntü", use_container_width=True)

                with col_cm2:
                    st.image(cm_result.mask, caption="İkili Tahmin Maskesi (uint8 0/255)", use_container_width=True)
                    st.caption("ℹ️ Beyaz pikseller (255), algoritma tarafından tespit edilen kaynak ve hedef kopyalama alanlarını gösterir.")

                with col_cm3:
                    overlay_bgr = create_mask_overlay(image_bgr, cm_result.mask, color=(0, 0, 255), alpha=0.45)
                    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                    st.image(overlay_rgb, caption="Maske Bindirmeli Görüntü (Overlay)", use_container_width=True)
                    st.caption("ℹ️ Kırmızı yarı saydam bölge şüpheli alanı, kırmızı sınır çizgileri ise alanın dış sınırını gösterir.")

                # Eşleşme Çizgileri Paneli
                if cm_result.num_inliers > 0:
                    st.markdown("#### 🔗 Geometrik Eşleşme Noktaları ve Vektörleri")
                    matches_bgr = draw_copy_move_matches(
                        image_bgr,
                        cm_result.matched_points_src,
                        cm_result.matched_points_dst,
                    )
                    matches_rgb = cv2.cvtColor(matches_bgr, cv2.COLOR_BGR2RGB)
                    st.image(matches_rgb, caption="Kaynak ve Hedef Anahtar Noktaları Arasındaki Eşleşme Çizgileri", use_container_width=True)
                    st.caption("ℹ️ Kırmızı daireler kaynak (source) noktaları, mavi daireler hedef (destination) noktaları, yeşil çizgiler ise aralarındaki afin transferi temsil eder.")

                # Dürüst Metrik Değerlendirmesi
                st.markdown("#### 📈 Piksel Düzeyi Başarım Metrikleri")
                cm_metrics = compute_optional_metrics(cm_result.mask, active_gt_mask)

                if cm_metrics is None:
                    st.warning(
                        "⚠️ **Metrikler Hesaplanamadı — Doğruluk Maskesi Sağlanmadı.**\n\n"
                        "Görüntünün gerçek manipülasyon maskesi (Ground Truth) yüklenmediğinde Precision, Recall, F1-Score "
                        "ve IoU metrikleri hesaplanamaz ve tahmin edilemez. Doğruluk ölçümü için lütfen görselle aynı boyutta bir zemin maskesi yükleyiniz."
                    )
                else:
                    st.session_state["metrics_copy_move"] = cm_metrics
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Precision (Kesinlik)", f"{cm_metrics['precision']*100:.2f}%")
                    mcol2.metric("Recall (Duyarlılık)", f"{cm_metrics['recall']*100:.2f}%")
                    mcol3.metric("F1-Score", f"{cm_metrics['f1_score']*100:.2f}%")
                    mcol4.metric("IoU (Jaccard)", f"{cm_metrics['iou']*100:.2f}%")
                    st.caption(f"Piksel Sayımları: TP={cm_metrics['tp']}, FP={cm_metrics['fp']}, FN={cm_metrics['fn']}")

    # -------------------------------------------------------------------------
    # TAB 2: Image Splicing Sonuçları
    # -------------------------------------------------------------------------
    if chosen_task in ("both", "splicing"):
        with tabs[tab_idx]:
            tab_idx += 1
            if sp_result is not None:
                st.subheader("Image Splicing Analizi (ELA & Gürültü Tutarsızlığı)")

                if sp_result.is_suspicious:
                    st.warning(
                        f"⚠️ **Şüpheli Bölgesel Tutarsızlıklar Bulundu!** "
                        f"({np.count_nonzero(sp_result.mask)} piksel şüpheli olarak işaretlendi. İşlem süresi: {sp_time:.3f} sn)"
                    )
                else:
                    st.success(
                        f"✅ **Belirgin Splicing Anomalisine Rastlanmadı.** "
                        f"(İşlem süresi: {sp_time:.3f} sn)"
                    )

                # Dörtlü Harita Paneli
                col_sp1, col_sp2 = st.columns(2)
                with col_sp1:
                    st.image(image_rgb, caption="Orijinal Görüntü", use_container_width=True)
                with col_sp2:
                    sp_overlay_bgr = create_mask_overlay(image_bgr, sp_result.mask, color=(0, 140, 255), alpha=0.45)
                    sp_overlay_rgb = cv2.cvtColor(sp_overlay_bgr, cv2.COLOR_BGR2RGB)
                    st.image(sp_overlay_rgb, caption=f"Füzyon Maske Bindirmesi ({sp_result.fusion_mode})", use_container_width=True)

                col_map1, col_map2 = st.columns(2)

                # ELA Haritası
                with col_map1:
                    st.markdown("##### 🔍 Error Level Analysis (ELA)")
                    if sp_result.ela_result.is_applicable:
                        st.image(sp_result.ela_result.normalized_map, caption="Normalize ELA Fark Haritası", use_container_width=True)
                        st.caption(
                            f"ℹ️ {sp_result.ela_result.status_message} Parlak pikseller, görüntünün geri kalanına göre "
                            "farklı sıkıştırma kaybı sergileyen şüpheli alanları gösterir."
                        )
                    else:
                        st.info(f"ℹ️ {sp_result.ela_result.status_message}")

                # Gürültü Tutarsızlığı Haritası
                with col_map2:
                    st.markdown("##### ⚡ Yerel Gürültü Tutarsızlığı (Noise Inconsistency)")
                    st.image(sp_result.noise_result.normalized_map, caption="Normalize Yerel Gürültü Haritası", use_container_width=True)
                    st.caption(
                        f"ℹ️ {sp_result.noise_result.status_message} Parlak bölgeler yerel standart sapmanın ortalamadan belirgin şekilde saptığı alanları temsil eder."
                    )

                # Dürüst Metrik Değerlendirmesi
                st.markdown("#### 📈 Piksel Düzeyi Başarım Metrikleri")
                sp_metrics = compute_optional_metrics(sp_result.mask, active_gt_mask)

                if sp_metrics is None:
                    st.warning(
                        "⚠️ **Metrikler Hesaplanamadı — Doğruluk Maskesi Sağlanmadı.**\n\n"
                        "Görüntünün gerçek manipülasyon maskesi (Ground Truth) yüklenmediğinde Precision, Recall, F1-Score "
                        "ve IoU metrikleri hesaplanamaz ve tahmin edilemez."
                    )
                else:
                    st.session_state["metrics_splicing"] = sp_metrics
                    smcol1, smcol2, smcol3, smcol4 = st.columns(4)
                    smcol1.metric("Precision (Kesinlik)", f"{sp_metrics['precision']*100:.2f}%")
                    smcol2.metric("Recall (Duyarlılık)", f"{sp_metrics['recall']*100:.2f}%")
                    smcol3.metric("F1-Score", f"{sp_metrics['f1_score']*100:.2f}%")
                    smcol4.metric("IoU (Jaccard)", f"{sp_metrics['iou']*100:.2f}%")
                    st.caption(f"Piksel Sayımları: TP={sp_metrics['tp']}, FP={sp_metrics['fp']}, FN={sp_metrics['fn']}")

    # -------------------------------------------------------------------------
    # TAB 3: İndirilebilir Çıktılar (Downloadable Outputs)
    # -------------------------------------------------------------------------
    with tabs[tab_idx]:
        st.subheader("📦 Analiz Çıktılarını İndir")
        st.write("Analiz sonucunda üretilen ikili maskeleri, görsel bindirmeleri ve teknik JSON raporunu indirebilirsiniz:")

        dcol1, dcol2, dcol3 = st.columns(3)

        # 1. Copy-Move Maske ve Overlay İndirme
        if cm_result is not None:
            with dcol1:
                st.markdown("**Copy-Move Dosyaları:**")
                st.download_button(
                    label="⬇️ Tahmin Maskesi (PNG)",
                    data=mask_to_png_bytes(cm_result.mask),
                    file_name="forensivision_copymove_mask.png",
                    mime="image/png",
                )
                cm_over = create_mask_overlay(image_bgr, cm_result.mask)
                st.download_button(
                    label="⬇️ Maske Overlay (PNG)",
                    data=image_to_png_bytes(cm_over),
                    file_name="forensivision_copymove_overlay.png",
                    mime="image/png",
                )

        # 2. Splicing Maske ve Harita İndirme
        if sp_result is not None:
            with dcol2:
                st.markdown("**Splicing Dosyaları:**")
                st.download_button(
                    label="⬇️ Splicing Maskesi (PNG)",
                    data=mask_to_png_bytes(sp_result.mask),
                    file_name="forensivision_splicing_mask.png",
                    mime="image/png",
                )
                if sp_result.ela_result.is_applicable:
                    st.download_button(
                        label="⬇️ ELA Haritası (PNG)",
                        data=image_to_png_bytes(sp_result.ela_result.normalized_map),
                        file_name="forensivision_ela_map.png",
                        mime="image/png",
                    )
                st.download_button(
                    label="⬇️ Gürültü Haritası (PNG)",
                    data=image_to_png_bytes(sp_result.noise_result.normalized_map),
                    file_name="forensivision_noise_map.png",
                    mime="image/png",
                )

        # 3. Teknik JSON Raporu
        with dcol3:
            st.markdown("**Akademik & Adli Rapor:**")
            cm_m = st.session_state.get("metrics_copy_move")
            sp_m = st.session_state.get("metrics_splicing")

            json_payload = generate_export_payload(
                analysis_type=chosen_task,
                image_shape=image_bgr.shape,
                detected_format=detected_format,
                copy_move_result=cm_result,
                splicing_result=sp_result,
                copy_move_time=cm_time,
                splicing_time=sp_time,
                copy_move_metrics=cm_m,
                splicing_metrics=sp_m,
                has_ground_truth=active_gt_mask is not None,
            )

            st.download_button(
                label="⬇️ Analiz Özeti (JSON)",
                data=json_payload,
                file_name="forensivision_analysis_report.json",
                mime="application/json",
            )

    # =========================================================================
    # 5. Alt Bilgilendirme ve Yöntem Detayları (Expander)
    # =========================================================================
    st.markdown("---")
    with st.expander("📚 Yöntem ve Algoritma Detayları (Akademik Özet)"):
        st.markdown(
            """
            - **Copy-Move Algoritması:** Görüntüden SIFT anahtar noktaları çıkarılır. Kendi kendine eşleşmeleri 
              engellemek için uzamsal Öklid mesafe filtrelemesi ve Lowe oran testi uygulanır. Eşleşen noktalar 
              RANSAC algoritması ile afin geometrik dönüşüme tabi tutulur; yalnızca fiziksel olarak tutarlı 
              inlier noktaların dışbükey zarfları (convex hull) şüpheli bölge olarak işaretlenir.
            - **Error Level Analysis (ELA):** Görüntü belirli bir kalite faktöründe (varsayılan 90) tekrar JPEG 
              olarak kaydedilir ve orijinal ile fark haritası çıkarılır. Farklı kaynaklardan gelen parçalar farklı 
              sıkıştırma oranına sahip olduğunda hata seviyesinde bölgesel sıçramalar oluşur.
            - **Yerel Gürültü Analizi:** Kamera sensörlerinin bıraktığı yüksek frekanslı gürültü artıklarının (noise residual) 
              yerel varyansı kayar pencere ile taranır ve arka plan gürültü seviyesinden belirgin şekilde sapan bölgeler yerelleştirilir.
            """
        )


if __name__ == "__main__":
    main()
