import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from alveolar_krest import alveolar_krest_analysis
from streamlit_image_comparison import image_comparison

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Alveolar AI (Pro)",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TASARIM ---
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 10px;
        color: #333333 !important;
    }
    .metric-card.success { border-left: 6px solid #4CAF50; } /* Yeşil */
    .metric-card.warning { border-left: 6px solid #FF9800; } /* Turuncu */
    .metric-card.danger  { border-left: 6px solid #F44336; } /* Kırmızı */
    
    .metric-title {
        color: #6c757d !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #2c3e50 !important;
        font-size: 1.6rem;
        font-weight: bold;
    }
    .metric-status {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        display: inline-block;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --- GÖRÜNTÜ İŞLEME VE ÇİZİM ---
def process_image(image_input, model, alpha_val, px_mm_val):
    img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # YOLO Tahmini
    results_list = model.predict(img_bgr, conf=0.5)
    res = results_list[0]

    # Maske Katmanı
    overlay = img_bgr.copy()
    COLOR_SINUS = (0, 255, 255) # Sarı
    COLOR_KRET  = (0, 0, 255)   # Kırmızı

    if res.masks is not None:
        polys = res.masks.xy
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for poly, cls in zip(polys, classes):
            pts = poly.astype(int)
            if cls == 3: cv2.fillPoly(overlay, [pts], COLOR_SINUS)
            elif cls == 0: cv2.fillPoly(overlay, [pts], COLOR_KRET)

    # Opaklık Uygula
    img_result = cv2.addWeighted(overlay, alpha_val, img_bgr, 1 - alpha_val, 0)
    
    # Analiz Yap (Yeni px_mm_ratio ile)
    analysis_results = alveolar_krest_analysis(res, img_result, px_to_mm_ratio=px_mm_val)
    
    # Orta Çizgi (Referans)
    mid_x = w // 2
    cv2.line(img_result, (mid_x, 0), (mid_x, h), (200, 200, 200), 1) 

    # --- ÇİZİM DÖNGÜSÜ ---
    for side in ["LEFT", "RIGHT"]:
        data = analysis_results[side]
        points = data["points"]
        
        if not points: continue

        for i, pt in enumerate(points):
            x, y_s, y_k = pt["coords"]
            mm_val = pt["mm"]
            
            # Ana Ölçüm Çizgisi (Yeşil)
            cv2.line(img_result, (x, y_s), (x, y_k), (0, 255, 0), 2)
            
            # Tırnaklar (Mavi ve Kırmızı)
            cv2.line(img_result, (x-10, y_s), (x+10, y_s), (255, 0, 0), 2) # Sinüs Tırnağı (Mavi)
            cv2.line(img_result, (x-10, y_k), (x+10, y_k), (0, 0, 255), 2) # Kret Tırnağı (Kırmızı)

            # --- ZIG-ZAG METİN YERLEŞİMİ ---
            # Yazıların üst üste binmesini önlemek için dikey kaydırma
            text_label = f"{mm_val}"
            vertical_offset = 25 + (i % 3) * 30 
            text_pos = (x - 20, y_k + vertical_offset)
            
            # Siyah kontur (okunabilirlik için)
           # cv2.putText(img_result, text_label, text_pos, 
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            # Beyaz yazı
            cv2.putText(img_result, text_label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3, cv2.LINE_AA)
            
            # Kılavuz Çizgisi (Yazı uzaktaysa)
            if vertical_offset > 25:
                cv2.line(img_result, (x, y_k), (x, y_k + vertical_offset - 10), (200, 200, 200), 1)

    return img_bgr, img_result, analysis_results 

# --- KART OLUŞTURUCU (YENİ PROTOKOL) ---
def create_card(side_name, info):
    min_val = info['min_mm']
    decision = info['global_decision']
    
    if min_val is None:
        return f"""
        <div class="metric-card danger">
            <div class="metric-title">{side_name}</div>
            <div class="metric-value">--</div>
            <div class="metric-status" style="background:#ffebee; color:#c62828;">Ölçüm Yok</div>
        </div>
        """
    
    # RENK VE İKON MANTIĞI (4 KADEME)
    if "LİFT GEREKMEZ" in decision:
        style_class = "success"
        bg_color = "#4CAF50" # Yeşil
        icon = "✅"
    elif "KAPALI LİFT" in decision:
        style_class = "warning"
        bg_color = "#FF9800" # Turuncu
        icon = "⚠️"
    elif "Tek Aşamalı" in decision:
        style_class = "danger"
        bg_color = "#FF5722" # Koyu Turuncu
        icon = "🚨"
    else: # Çift Aşamalı (0-3mm)
        style_class = "danger"
        bg_color = "#D32F2F" # Koyu Kırmızı
        icon = "🛑"
    
    return f"""
    <div class="metric-card {style_class}">
        <div class="metric-title">{side_name} (En Kritik)</div>
        <div class="metric-value">{min_val} <span style="font-size:1rem; color:#999">mm</span></div>
        <div class="metric-status" style="background:{bg_color};">
            {icon} {decision}
        </div>
        <div style="font-size:0.8rem; color:#666; margin-top:5px;">
            (Bölgedeki en düşük ölçüm)
        </div>
    </div>
    """

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60) 
    st.title("Alveolar AI")
    st.caption("Dental Radyoloji Asistanı v5.0")
    st.divider()
    
    st.subheader("📏 Kalibrasyon")
    px_to_mm = st.number_input("1 Piksel kaç mm?", 0.001, 5.0, 0.100, 0.001, "%.3f")
    alpha = st.slider("Maske Opaklığı", 0.0, 1.0, 0.4, step=0.05)
    
    st.divider()
    st.subheader("📋 Yeni Protokol")
    st.info("0-3mm: Açık Lift (Çift)")
    st.warning("3-5mm: Açık Lift (Tek)")
    st.warning("6-8mm: Kapalı Lift")
    st.success("8mm+: Gerekmez")
    
    st.divider()
    st.caption("Dr. Muhammed ÇELİK")

# --- ANA EKRAN ---
st.title("🦷 Otomatik İmplant Planlama")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    try:
        # Model dosyasının adı 'best.pt' olarak varsayıldı
        model = load_model("best.pt")
    except Exception as e:
        st.error(f"Model yüklenemedi! Hata: {e}")
        st.stop()

    # Görüntüyü İşle
    orig_img, proc_img, data = process_image(image, model, alpha, px_to_mm)
    
    # Streamlit için RGB dönüşümü
    img1 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    st.divider()

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("👁️ Görüntü Analizi")
        # Görüntü boyutları aynıysa kaydırma çubuğu göster, değilse sadece sonucu göster
        if img1.shape == img2.shape:
            image_comparison(
                img1=img1, img2=img2,
                label1="Orijinal", label2="Analiz",
                width=800, starting_position=2,
                show_labels=True, make_responsive=True, in_memory=True
            )
        else:
            st.image(img2, use_container_width=True)

    with col_right:
        st.subheader("📋 Klinik Rapor")
        # Sağ ve Sol taraf için kartları oluştur
        st.markdown(create_card("HASTA SAĞ", data["LEFT"]), unsafe_allow_html=True)
        st.write("") 
        st.markdown(create_card("HASTA SOL", data["RIGHT"]), unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="border: 2px dashed #ccc; padding: 40px; border-radius: 10px; text-align: center; color: gray; margin-top: 20px;">
        <h3>Röntgen Yükleyin</h3>
        <p>Otomatik Segmentasyon ve Cerrahi Planlama Önerisi</p>
    </div>
    """, unsafe_allow_html=True)
