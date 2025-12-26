import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from alveolar_krest import alveolar_krest_analysis
from streamlit_image_comparison import image_comparison

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Alveolar AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TASARIM ---
st.markdown("""
<style>
    /* Kart Tasarımları */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
        color: #333333 !important;
    }
    .metric-card.danger {
        border-left: 5px solid #FF5252;
    }
    .metric-title {
        color: #6c757d !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #2c3e50 !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-status {
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 5px;
        padding: 4px 8px;
        border-radius: 15px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- FONKSİYONLAR ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

def process_image(image_input, model, alpha_val, thresh_val):
    # PIL -> OpenCV (BGR)
    img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Tahmin
    results_list = model.predict(img_bgr, conf=0.5)
    res = results_list[0]

    # Maske Katmanı
    overlay = img_bgr.copy()
    COLOR_SINUS = (0, 255, 255)  # Sarı
    COLOR_KRET  = (0, 0, 255)    # Kırmızı

    if res.masks is not None:
        polys = res.masks.xy
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for poly, cls in zip(polys, classes):
            pts = poly.astype(int)
            if cls == 3: cv2.fillPoly(overlay, [pts], COLOR_SINUS)
            elif cls == 0: cv2.fillPoly(overlay, [pts], COLOR_KRET)

    # Birleştirme
    img_result = cv2.addWeighted(overlay, alpha_val, img_bgr, 1 - alpha_val, 0)

    # Analiz
    analysis_results = alveolar_krest_analysis(res, img_result, threshold_px=thresh_val)
    
    # Çizimler
    mid_x = w // 2
    cv2.line(img_result, (mid_x, 0), (mid_x, h), (200, 200, 200), 1) 

    for side in ["LEFT", "RIGHT"]:
        r = analysis_results[side]
        if r["thickness_px"] is not None:
            x, y_s, y_k = r["x_col"], r["sinus_y"], r["kret_y"]
            cv2.line(img_result, (x, y_s), (x, y_k), (0, 255, 0), 3) 
            cv2.line(img_result, (x-25, y_s), (x+25, y_s), (255, 0, 0), 2) 
            cv2.line(img_result, (x-25, y_k), (x+25, y_k), (0, 0, 255), 2) 

    return img_bgr, img_result, analysis_results 

# --- YARDIMCI HTML KART OLUŞTURUCU ---
def create_card(side_name, info):
    if info['thickness_px'] is None:
        return f"""
        <div class="metric-card danger">
            <div class="metric-title">{side_name}</div>
            <div class="metric-value">--</div>
            <div class="metric-status" style="background:#ffebee; color:#c62828;">Ölçüm Yok</div>
        </div>
        """
    
    px = info['thickness_px']
    decision = info['decision']
    is_safe = "GEREKMEZ" in decision
    
    color_class = "" if is_safe else "danger"
    status_bg = "#e8f5e9" if is_safe else "#ffebee"
    status_text = "#2e7d32" if is_safe else "#c62828"
    icon = "✅" if is_safe else "⚠️"
    
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-title">{side_name}</div>
        <div class="metric-value">{px} <span style="font-size:1rem; color:#999">px</span></div>
        <div class="metric-status" style="background:{status_bg}; color:{status_text};">
            {icon} {decision}
        </div>
    </div>
    """

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60) 
    st.title("Alveolar AI")
    st.caption("Dental Radyoloji Asistanı v1.0")
    st.divider()
    
    st.subheader("⚙️ Parametreler")
    thresh_val = st.slider("Karar Eşiği (px)", 5, 100, 20)
    alpha = st.slider("Maske Opaklığı", 0.0, 1.0, 0.4)
    # SLIDER KONUM AYARI KALDIRILDI
    
    st.divider()
    st.info("Hekim kontrolü şarttır.")

# --- ANA EKRAN ---
st.title("🦷 Akıllı Kemik Analizi")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    try:
        model = load_model("best.pt")
    except:
        st.error("Model yüklenemedi! 'best.pt' dosyasını kontrol edin.")
        st.stop()

    # Analiz
    orig_img, proc_img, data = process_image(image, model, alpha, thresh_val)
    
    img1 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    st.divider()

    # --- YERLEŞİM DÜZENİ ---
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("👁️ Görüntü Analizi")
        
        if img1.shape == img2.shape:
            image_comparison(
                img1=img1,
                img2=img2,
                label1="Orijinal",
                label2="Analiz",
                width=800, 
                starting_position=2, # SABİT: EN SOL (%2)
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
        else:
            st.image(img2, use_container_width=True)

    with col_right:
        st.subheader("📋 Rapor")
        st.markdown(create_card("HASTA SAĞ", data["LEFT"]), unsafe_allow_html=True)
        st.write("") 
        st.markdown(create_card("HASTA SOL", data["RIGHT"]), unsafe_allow_html=True)

else:
    # Boş durum
    st.markdown("""
    <div style="
        border: 2px dashed #ccc; 
        padding: 40px; 
        border-radius: 10px; 
        text-align: center; 
        color: gray;
        margin-top: 20px;">
        <h3>Röntgen Yükleyin</h3>
        <p>Sürükleyip bırakın</p>
    </div>
    """, unsafe_allow_html=True)