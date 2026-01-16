import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from alveolar_krest import alveolar_krest_analysis
from streamlit_image_comparison import image_comparison

st.set_page_config(
    page_title="Alveolar AI (Multi-Point)",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
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
    .metric-card.success { border-left: 6px solid #4CAF50; }
    .metric-card.warning { border-left: 6px solid #FF9800; }
    .metric-card.danger  { border-left: 6px solid #F44336; }
    
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
        font-size: 1rem;
        font-weight: 600;
        margin-top: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        display: inline-block;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return YOLO(path)

def process_image(image_input, model, alpha_val, px_mm_val):
    img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    results_list = model.predict(img_bgr, conf=0.5)
    res = results_list[0]

    overlay = img_bgr.copy()
    COLOR_SINUS = (0, 255, 255)
    COLOR_KRET  = (0, 0, 255)

    if res.masks is not None:
        polys = res.masks.xy
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for poly, cls in zip(polys, classes):
            pts = poly.astype(int)
            if cls == 3: cv2.fillPoly(overlay, [pts], COLOR_SINUS)
            elif cls == 0: cv2.fillPoly(overlay, [pts], COLOR_KRET)

    img_result = cv2.addWeighted(overlay, alpha_val, img_bgr, 1 - alpha_val, 0)
    analysis_results = alveolar_krest_analysis(res, img_result, px_to_mm_ratio=px_mm_val)
    
    mid_x = w // 2
    cv2.line(img_result, (mid_x, 0), (mid_x, h), (200, 200, 200), 1) 

    # --- ÇOKLU ÇİZİM DÖNGÜSÜ ---
    for side in ["LEFT", "RIGHT"]:
        data = analysis_results[side]
        points = data["points"] # Artık bu bir liste
        
        if not points: continue

        for i, pt in enumerate(points):
            x, y_s, y_k = pt["coords"]
            mm_val = pt["mm"]
            
            # Çizgiler
            cv2.line(img_result, (x, y_s), (x, y_k), (0, 255, 0), 2)
            # Sınır çizgilerini biraz küçültelim ki karışmasın
            cv2.line(img_result, (x-10, y_s), (x+10, y_s), (255, 0, 0), 2) 
            cv2.line(img_result, (x-10, y_k), (x+10, y_k), (0, 0, 255), 2) 

            # Yazı (Çakışmayı önlemek için Y pozisyonunu hafif kaydırabiliriz)
            text_label = f"{mm_val}"
            mid_y = (y_s + y_k) // 2
            
            # Yazıyı çizginin sağına koy
            text_pos = (x + 5, mid_y)
            
            # Küçük font ile değerleri yaz
            cv2.putText(img_result, text_label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img_result, text_label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img_bgr, img_result, analysis_results 

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
    
    if decision == "LİFT GEREKMEZ":
        style_class = "success"
        bg_color = "#4CAF50"
        icon = "✅"
    elif decision == "KAPALI LİFT":
        style_class = "warning"
        bg_color = "#FF9800"
        icon = "⚠️"
    else: 
        style_class = "danger"
        bg_color = "#F44336"
        icon = "🚨"
    
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

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60) 
    st.title("Alveolar AI")
    st.caption("Dental Radyoloji Asistanı v4.0")
    st.divider()
    
    st.subheader("📏 Ayarlar")
    px_to_mm = st.number_input("1 Piksel kaç mm?", 0.001, 5.0, 0.100, 0.001, "%.3f")
    alpha = st.slider("Maske Opaklığı", 0.0, 1.0, 0.4, step=0.05)
    
    st.divider()
    st.subheader("📋 Protokol")
    st.info("≤5mm: Açık | 6-8mm: Kapalı | ≥8mm: Gerekmez")
    st.divider()
    st.caption("Dr. Muhammed ÇELİK")

st.title("🦷 Çoklu Nokta Analizi")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    try:
        model = load_model("best.pt")
    except:
        st.error("Model yüklenemedi!")
        st.stop()

    orig_img, proc_img, data = process_image(image, model, alpha, px_to_mm)
    
    img1 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    st.divider()

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("👁️ Görüntü Analizi")
        if img1.shape == img2.shape:
            image_comparison(
                img1=img1, img2=img2,
                label1="Orijinal", label2="Çoklu Analiz",
                width=800, starting_position=2,
                show_labels=True, make_responsive=True, in_memory=True
            )
        else:
            st.image(img2, use_container_width=True)

    with col_right:
        st.subheader("📋 Kritik Rapor")
        st.markdown(create_card("HASTA SAĞ", data["LEFT"]), unsafe_allow_html=True)
        st.write("") 
        st.markdown(create_card("HASTA SOL", data["RIGHT"]), unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="border: 2px dashed #ccc; padding: 40px; border-radius: 10px; text-align: center; color: gray; margin-top: 20px;">
        <h3>Röntgen Yükleyin</h3>
        <p>Otomatik Bölgesel Tarama ve Çoklu Ölçüm</p>
    </div>
    """, unsafe_allow_html=True)
