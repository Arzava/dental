import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from alveolar_krest import alveolar_krest_analysis
from streamlit_image_comparison import image_comparison

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Alveolar AI (Pro)",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS DESIGN ---
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

# --- MODEL LOADING ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --- IMAGE PROCESSING AND DRAWING ---
def process_image(image_input, model, alpha_val, px_mm_val):
    img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # YOLO Prediction
    results_list = model.predict(img_bgr, conf=0.5)
    res = results_list[0]

    # Mask Overlay
    overlay = img_bgr.copy()
    COLOR_SINUS = (0, 255, 255) # Yellow
    COLOR_KRET  = (0, 0, 255)   # Red

    if res.masks is not None:
        polys = res.masks.xy
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for poly, cls in zip(polys, classes):
            pts = poly.astype(int)
            if cls == 3: cv2.fillPoly(overlay, [pts], COLOR_SINUS)
            elif cls == 0: cv2.fillPoly(overlay, [pts], COLOR_KRET)

    # Apply Opacity
    img_result = cv2.addWeighted(overlay, alpha_val, img_bgr, 1 - alpha_val, 0)
    
    # Perform Analysis (with new px_mm_ratio)
    analysis_results = alveolar_krest_analysis(res, img_result, px_to_mm_ratio=px_mm_val)
    
    # Midline (Reference)
    mid_x = w // 2
    cv2.line(img_result, (mid_x, 0), (mid_x, h), (200, 200, 200), 1) 

    # --- DRAWING LOOP ---
    for side in ["LEFT", "RIGHT"]:
        data = analysis_results[side]
        points = data["points"]
        
        if not points: 
            continue

        for i, pt in enumerate(points):
            x, y_s, y_k = pt["coords"]
            mm_val = pt["mm"]
            
            # Main Measurement Line (Green)
            cv2.line(img_result, (x, y_s), (x, y_k), (0, 255, 0), 2)
            
            # Ticks (Blue and Red)
            cv2.line(img_result, (x-10, y_s), (x+10, y_s), (255, 0, 0), 2)
            cv2.line(img_result, (x-10, y_k), (x+10, y_k), (0, 0, 255), 2)

            # ZIG-ZAG TEXT PLACEMENT
            text_label = f"{mm_val}"
            vertical_offset = 25 + (i % 3) * 30 
            text_pos = (x - 20, y_k + vertical_offset)
            
            cv2.putText(
                img_result,
                text_label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                3,
                cv2.LINE_AA
            )
            
            # Guide Line
            if vertical_offset > 25:
                cv2.line(
                    img_result,
                    (x, y_k),
                    (x, y_k + vertical_offset - 10),
                    (200, 200, 200),
                    1
                )

    return img_bgr, img_result, analysis_results 

# --- CARD GENERATOR (NEW PROTOCOL) ---
def create_card(side_name, info):
    min_val = info['min_mm']
    decision = info['global_decision']
    
    if min_val is None:
        return f"""
        <div class="metric-card danger">
            <div class="metric-title">{side_name}</div>
            <div class="metric-value">--</div>
            <div class="metric-status" style="background:#ffebee; color:#c62828;">
                No Measurement
            </div>
        </div>
        """
    
    if "NO LIFT REQUIRED" in decision:
        style_class = "success"
        bg_color = "#4CAF50"
        icon = "✅"
    elif "CLOSED LIFT" in decision:
        style_class = "warning"
        bg_color = "#FF9800"
        icon = "⚠️"
    elif "Single-Stage" in decision:
        style_class = "danger"
        bg_color = "#FF5722"
        icon = "🚨"
    else:
        style_class = "danger"
        bg_color = "#D32F2F"
        icon = "🛑"
    
    return f"""
    <div class="metric-card {style_class}">
        <div class="metric-title">{side_name} (Most Critical)</div>
        <div class="metric-value">{min_val} <span style="font-size:1rem; color:#999">mm</span></div>
        <div class="metric-status" style="background:{bg_color};">
            {icon} {decision}
        </div>
        <div style="font-size:0.8rem; color:#666; margin-top:5px;">
            (Lowest measurement in the region)
        </div>
    </div>
    """

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60) 
    st.title("Alveolar AI")
    st.caption("Dental Radiology Assistant v5.0")
    st.divider()
    
    st.subheader("📏 Calibration")
    px_to_mm = st.number_input("How many mm per pixel?", 0.001, 5.0, 0.100, 0.001, "%.3f")
    alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.4, step=0.05)
    
    st.divider()
    st.subheader("📋 New Protocol")
    st.info("0–3mm: Open Lift (Two-Stage)")
    st.warning("3–5mm: Open Lift (Single-Stage)")
    st.warning("6–8mm: Closed Lift")
    st.success("8mm+: Not Required")
    
    st.divider()
    st.caption("Dr. Muhammed ÇELİK")

# --- MAIN SCREEN ---
st.title("🦷 Automatic Implant Planning")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    try:
        model = load_model("best.pt")
    except Exception as e:
        st.error(f"Model could not be loaded! Error: {e}")
        st.stop()

    orig_img, proc_img, data = process_image(image, model, alpha, px_to_mm)
    
    img1 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

    st.divider()

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("👁️ Image Analysis")
        if img1.shape == img2.shape:
            image_comparison(
                img1=img1,
                img2=img2,
                label1="Original",
                label2="Analysis",
                width=800,
                starting_position=2,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
        else:
            st.image(img2, use_container_width=True)

    with col_right:
        st.subheader("📋 Clinical Report")
        st.markdown(create_card("PATIENT RIGHT", data["LEFT"]), unsafe_allow_html=True)
        st.write("") 
        st.markdown(create_card("PATIENT LEFT", data["RIGHT"]), unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="border: 2px dashed #ccc; padding: 40px; border-radius: 10px; text-align: center; color: gray; margin-top: 20px;">
        <h3>Upload an X-ray</h3>
        <p>Automatic Segmentation and Surgical Planning Recommendation</p>
    </div>
    """, unsafe_allow_html=True)
