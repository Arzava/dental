import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from alveolar_krest import alveolar_krest_analysis
from streamlit_image_comparison import image_comparison

# ... (Sayfa yapılandırması aynı kalabilir) ...

# --- KART OLUŞTURUCU (GÜNCELLENDİ) ---
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
    
    # YENİ RENK VE İKON MANTIĞI
    if "LİFT GEREKMEZ" in decision:
        style_class = "success"
        bg_color = "#4CAF50" # Yeşil
        icon = "✅"
    elif "KAPALI LİFT" in decision:
        style_class = "warning"
        bg_color = "#FF9800" # Turuncu (Uyarı)
        icon = "⚠️"
    elif "Tek Aşamalı" in decision:
        style_class = "danger"
        bg_color = "#FF5722" # Koyu Turuncu / Açık Kırmızı
        icon = "🚨"
    else: # Çift Aşamalı (En Kritik)
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

# ... (Geri kalan process_image ve ana ekran kodları aynı kalacak) ...
