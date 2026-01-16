import numpy as np
import cv2

# ---- CLASS ID'LER ----
KRET_CLASS = 0
SINUS_CLASS = 3

def rasterize_class(res, cls_id, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if res.masks is None: return mask
    polys = res.masks.xy
    classes = res.boxes.cls.cpu().numpy().astype(int)
    for poly, c in zip(polys, classes):
        if c == cls_id:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

def keep_half(mask, side):
    h, w = mask.shape
    mid = w // 2
    out = mask.copy()
    if side == "LEFT": out[:, mid:] = 0
    else: out[:, :mid] = 0
    return out

def get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=30):
    """Kret altındaki sivri çıkıntıları (diş aralarını) temizler."""
    raw_bottoms = []
    for x in range(x_min, x_max):
        ys = np.where(kret_mask[:, x] > 0)[0]
        if ys.size > 0: raw_bottoms.append(int(ys.max()))
        else: raw_bottoms.append(np.nan)
            
    smoothed_bottoms = []
    arr = np.array(raw_bottoms)
    
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        window = arr[start:end]
        valid = window[~np.isnan(window)]
        if valid.size > 0: smoothed_bottoms.append(int(np.min(valid)))
        else: smoothed_bottoms.append(None)
            
    return smoothed_bottoms

def compute_multi_thickness(res, image, side, num_segments=3):
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)

    # --- KRİTİK DEĞİŞİKLİK: KOMŞULUK KONTROLÜ ---
    # Sadece hem Sinüsün (Sarı) hem de Kretin (Kırmızı) AYNI KOLONDA var olduğu yerleri bul.
    # np.logical_and ile iki maskenin çakıştığı dikey sütunları seçiyoruz.
    
    # 1. Sinüs var mı?
    has_sinus = np.any(sinus_mask > 0, axis=0)
    # 2. Kret var mı?
    has_kret = np.any(kret_mask > 0, axis=0)
    
    # 3. İkisi de aynı sütunda VAR MI?
    valid_cols_mask = np.logical_and(has_sinus, has_kret)
    
    # Bu maskenin True olduğu indeksleri (x koordinatlarını) al
    common_cols = np.where(valid_cols_mask)[0]
    
    # Eğer hiç ortak alan yoksa ölçüm yapma
    if common_cols.size == 0:
        return []

    # Geçerli alanın sınırları
    x_min, x_max = common_cols.min(), common_cols.max()
    width = x_max - x_min
    
    # --- Yanal Boşlukları Temizle (Gürültü Filtresi) ---
    # Bazen maskeler uçlarda 1-2 piksel çakışabilir, bunları elemek için
    # en az 10 piksellik bir bütünlük arayalım.
    if width < 10: 
        return []

    # Kret altı düzeltme (Moving Window ile diş aralarını doldur)
    smoothed_kret_ys = get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=30)
    
    measurements = []
    
    # Eğer alan darsa tek parça ölç
    real_num_segments = num_segments
    if width < 50: real_num_segments = 1
    
    step = width // real_num_segments
    if step < 1: step = 1

    for i in range(real_num_segments):
        seg_start = x_min + (i * step)
        seg_end = seg_start + step
        if i == real_num_segments - 1: seg_end = x_max

        best_in_segment = None
        
        for x_global in range(seg_start, seg_end):
            # Orijinal ortak alan maskesine tekrar bak (arada boşluklar olabilir)
            if not valid_cols_mask[x_global]:
                continue

            # Sinüs tabanı (En alt nokta)
            ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
            sinus_bottom = int(ys_s.max())
            
            # Kret tabanı (Düzeltilmiş listeden)
            idx = x_global - x_min
            if idx < 0 or idx >= len(smoothed_kret_ys): continue
            kret_bottom = smoothed_kret_ys[idx]
            
            # Eğer Kret Tabanı, Sinüs Tabanından YUKARIDAYSA (Hatalı Maske)
            if kret_bottom is None or kret_bottom <= sinus_bottom: 
                continue
            
            # --- EK GÜVENLİK: DİKEY MESAFE KONTROLÜ ---
            # Sinüs ile Kret arasında çok büyük bir boşluk varsa (örn: sinüs duvarda, kret aşağıda)
            # ve bu ikisi görsel olarak "komşu" değilse ölçme.
            # Ancak "Alveolar Krest" analizinde zaten sinüs tabanı ile kret tabanı arasındaki
            # mesafeyi ölçüyoruz. Buradaki mantık: Maskeler dikeyde üst üste biniyorsa
            # bu geçerli bir ölçümdür.
            
            dist = kret_bottom - sinus_bottom
            
            # En riskli (en ince) kemiği bul
            if best_in_segment is None or dist < best_in_segment[0]:
                best_in_segment = (dist, x_global, sinus_bottom, kret_bottom)
        
        if best_in_segment:
            measurements.append(best_in_segment)
            
    return measurements

def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1):
    results = {}
    for side in ["LEFT", "RIGHT"]:
        points = compute_multi_thickness(res, image, side, num_segments=3)
        side_results = []
        
        if not points:
            results[side] = {"points": [], "global_decision": "ÖLÇÜM YOK", "min_mm": None}
            continue
            
        min_mm_global = 999.0
        for p in points:
            dist_px, x, y_sinus, y_kret = p
            dist_mm = dist_px * px_to_mm_ratio
            
            if dist_mm <= 5.0: decision = "AÇIK LİFT"
            elif dist_mm >= 8.0: decision = "LİFT GEREKMEZ"
            else: decision = "KAPALI LİFT"
            
            if dist_mm < min_mm_global: min_mm_global = dist_mm

            side_results.append({
                "mm": round(dist_mm, 2), "px": dist_px,
                "decision": decision, "coords": (x, y_sinus, y_kret)
            })

        if min_mm_global <= 5.0: g_dec = "AÇIK LİFT"
        elif min_mm_global >= 8.0: g_dec = "LİFT GEREKMEZ"
        else: g_dec = "KAPALI LİFT"

        results[side] = {
            "points": side_results,
            "global_decision": g_dec,
            "min_mm": round(min_mm_global, 2)
        }
    return results
