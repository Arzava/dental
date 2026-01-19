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

def get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=12):
    """Kret yüzeyini yumuşatır."""
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

def compute_multi_thickness(res, image, side, num_points=3):
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)
    
    # 1. GEÇERLİ ALANI BUL
    # Filtreleri kaldırdık, maske nerede varsa orası geçerlidir.
    has_sinus = np.any(sinus_mask > 0, axis=0)
    has_kret = np.any(kret_mask > 0, axis=0)
    valid_cols_mask = np.logical_and(has_sinus, has_kret)
    
    common_cols = np.where(valid_cols_mask)[0]
    if common_cols.size == 0: return []

    x_min, x_max = common_cols.min(), common_cols.max()
    width = x_max - x_min
    if width < 10: return []

    # 2. PROFİL ÇIKARMA
    smoothed_kret_ys = get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=12)
    
    # Tüm geçerli noktaları kaydet
    full_profile = [] # (dist, x, y_s, y_k)
    
    for x_global in range(x_min, x_max):
        if not valid_cols_mask[x_global]: continue

        ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
        if ys_s.size == 0: continue
        sinus_bottom = int(ys_s.max())
        
        idx = x_global - x_min
        if idx < 0 or idx >= len(smoothed_kret_ys): continue
        kret_bottom = smoothed_kret_ys[idx]
        
        if kret_bottom is None or kret_bottom <= sinus_bottom: continue
        
        dist = kret_bottom - sinus_bottom
        full_profile.append({'dist': dist, 'x': x_global, 'y_s': sinus_bottom, 'y_k': kret_bottom})

    if not full_profile: return []

    # --- YENİ STRATEJİ: HİBRİT SEÇİM (Zorunlu Konumlar + En Düşükler) ---
    
    candidates = []

    # A. ZORUNLU KONUMLAR (Baş, Orta, Son)
    # Maske kenarlarındaki hatalardan kaçınmak için %15, %50, %85 noktalarını hedefle
    target_ratios = [0.15, 0.50, 0.85]
    
    # Hızlı erişim için profili X'e göre indeksle
    profile_by_x = {p['x']: p for p in full_profile}
    valid_xs = sorted(profile_by_x.keys())
    valid_xs_arr = np.array(valid_xs)
    
    for ratio in target_ratios:
        target_x = int(x_min + width * ratio)
        
        # Bu hedefe en yakın geçerli noktayı bul
        idx = (np.abs(valid_xs_arr - target_x)).argmin()
        nearest_x = valid_xs_arr[idx]
        
        # Eğer çok uzak değilse (örn: arada büyük boşluk yoksa) al
        if abs(nearest_x - target_x) < 30:
            candidates.append(profile_by_x[nearest_x])

    # B. EN DÜŞÜK NOKTALAR (RİSK GRUBU)
    # Tüm profilin en düşük noktasını (Global Min) kesinlikle ekle
    full_profile.sort(key=lambda p: p['dist'])
    global_min = full_profile[0]
    candidates.append(global_min)
    
    # Varsa ikinci bir derin çukuru da ekleyebiliriz (opsiyonel)
    # Ama şimdilik Global Min yeterli, kalabalık olmasın.

    # C. BİRLEŞTİRME VE TEMİZLEME (MERGE & DEDUPLICATE)
    # Aday noktalar birbirine çok yakınsa, daha düşük (riskli) olanı tut.
    
    final_points = []
    min_separation = 40 # 40 pikselden yakın noktaları birleştir
    
    # Adayları mesafeye (riske) göre sırala ki öncelik en düşükte olsun
    candidates.sort(key=lambda p: p['dist'])
    
    for cand in candidates:
        is_duplicate = False
        for i, existing in enumerate(final_points):
            if abs(cand['x'] - existing['x']) < min_separation:
                # Çakışma var! 
                # Zaten listeye 'en düşükten' başlayarak eklediğimiz için
                # listedeki nokta muhtemelen daha iyidir veya aynısıdır.
                # O yüzden bu adayı pas geçiyoruz.
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_points.append(cand)
            
    # 5. GÖSTERİM SIRALAMASI (Soldan Sağa)
    final_output = []
    final_points.sort(key=lambda p: p['x'])
    
    for p in final_points:
        final_output.append((p['dist'], p['x'], p['y_s'], p['y_k']))
            
    return final_output

def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1):
    results = {}
    for side in ["LEFT", "RIGHT"]:
        points = compute_multi_thickness(res, image, side, num_points=3)
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
