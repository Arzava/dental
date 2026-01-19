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

def get_sinus_floor_boundaries_finetuned(sinus_mask, x_min, x_max):
    """
    FINE-TUNING YAPILMIŞ HİBRİT SINIR TESPİTİ
    Amaç: Derin cepleri (kavisleri) koru ama dik duvarları kes.
    Yöntem: 
    1. Profili "ütüle" (smooth) ki anlık hatalar yanıltmasın.
    2. Merkezden dışa doğru yürü.
    3. Yükseklik 45px'i geçerse VEYA Eğim (Slope) 0.6'yı (duvar açısı) geçerse dur.
    4. Hata durumunda 'Güvenli Mod' (30px) devreye girer.
    """
    
    # 1. Ham Profili Çıkar
    sinus_profile = []
    valid_xs = []
    for x in range(x_min, x_max):
        ys = np.where(sinus_mask[:, x] > 0)[0]
        if ys.size > 0:
            sinus_profile.append(int(ys.max())) 
            valid_xs.append(x)
        else:
            sinus_profile.append(None)
            valid_xs.append(x)
            
    # Eğer veri çok azsa direkt dön
    if len([y for y in sinus_profile if y is not None]) < 10:
        return x_min, x_max

    # 2. Yumuşatma (Smoothing) - Gürültüyü engellemek için kritik
    clean_y = []
    last_valid = 0
    # None boşluklarını doldur
    for y in sinus_profile:
        if y is not None: 
            clean_y.append(y)
            last_valid = y
        else: 
            clean_y.append(last_valid)
    
    # Kernel size 9 ile yumuşat (Keskin çıkışları törpüler)
    smooth_kernel_size = 9
    smooth_y = np.convolve(clean_y, np.ones(smooth_kernel_size)/smooth_kernel_size, mode='same')
    
    # 3. En derin noktayı bul
    deepest_val = max(clean_y) # Görüntü koordinatında en büyük Y = En alt nokta
    
    # En derine yakın (2px tolerans) noktaları bul ve ortadakini merkez seç
    deepest_indices = [i for i, y in enumerate(smooth_y) if y >= deepest_val - 2]
    if not deepest_indices: deepest_indices = [len(smooth_y)//2]
    center_idx = deepest_indices[len(deepest_indices)//2]

    # ---- AYARLAR (FINE TUNING) ----
    MAX_HEIGHT_LIMIT = 45   # 30 az geliyordu, 50 çoktu. 45 ideal.
    MAX_SLOPE_LIMIT = 0.60  # Duvar kabul etme eğimi (0.6 ~ 30 derece)
    SLOPE_LOOKBACK = 15     # Eğimi hesaplarken kaç piksel geriye bakılsın? (Geniş bakış)
    
    # 4. SOLA TARA
    start_idx = 0
    # Merkezden biraz (10px) uzaklaşana kadar slope kontrolü yapma (Dip düzdür)
    for i in range(center_idx, 0, -1):
        # A. Yükseklik Kontrolü
        current_h = deepest_val - smooth_y[i]
        if current_h > MAX_HEIGHT_LIMIT:
            start_idx = i + 1
            break
        
        # B. Eğim Kontrolü (Sadece merkezden uzaklaşınca bak)
        if abs(center_idx - i) > 10:
            lookback_idx = min(len(smooth_y)-1, i + SLOPE_LOOKBACK) 
            # Sola giderken "ilerisi" (i+lookback) daha yüksek (küçük Y) olmalı
            dy = abs(smooth_y[lookback_idx] - smooth_y[i])
            dx = abs(lookback_idx - i)
            if dx > 0:
                slope = dy / dx
                if slope > MAX_SLOPE_LIMIT:
                    start_idx = i + 1 # Duvar başladı, kes.
                    break
    
    # 5. SAĞA TARA
    end_idx = len(valid_xs) - 1
    for i in range(center_idx, len(valid_xs) - 1):
        # A. Yükseklik Kontrolü
        current_h = deepest_val - smooth_y[i]
        if current_h > MAX_HEIGHT_LIMIT:
            end_idx = i - 1
            break
            
        # B. Eğim Kontrolü
        if abs(i - center_idx) > 10:
            lookback_idx = max(0, i - SLOPE_LOOKBACK)
            dy = abs(smooth_y[lookback_idx] - smooth_y[i])
            dx = abs(i - lookback_idx)
            if dx > 0:
                slope = dy / dx
                if slope > MAX_SLOPE_LIMIT:
                    end_idx = i - 1 # Duvar başladı, kes.
                    break

    # 6. GÜVENLİK SİGORTASI (Fallback)
    # Eğer akıllı algoritma çok dar bir alan (<15px) seçerse hata yapmış demektir.
    # Bu durumda "Basit 30px Kesimi"ne geri dön. Hepsini silmesin.
    if (end_idx - start_idx) < 15:
        # Fallback: Sadece yükseklik 30px
        safe_limit = 30
        
        # Sola basit tara
        s_safe = 0
        for i in range(center_idx, -1, -1):
            if (deepest_val - smooth_y[i]) > safe_limit:
                s_safe = i
                break
        # Sağa basit tara
        e_safe = len(valid_xs) - 1
        for i in range(center_idx, len(valid_xs)):
            if (deepest_val - smooth_y[i]) > safe_limit:
                e_safe = i
                break
        
        return valid_xs[s_safe], valid_xs[e_safe]

    return valid_xs[start_idx], valid_xs[end_idx]

def compute_multi_thickness(res, image, side, px_to_mm_ratio, num_points=3):
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)
    
    # 1. ORTAK ALAN
    has_sinus = np.any(sinus_mask > 0, axis=0)
    has_kret = np.any(kret_mask > 0, axis=0)
    valid_cols_mask = np.logical_and(has_sinus, has_kret)
    
    common_cols = np.where(valid_cols_mask)[0]
    if common_cols.size == 0: return []

    x_min, x_max = common_cols.min(), common_cols.max()
    
    # --- YENİ FINE-TUNED DUVAR KESME ---
    floor_start, floor_end = get_sinus_floor_boundaries_finetuned(
        sinus_mask, x_min, x_max
    )
    
    width = floor_end - floor_start
    if width < 10: return [] 

    # 2. PROFİL
    smoothed_kret_ys = get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=12)
    
    full_profile = [] 
    
    for x_global in range(floor_start, floor_end):
        if not valid_cols_mask[x_global]: continue

        ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
        if ys_s.size == 0: continue
        sinus_bottom = int(ys_s.max())
        
        idx = x_global - x_min
        if idx < 0 or idx >= len(smoothed_kret_ys): continue
        kret_bottom = smoothed_kret_ys[idx]
        
        if kret_bottom is None or kret_bottom <= sinus_bottom: continue
        
        dist_px = kret_bottom - sinus_bottom
        dist_mm = dist_px * px_to_mm_ratio
        
        # Ekstra güvenlik: 20mm üstü kesin duvardır
        if dist_mm > 20.0: continue 
            
        full_profile.append({'dist': dist_px, 'x': x_global, 'y_s': sinus_bottom, 'y_k': kret_bottom})

    if not full_profile: return []

    # 3. NOKTA SEÇİMİ
    candidates = []

    # A. ZORUNLU KONUMLAR (%15, %50, %85)
    target_ratios = [0.15, 0.50, 0.85]
    
    profile_by_x = {p['x']: p for p in full_profile}
    valid_xs = sorted(profile_by_x.keys())
    valid_xs_arr = np.array(valid_xs)
    
    if len(valid_xs) > 0:
        for ratio in target_ratios:
            target_x = int(floor_start + width * ratio)
            idx = (np.abs(valid_xs_arr - target_x)).argmin()
            nearest_x = valid_xs_arr[idx]
            
            # 40 piksel toleransla hedefi bul
            if abs(nearest_x - target_x) < 40:
                candidates.append(profile_by_x[nearest_x])

    # B. GLOBAL MİNİMUM
    full_profile.sort(key=lambda p: p['dist'])
    global_min = full_profile[0]
    candidates.append(global_min)
    
    # C. BİRLEŞTİRME
    final_points = []
    min_separation = 30 
    
    candidates.sort(key=lambda p: p['dist'])
    
    for cand in candidates:
        is_duplicate = False
        for existing in final_points:
            if abs(cand['x'] - existing['x']) < min_separation:
                is_duplicate = True
                break
        if not is_duplicate:
            final_points.append(cand)
            
    final_output = []
    final_points.sort(key=lambda p: p['x'])
    
    for p in final_points:
        final_output.append((p['dist'], p['x'], p['y_s'], p['y_k']))
            
    return final_output

def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1):
    results = {}
    for side in ["LEFT", "RIGHT"]:
        points = compute_multi_thickness(res, image, side, px_to_mm_ratio, num_points=3)
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
