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

def get_sinus_floor_boundaries(sinus_mask, x_min, x_max, wall_height_limit=30):
    """
    GÜNCELLENDİ: wall_height_limit = 30
    Sinüsün en derin noktasından 30 piksel yukarı çıkılmasına izin verilir.
    """
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
            
    if not any(y is not None for y in sinus_profile):
        return x_min, x_max

    valid_ys = [y for y in sinus_profile if y is not None]
    if not valid_ys: return x_min, x_max
    
    deepest_y = max(valid_ys) 
    
    # Sınır: En derin noktadan 30 piksel yukarısı
    ceiling_y = deepest_y - wall_height_limit
    
    deepest_indices = [i for i, y in enumerate(sinus_profile) if y == deepest_y]
    center_idx = deepest_indices[len(deepest_indices)//2]
    
    # SOLA TARA
    start_idx = 0
    for i in range(center_idx, -1, -1):
        y = sinus_profile[i]
        if y is None or y < ceiling_y: 
            start_idx = i + 1 
            break
            
    # SAĞA TARA
    end_idx = len(sinus_profile) - 1
    for i in range(center_idx, len(sinus_profile)):
        y = sinus_profile[i]
        if y is None or y < ceiling_y: 
            end_idx = i - 1
            break
            
    real_x_start = valid_xs[start_idx]
    real_x_end = valid_xs[end_idx]
    
    return real_x_start, real_x_end

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
    
    # --- DUVAR KESME (30px LİMİT) ---
    floor_start, floor_end = get_sinus_floor_boundaries(
        sinus_mask, x_min, x_max, wall_height_limit=30
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
        
        # 20 mm üstü (duvar kalıntısı varsa) at
        dist_mm = dist_px * px_to_mm_ratio
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
            
            if abs(nearest_x - target_x) < 30:
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
