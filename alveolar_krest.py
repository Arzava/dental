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

def get_valid_sinus_floor_range(sinus_mask, x_min, x_max, slope_threshold=1.0, height_tolerance_px=80):
    sinus_bottoms = []
    valid_indices = [] 
    
    for x in range(x_min, x_max):
        ys = np.where(sinus_mask[:, x] > 0)[0]
        if ys.size > 0:
            val = int(ys.max())
            sinus_bottoms.append(val)
            valid_indices.append(x)
    
    if not sinus_bottoms: return []

    y_vals = np.array(sinus_bottoms)
    
    deepest_y = np.max(y_vals)
    cutoff_y = deepest_y - height_tolerance_px
    is_in_height_range = y_vals > cutoff_y
    
    kernel_size = 15
    if len(y_vals) > kernel_size:
        y_smooth = np.convolve(y_vals, np.ones(kernel_size)/kernel_size, mode='same')
    else:
        y_smooth = y_vals

    gradients = np.gradient(y_smooth)
    is_flat_slope = np.abs(gradients) < slope_threshold
    
    is_valid_floor = is_in_height_range & is_flat_slope
    
    if not np.any(is_valid_floor): return [] 

    valid_indices_arr = np.array(valid_indices)
    floor_indices = np.where(is_valid_floor)[0]
    
    from itertools import groupby
    from operator import itemgetter
    
    segments = []
    for k, g in groupby(enumerate(floor_indices), lambda ix: ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))
        
    longest_segment_indices = max(segments, key=len)
    final_range = valid_indices_arr[longest_segment_indices]
    
    return final_range.tolist()

def compute_multi_thickness(res, image, side, num_segments=3):
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)
    
    # ORTA HAT FİLTRESİ
    center_x = w // 2
    dead_zone_width = int(w * 0.10) 
    safe_zone_start = center_x - dead_zone_width
    safe_zone_end   = center_x + dead_zone_width

    has_sinus = np.any(sinus_mask > 0, axis=0)
    has_kret = np.any(kret_mask > 0, axis=0)
    valid_cols_mask = np.logical_and(has_sinus, has_kret)
    valid_cols_mask[safe_zone_start:safe_zone_end] = False
    
    common_cols = np.where(valid_cols_mask)[0]
    if common_cols.size == 0: return []

    x_min, x_max = common_cols.min(), common_cols.max()
    
    valid_x_range = get_valid_sinus_floor_range(
        sinus_mask, x_min, x_max, 
        slope_threshold=1.0, 
        height_tolerance_px=80
    )
    
    if not valid_x_range: return []
        
    real_x_min = min(valid_x_range)
    real_x_max = max(valid_x_range)
    width = real_x_max - real_x_min
    
    if width < 10: return []

    smoothed_kret_ys = get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=12)
    
    # --- YENİ MANTIK: TÜM ALANI TARA VE SEÇ ---
    # 1. Önce tüm geçerli noktaları bir listeye topla
    all_possible_points = []
    
    for x_global in range(real_x_min, real_x_max):
        if not valid_cols_mask[x_global]: continue

        ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
        if ys_s.size == 0: continue

        sinus_bottom = int(ys_s.max())
        
        idx = x_global - x_min
        if idx < 0 or idx >= len(smoothed_kret_ys): continue
        kret_bottom = smoothed_kret_ys[idx]
        
        if kret_bottom is None or kret_bottom <= sinus_bottom: continue
        
        dist = kret_bottom - sinus_bottom
        all_possible_points.append((dist, x_global, sinus_bottom, kret_bottom))
    
    if not all_possible_points:
        return []

    # 2. Mesafeye göre sırala (En küçük/riskli en başta)
    all_possible_points.sort(key=lambda x: x[0])
    
    # 3. Birbirinden uzak 3 noktayı seç
    selected_points = []
    # Noktaların birbirine minimum uzaklığı (Genişliğin %15'i kadar olsun)
    min_separation = max(15, width * 0.15) 
    
    for pt in all_possible_points:
        dist, x, y_s, y_k = pt
        
        # Eğer yeterince nokta seçtiysek dur
        if len(selected_points) >= num_segments:
            break
            
        # Çakışma kontrolü
        is_far_enough = True
        for existing in selected_points:
            existing_x = existing[1]
            if abs(x - existing_x) < min_separation:
                is_far_enough = False
                break
        
        if is_far_enough:
            selected_points.append(pt)
            
    # Eğer hiç nokta seçemediysek (mesela alan çok dar ve min_separation büyük geldi)
    # En iyi tek noktayı al
    if not selected_points and all_possible_points:
        selected_points.append(all_possible_points[0])
        
    # 4. Seçilen noktaları X koordinatına göre (Soldan Sağa) sırala
    # Böylece ekranda Start - Mid - End gibi görünürler
    selected_points.sort(key=lambda x: x[1])
            
    return selected_points

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
