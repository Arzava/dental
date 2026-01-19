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
    """
    Diş kökü boşluklarını doldurmak için 12 piksellik pencere.
    """
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
    
    # ORTA HAT FİLTRESİ (%10)
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
    
    # EĞİM VE DUVAR FİLTRESİ
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
    
    measurements = []

    # --- YENİ STRATEJİ: AYRIK PENCERELER (Baş - Orta - Son) ---
    # Eğer alan yeterince genişse (>60px), zorla ayrıştırılmış 3 bölgeye bak.
    # Değilse tek bölgeye bak.
    
    zones = []
    
    if width < 60:
        # Alan çok dar, tek bir ölçüm al (Tamamı)
        zones.append((real_x_min, real_x_max))
    else:
        # Alan geniş, 3 ayrık pencere tanımla
        # Pencere genişliği: Toplam genişliğin %25'i
        zone_w = int(width * 0.25)
        
        # 1. BAŞ (START): İlk %25
        zones.append((real_x_min, real_x_min + zone_w))
        
        # 2. ORTA (MID): Tam ortadaki %25
        mid_center = real_x_min + width // 2
        zones.append((mid_center - zone_w // 2, mid_center + zone_w // 2))
        
        # 3. SON (END): Son %25
        zones.append((real_x_max - zone_w, real_x_max))

    # Tanımlanan her bölge (zone) içinde EN İNCE kemiği bul
    for z_start, z_end in zones:
        best_in_zone = None
        
        for x_global in range(z_start, z_end):
            if not valid_cols_mask[x_global]: continue

            ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
            if x_global < real_x_min or x_global > real_x_max: continue
            if ys_s.size == 0: continue

            sinus_bottom = int(ys_s.max())
            
            idx = x_global - x_min
            if idx < 0 or idx >= len(smoothed_kret_ys): continue
            kret_bottom = smoothed_kret_ys[idx]
            
            if kret_bottom is None or kret_bottom <= sinus_bottom: continue
            
            dist = kret_bottom - sinus_bottom
            
            if best_in_zone is None or dist < best_in_zone[0]:
                best_in_zone = (dist, x_global, sinus_bottom, kret_bottom)
        
        if best_in_zone:
            measurements.append(best_in_zone)
            
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
