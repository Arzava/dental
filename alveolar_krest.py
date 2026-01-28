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
    
    # 2. HAM VERİLERİ HAZIRLA
    # Önce tüm geçerli aralık için ham verileri (y_sinus, y_kret) çıkaralım
    # Böylece algoritma içinde hızlıca gezinebiliriz.
    
    smoothed_kret_ys = get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=12)
    
    # Data structure: index -> {x, y_s, y_k, dist_px, dist_mm}
    # x_min'den başlayarak 0, 1, 2... diye indeksleyeceğiz.
    data_line = []
    
    for x_global in range(x_min, x_max):
        if not valid_cols_mask[x_global]: 
            data_line.append(None)
            continue

        ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
        if ys_s.size == 0: 
            data_line.append(None)
            continue
            
        sinus_bottom = int(ys_s.max())
        
        idx = x_global - x_min
        if idx < 0 or idx >= len(smoothed_kret_ys): 
            data_line.append(None)
            continue
            
        kret_bottom = smoothed_kret_ys[idx]
        
        if kret_bottom is None or kret_bottom <= sinus_bottom: 
            data_line.append(None)
            continue
        
        dist_px = kret_bottom - sinus_bottom
        dist_mm = dist_px * px_to_mm_ratio
        
        data_line.append({
            'x': x_global,
            'y_s': sinus_bottom,
            'y_k': kret_bottom,
            'dist_px': dist_px,
            'dist_mm': dist_mm
        })

    # 3. MERKEZDEN DIŞA GENİŞLEME (CENTER-OUT EXPANSION)
    # En derin noktayı bul (Valid olanlar içinden)
    valid_points = [d for d in data_line if d is not None]
    if not valid_points: return []
    
    # En büyük y_s (ekranda en alt) en derin noktadır.
    deepest_point = max(valid_points, key=lambda p: p['y_s'])
    deepest_y = deepest_point['y_s']
    
    # Deepest point'in data_line içindeki indeksini bul (x_global değil, array index)
    # x = x_min + index olduğu için: index = x - x_min
    start_idx = deepest_point['x'] - x_min
    
    # KURALLAR
    MAX_HEIGHT_DIFF_PX = 50   # 50 pikselden fazla yukarı çıkarsa DUR.
    MAX_BONE_THICKNESS_MM = 13.0 # 13mm'den kalın kemik görürsen DUR.
    
    # Sola Yürü
    left_boundary_idx = start_idx
    for i in range(start_idx, -1, -1):
        pt = data_line[i]
        if pt is None: continue # Maske kopuksa atla ama durma (belki küçük bir deliktir)
        
        # Kural 1: Yükseklik Kontrolü (Duvar mı?)
        # Sinüs tabanı (y_s) Deepest'ten çok küçükse (yukarıdaysa) dur.
        if (deepest_y - pt['y_s']) > MAX_HEIGHT_DIFF_PX:
            break # Duvar başladı, kes.
            
        # Kural 2: Kalınlık Kontrolü (Komşuluk bitti mi?)
        if pt['dist_mm'] > MAX_BONE_THICKNESS_MM:
            break # Kemik çok kalınlaştı, komşuluk bitti, kes.
            
        left_boundary_idx = i

    # Sağa Yürü
    right_boundary_idx = start_idx
    for i in range(start_idx, len(data_line)):
        pt = data_line[i]
        if pt is None: continue
        
        if (deepest_y - pt['y_s']) > MAX_HEIGHT_DIFF_PX:
            break
            
        if pt['dist_mm'] > MAX_BONE_THICKNESS_MM:
            break
            
        right_boundary_idx = i
        
    # 4. YENİ GEÇERLİ LİSTE (Filtered Profile)
    # Sadece sol ve sağ sınırın içindeki noktaları al
    final_profile = []
    for i in range(left_boundary_idx, right_boundary_idx + 1):
        if data_line[i] is not None:
            final_profile.append(data_line[i])
            
    if not final_profile: return []

    # 5. NOKTA SEÇİMİ (BAŞ - ORTA - SON + EN DÜŞÜK)
    candidates = []
    
    # Profil genişliği
    p_width = final_profile[-1]['x'] - final_profile[0]['x']
    p_start_x = final_profile[0]['x']
    
    # Profil dictionary lookup (hızlı erişim için)
    profile_by_x = {p['x']: p for p in final_profile}
    valid_xs = sorted(profile_by_x.keys())
    valid_xs_arr = np.array(valid_xs)

    # A. ZORUNLU KONUMLAR (%15, %50, %85)
    target_ratios = [0.15, 0.50, 0.85]
    if len(valid_xs) > 0:
        for ratio in target_ratios:
            target_x = int(p_start_x + p_width * ratio)
            # En yakın geçerli noktayı bul
            idx = (np.abs(valid_xs_arr - target_x)).argmin()
            nearest_x = valid_xs_arr[idx]
            
            # Hedefe makul mesafedeyse ekle (30px tolerans)
            if abs(nearest_x - target_x) < 30:
                candidates.append(profile_by_x[nearest_x])

    # B. GLOBAL MİNİMUM (En İnce) - Mutlaka olmalı
    final_profile.sort(key=lambda p: p['dist_px'])
    global_min = final_profile[0]
    candidates.append(global_min)
    
    # C. BİRLEŞTİRME VE TEMİZLEME
    final_points = []
    min_separation = 30 
    
    # Öncelik en ince olanda olsun diye mesafeye göre sırala
    candidates.sort(key=lambda p: p['dist_px'])
    
    for cand in candidates:
        is_duplicate = False
        for existing in final_points:
            if abs(cand['x'] - existing['x']) < min_separation:
                is_duplicate = True
                break
        if not is_duplicate:
            final_points.append(cand)
            
    # Sonuçları Formatla
    final_output = []
    final_points.sort(key=lambda p: p['x']) # Soldan sağa çizilsin
    
    for p in final_points:
        final_output.append((p['dist_px'], p['x'], p['y_s'], p['y_k']))
            
    return final_output
"""
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
            
            # --- YENİ KARAR MEKANİZMASI ---
            if dist_mm <= 3.0:
                decision = "AÇIK LİFT (Çift Aşamalı)"
            elif dist_mm <= 5.0:
                decision = "AÇIK LİFT (Tek Aşamalı)"
            elif dist_mm < 8.0:
                decision = "KAPALI LİFT"
            else:
                decision = "LİFT GEREKMEZ"
            
            if dist_mm < min_mm_global: min_mm_global = dist_mm

            side_results.append({
                "mm": round(dist_mm, 2), "px": dist_px,
                "decision": decision, "coords": (x, y_sinus, y_kret)
            })

        # Global Karar (En kritik değere göre)
        if min_mm_global <= 3.0:
            g_dec = "AÇIK LİFT (Çift Aşamalı)"
        elif min_mm_global <= 5.0:
            g_dec = "AÇIK LİFT (Tek Aşamalı)"
        elif min_mm_global < 8.0:
            g_dec = "KAPALI LİFT"
        else:
            g_dec = "LİFT GEREKMEZ"

        results[side] = {
            "points": side_results,
            "global_decision": g_dec,
            "min_mm": round(min_mm_global, 2)
        }
    return results
    """
    def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1):
    results = {}
    for side in ["LEFT", "RIGHT"]:
        points = compute_multi_thickness(res, image, side, px_to_mm_ratio, num_points=3)
        side_results = []
        
        if not points:
            results[side] = {
                "points": [],
                "global_decision": "NO MEASUREMENT",
                "min_mm": None
            }
            continue
            
        min_mm_global = 999.0
        for p in points:
            dist_px, x, y_sinus, y_kret = p
            dist_mm = dist_px * px_to_mm_ratio
            
            # --- DECISION MECHANISM ---
            if dist_mm <= 3.0:
                decision = "OPEN LIFT (TWO-STAGE)"
            elif dist_mm <= 5.0:
                decision = "OPEN LIFT (SINGLE-STAGE)"
            elif dist_mm < 8.0:
                decision = "CLOSED LIFT"
            else:
                decision = "NO LIFT REQUIRED"
            
            if dist_mm < min_mm_global:
                min_mm_global = dist_mm

            side_results.append({
                "mm": round(dist_mm, 2),
                "px": dist_px,
                "decision": decision,
                "coords": (x, y_sinus, y_kret)
            })

        # Global Decision (based on the most critical value)
        if min_mm_global <= 3.0:
            g_dec = "OPEN LIFT (TWO-STAGE)"
        elif min_mm_global <= 5.0:
            g_dec = "OPEN LIFT (SINGLE-STAGE)"
        elif min_mm_global < 8.0:
            g_dec = "CLOSED LIFT"
        else:
            g_dec = "NO LIFT REQUIRED"

        results[side] = {
            "points": side_results,
            "global_decision": g_dec,
            "min_mm": round(min_mm_global, 2)
        }

    return results
