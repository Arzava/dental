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
    
    # 1. KESİŞİM ALANI BUL (Filtresiz)
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
    
    # Tüm geçerli noktaları ve mesafeleri kaydet
    profile = [] # (dist, x, y_s, y_k)
    
    for x_global in range(x_min, x_max):
        if not valid_cols_mask[x_global]: continue

        # Sinüs tabanı (En alt nokta)
        ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
        if ys_s.size == 0: continue
        sinus_bottom = int(ys_s.max())
        
        # Kret tabanı
        idx = x_global - x_min
        if idx < 0 or idx >= len(smoothed_kret_ys): continue
        kret_bottom = smoothed_kret_ys[idx]
        
        # Hatalı maske kontrolü
        if kret_bottom is None or kret_bottom <= sinus_bottom: continue
        
        dist = kret_bottom - sinus_bottom
        profile.append({'dist': dist, 'x': x_global, 'y_s': sinus_bottom, 'y_k': kret_bottom})

    if not profile: return []

    # 3. LOKAL MİNİMUMLARI (ÇUKURLARI) BUL
    # Bir nokta, sağındaki ve solundaki komşularından daha düşükse (veya eşitse) lokal minimumdur.
    local_minima = []
    
    # Gürültüyü azaltmak için mesafeler üzerinde hafif bir yumuşatma yapabiliriz ama
    # burada direkt ham veri üzerinden gidelim ki ince detaylar kaçmasın.
    
    for i in range(1, len(profile) - 1):
        prev_p = profile[i-1]['dist']
        curr_p = profile[i]['dist']
        next_p = profile[i+1]['dist']
        
        # Çukur tespiti
        if curr_p <= prev_p and curr_p <= next_p:
            local_minima.append(profile[i])
            
    # Eğer hiç lokal minimum bulunamazsa (dümdüz veya hep artan/azalan grafik),
    # en düşük noktayı ekle.
    if not local_minima:
        # Listeyi mesafeye göre sırala en küçüğü al
        profile.sort(key=lambda p: p['dist'])
        local_minima.append(profile[0])

    # 4. AKILLI SEÇİM (SELECTION)
    # Bulunan çukurları "En Riskli (En İnce)"den başlayarak sırala
    local_minima.sort(key=lambda p: p['dist'])
    
    selected_points = []
    # Noktalar arası minimum mesafe (Piksel cinsinden). 
    # Birbirine çok yakın iki çukuru aynı anda göstermemek için.
    min_spatial_dist = 40 
    
    for candidate in local_minima:
        if len(selected_points) >= num_points:
            break
            
        # Bu aday nokta, daha önce seçilenlere çok yakın mı?
        too_close = False
        for chosen in selected_points:
            if abs(candidate['x'] - chosen['x']) < min_spatial_dist:
                too_close = True
                break
        
        if not too_close:
            selected_points.append(candidate)
    
    # 5. GÖSTERİM SIRALAMASI
    # Seçilen noktaları ekranda soldan sağa düzgün durması için X koordinatına göre sırala
    # Veri formatını eski yapıya çevir (tuple)
    final_output = []
    selected_points.sort(key=lambda p: p['x'])
    
    for p in selected_points:
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
