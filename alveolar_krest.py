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
    """
    Kretin alt sınırını (dişlere bakan tarafı) çıkarır ve 
    diş aralarına giren 'ince derinlikleri' (spike) temizler.
    
    Mantık: Bir noktanın kret tabanı, komşularının 'en sığ' (yukarıda) 
    olanına yakın olmalıdır.
    """
    raw_bottoms = []
    
    # 1. Ham veriyi topla
    for x in range(x_min, x_max):
        ys = np.where(kret_mask[:, x] > 0)[0]
        if ys.size > 0:
            # Maskenin o kolondaki en alt noktası (resimde en büyük Y)
            raw_bottoms.append(int(ys.max()))
        else:
            raw_bottoms.append(np.nan)
            
    # 2. Filtrele (Moving Minimum Filter)
    # Bu işlem, aşağı sarkan ince çıkıntıları (diş aralarını) budar.
    smoothed_bottoms = []
    arr = np.array(raw_bottoms)
    
    for i in range(len(arr)):
        # Pencere sınırları
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        window = arr[start:end]
        
        # Geçerli (NaN olmayan) komşuları al
        valid = window[~np.isnan(window)]
        
        if valid.size > 0:
            # Püf Nokta: 'min' alıyoruz. Çünkü görüntü koordinatında Y aşağı artar.
            # Min Y demek, ekranın yukarısı (yani Kret Tepesi) demektir.
            # Böylece aşağı sarkan aykırı değerleri eliyoruz.
            smoothed_bottoms.append(int(np.min(valid)))
        else:
            smoothed_bottoms.append(None)
            
    return smoothed_bottoms

def compute_multi_thickness(res, image, side, num_segments=3):
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)

    # Ortak alan sınırları
    common_cols = np.where(
        (np.any(sinus_mask > 0, axis=0)) & (np.any(kret_mask > 0, axis=0))
    )[0]
    
    if common_cols.size == 0: return []

    x_min, x_max = common_cols.min(), common_cols.max()
    width = x_max - x_min
    
    # --- İYİLEŞTİRME: Tüm bölge için düzeltilmiş kret hattını baştan hesapla ---
    # window_size=30 piksel (yaklaşık bir diş kökü genişliği kadar filtreleme yapar)
    smoothed_kret_ys = get_smoothed_kret_bottom(kret_mask, x_min, x_max, window_size=30)
    
    measurements = []
    step = width // num_segments
    if step < 1: step = 1

    for i in range(num_segments):
        seg_start_rel = i * step
        seg_end_rel = seg_start_rel + step
        if i == num_segments - 1: seg_end_rel = width # Son parçayı tamamla
        
        # Segment içindeki en iyi (en riskli/en ince) ölçümü bul
        best_in_segment = None
        
        for j in range(seg_start_rel, seg_end_rel):
            x_global = x_min + j
            
            # Sinüs tabanını bul (Sarı maskenin en altı)
            ys_s = np.where(sinus_mask[:, x_global] > 0)[0]
            if ys_s.size == 0: continue
            sinus_bottom = int(ys_s.max())
            
            # Düzeltilmiş Kret tabanını al
            kret_bottom = smoothed_kret_ys[j]
            
            if kret_bottom is None or kret_bottom <= sinus_bottom:
                continue
                
            dist = kret_bottom - sinus_bottom
            
            # En küçük mesafeyi (en riskli yeri) kaydet
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
            results[side] = {
                "points": [],
                "global_decision": "ÖLÇÜM YOK",
                "min_mm": None
            }
            continue
            
        min_mm_global = 999.0
        
        for p in points:
            dist_px, x, y_sinus, y_kret = p
            dist_mm = dist_px * px_to_mm_ratio
            
            if dist_mm <= 5.0: decision = "AÇIK LİFT"
            elif dist_mm >= 8.0: decision = "LİFT GEREKMEZ"
            else: decision = "KAPALI LİFT"
            
            if dist_mm < min_mm_global:
                min_mm_global = dist_mm

            side_results.append({
                "mm": round(dist_mm, 2),
                "px": dist_px,
                "decision": decision,
                "coords": (x, y_sinus, y_kret)
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
