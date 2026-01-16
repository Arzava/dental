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

def get_segment_minima(sinus_mask, kret_mask, x_start, x_end):
    """Belirli bir aralıktaki (x_start, x_end) en küçük mesafeyi bulur."""
    best = None # (dist, x, sinus_y, kret_y)
    
    for x in range(x_start, x_end):
        ys_s = np.where(sinus_mask[:, x] > 0)[0]
        ys_k = np.where(kret_mask[:, x] > 0)[0]
        
        if ys_s.size == 0 or ys_k.size == 0: continue

        sinus_bottom = int(ys_s.max())
        kret_bottom  = int(ys_k.max())

        if kret_bottom <= sinus_bottom: continue # Hatalı maske durumu

        dist = kret_bottom - sinus_bottom
        
        # En küçüğü (en riskliyi) buluyoruz
        if best is None or dist < best[0]:
            best = (dist, x, sinus_bottom, kret_bottom)
            
    return best

def compute_multi_thickness(res, image, side, num_segments=3):
    """
    Ölçüm alanını 'num_segments' kadar parçaya bölüp her parçadan ölçüm alır.
    """
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)

    # Ölçüm yapılabilir alanın sınırlarını bul (x_min, x_max)
    # Hem sinüsün hem kretin olduğu ortak kolonlar
    common_cols = np.where(
        (np.any(sinus_mask > 0, axis=0)) & (np.any(kret_mask > 0, axis=0))
    )[0]
    
    if common_cols.size == 0:
        return []

    x_min, x_max = common_cols.min(), common_cols.max()
    width = x_max - x_min
    
    # Alan çok darsa tek ölçüm yap geç
    if width < 50: 
        val = get_segment_minima(sinus_mask, kret_mask, x_min, x_max)
        return [val] if val else []

    # Alanı parçalara böl (Zonal Analysis)
    step = width // num_segments
    measurements = []
    
    for i in range(num_segments):
        seg_start = x_min + (i * step)
        seg_end = seg_start + step
        
        # Son parçada kalan küsuratı da al
        if i == num_segments - 1:
            seg_end = x_max

        val = get_segment_minima(sinus_mask, kret_mask, seg_start, seg_end)
        if val:
            measurements.append(val)
            
    return measurements

def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1):
    """
    Çoklu nokta analizi döndürür.
    Dönen yapı: { "LEFT": [ölçüm1, ölçüm2...], "RIGHT": [...] }
    Her ölçüm: {'mm': 5.2, 'px': 52, 'decision': '...', coords: ...}
    """
    results = {}

    for side in ["LEFT", "RIGHT"]:
        points = compute_multi_thickness(res, image, side, num_segments=3) # 3 Noktadan ölç
        
        side_results = []
        
        if not points:
            # Hiç ölçüm yoksa boş liste
            results[side] = {
                "points": [],
                "global_decision": "ÖLÇÜM YOK",
                "min_mm": None
            }
            continue
            
        # Tüm noktaları işle
        min_mm_global = 999.0
        
        for p in points:
            dist_px, x, y_sinus, y_kret = p
            dist_mm = dist_px * px_to_mm_ratio
            
            # Karar mantığı (Her nokta için ayrı)
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

        # Global Karar (En düşük değere göre)
        if min_mm_global <= 5.0: g_dec = "AÇIK LİFT"
        elif min_mm_global >= 8.0: g_dec = "LİFT GEREKMEZ"
        else: g_dec = "KAPALI LİFT"

        results[side] = {
            "points": side_results,
            "global_decision": g_dec,
            "min_mm": round(min_mm_global, 2)
        }

    return results
