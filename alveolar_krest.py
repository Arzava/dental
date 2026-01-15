import numpy as np
import cv2

# ---- CLASS ID'LER ----
KRET_CLASS = 0
SINUS_CLASS = 3

def rasterize_class(res, cls_id, h, w):
    """Polygonları tek bir binary maskeye çevirir."""
    mask = np.zeros((h, w), dtype=np.uint8)

    if res.masks is None:
        return mask

    polys = res.masks.xy
    classes = res.boxes.cls.cpu().numpy().astype(int)

    for poly, c in zip(polys, classes):
        if c != cls_id:
            continue
        pts = poly.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask

def keep_half(mask, side):
    """Maskeyi sağ / sol yarıya kırpar."""
    h, w = mask.shape
    mid = w // 2
    out = mask.copy()

    if side == "LEFT":
        out[:, mid:] = 0
    else:
        out[:, :mid] = 0

    return out

def compute_local_thickness(res, image, side):
    """LOKAL KOLON TABANLI ÖLÇÜM (Piksel olarak en kısa mesafeyi bulur)"""
    h, w = image.shape[:2]

    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)

    # sinüs bulunan kolonlar
    sinus_cols = np.where(np.any(sinus_mask > 0, axis=0))[0]
    if sinus_cols.size == 0:
        return None, None, None, None

    x_min, x_max = sinus_cols.min(), sinus_cols.max()

    best = None  # (dist, x, sinus_y, kret_y)

    for x in range(x_min, x_max + 1):
        ys_s = np.where(sinus_mask[:, x] > 0)[0]
        ys_k = np.where(kret_mask[:, x] > 0)[0]

        if ys_s.size == 0 or ys_k.size == 0:
            continue

        sinus_bottom = int(ys_s.max())   # sinüs tabanı
        kret_bottom  = int(ys_k.max())   # alveoler kret TABANI

        if kret_bottom <= sinus_bottom:
            continue

        dist = kret_bottom - sinus_bottom

        if best is None or dist < best[0]:
            best = (dist, x, sinus_bottom, kret_bottom)

    if best is None:
        return None, None, None, None

    return best


def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1, threshold_mm=5.0):
    """
    Sağ ve sol yarı için analiz.
    px_to_mm_ratio: 1 pikselin kaç mm'ye denk geldiği (Katsayı).
    threshold_mm: Karar vermek için mm cinsinden sınır değeri.
    """
    results = {}

    for side in ["LEFT", "RIGHT"]:
        dist_px, x, y_sinus, y_kret = compute_local_thickness(res, image, side)

        if dist_px is None:
            results[side] = {
                "thickness_px": None,
                "thickness_mm": None,
                "decision": "ÖLÇÜM YOK",
                "x_col": None, "sinus_y": None, "kret_y": None
            }
        else:
            # --- KRİTİK DEĞİŞİKLİK BURADA ---
            # Pikseli mm'ye çeviriyoruz
            dist_mm = dist_px * px_to_mm_ratio
            
            # Kararı MM üzerinden veriyoruz
            decision = "GRAFT GEREKLİ" if dist_mm < threshold_mm else "GRAFT GEREKMEZ"

            results[side] = {
                "thickness_px": int(dist_px),
                "thickness_mm": round(dist_mm, 2), # Virgülden sonra 2 hane (Örn: 4.52 mm)
                "decision": decision,
                "x_col": x,
                "sinus_y": y_sinus,
                "kret_y": y_kret
            }

    return results
