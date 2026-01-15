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

def compute_local_thickness(res, image, side):
    h, w = image.shape[:2]
    sinus_mask = rasterize_class(res, SINUS_CLASS, h, w)
    kret_mask  = rasterize_class(res, KRET_CLASS,  h, w)

    sinus_mask = keep_half(sinus_mask, side)
    kret_mask  = keep_half(kret_mask,  side)

    sinus_cols = np.where(np.any(sinus_mask > 0, axis=0))[0]
    if sinus_cols.size == 0: return None, None, None, None

    x_min, x_max = sinus_cols.min(), sinus_cols.max()
    best = None 

    for x in range(x_min, x_max + 1):
        ys_s = np.where(sinus_mask[:, x] > 0)[0]
        ys_k = np.where(kret_mask[:, x] > 0)[0]
        if ys_s.size == 0 or ys_k.size == 0: continue

        sinus_bottom = int(ys_s.max())
        kret_bottom  = int(ys_k.max())

        if kret_bottom <= sinus_bottom: continue

        dist = kret_bottom - sinus_bottom
        if best is None or dist < best[0]:
            best = (dist, x, sinus_bottom, kret_bottom)

    if best is None: return None, None, None, None
    return best

def alveolar_krest_analysis(res, image, px_to_mm_ratio=0.1):
    """
    3 Aşamalı Klinik Karar:
    - 5mm ve altı: AÇIK LİFT
    - 6-8 mm arası: KAPALI LİFT
    - 8mm ve üzeri: LİFT GEREKMEZ
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
            dist_mm = dist_px * px_to_mm_ratio
            
            # --- YENİ KLİNİK KARAR MANTIĞI ---
            if dist_mm <= 5.0:
                decision = "AÇIK LİFT"
            elif dist_mm >= 8.0:
                decision = "LİFT GEREKMEZ"
            else:
                decision = "KAPALI LİFT" # 5.0 ile 8.0 arası

            results[side] = {
                "thickness_px": int(dist_px),
                "thickness_mm": round(dist_mm, 2),
                "decision": decision,
                "x_col": x,
                "sinus_y": y_sinus,
                "kret_y": y_kret
            }

    return results
