"""Konfigürasyon sözlüğüyle tanımlanan klasik hücre sayma pipeline'ı.

Her pipeline şu aşamalardan oluşur:
    yoğunluk dönüşümü -> filtre -> segmentasyon -> morfoloji -> (ayırma) -> sayım

Örnek konfigürasyon:
    {
      "intensity": {"method": "clahe", "clip": 2.0, "tile": 8},
      "filter":    {"method": "gaussian", "ksize": 5},
      "segment":   {"method": "otsu"},
      "morph":     [["open", 5], ["fill", 0]],
      "split":     {"method": "watershed", "min_distance": 15},
      "min_area":  150
    }
"""
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


# ----------------------------------------------------------------- yoğunluk
def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def contrast_stretch(img, p_low=1, p_high=99.8):
    lo, hi = np.percentile(img, (p_low, p_high))
    if hi <= lo:
        return img.copy()
    out = (img.astype(np.float32) - lo) / (hi - lo) * 255.0
    return to_uint8(out)


def apply_intensity(img, cfg, ref_img=None):
    m = cfg.get("method", "none")
    if m == "none":
        return img.copy()
    if m == "stretch":
        return contrast_stretch(img, cfg.get("p_low", 1), cfg.get("p_high", 99.8))
    if m == "histeq":
        return cv2.equalizeHist(img)
    if m == "clahe":
        clahe = cv2.createCLAHE(clipLimit=cfg.get("clip", 2.0),
                                tileGridSize=(cfg.get("tile", 8),) * 2)
        return clahe.apply(img)
    if m == "match":
        if ref_img is None:
            raise ValueError("histogram matching için referans görüntü gerekli")
        return to_uint8(exposure.match_histograms(img, ref_img))
    if m == "gamma":
        # önce percentile germe, ardından gamma < 1 ile soluk hücreleri parlatma
        s = contrast_stretch(img, cfg.get("p_low", 1), cfg.get("p_high", 99.8))
        return to_uint8(255.0 * (s / 255.0) ** cfg.get("gamma", 0.5))
    raise ValueError(f"bilinmeyen yoğunluk yöntemi: {m}")


# ------------------------------------------------------------------- filtre
SHARPEN_KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)


def apply_filter(img, cfg):
    m = cfg.get("method", "none")
    k = cfg.get("ksize", 5)
    if m == "none":
        return img.copy()
    if m == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    if m == "median":
        return cv2.medianBlur(img, k)
    if m == "bilateral":
        return cv2.bilateralFilter(img, k, cfg.get("sigma_color", 30), cfg.get("sigma_space", 5))
    if m == "sharpen":
        # konvolüsyon tabanlı Laplacian keskinleştirme (merkez 5, 4-komşu -1)
        return to_uint8(cv2.filter2D(img.astype(np.float32), -1, SHARPEN_KERNEL))
    if m == "gauss_sharpen":
        # önce gürültüyü bastır, sonra unsharp masking ile kenarları güçlendir
        blur = cv2.GaussianBlur(img, (k, k), 0)
        amount = cfg.get("amount", 1.0)
        base = cv2.GaussianBlur(blur, (0, 0), cfg.get("sigma", 3))
        return to_uint8(blur.astype(np.float32) * (1 + amount) - base.astype(np.float32) * amount)
    raise ValueError(f"bilinmeyen filtre: {m}")


# ------------------------------------------------------------- segmentasyon
def apply_segment(img, cfg):
    m = cfg.get("method", "otsu")
    if m == "fixed":
        return img > cfg.get("thresh", 30)
    if m == "otsu":
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img > t
    if m == "triangle":
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return img > t
    if m == "li":
        return img > filters.threshold_li(img)
    if m == "adaptive":
        # adaptif ortalama eşikleme + karanlık arka planda yanlış pozitifleri
        # önlemek için düşük bir global taban eşik
        b = cfg.get("block", 151)
        c = cfg.get("C", -5)
        local = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, b, c) > 0
        return local & (img > cfg.get("floor", 10))
    if m == "noise":
        # arka plan istatistiğine dayalı otomatik eşik: medyan + k * sigma
        # sigma, MAD (median absolute deviation) ile gürbüz biçimde kestirilir.
        # Pikselin çoğu arka plan olduğundan medyan/MAD arka planı temsil eder.
        med = float(np.median(img))
        mad = float(np.median(np.abs(img.astype(np.float32) - med)))
        sigma = max(1.4826 * mad, cfg.get("min_sigma", 0.5))
        return img > med + cfg.get("k", 3.0) * sigma
    if m == "kmeans":
        k = cfg.get("k", 3)
        Z = img.reshape(-1, 1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(0)
        _, lab, centers = cv2.kmeans(Z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(centers.ravel())
        bg = order[0]  # en karanlık küme arka plan
        return (lab.reshape(img.shape) != bg)
    raise ValueError(f"bilinmeyen segmentasyon: {m}")


# ---------------------------------------------------------------- morfoloji
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def apply_morph(mask, ops):
    m = mask.astype(np.uint8)
    for op, r in ops:
        if op == "erode":
            m = cv2.erode(m, disk(r))
        elif op == "dilate":
            m = cv2.dilate(m, disk(r))
        elif op == "open":
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(r))
        elif op == "close":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(r))
        elif op == "fill":
            m = ndi.binary_fill_holes(m).astype(np.uint8)
        else:
            raise ValueError(op)
    return m.astype(bool)


# ------------------------------------------------------------------- ayırma
def apply_split(mask, cfg):
    m = cfg.get("method", "none")
    if m == "none":
        return measure.label(mask, connectivity=2)
    if m == "watershed":
        dist = ndi.distance_transform_edt(mask)
        # h-maxima benzeri bastırma: küçük tepeleri yumuşatmak için hafif blur
        dist_s = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), cfg.get("sigma", 2))
        lab0 = measure.label(mask)
        coords = peak_local_max(dist_s, min_distance=cfg.get("min_distance", 15),
                                threshold_abs=cfg.get("min_peak", 5), labels=lab0,
                                exclude_border=False)
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        lab = segmentation.watershed(-dist_s, markers, mask=mask)
        # marker'ı olmayan küçük bileşenler kaybolmasın
        missing = mask & (lab == 0)
        if missing.any():
            extra = measure.label(missing)
            lab[extra > 0] = extra[extra > 0] + lab.max()
        return lab
    raise ValueError(f"bilinmeyen ayırma yöntemi: {m}")


# ------------------------------------------------------------------ çalıştır
def remove_small(lab, min_area):
    if min_area <= 0:
        return lab
    out = lab.copy()
    areas = np.bincount(lab.ravel())
    small = np.where(areas < min_area)[0]
    small = small[small > 0]
    out[np.isin(lab, small)] = 0
    out, _, _ = segmentation.relabel_sequential(out)
    return out


def clear_border_partial(lab, max_area):
    """Görüntü kenarına değen ve alanı max_area'dan küçük (yani büyük kısmı
    görüş alanı dışında kalan) nesneleri siler. Referans anotasyonlarda bu
    kısmi hücreler çoğunlukla işaretlenmemiştir."""
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    if len(edge) == 0:
        return lab
    areas = np.bincount(lab.ravel())
    drop = [i for i in edge if areas[i] < max_area]
    out = lab.copy()
    out[np.isin(lab, drop)] = 0
    out, _, _ = segmentation.relabel_sequential(out)
    return out


def run_pipeline(img, cfg, ref_img=None, return_steps=False):
    t0 = time.perf_counter()
    steps = {"original": img}
    x = apply_intensity(img, cfg.get("intensity", {"method": "none"}), ref_img)
    steps["intensity"] = x
    x = apply_filter(x, cfg.get("filter", {"method": "none"}))
    steps["filter"] = x
    mask = apply_segment(x, cfg.get("segment", {"method": "otsu"}))
    steps["segment"] = mask
    mask = apply_morph(mask, cfg.get("morph", []))
    steps["morph"] = mask
    lab = apply_split(mask, cfg.get("split", {"method": "none"}))
    lab = remove_small(lab, cfg.get("min_area", 0))
    if cfg.get("clear_border", False):
        lab = clear_border_partial(lab, cfg.get("border_max_area", 2500))
    steps["labels"] = lab
    count = int(lab.max())
    elapsed = time.perf_counter() - t0
    if return_steps:
        return count, elapsed, steps
    return count, elapsed


# Baseline: gri görüntü -> Otsu -> bağlı bileşen -> sabit min alan filtresi
BASELINE = {
    "name": "Baseline",
    "intensity": {"method": "none"},
    "filter": {"method": "none"},
    "segment": {"method": "otsu"},
    "morph": [],
    "split": {"method": "none"},
    "min_area": 100,
}


def describe(cfg):
    """Deney tablosu için okunabilir kısa açıklamalar."""
    def kv(d):
        d = {k: v for k, v in d.items() if k != "method"}
        return ",".join(f"{k}={v}" for k, v in d.items())
    it, fl, sg = cfg["intensity"], cfg["filter"], cfg["segment"]
    morph = " -> ".join(f"{o}({r})" if o != "fill" else "fill" for o, r in cfg["morph"]) or "yok"
    sp = cfg["split"]
    return {
        "intensity": it["method"],
        "filter": fl["method"] + (f"{fl['ksize']}x{fl['ksize']}" if "ksize" in fl and fl["method"] != "none" else ""),
        "segmentation": sg["method"],
        "morphology": morph,
        "split": sp["method"] + (f"(md={sp.get('min_distance')})" if sp["method"] == "watershed" else ""),
        "params": "; ".join(x for x in [kv(it), kv(fl), kv(sg), f"min_area={cfg['min_area']}"] if x),
    }
