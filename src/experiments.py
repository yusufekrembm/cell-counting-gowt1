"""Geliştirme kümesi (Sequence 01, 10 kare) üzerinde kontrollü deneyler.

Deneyler aşamalar halinde tasarlandı; her aşamada bir önceki aşamanın en
iyi sonucunu veren ayar sabit tutulup yalnızca incelenen bileşen değiştirildi:

  S0  Baseline
  S1  Yoğunluk dönüşümleri      (filtre G5, Otsu, open3+fill sabit)
  S2  Segmentasyon yöntemleri   (yoğunluk yok, G5, open3+fill sabit)
  S3  Kenara değen kısmi hücrelerin silinmesi
  S4  Filtre ve kernel boyutu   (sabit eşik T=4, open3+fill, kenar kuralı)
  S5  Morfoloji ve işlem sırası (G9, T=4, kenar kuralı)
  S6  Kernel boyutu x açma yarıçapı ızgarası ve eşik taraması
  S7  Watershed ile temas eden hücrelerin ayrılması
  P   Aday tam pipeline'lar

Çıktı: experiments.csv (her deney için parametreler, kare bazında sayılar,
MAE, RMSE, Bias ve ortalama süre).
"""
import copy
import json

import numpy as np
import pandas as pd

from config import RESULTS_DIR, REPO_DIR
from data_io import read_image, reference_count
from metrics import count_metrics
from pipeline import BASELINE, run_pipeline, describe


def load_split(split):
    sel = pd.read_csv(REPO_DIR / "selected_frames.csv", dtype={"sequence": str})
    return sel[sel.split == split]


def make(base, **changes):
    c = copy.deepcopy(base)
    c.update(copy.deepcopy(changes))
    return c


G5 = {"method": "gaussian", "ksize": 5}
G15 = {"method": "gaussian", "ksize": 15}
OPEN3_FILL = [["open", 3], ["fill", 0]]
OPEN5_FILL = [["open", 5], ["fill", 0]]
WS = {"method": "watershed", "min_distance": 30, "min_peak": 10, "sigma": 2}

# Ortak başlangıç ayarı (S1 için): yalnızca yoğunluk dönüşümü değişecek
S1_BASE = dict(BASELINE, filter=G5, morph=OPEN3_FILL, min_area=400)
# S2: yoğunluk dönüşümü yok, segmentasyon değişecek
S2_BASE = S1_BASE
# S3: S2'nin en iyisi (sabit T=4) + kenar nesnesi kuralı
S3_BASE = make(S1_BASE, segment={"method": "fixed", "thresh": 4})
CB = {"clear_border": True, "border_max_area": 2500}
# S4: kenar kuralı açık, filtre değişecek
S4_BASE = make(S3_BASE, **CB)
# S5: S4'te en iyi filtre (G9) ile morfoloji değişecek
S5_BASE = make(S4_BASE, filter={"method": "gaussian", "ksize": 9})
# S7: kernel x açma yarıçapı ızgarasında en iyi nokta (G15, open5)
S7_BASE = make(S4_BASE, filter=G15, morph=OPEN5_FILL)


def experiment_list():
    E = []

    def add(eid, stage, note, cfg):
        E.append((eid, stage, note, cfg))

    add("E00", "S0 baseline", "Otsu + CCL + min alan 100", copy.deepcopy(BASELINE))

    # ---- S1 yoğunluk dönüşümleri
    add("E01", "S1 yoğunluk", "orijinal (dönüşüm yok)", make(S1_BASE))
    add("E02", "S1 yoğunluk", "contrast stretching %1-%99.8",
        make(S1_BASE, intensity={"method": "stretch", "p_low": 1, "p_high": 99.8}))
    add("E03", "S1 yoğunluk", "histogram eşitleme",
        make(S1_BASE, intensity={"method": "histeq"}))
    add("E04", "S1 yoğunluk", "CLAHE clip=2, tile=8",
        make(S1_BASE, intensity={"method": "clahe", "clip": 2.0, "tile": 8}))
    add("E05", "S1 yoğunluk", "histogram matching (ref: dev t003)",
        make(S1_BASE, intensity={"method": "match", "ref_frame": 3}))
    add("E06", "S1 yoğunluk", "stretch + gamma 0.5",
        make(S1_BASE, intensity={"method": "gamma", "gamma": 0.5, "p_low": 1, "p_high": 99.8}))

    # ---- S2 segmentasyon
    for i, t in enumerate([4, 10, 20, 35]):
        add(f"E{7 + i:02d}", "S2 segmentasyon", f"sabit eşik T={t}",
            make(S2_BASE, segment={"method": "fixed", "thresh": t}))
    add("E11", "S2 segmentasyon", "Otsu", make(S2_BASE, segment={"method": "otsu"}))
    add("E12", "S2 segmentasyon", "Li (minimum cross-entropy)", make(S2_BASE, segment={"method": "li"}))
    add("E13", "S2 segmentasyon", "adaptif ortalama b=151, C=-5 (CLAHE sonrası)",
        make(S2_BASE, intensity={"method": "clahe", "clip": 2.0, "tile": 8},
             segment={"method": "adaptive", "block": 151, "C": -5, "floor": 10}))
    add("E14", "S2 segmentasyon", "K-means k=3",
        make(S2_BASE, segment={"method": "kmeans", "k": 3}))

    # ---- S3 kenara değen kısmi hücreler
    add("E15", "S3 kenar", "E07 + kısmi kenar nesnelerini sil (alan<2500)", make(S3_BASE, **CB))

    # ---- S4 filtre ve kernel
    add("E16", "S4 filtre", "filtre yok", make(S4_BASE, filter={"method": "none"}))
    for i, k in enumerate([3, 5, 9, 15, 21]):
        add(f"E{17 + i:02d}", "S4 filtre", f"Gaussian {k}x{k}",
            make(S4_BASE, filter={"method": "gaussian", "ksize": k}))
    add("E22", "S4 filtre", "median 5x5", make(S4_BASE, filter={"method": "median", "ksize": 5}))
    add("E23", "S4 filtre", "median 9x9", make(S4_BASE, filter={"method": "median", "ksize": 9}))
    add("E24", "S4 filtre", "Laplacian keskinleştirme (3x3 kernel)",
        make(S4_BASE, filter={"method": "sharpen"}))
    add("E25", "S4 filtre", "Gaussian 9x9 + unsharp masking",
        make(S4_BASE, filter={"method": "gauss_sharpen", "ksize": 9, "amount": 1.0, "sigma": 3}))

    # ---- S5 morfoloji
    morphs = [
        ("morfoloji yok", []),
        ("erosion r=3", [["erode", 3]]),
        ("dilation r=3", [["dilate", 3]]),
        ("opening r=3 + fill", [["open", 3], ["fill", 0]]),
        ("opening r=5 + fill", [["open", 5], ["fill", 0]]),
        ("opening r=9 + fill", [["open", 9], ["fill", 0]]),
        ("closing r=5 + fill", [["close", 5], ["fill", 0]]),
        ("closing r=5 -> opening r=5 + fill", [["close", 5], ["open", 5], ["fill", 0]]),
        ("opening r=5 -> closing r=5 + fill", [["open", 5], ["close", 5], ["fill", 0]]),
        ("erosion r=5 -> dilation r=5", [["erode", 5], ["dilate", 5]]),
        ("dilation r=5 -> erosion r=5", [["dilate", 5], ["erode", 5]]),
    ]
    for i, (note, m) in enumerate(morphs):
        add(f"E{26 + i:02d}", "S5 morfoloji", note, make(S5_BASE, morph=m))

    # ---- S6 ızgara sonucu ve S7 ayırma
    add("E37", "S6 ızgara", "Gaussian 15x15 + opening r=5 (ızgaranın en iyisi)", make(S7_BASE))
    add("E38", "S7 ayırma", "E37 + watershed md=15, peak>=5",
        make(S7_BASE, split={"method": "watershed", "min_distance": 15, "min_peak": 5, "sigma": 2}))
    add("E39", "S7 ayırma", "E37 + watershed md=30, peak>=10", make(S7_BASE, split=WS))
    return E


def candidate_pipelines():
    """Doğrulamaya gidecek dört aday pipeline."""
    A = make(S7_BASE)
    A["name"] = "Pipeline A"
    A["summary"] = "G15 + sabit eşik T=4 + open5/fill + kenar temizleme"

    B = make(A, split=WS)
    B["name"] = "Pipeline B"
    B["summary"] = "A + distance transform / watershed ayırma"

    C = make(B, intensity={"method": "stretch", "p_low": 1, "p_high": 99.8},
             segment={"method": "fixed", "thresh": 10})
    C["name"] = "Pipeline C"
    C["summary"] = "percentile germe + G15 + göreli eşik T=10 + open5/fill + watershed + kenar"

    D = make(B, intensity={"method": "histeq"}, filter=G5,
             segment={"method": "otsu"}, morph=OPEN3_FILL)
    D["name"] = "Pipeline D"
    D["summary"] = "histogram eşitleme + G5 + Otsu + open3/fill + watershed + kenar"
    return [A, B, C, D]


def parameter_grid(dev, refs, imgs):
    """Gaussian kernel boyutu x açma yarıçapı ve eşik değeri taramaları."""
    rows = []
    for k in [3, 5, 9, 15, 21]:
        for r in [0, 3, 5, 9]:
            morph = ([["open", r]] if r else []) + [["fill", 0]]
            c = make(S4_BASE, filter={"method": "gaussian", "ksize": k}, morph=morph)
            p, _ = evaluate(c, dev, img_cache=imgs)
            m = count_metrics(p, refs)
            rows.append({"scan": "kernel_x_open", "ksize": k, "open_r": r, "thresh": 4,
                         "MAE": m["MAE"], "RMSE": m["RMSE"], "Bias": m["Bias"]})
    for t in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
        for cb in [False, True]:
            c = make(S7_BASE, segment={"method": "fixed", "thresh": t}, clear_border=cb)
            p, _ = evaluate(c, dev, img_cache=imgs)
            m = count_metrics(p, refs)
            rows.append({"scan": "threshold", "ksize": 15, "open_r": 5, "thresh": t,
                         "clear_border": cb, "MAE": m["MAE"], "RMSE": m["RMSE"], "Bias": m["Bias"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "parameter_grid.csv", index=False)
    return df


def evaluate(cfg, frames, seq="01", ref_cache=None, img_cache=None):
    ref_img = None
    if cfg["intensity"]["method"] == "match":
        ref_img = read_image("01", cfg["intensity"].get("ref_frame", 3))
    preds, times = [], []
    for f in frames:
        img = img_cache[f] if img_cache else read_image(seq, f)
        n, t = run_pipeline(img, cfg, ref_img=ref_img)
        preds.append(n)
        times.append(t)
    return preds, float(np.mean(times))


def main():
    dev = load_split("development").frame.tolist()
    refs = [reference_count("01", f) for f in dev]
    imgs = {f: read_image("01", f) for f in dev}
    rows = []
    for eid, stage, note, cfg in experiment_list():
        preds, t = evaluate(cfg, dev, img_cache=imgs)
        m = count_metrics(preds, refs)
        d = describe(cfg)
        rows.append({
            "exp_id": eid, "stage": stage, "note": note, **d,
            "clear_border": cfg.get("clear_border", False),
            "dev_MAE": round(m["MAE"], 2), "dev_RMSE": round(m["RMSE"], 2),
            "dev_Bias": round(m["Bias"], 2), "mean_time_s": round(t, 4),
            "pred_counts": " ".join(map(str, preds)),
            "ref_counts": " ".join(map(str, refs)),
            "config_json": json.dumps(cfg, ensure_ascii=False),
        })
        print(f"{eid} {note:45s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} bias={m['Bias']:+.2f}  {preds}")
    df = pd.DataFrame(rows)
    df.to_csv(REPO_DIR / "experiments.csv", index=False)

    # aday pipeline'ların geliştirme sonuçları (bilgi amaçlı)
    cand_rows = []
    for c in candidate_pipelines():
        preds, t = evaluate(c, dev, img_cache=imgs)
        m = count_metrics(preds, refs)
        cand_rows.append({"pipeline": c["name"], "summary": c["summary"],
                          "dev_MAE": round(m["MAE"], 2), "dev_RMSE": round(m["RMSE"], 2),
                          "pred_counts": " ".join(map(str, preds))})
        print(c["name"], m)
    pd.DataFrame(cand_rows).to_csv(RESULTS_DIR / "candidates_dev.csv", index=False)
    print(parameter_grid(dev, refs, imgs).to_string())
    return df


if __name__ == "__main__":
    main()
