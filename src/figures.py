"""Rapor şekillerinin üretimi.

Veri setine ait görüntü içeren şekiller results/figures_private/ klasörüne
yazılır (bu klasör .gitignore ile GitHub'a gönderilmez; ödev Bölüm 18.5).
Yalnızca grafik/diyagram içeren şekiller results/figures/ altına yazılır.
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import segmentation

from config import REPO_DIR, RESULTS_DIR, FIG_DIR
from data_io import read_image, read_tra_labels
from experiments import S1_BASE, S5_BASE, S7_BASE, make, load_split
from pipeline import (apply_intensity, run_pipeline, contrast_stretch, to_uint8,
                      BASELINE)

PRIV = RESULTS_DIR / "figures_private"
PRIV.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})


def show(img):
    """Görselleştirme için sabit bir gösterim dönüşümü (yalnızca çizim)."""
    s = contrast_stretch(img, 0.5, 99.9)
    return to_uint8(255 * (s / 255.0) ** 0.6)


def overlay(img, lab, gt=None, crop=None):
    v = cv2.cvtColor(show(img), cv2.COLOR_GRAY2RGB)
    bd = cv2.dilate(segmentation.find_boundaries(lab, mode="outer").astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    v[bd] = (0, 230, 0)
    if gt is not None:
        for i in np.unique(gt)[1:]:
            ys, xs = np.where(gt == i)
            cv2.circle(v, (int(xs.mean()), int(ys.mean())), 7, (255, 40, 40), -1)
    if crop:
        y0, y1, x0, x1 = crop
        v = v[y0:y1, x0:x1]
    return v


def fig0_data(sel):
    """Veri inceleme: iki sekansın yoğunluk istatistikleri ve histogramları."""
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for seq, col in [("01", "#1f77b4"), ("02", "#d62728")]:
        acc = np.zeros(256)
        for f in range(0, 92, 7):
            im = read_image(seq, f)
            acc += np.bincount(im.ravel(), minlength=256)
            p = np.percentile(im, [50, 95, 99, 99.9])
            rows.append({"sequence": seq, "frame": f, "dtype": str(im.dtype),
                         "shape": f"{im.shape[0]}x{im.shape[1]}", "min": int(im.min()),
                         "max": int(im.max()), "mean": round(float(im.mean()), 2),
                         "p50": p[0], "p95": p[1], "p99": p[2], "p99.9": p[3]})
        acc /= acc.sum()
        ax[0].semilogy(acc, color=col, label=f"Sequence {seq}")
        ax[1].plot(np.cumsum(acc), color=col, label=f"Sequence {seq}")
    ax[0].set_title("Ortalama normalize histogram (log ölçek)")
    ax[0].set_xlabel("Gri seviye")
    ax[1].set_title("Kümülatif dağılım")
    ax[1].set_xlabel("Gri seviye")
    ax[1].set_xlim(0, 120)
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_dataset_histograms.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "dataset_stats.csv", index=False)

    # örnek kareler (özel)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    for a, (seq, f) in zip(ax, [("01", 3), ("02", 50)]):
        a.imshow(read_image(seq, f), cmap="gray", vmin=0, vmax=80)
        a.set_title(f"Sequence {seq}, t{f:03d} (gösterim aralığı 0–80)")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(PRIV / "fig0b_examples.png", dpi=150)
    plt.close(fig)


def fig2_intensity(frame=3):
    img = read_image("01", frame)
    ref = read_image("01", 3)
    methods = [("Orijinal", {"method": "none"}),
               ("Contrast stretching (%1–%99.8)", {"method": "stretch", "p_low": 1, "p_high": 99.8}),
               ("Histogram eşitleme", {"method": "histeq"}),
               ("CLAHE (clip=2, 8×8)", {"method": "clahe", "clip": 2.0, "tile": 8}),
               ("Stretch + gamma 0.5", {"method": "gamma", "gamma": 0.5}),
               ("Histogram matching (ref t003)", {"method": "match"})]
    exp = pd.read_csv(REPO_DIR / "experiments.csv")
    maes = dict(zip(exp.exp_id, exp.dev_MAE))
    ids = ["E01", "E02", "E03", "E04", "E06", "E05"]
    fig, ax = plt.subplots(2, 6, figsize=(17, 6))
    crop = (550, 1024, 550, 1024)  # sönük hücrelerin bulunduğu bölge
    for j, ((name, cfg), eid) in enumerate(zip(methods, ids)):
        out = apply_intensity(img, cfg, ref_img=ref if cfg["method"] == "match" else None)
        ax[0, j].imshow(out[crop[0]:crop[1], crop[2]:crop[3]], cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(f"{name}\n{eid}, dev MAE={maes[eid]:.1f}", fontsize=8)
        ax[0, j].axis("off")
        h = np.bincount(out.ravel(), minlength=256)
        ax[1, j].bar(np.arange(256), h, width=1, color="#444")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlim(0, 255)
        ax[1, j].set_xlabel("Gri seviye")
    fig.tight_layout()
    fig.savefig(PRIV / "fig2_intensity.png", dpi=140)
    plt.close(fig)


def fig3_seg_morph(frame=3):
    img = read_image("01", frame)
    gt = read_tra_labels("01", frame)
    crop = (380, 1024, 380, 1024)
    exp = pd.read_csv(REPO_DIR / "experiments.csv").set_index("exp_id")
    seg_ids = ["E07", "E11", "E12", "E13", "E14"]
    mor_ids = ["E26", "E27", "E29", "E31", "E33", "E36"]
    fig, ax = plt.subplots(2, 6, figsize=(18, 7.2))
    for j, eid in enumerate(seg_ids + [None]):
        a = ax[0, j]
        if eid is None:
            # baseline
            n, _, st = run_pipeline(img, BASELINE, return_steps=True)
            title = f"E00 baseline (Otsu)\nsayı={n}, ref={len(np.unique(gt)) - 1}"
        else:
            cfg = json.loads(exp.loc[eid, "config_json"])
            n, _, st = run_pipeline(img, cfg, return_steps=True)
            title = f"{eid} {exp.loc[eid, 'note'][:30]}\nsayı={n}, dev MAE={exp.loc[eid, 'dev_MAE']}"
        a.imshow(overlay(img, st["labels"], gt, crop))
        a.set_title(title, fontsize=8)
        a.axis("off")
    for j, eid in enumerate(mor_ids):
        cfg = json.loads(exp.loc[eid, "config_json"])
        n, _, st = run_pipeline(img, cfg, return_steps=True)
        ax[1, j].imshow(overlay(img, st["labels"], gt, crop))
        ax[1, j].set_title(f"{eid} {exp.loc[eid, 'note'][:32]}\nsayı={n}, dev MAE={exp.loc[eid, 'dev_MAE']}", fontsize=8)
        ax[1, j].axis("off")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(PRIV / "fig3_seg_morph.png", dpi=130)
    plt.close(fig)


def fig_filters(frame=3):
    img = read_image("01", frame)
    gt = read_tra_labels("01", frame)
    crop = (550, 1024, 550, 1024)
    exp = pd.read_csv(REPO_DIR / "experiments.csv").set_index("exp_id")
    ids = ["E16", "E17", "E19", "E20", "E22", "E24"]
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.8))
    for j, eid in enumerate(ids):
        cfg = json.loads(exp.loc[eid, "config_json"])
        n, _, st = run_pipeline(img, cfg, return_steps=True)
        f = st["filter"][crop[0]:crop[1], crop[2]:crop[3]]
        ax[0, j].imshow(f, cmap="gray", vmin=0, vmax=40)
        ax[0, j].set_title(f"{eid} {exp.loc[eid, 'note']}", fontsize=8)
        ax[1, j].imshow(overlay(img, st["labels"], gt, crop))
        ax[1, j].set_title(f"sayı={n}, dev MAE={exp.loc[eid, 'dev_MAE']}", fontsize=8)
        for a in ax[:, j]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(PRIV / "fig_filters.png", dpi=130)
    plt.close(fig)


def fig_params():
    g = pd.read_csv(RESULTS_DIR / "parameter_grid.csv")
    exp = pd.read_csv(REPO_DIR / "experiments.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    # (a) kernel x open heatmap
    h = g[g.scan == "kernel_x_open"].pivot(index="ksize", columns="open_r", values="MAE")
    im = ax[0].imshow(h.values, cmap="viridis_r", aspect="auto")
    ax[0].set_xticks(range(len(h.columns)), [f"r={c}" if c else "yok" for c in h.columns])
    ax[0].set_yticks(range(len(h.index)), [f"G{k}×{k}" for k in h.index])
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            ax[0].text(j, i, f"{h.values[i, j]:.1f}", ha="center", va="center",
                       color="w" if h.values[i, j] > 1.5 else "k")
    ax[0].set_title("(a) Gaussian kernel × opening yarıçapı\n(dev MAE, T=4, kenar kuralı)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # (b) threshold sweep
    t = g[g.scan == "threshold"]
    for cb, st, lab in [(False, "o--", "kenar kuralı yok"), (True, "s-", "kenar kuralı var")]:
        s = t[t.clear_border == cb]
        ax[1].plot(s.thresh, s.MAE, st, label=f"MAE ({lab})")
        ax[1].plot(s.thresh, s.Bias, st, alpha=0.4, label=f"Bias ({lab})")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xticks([2, 3, 4, 6, 10, 20, 30], ["2", "3", "4", "6", "10", "20", "30"])
    ax[1].set_xlabel("Sabit eşik T (G15, open5)")
    ax[1].set_title("(b) Eşik değeri taraması")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    # (c) experiment MAE bar
    e = exp[exp.exp_id != "E24"]
    colors = {"S0": "#7f7f7f", "S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#9467bd", "S4": "#2ca02c",
              "S5": "#d62728", "S6": "#8c564b", "S7": "#e377c2"}
    ax[2].bar(range(len(e)), e.dev_MAE, color=[colors[s[:2]] for s in e.stage])
    ax[2].set_xticks(range(len(e)), e.exp_id, rotation=90, fontsize=6)
    ax[2].set_ylabel("Dev MAE")
    ax[2].set_title("(c) Tüm deneylerin geliştirme MAE değerleri\n(E24 = 22.2 dışarıda bırakıldı)")
    from matplotlib.patches import Patch
    ax[2].legend(handles=[Patch(color=c, label=s) for s, c in colors.items()], fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_params.png", dpi=160)
    plt.close(fig)


def fig1_flowchart(final):
    steps = [
        ("Girdi", "8-bit gri görüntü\n1024×1024"),
        ("Yoğunluk", f"Percentile germe\n%{final['intensity']['p_low']} – %{final['intensity']['p_high']}"),
        ("Filtre", f"Gaussian\n{final['filter']['ksize']}×{final['filter']['ksize']}"),
        ("Segmentasyon", f"Sabit (göreli) eşik\nT = {final['segment']['thresh']}"),
        ("Morfoloji", "Opening (disk r=5)\n+ delik doldurma"),
        ("Ayırma", f"Dist. transform +\nwatershed (md={final['split']['min_distance']})"),
        ("Son işlem", f"alan<{final['min_area']} sil\nkenar ∧ alan<{final['border_max_area']} sil"),
        ("Çıktı", "Etiket sayısı\n= hücre sayısı"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 1.9))
    ax.set_xlim(0, len(steps) * 1.9)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    for i, (h, t) in enumerate(steps):
        x = i * 1.9 + 0.1
        col = "#dbe9f6" if 0 < i < len(steps) - 1 else "#f2f2f2"
        ax.add_patch(FancyBboxPatch((x, 0.15), 1.55, 1.2, boxstyle="round,pad=0.03",
                                    fc=col, ec="#2b5c8a", lw=1.2))
        ax.text(x + 0.775, 1.12, h, ha="center", va="center", fontweight="bold", fontsize=10.5)
        ax.text(x + 0.775, 0.6, t, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.88, 0.75), xytext=(x + 1.62, 0.75),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#2b5c8a"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_flowchart.png", dpi=170)
    plt.close(fig)


def fig4_stages(final, frame=45):
    img = read_image("01", frame)
    gt = read_tra_labels("01", frame)
    n, _, st = run_pipeline(img, final, return_steps=True)
    dist = ndi.distance_transform_edt(st["morph"])
    panels = [("1) Orijinal (ham, 0–255)", st["original"], dict(cmap="gray", vmin=0, vmax=255)),
              ("2) Percentile germe", st["intensity"], dict(cmap="gray", vmin=0, vmax=255)),
              ("3) Gaussian 15×15", st["filter"], dict(cmap="gray", vmin=0, vmax=255)),
              ("4) Eşikleme T=10", st["segment"], dict(cmap="gray")),
              ("5) Opening r=5 + fill", st["morph"], dict(cmap="gray")),
              ("6) Distance transform", dist, dict(cmap="magma")),
              ("7) Watershed etiketleri", st["labels"], dict(cmap="nipy_spectral", interpolation="nearest")),
              (f"8) Sonuç: {n} hücre (ref {len(np.unique(gt)) - 1})", overlay(img, st["labels"], gt), {})]
    fig, ax = plt.subplots(2, 4, figsize=(15, 8.6))
    for a, (t, im, kw) in zip(ax.ravel(), panels):
        a.imshow(im, **kw)
        a.set_title(t)
        a.axis("off")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(PRIV / "fig4_stages.png", dpi=120)
    plt.close(fig)


def fig6_examples(final):
    res = pd.read_csv(REPO_DIR / "test_results.csv")
    cases = [("Başarılı", 86, None), ("Hatalı (aşırı sayım)", 12, None), ("Zorlayıcı (temas/sönük)", 68, None)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for j, (name, f, _) in enumerate(cases):
        img = read_image("02", f)
        gt = read_tra_labels("02", f)
        r = res[res.frame == f].iloc[0]
        nb, _, sb = run_pipeline(img, BASELINE, return_steps=True)
        nf, _, sf = run_pipeline(img, final, return_steps=True)
        ax[0, j].imshow(overlay(img, sb["labels"], gt))
        ax[0, j].set_title(f"{name} – 02/t{f:03d}\nBaseline: {nb} (ref {r.reference})")
        ax[1, j].imshow(overlay(img, sf["labels"], gt))
        ax[1, j].set_title(f"Nihai pipeline: {nf} (ref {r.reference})")
        for a in ax[:, j]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(PRIV / "fig6_examples.png", dpi=110)
    plt.close(fig)


def main():
    sel = load_split("development")
    final = json.load(open(REPO_DIR / "final_pipeline.json", encoding="utf-8"))["final_pipeline"]
    fig0_data(sel)
    fig1_flowchart(final)
    fig2_intensity()
    fig3_seg_morph()
    fig_filters()
    fig_params()
    fig4_stages(final)
    if (REPO_DIR / "test_results.csv").exists():
        fig6_examples(final)
    print("Şekiller üretildi.")


if __name__ == "__main__":
    main()
