"""Kilitli test tahminlerinin referans sayılarla değerlendirilmesi (Bölüm 13-14).

test_predictions_locked.csv değiştirilmez; yalnızca okunur. Önce dosyadaki
SHA-256 özetinin mevcut final_pipeline.json ile aynı olduğu kontrol edilir.
Çıktılar: test_results.csv, results/test_summary.csv, results/figures/fig5_temporal.png
"""
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import REPO_DIR, RESULTS_DIR, FIG_DIR
from data_io import reference_count
from metrics import count_metrics


def main():
    locked = pd.read_csv(REPO_DIR / "test_predictions_locked.csv", dtype={"sequence": str})
    raw = (REPO_DIR / "final_pipeline.json").read_bytes()
    # satır sonları normalize edilir (Windows CRLF / Linux LF aynı özeti verir)
    sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if not (locked.final_pipeline_sha256 == sha).all():
        raise RuntimeError("final_pipeline.json kilitli tahminlerden sonra değiştirilmiş!")

    df = locked.copy()
    df["reference"] = [reference_count(s, f) for s, f in zip(df.sequence, df.frame)]
    df["baseline_abs_err"] = (df.baseline_pred - df.reference).abs()
    df["final_abs_err"] = (df.final_pred - df.reference).abs()
    cols = ["block", "sequence", "frame", "reference", "baseline_pred", "final_pred",
            "baseline_abs_err", "final_abs_err", "baseline_time_s", "final_time_s"]
    df[cols].to_csv(REPO_DIR / "test_results.csv", index=False)

    summ = []
    for name, col, tcol in [("Baseline", "baseline_pred", "baseline_time_s"),
                            ("Nihai pipeline", "final_pred", "final_time_s")]:
        m = count_metrics(df[col], df.reference)
        summ.append({"method": name, **{k: round(v, 3) for k, v in m.items()},
                     "mean_time_s": round(df[tcol].mean(), 4)})
    # blok bazında MAE
    for b, g in df.groupby("block"):
        for name, col in [("Baseline", "baseline_pred"), ("Nihai pipeline", "final_pred")]:
            m = count_metrics(g[col], g.reference)
            summ.append({"method": f"{name} [{b}]", **{k: round(v, 3) for k, v in m.items()}})
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS_DIR / "test_summary.csv", index=False)
    print(df[cols].to_string(index=False))
    print(summ.to_string(index=False))

    # SONRADAN ANALİZ (seçimi etkilemez): diğer adaylar testte nasıl olurdu?
    # Yalnızca tartışma için; nihai pipeline ve kilitli tahminler değiştirilmez.
    from experiments import candidate_pipelines, evaluate
    ph = []
    for c in candidate_pipelines():
        p, _ = evaluate(c, df.frame.tolist(), seq="02")
        m = count_metrics(p, df.reference)
        ph.append({"pipeline": c["name"], **{k: round(v, 3) for k, v in m.items()},
                   "pred_counts": " ".join(map(str, p))})
    pd.DataFrame(ph).to_csv(RESULTS_DIR / "posthoc_all_candidates_test.csv", index=False)

    # Şekil 5: zamansal grafik
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (b, g) in zip(axes, df.groupby("block")):
        ax.plot(g.frame, g.reference, "k-o", lw=2, label="Referans (TRA)")
        ax.plot(g.frame, g.baseline_pred, "s--", color="#c0504d", label="Baseline")
        ax.plot(g.frame, g.final_pred, "^-", color="#1f77b4", label="Nihai pipeline")
        ax.set_title(f"Test bloğu {b}: kare {g.frame.min()}–{g.frame.max()}")
        ax.set_xlabel("Kare numarası (Sequence 02)")
        ax.set_xticks(g.frame)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Hücre sayısı")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_temporal.png", dpi=160)
    return df, summ


if __name__ == "__main__":
    main()
