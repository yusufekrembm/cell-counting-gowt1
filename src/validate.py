"""Aday pipeline'ların 5 doğrulama karesinde karşılaştırılması ve nihai seçim.

Seçim kuralı (ödev Bölüm 11):
  1) en düşük doğrulama MAE
  2) eşitlikte en düşük RMSE
  3) hâlâ eşitlikte: ortalama süre (yalnızca fark %20'den büyükse anlamlı
     kabul edilir, çünkü süre ölçümü çalıştırmadan çalıştırmaya değişir)
  4) son olarak pipeline karmaşıklığı (daha az işlem adımı tercih edilir)

Çıktılar: validation_results.csv, final_pipeline.json
"""
import json

import numpy as np
import pandas as pd

from config import REPO_DIR, SEED
from data_io import read_image, reference_count
from experiments import candidate_pipelines, evaluate, load_split
from metrics import count_metrics
from pipeline import BASELINE


def complexity(cfg):
    n = 0
    n += cfg["intensity"]["method"] != "none"
    n += cfg["filter"]["method"] != "none"
    n += 1  # segmentasyon
    n += len(cfg["morph"])
    n += cfg["split"]["method"] != "none"
    n += bool(cfg.get("clear_border", False))
    n += cfg.get("min_area", 0) > 0
    return int(n)


def main(n_repeat=3):
    val = load_split("validation").frame.tolist()
    refs = [reference_count("01", f) for f in val]
    imgs = {f: read_image("01", f) for f in val}
    rows = []
    cands = candidate_pipelines()
    for c in cands:
        # süre ölçümünü kararlı hale getirmek için 3 tekrarın medyanı
        times = []
        for _ in range(n_repeat):
            preds, t = evaluate(c, val, img_cache=imgs)
            times.append(t)
        m = count_metrics(preds, refs)
        rows.append({
            "pipeline": c["name"], "summary": c["summary"],
            "val_MAE": round(m["MAE"], 3), "val_RMSE": round(m["RMSE"], 3),
            "val_Bias": round(m["Bias"], 3), "val_ExactAcc": round(m["ExactAcc"], 3),
            "mean_time_s": round(float(np.median(times)), 4),
            "complexity": complexity(c),
            "pred_counts": " ".join(map(str, preds)),
            "ref_counts": " ".join(map(str, refs)),
        })
    df = pd.DataFrame(rows)

    # seçim
    best = df.sort_values(["val_MAE", "val_RMSE"]).iloc[0]
    tied = df[(df.val_MAE == best.val_MAE) & (df.val_RMSE == best.val_RMSE)]
    if len(tied) > 1:
        tmin = tied.mean_time_s.min()
        tied = tied[tied.mean_time_s <= tmin * 1.2]
        tied = tied.sort_values(["complexity", "mean_time_s"])
    winner = tied.iloc[0].pipeline
    df["selected"] = np.where(df.pipeline == winner, "SEÇİLDİ", "")
    df.to_csv(REPO_DIR / "validation_results.csv", index=False)
    print(df.drop(columns=["summary"]).to_string())

    final = next(c for c in cands if c["name"] == winner)
    out = {
        "seed": SEED,
        "selected_pipeline": winner,
        "selection_rule": "min val MAE -> min val RMSE -> süre (>%20 fark) -> karmaşıklık",
        "final_pipeline": final,
        "baseline": BASELINE,
        "validation_frames": val,
    }
    with open(REPO_DIR / "final_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Nihai pipeline:", winner)
    return df, final


if __name__ == "__main__":
    main()
