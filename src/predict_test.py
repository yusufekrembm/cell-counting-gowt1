"""KİLİTLİ TEST TAHMİNİ (ödev Bölüm 12).

Bu betik test referanslarını (02_GT) OKUMAZ. Yalnızca final_pipeline.json'daki
sabitlenmiş nihai pipeline ve baseline ile 15 test karesinin sayımlarını
üretir ve test_predictions_locked.csv dosyasına yazar. Dosyaya ayrıca
final_pipeline.json'ın SHA-256 özeti eklenir; böylece tahminlerin hangi
konfigürasyonla üretildiği sonradan doğrulanabilir.
"""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from config import REPO_DIR
from data_io import read_image
from experiments import load_split
from pipeline import run_pipeline


def main(force=False):
    out_path = REPO_DIR / "test_predictions_locked.csv"
    if out_path.exists() and not force:
        print("test_predictions_locked.csv zaten var; kilitli tahminler yeniden yazılmadı.")
        return pd.read_csv(out_path)
    cfg_path = REPO_DIR / "final_pipeline.json"
    raw = cfg_path.read_bytes()
    # satır sonları normalize edilir (Windows CRLF / Linux LF aynı özeti verir)
    sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    conf = json.loads(raw.decode("utf-8"))
    final, base = conf["final_pipeline"], conf["baseline"]

    test = load_split("test")
    ref_img = read_image("01", final["intensity"].get("ref_frame", 3)) \
        if final["intensity"]["method"] == "match" else None
    rows = []
    for _, r in test.iterrows():
        img = read_image(r.sequence, r.frame)
        nb, tb = run_pipeline(img, base)
        nf, tf = run_pipeline(img, final, ref_img=ref_img)
        rows.append({"sequence": r.sequence, "frame": r.frame, "block": r.block,
                     "baseline_pred": nb, "final_pred": nf,
                     "baseline_time_s": round(tb, 4), "final_time_s": round(tf, 4)})
    df = pd.DataFrame(rows)
    df["final_pipeline_sha256"] = sha
    df["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(out_path, index=False)
    print(df[["block", "frame", "baseline_pred", "final_pred"]].to_string(index=False))
    print("final_pipeline.json sha256:", sha)


if __name__ == "__main__":
    main()
