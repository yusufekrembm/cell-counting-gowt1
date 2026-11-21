"""Geliştirme ve doğrulama referans hücre sayılarının hazırlanması (Bölüm 4).

Test referansları burada OKUNMAZ; onlar yalnızca evaluate_test.py içinde,
kilitli tahminler kaydedildikten sonra okunur.
Çıktı: dev_val_references.csv
"""
import pandas as pd

from config import REPO_DIR
from data_io import reference_count


def main():
    sel = pd.read_csv(REPO_DIR / "selected_frames.csv", dtype={"sequence": str})
    dv = sel[sel.split != "test"].copy()
    dv["reference_count"] = [reference_count(s, f) for s, f in zip(dv.sequence, dv.frame)]
    dv["source"] = dv.apply(lambda r: f"{r.sequence}_GT/TRA/man_track{r.frame:03d}.tif", axis=1)
    dv = dv[["split", "sequence", "frame", "filename", "source", "reference_count"]]
    dv.to_csv(REPO_DIR / "dev_val_references.csv", index=False)
    print(dv.to_string(index=False))
    return dv


if __name__ == "__main__":
    main()
