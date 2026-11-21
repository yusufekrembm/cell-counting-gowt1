"""Seed'e bağlı, tekrarlanabilir geliştirme / doğrulama / test kare seçimi.

Kurallar (ödev Bölüm 3):
  * Sequence 01: 5 doğrulama + 10 geliştirme karesi, çakışmasız ve iki grup
    arasında en az 5 karelik boşluk (|i - j| - 1 >= 5).
  * Sequence 02: 3 farklı başlangıç noktası, her birinden ardışık 5 kare;
    bloklar arasında en az 5 karelik boşluk.
  * Koşul sağlanamazsa aynı RNG ile (aynı seed) yeniden çekim yapılır.
    RNG durumu her denemede ilerlediği için prosedür deterministiktir.
"""
import numpy as np
import pandas as pd

from config import SEED, N_FRAMES, MIN_GAP, RESULTS_DIR

MAX_TRIES = 1000


def gap_ok(a, b):
    return abs(a - b) - 1 >= MIN_GAP


def select_seq01(rng):
    frames = np.arange(N_FRAMES)
    for attempt in range(1, MAX_TRIES + 1):
        val = np.sort(rng.choice(frames, size=5, replace=False))
        pool = [f for f in frames if all(gap_ok(f, v) for v in val)]
        if len(pool) < 10:
            continue
        dev = np.sort(rng.choice(pool, size=10, replace=False))
        return dev.tolist(), val.tolist(), attempt
    raise RuntimeError("Sequence 01 için uygun seçim bulunamadı")


def select_seq02(rng, n_blocks=3, block_len=5):
    starts_all = np.arange(N_FRAMES - block_len + 1)
    for attempt in range(1, MAX_TRIES + 1):
        starts = np.sort(rng.choice(starts_all, size=n_blocks, replace=False))
        ok = all(starts[i + 1] - (starts[i] + block_len - 1) - 1 >= MIN_GAP
                 for i in range(n_blocks - 1))
        if ok:
            return starts.tolist(), attempt
    raise RuntimeError("Sequence 02 için uygun seçim bulunamadı")


def main():
    rng = np.random.default_rng(SEED)
    dev, val, a1 = select_seq01(rng)   # önce doğrulama, sonra geliştirme
    starts, a2 = select_seq02(rng)

    rows = []
    for f in dev:
        rows.append(dict(split="development", sequence="01", frame=f, block=""))
    for f in val:
        rows.append(dict(split="validation", sequence="01", frame=f, block=""))
    for b, s in enumerate(starts, 1):
        for f in range(s, s + 5):
            rows.append(dict(split="test", sequence="02", frame=f, block=f"B{b}"))
    df = pd.DataFrame(rows)
    df["filename"] = df.apply(lambda r: f"{r.sequence}/t{r.frame:03d}.tif", axis=1)
    df["seed"] = SEED
    df.to_csv(RESULTS_DIR.parent / "selected_frames.csv", index=False)
    print(f"seed={SEED}  seq01 deneme={a1}  seq02 deneme={a2}")
    print("Geliştirme :", dev)
    print("Doğrulama  :", val)
    print("Test blokları başlangıç:", starts)
    return df


if __name__ == "__main__":
    main()
