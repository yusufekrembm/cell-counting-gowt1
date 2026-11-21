"""Proje genelinde kullanılan sabitler ve yol tanımları."""
import os
from pathlib import Path

STUDENT_NO = "254329042"
SEED = int(STUDENT_NO[-5:]) + 2026  # 29042 + 2026 = 31068

REPO_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Veri seti klasörü: önce GOWT1_DIR ortam değişkenine bakılır, yoksa
# aşağıdaki olası konumlar sırayla denenir. Klasörün içinde 01, 02,
# 01_GT ve 02_GT alt klasörleri bulunmalıdır.
_CANDIDATES = [
    os.environ.get("GOWT1_DIR", ""),
    REPO_DIR / "data" / "Fluo-N2DH-GOWT1",
    REPO_DIR.parent / "Fluo-N2DH-GOWT1" / "Fluo-N2DH-GOWT1",
    REPO_DIR.parent / "Fluo-N2DH-GOWT1",
    Path("/content/Fluo-N2DH-GOWT1"),  # Colab
]


def find_data_dir():
    for c in _CANDIDATES:
        if c and (Path(c) / "01").is_dir() and (Path(c) / "01_GT").is_dir():
            return Path(c)
    raise FileNotFoundError(
        "Fluo-N2DH-GOWT1 klasörü bulunamadı. README'deki indirme adımlarını "
        "izleyin veya GOWT1_DIR ortam değişkenini ayarlayın."
    )


N_FRAMES = 92       # her sekanstaki kare sayısı (t000 - t091)
MIN_GAP = 5         # gruplar/bloklar arasında bırakılacak en az kare sayısı

for d in (RESULTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
