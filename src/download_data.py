"""Veri seti bulunamazsa resmi adresten indirip repo içindeki data/ klasörüne açar.

Kullanım:  python download_data.py
Windows, Linux, macOS ve Colab'da çalışır (yalnızca Python standart kütüphanesi).
"""
import urllib.request
import zipfile

from config import REPO_DIR, find_data_dir

URL = "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip"


def ensure_dataset(url=URL):
    try:
        d = find_data_dir()
        print("Veri seti bulundu:", d)
        return d
    except FileNotFoundError:
        pass
    data_dir = REPO_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    zip_path = data_dir / "Fluo-N2DH-GOWT1.zip"
    if not zip_path.exists():
        print("Veri seti indiriliyor (~60 MB):", url)
        urllib.request.urlretrieve(url, zip_path)
    print("Arşiv açılıyor...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    d = find_data_dir()
    print("Veri seti hazır:", d)
    return d


if __name__ == "__main__":
    ensure_dataset()
