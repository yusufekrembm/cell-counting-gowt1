# Klasik Görüntü İşleme ile Hücre Sayımı – Fluo-N2DH-GOWT1

Görüntü İşleme dersi yüksek lisans ara sınav projesi.
**Yusuf Ekrem Ünlü – 254329042**

Cell Tracking Challenge **Fluo-N2DH-GOWT1** veri setindeki floresan mikroskopi görüntülerinde
hücre (çekirdek) sayımı, yalnızca klasik görüntü işleme yöntemleriyle yapılmıştır
(yoğunluk dönüşümü, uzamsal filtreleme, eşikleme, morfoloji, distance transform + watershed).
Derin öğrenme veya önceden eğitilmiş model kullanılmamıştır.

## Veri seti

* Sayfa: https://celltrackingchallenge.net/2d-datasets/
* Eğitim arşivi: https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

Veri seti bu repository'de **yoktur** (ödev kuralı 18.5). Ayrıca indirmeye gerek yok:
`main.ipynb`'nin ilk hücresi (ya da `python src/download_data.py`) veri setini bulamazsa resmi adresten
indirip `data/` klasörüne açar. Elle indirilmiş bir kopya varsa kod, içinde `01`, `02`, `01_GT`, `02_GT`
bulunan klasörü sırasıyla şu konumlarda arar:

1. `GOWT1_DIR` ortam değişkeni
2. `data/Fluo-N2DH-GOWT1/`
3. `../Fluo-N2DH-GOWT1/Fluo-N2DH-GOWT1/` (repo ile aynı klasörde açılmış arşiv)
4. `../Fluo-N2DH-GOWT1/`

## Random seed

`seed = öğrenci numarasının son 5 hanesi + 2026 = 29042 + 2026 = 31068`

| Grup | Sekans | Kareler |
|---|---|---|
| Geliştirme (10) | 01 | 3, 7, 10, 27, 31, 59, 77, 85, 86, 90 |
| Doğrulama (5) | 01 | 20, 21, 45, 52, 67 |
| Test (3×5) | 02 | B1: 11–15, B2: 65–69, B3: 86–90 |

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
```

**Jupyter:** `main.ipynb` dosyasını açıp hücreleri yukarıdan aşağıya çalıştırın.
**Google Colab:** `!git clone https://github.com/yusufekrembm/cell-counting-gowt1.git`, `%cd cell-counting-gowt1`, `!pip -q install -r requirements.txt` komutlarından sonra `main.ipynb`'yi açın. İlk hücre veri setini kendisi indirir.

**Komut satırı** (`src/` klasöründen):

```bash
python run_all.py --stage locked   # 1) seçim, deneyler, doğrulama, KİLİTLİ test tahmini
python run_all.py --stage eval     # 2) test değerlendirmesi, zamansal analiz, şekiller
```

Tek tek çalıştırma sırası: (`download_data.py`) → `select_frames.py` → `references.py` → `experiments.py` → `validate.py` →
`predict_test.py` → `evaluate_test.py` → `figures.py`.

`predict_test.py`, `test_predictions_locked.csv` zaten varsa üzerine yazmaz. `evaluate_test.py` ise
kilitli dosyadaki SHA-256 özetini `final_pipeline.json` ile karşılaştırır. Konfigürasyon sonradan
değiştirilmişse değerlendirme hata verir.

## Nihai pipeline (Pipeline C)

```
Percentile germe (%1–%99.8) → Gaussian 15×15 → sabit eşik T=10 (germe sonrası, yani göreli eşik)
→ opening (disk r=5) + delik doldurma → distance transform + watershed (min_distance=30)
→ alanı < 400 px olan nesneleri sil → kenara değen ve alanı < 2500 px olan kısmi nesneleri sil → sayım
```

Dört aday pipeline arasından en düşük doğrulama MAE değeriyle seçildi (val MAE = 0.20).

| | MAE | RMSE | Bias | Exact-count |
|---|---|---|---|---|
| Baseline (Otsu + CCL), test | 11.33 | 11.84 | −11.33 | 0.00 |
| Nihai pipeline, test | 2.40 | 3.52 | +1.47 | 0.20 |

## Dosyalar

| Dosya | İçerik |
|---|---|
| `main.ipynb` | Tüm adımları çalıştıran notebook |
| `src/config.py` | Seed, yol ayarları |
| `src/download_data.py` | Veri seti yoksa resmi adresten indirme |
| `src/data_io.py` | Görüntü ve TRA etiketi okuma, referans sayım |
| `src/select_frames.py` | Seed'e bağlı deterministik kare seçimi |
| `src/references.py` | Geliştirme/doğrulama referans sayıları (TRA) |
| `src/pipeline.py` | Konfigürasyonla tanımlanan pipeline ve baseline |
| `src/experiments.py` | 40 kontrollü deney (E00–E39), parametre taramaları, aday pipeline'lar |
| `src/validate.py` | Doğrulama karşılaştırması ve nihai seçim |
| `src/predict_test.py` | Referans okumadan kilitli test tahmini |
| `src/evaluate_test.py` | Test metrikleri ve zamansal grafik |
| `src/figures.py` | Rapor şekilleri |
| `selected_frames.csv` | Seçilen kareler, gruplar, seed |
| `dev_val_references.csv` | Geliştirme ve doğrulama karelerinin referans hücre sayıları |
| `experiments.csv` | Deneylerin yöntemleri, parametreleri, kare bazında sayıları ve MAE/RMSE/Bias değerleri |
| `validation_results.csv` | Aday pipeline doğrulama sonuçları |
| `final_pipeline.json` | Nihai pipeline + baseline konfigürasyonu |
| `test_predictions_locked.csv` | Test referansları okunmadan kaydedilen tahminler (+ SHA-256) |
| `test_results.csv` | Test referansları, tahminler, mutlak hatalar |
| `results/test_summary.csv` | Özet test metrikleri (genel ve blok bazında) |
| `results/parameter_grid.csv` | Kernel × opening ızgarası ve eşik taraması |
| `results/candidates_dev.csv` | Aday pipeline'ların geliştirme sonuçları |
| `results/dataset_stats.csv` | Sekansların yoğunluk istatistikleri |
| `results/figures/` | Veri görüntüsü içermeyen grafikler (akış diyagramı, parametre analizi, zamansal grafik) |

`results/figures_private/` klasöründeki, veri setine ait görüntü içeren şekiller yalnızca raporda
kullanılır ve `.gitignore` ile repo dışında tutulur.

## Notlar

* Süre ölçümleri (`mean_time_s`) donanıma bağlıdır ve her çalıştırmada küçük farklar gösterir.
  Sayısal sayım sonuçları ise aynı seed ile birebir tekrarlanır.
* Görüntüler LZW sıkıştırmalı TIFF'tir ve OpenCV (`cv2.IMREAD_UNCHANGED`) ile okunur.
  `tifffile` kullanılacaksa `imagecodecs` paketi gerekir.
