"""Tüm adımları sırayla çalıştırır.

  python run_all.py --stage locked   -> seçim, deneyler, doğrulama, kilitli test tahmini
  python run_all.py --stage eval     -> test değerlendirmesi ve şekiller
  python run_all.py                  -> hepsi
"""
import argparse

import select_frames, references, experiments, validate, predict_test, evaluate_test, figures

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["locked", "eval", "all"], default="all")
a = p.parse_args()
if a.stage in ("locked", "all"):
    select_frames.main()
    references.main()
    experiments.main()
    validate.main()
    predict_test.main()
if a.stage in ("eval", "all"):
    evaluate_test.main()
    figures.main()
