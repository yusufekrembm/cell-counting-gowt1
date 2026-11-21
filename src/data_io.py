"""Görüntü ve referans etiketlerini okuma fonksiyonları."""
import cv2
import numpy as np

from config import find_data_dir


def image_path(seq, frame):
    return find_data_dir() / seq / f"t{frame:03d}.tif"


def read_image(seq, frame):
    """8-bit gri seviyeli görüntüyü okur (LZW sıkıştırmalı TIFF, OpenCV ile)."""
    img = cv2.imread(str(image_path(seq, frame)), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(image_path(seq, frame))
    return img


def read_tra_labels(seq, frame):
    p = find_data_dir() / f"{seq}_GT" / "TRA" / f"man_track{frame:03d}.tif"
    lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(p)
    return lab


def reference_count(seq, frame):
    """TRA etiket görüntüsündeki farklı pozitif hücre kimliklerinin sayısı."""
    lab = read_tra_labels(seq, frame)
    ids = np.unique(lab)
    return int((ids > 0).sum())
