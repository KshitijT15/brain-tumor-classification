# 🧠 Brain Tumor MRI — Augmented Figshare Dataset

> A clean, augmented, and pre-split version of the Figshare brain tumor MRI dataset — ready for deep learning pipelines.

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/userisakid/augmented-figshare-dataset)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## 📌 Overview

This repository contains the augmentation pipeline used to generate the **Augmented Figshare Brain Tumor Dataset** published on Kaggle. The raw [Figshare dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contains `.mat` (MATLAB) MRI scans across 4 tumor classes. This pipeline applies 6 augmentation transforms to each training image and produces a balanced 60/20/20 train/val/test split.

**Grand Total: 11,200 `.mat` files across all splits.**

---

## 📊 Dataset Statistics

| Split | Per Class | Total | Notes |
|-------|-----------|-------|-------|
| `train` | 2,520 | 10,080 | 6 augmented variants per original image |
| `val` | 140 | 560 | Original, unaugmented scans |
| `test` | 140 | 560 | Original, unaugmented scans |
| **Total** | **2,800** | **11,200** | |

**Classes:** `glioma` · `meningioma` · `pituitary` · `no_tumor`

---

## 🗂️ Dataset Structure

```
Augmented_v2/
├── train/
│   ├── glioma/          # 2,520 files (GLIOMA_0001_hflip.mat, ...)
│   ├── meningioma/      # 2,520 files
│   ├── pituitary/       # 2,520 files
│   └── no_tumor/        # 2,520 files
├── val/
│   ├── glioma/          # 140 files (GLIOMA_0001.mat, ...)
│   ├── meningioma/      # 140 files
│   ├── pituitary/       # 140 files
│   └── no_tumor/        # 140 files
└── test/
    ├── glioma/          # 140 files
    ├── meningioma/      # 140 files
    ├── pituitary/       # 140 files
    └── no_tumor/        # 140 files
```

---

## 🔬 Augmentation Pipeline

Each training image is expanded into **6 variants** using `torchvision.transforms`:

| Variant | Transform Applied |
|---------|-------------------|
| `original` | Identity (no change) |
| `hflip` | Horizontal Flip (`p=1.0`) |
| `vflip` | Vertical Flip (`p=1.0`) |
| `rotation20` | Fixed 20° Rotation |
| `colorjitter` | Brightness ±0.3, Contrast ±0.3, Saturation ±0.2, Hue ±0.1 |
| `affine` | Rotate ±10°, Translate ±10%, Scale 0.9–1.1×, Shear ±10° |

> Val and test splits are **never augmented** — only original scans.

---

## 📁 `.mat` File Format

Each `.mat` file contains a `cjdata` struct with the following fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `image` | `(512, 512)` | MRI scan (float64 / int16) |
| `label` | `(1, 1)` | Tumor class label |
| `tumorMask` | `(512, 512)` | Binary segmentation mask |
| `tumorBorder` | `(1, 64)` | Tumor border coordinates |
| `PID` | `(6, 1)` | Patient ID |

Both **HDF5 (v7.3)** and **classic MATLAB (v5)** formats are preserved from the source.

---

## ⚙️ How to Run

### Prerequisites

```bash
pip install scipy numpy Pillow torchvision tqdm h5py matplotlib
```

### Setup

1. Clone this repository:
```bash
git clone https://github.com/KshitijT15/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Download the original Figshare raw dataset and set your paths inside the notebook:
```python
SOURCE_ROOT = r"path/to/Raw_dataset"   # folder with 4 class subfolders
OUTPUT_ROOT = r"path/to/Augmented_v2"  # output location
```

3. Run the notebook:
```bash
jupyter notebook Augmented_dataset.ipynb
```

### What the notebook does

1. **Validates** all source `.mat` files (skips corrupt ones)
2. **Splits** each class 60/20/20 (train trimmed to nearest multiple of 6 for clean augmentation)
3. **Augments** training images with 6 transforms each
4. **Copies** val and test images without modification
5. **Verifies** file counts match expectations
6. **Zips** everything into `brain_tumor_augmented.zip`

---

## 🔢 Loading the Data (Python)

```python
import h5py
import scipy.io as sio
import numpy as np

def load_mat(path):
    try:
        with h5py.File(path, 'r') as f:
            img = f['cjdata']['image'][()].astype(np.float64)
            label = f['cjdata']['label'][()]
        return img, label
    except:
        m = sio.loadmat(path)
        img = m['cjdata']['image'][0, 0].astype(np.float64)
        label = m['cjdata']['label'][0, 0]
        return img, label

img, label = load_mat("train/glioma/GLIOMA_0001_original.mat")
print(img.shape, label)  # (512, 512) [[1]]
```

---

## 🧪 Use Cases

- Brain tumor classification (CNN, ResNet, EfficientNet, ViT)
- Medical image segmentation using `tumorMask`
- Transfer learning benchmarks
- Augmentation strategy comparison studies
- Class imbalance research

---

## 📦 Kaggle Dataset

The generated dataset is published on Kaggle:

🔗 [Augmented Figshare Dataset on Kaggle](https://www.kaggle.com/datasets/userisakid/augmented-figshare-dataset)

---

## 📚 Source & Citation

Original dataset by Cheng et al., hosted on Figshare:

> Jun Cheng. (2017). *brain tumor dataset*. figshare. [https://doi.org/10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427)

---

## 👤 Author

**Kshitij Thorat**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kshitij-thorat-15july2005)
[![GitHub](https://img.shields.io/badge/GitHub-KshitijT15-181717?logo=github&logoColor=white)](https://github.com/KshitijT15)
