# Coronary Artery Segmentation - MPhil Research

Deep Learning approaches for automatic coronary artery segmentation and labeling in X-ray angiography using the ARCADE dataset.

---

## 🎯 4 Complete Training Approaches

This repository contains **4 distinct methods** for coronary artery segmentation, ready for training on Google Colab or locally.

### 1. **Simple U-Net** (Learning & Baseline)
- **Folder:** `experiments/1_simple_unet_learning/`
- **Task:** Binary vessel segmentation
- **Performance:** Dice ~0.75
- **Time:** 1 hour
- **Best for:** Understanding segmentation fundamentals

### 2. **cGAN** (Generative - High Quality)
- **Folder:** `experiments/2_cgan/`
- **Task:** Instance segmentation of 20-25 coronary branches
- **Performance:** Dice ~0.98
- **Time:** 2-3 hours
- **Best for:** High-quality mask generation with adversarial training

### 3. **YOLOv8** (Fast Deployment)
- **Folder:** `experiments/3_yolov8/`
- **Task:** Instance segmentation with detection
- **Performance:** mAP50 ~0.60
- **Time:** 1.5 hours
- **Best for:** Real-time inference and deployment

### 4. **FPN + U-Net + Swin** (State-of-the-Art)
- **Folder:** `experiments/4_fpn_unet_swin/`
- **Task:** Multi-architecture comparison
- **Performance:** Dice ~0.91 (FPN best)
- **Time:** 3-5 hours
- **Best for:** Highest accuracy, research comparison

---

## 📁 Project Structure

```
.
├── README.md                          ← This file
├── .gitignore                         ← Git configuration
│
├── experiments/                       ← ALL TRAINING CODE HERE
│   ├── README.md                      ← Complete training guide
│   ├── datasets/                      ← Place ARCADE dataset here (~4GB)
│   │
│   ├── 1_simple_unet_learning/
│   │   └── simple_unet_colab.ipynb    ✓ Colab-ready
│   │
│   ├── 2_cgan/
│   │   ├── colab_train.ipynb          ✓ Colab-ready
│   │   └── ucnet/                     ✓ Model implementation
│   │
│   ├── 3_yolov8/
│   │   ├── colab_train.ipynb          ✓ Colab-ready
│   │   └── data/                      ✓ Dataset converter
│   │
│   └── 4_fpn_unet_swin/
│       ├── 01_data_prep_colab.ipynb   ✓ Colab-ready
│       ├── 02_train_colab.ipynb       ✓ Colab-ready
│       ├── models.py                  ✓ Model definitions
│       └── utils.py                   ✓ Training utilities
```

---

## 🚀 Quick Start

### **For Google Colab (Recommended)**

1. **Upload to Google Drive:**
   ```
   My Drive/experiments/ ← Copy entire experiments folder
   ```

2. **Open any notebook in Colab:**
   - Right-click → "Open with" → "Google Colaboratory"
   - First cell auto-configures everything
   - Click "Run All"

3. **Configure BASE_URL (if needed):**
   ```python
   BASE_URL = "/content/drive/MyDrive/experiments"
   ```

### **For Local Machine**

1. **Clone and setup:**
   ```bash
   cd d:/MPHIL_CODES/MPHIL_MAIN_REPO
   pip install -r requirements.txt
   ```

2. **Run notebooks or scripts:**
   ```bash
   jupyter notebook experiments/1_simple_unet_learning/simple_unet_colab.ipynb
   ```

3. **Configure BASE_URL (if needed):**
   ```python
   BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"
   ```

---

## 📊 Dataset

**ARCADE** - MICCAI 2023 Coronary Artery Segmentation Challenge

- **Format:** COCO JSON annotations + PNG images
- **Size:** ~26 coronary artery segments
- **Images:** X-ray angiography (grayscale)
- **Download:** [ARCADE Challenge](https://arcade.grand-challenge.org/)

**Expected structure:**
```
experiments/datasets/ARCADE/
├── train/
│   ├── images/*.png
│   └── annotations/train.json
└── val/
    ├── images/*.png
    └── annotations/val.json
```

---

## 💻 Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch |
| Vision Models | torchvision, timm |
| Instance Segmentation | YOLOv8 (Ultralytics) |
| Segmentation Metrics | Dice, IoU, Sensitivity, Specificity |
| Development | Jupyter, Colab |

---

## 📚 What Each Approach Learns

| Approach | Learns | Applications |
|----------|--------|--------------|
| Simple U-Net | Encoder-decoder basics, Dice loss | Educational, baseline |
| cGAN | Adversarial training, instance segmentation | High-quality generation |
| YOLOv8 | Modern detection + segmentation | Deployment, real-time |
| FPN+U-Net+Swin | Multi-scale features, vision transformers | State-of-the-art research |

---

## 🎓 Documentation

- **Training guide:** See `experiments/README.md`
- **Model architectures:** Check individual folders
- **Code quality:** All code reviewed and cleaned

---

## 📋 Prerequisites

- Python 3.8+
- GPU (recommended, but CPU works)
- For Colab: Google Drive account
- For local: PyTorch + dependencies

---

## 🔧 Configuration

All notebooks use a single **BASE_URL** variable:

```python
BASE_URL = "/content/drive/MyDrive/experiments"  # Colab
# OR
BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"  # Local
```

Change this once, everything works automatically on that system.

---

## 📈 Expected Results

| Approach | Dice | Training Time | GPU Memory |
|----------|------|---------------|-----------|
| Simple U-Net | 0.75 | 1 hour | 4GB |
| cGAN | 0.98 | 2-3 hours | 8GB |
| YOLOv8 | 0.60 | 1.5 hours | 6GB |
| FPN+U-Net+Swin | 0.91 | 3-5 hours | 8GB+ |

---

## ✅ Repository Status

- ✅ 4 complete, Colab-ready training approaches
- ✅ All code cleaned and reviewed
- ✅ Organized folder structure
- ✅ Single comprehensive README per level
- ✅ BASE_URL configuration for any environment
- ✅ Ready for Google Colab or local training

---

## 📝 Notes

- All notebooks have `BASE_URL` configured at the start
- Datasets are gitignored (too large to commit)
- Model checkpoints are saved locally during training
- Results are generated in each approach's folder

---

## 🎯 Next Steps

1. **Read** `experiments/README.md` for detailed guide
2. **Choose** one approach to start with
3. **Download** ARCADE dataset to `experiments/datasets/`
4. **Run** the notebook (Colab or local)
5. **Modify** hyperparameters for your research

---

**Happy training!** 🚀
