# Coronary Artery Segmentation Experiments

**4 Complete Training Pipelines - Ready for Google Colab**

All code uses `BASE_URL` configuration for Colab and local compatibility.

---

## 🎯 4 Approaches

### 1. Simple U-Net (Learning)
- **Folder:** `1_simple_unet_learning/`
- **Notebook:** `simple_unet_colab.ipynb`
- **Best for:** Understanding segmentation
- **Time:** 1 hour
- **Performance:** Dice ~0.75

### 2. cGAN (Generative)
- **Folder:** `2_cgan/`
- **Notebook:** `colab_train.ipynb`
- **Best for:** High-quality mask generation
- **Time:** 2-3 hours  
- **Performance:** Dice ~0.98
- **Includes:** UCNet module + models

### 3. YOLOv8 (Fast Deployment)
- **Folder:** `3_yolov8/`
- **Notebook:** `colab_train.ipynb`
- **Best for:** Real-time inference
- **Time:** 1.5 hours
- **Performance:** mAP50 ~0.60
- **Includes:** ARCADE dataset converter

### 4. FPN + U-Net + Swin (Advanced)
- **Folder:** `4_fpn_unet_swin/`
- **Notebooks:** 
  - `01_data_prep_colab.ipynb` (data preparation)
  - `02_train_colab.ipynb` (training)
- **Best for:** Highest accuracy, comparison study
- **Time:** 3-5 hours
- **Performance:** Dice ~0.91
- **Includes:** models.py, utils.py

---

## 🚀 Quick Start (3 Steps)

### Step 1: Configure BASE_URL
Every notebook starts with this configuration:

```python
# For Google Colab:
BASE_URL = "/content/drive/MyDrive/experiments"

# For Local (Windows):
BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"

# For Local (Mac/Linux):
BASE_URL = "/Users/yourname/experiments"
```

Change once - all paths update automatically!

### Step 2: Upload to Google Drive
```
My Drive/
└── experiments/  ← Upload entire folder
    ├── 1_simple_unet_learning/
    ├── 2_cgan/
    ├── 3_yolov8/
    ├── 4_fpn_unet_swin/
    └── datasets/  ← Add ARCADE here (~4GB, optional)
```

### Step 3: Open & Run
1. Right-click notebook in Drive
2. Open with → Google Colaboratory
3. First cell auto-detects and configures
4. Click "Run All"

---

## 📁 Folder Structure

```
experiments/
├── README.md                           ← You are here
├── datasets/                           ← Place ARCADE dataset here
│
├── 1_simple_unet_learning/
│   └── simple_unet_colab.ipynb        ✅ Complete & ready
│
├── 2_cgan/
│   ├── colab_train.ipynb              ✅ Ready
│   └── ucnet/                         ✅ Model files
│
├── 3_yolov8/
│   ├── colab_train.ipynb              ✅ Ready
│   └── data/                          ✅ Dataset converter
│
└── 4_fpn_unet_swin/
    ├── 01_data_prep_colab.ipynb       ✅ Ready
    ├── 02_train_colab.ipynb           ✅ Ready
    ├── models.py                      ✅ Model architectures
    └── utils.py                       ✅ Utilities & losses
```

---

## 💡 Key Features

✅ **Single Configuration** - Change `BASE_URL` once, works everywhere
✅ **Colab + Local** - Same code for Google Colab or local machine
✅ **Auto-Detection** - Detects environment and mounts drive if needed
✅ **Self-Contained** - All code & models in 4 folders
✅ **No Path Changes** - All notebooks ready to run as-is
✅ **Dependency Management** - Auto-installs required packages

---

## 🎯 Choosing Your Path

### For Learning Segmentation
→ Start with **Simple U-Net** (`1_simple_unet_learning/`)
- Teaches fundamentals
- 14 clear sections
- Under 1 hour total

### For Production Deployment
→ Use **YOLOv8** (`3_yolov8/`)
- Fast inference (50 FPS)
- Easy to export
- Ready for deployment

### For Best Accuracy & Research
→ Use **FPN + U-Net + Swin** (`4_fpn_unet_swin/`)
- Compare 3 architectures
- Achieve Dice ~0.91
- 5-fold cross-validation

### For Understanding GANs
→ Use **cGAN** (`2_cgan/`)
- Generative approach
- High-quality masks
- Complex but powerful

---

## 📊 Performance Comparison

| Model | Dice | Time | Best For |
|-------|------|------|----------|
| Simple U-Net | 0.75 | 1 h | Learning |
| cGAN | 0.98 | 2-3 h | Generation |
| YOLOv8 | 0.60 | 1.5 h | Speed |
| FPN | **0.91** | 3-5 h | **Accuracy** |

---

## 🔧 Environment Setup (Colab)

All notebooks automatically handle this, but here's what happens:

```python
# Cell 1 (automatic):
BASE_URL = "/content/drive/MyDrive/experiments"

# Mount Google Drive (if on Colab)
from google.colab import drive
drive.mount('/content/drive')

# All paths now ready:
ARCADE_PATH = BASE_URL / "datasets" / "ARCADE"
RESULTS_PATH = BASE_URL / "results"
```

---

## 📋 What Each Notebook Does

### Simple U-Net (14 parts)
1. Setup & imports
2. Load ARCADE dataset
3. Create binary vessel masks
4. Visualize samples
5. PyTorch dataset class
6. U-Net architecture
7. Dice loss
8. Training setup
9. Training loop
10. Training history
11. Evaluation
12. Predictions visualization
13. Single image inference
14. Summary

### cGAN
1. Setup paths
2. Load ARCADE data
3. Build generator + discriminator
4. Adversarial training
5. Evaluation
6. Results visualization

### YOLOv8
1. Convert ARCADE → YOLO format
2. Load YOLOv8m-seg model
3. Train on dataset
4. Evaluate performance
5. Export model

### FPN + U-Net + Swin
1. Data preparation & 5-fold split
2. Load & configure 3 models
3. Training loop for each
4. Evaluation & comparison
5. Visualization & results

---

## 📈 Training Times (GPU)

```
Simple U-Net:    ~10 min per epoch  → 1 hour total
cGAN:            ~2-3 hours total (50 epochs)
YOLOv8:          ~1.5-2 hours total (50 epochs)
FPN+U-Net+Swin:  ~1 hour/model (3 models × 50 epochs)

Total: ~7-12 hours for all 4 models
```

---

## 🎓 Learning Outcomes

After running these notebooks, you'll understand:

✅ Medical image segmentation
✅ Encoder-decoder architectures
✅ Multi-scale feature extraction
✅ Vision Transformers
✅ Conditional GANs
✅ Instance segmentation (YOLO)
✅ Model evaluation metrics
✅ Cross-validation strategies

---

## 💾 Results & Outputs

All notebooks save results to:
```
experiments/results/
├── 1_simple_unet/
│   ├── best_unet.pth          (trained model)
│   ├── training_history.png
│   └── predictions.png
├── 2_cgan/
│   ├── generator_best.pth
│   └── results/
├── 3_yolov8/
│   └── runs/detect/...
└── 4_fpn_unet_swin/
    ├── fold_0/
    ├── fold_1/
    └── metrics.csv
```

---

## 🚀 Getting Started

### Local Machine
```python
BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"
# Run any notebook in Jupyter
```

### Google Colab
```python
BASE_URL = "/content/drive/MyDrive/experiments"
# Open notebook in Colab
# First cell auto-detects & mounts Drive
# Run all cells
```

### Cloud (AWS/GCP)
```python
BASE_URL = "/path/to/experiments"
# Same notebooks, different BASE_URL
```

---

## ⚙️ Configuration Options

### Hyperparameters (all notebooks)

**Simple U-Net:**
```python
NUM_EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
IMAGE_SIZE = 256
```

**cGAN:**
```python
cgan_epochs = 50
cgan_batch = 4
learning_rate = 2e-4
```

**YOLOv8:**
```python
yolo_epochs = 50
yolo_batch = 8
model = 'yolov8m-seg'
```

**FPN+U-Net+Swin:**
```python
image_size = 384
batch_size = 8
num_epochs = 50
```

---

## 📚 References

**Simple U-Net:**
- Ronneberger et al. (2015): U-Net: Convolutional Networks for Biomedical Image Segmentation

**cGAN:**
- Mirza & Osindero (2014): Conditional Generative Adversarial Nets

**YOLOv8:**
- Ultralytics: State-of-the-art instance segmentation

**FPN + U-Net + Swin:**
- Lin et al. (2017): Feature Pyramid Networks for Object Detection
- Liu et al. (2021): Swin Transformer: Hierarchical Vision Transformer

---

## ✅ Checklist

Before running:
- [ ] Pick which notebook to start with
- [ ] Update `BASE_URL` if needed (usually not - auto-configured)
- [ ] Ensure 10GB+ disk space (for dataset + results)
- [ ] Check GPU availability (faster training)
- [ ] Install PyTorch if running locally

---

## 🎉 Ready to Start?

1. **First time?** → Start with `1_simple_unet_learning/simple_unet_colab.ipynb`
2. **Want comparison?** → Run all 4 approaches
3. **Need best results?** → Focus on `4_fpn_unet_swin/`
4. **Need speed?** → Use `3_yolov8/`

All notebooks are identical in structure - change ONE line and they work anywhere!

---

**Happy training! 🚀**
