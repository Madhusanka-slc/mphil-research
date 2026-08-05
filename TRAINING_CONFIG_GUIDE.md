# Training Configuration Guide

## 🎯 How to Configure Training Hyperparameters

All training notebooks now support easy configuration of key hyperparameters at the top of the notebook.

---

## 📋 Configuration Variables

### 1. **BASE_URL** (Data Location)
Location of experiments folder (required for all notebooks)

```python
# For Google Colab:
BASE_URL = "/content/drive/MyDrive/experiments"

# For Windows local:
BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"

# For Mac/Linux local:
BASE_URL = "/Users/yourname/experiments"
```

### 2. **EPOCHS** (Training Duration)
Number of training epochs (iterations through dataset)

```python
EPOCHS = 20  # Recommended defaults by model
```

| Model | Default | Range | Notes |
|-------|---------|-------|-------|
| Simple U-Net | 20 | 10-100 | Faster training |
| cGAN | 50 | 30-200 | Needs more epochs |
| YOLOv8 | 50 | 20-100 | Detection training |
| FPN+Swin | 50 | 30-100 | Complex model |
| SAM-VMNet | 50 | 30-100 | Foundation model |

### 3. **BATCH_SIZE** (Memory Usage)
Number of images per batch

```python
BATCH_SIZE = 4  # Recommended for 8GB+ GPU
```

| GPU Memory | Recommended | Notes |
|------------|-------------|-------|
| 4GB | 2 | Very tight |
| 8GB | 4 | Standard |
| 12GB+ | 8 | Can increase |
| 16GB+ | 16 | More GPU power |

### 4. **LEARNING_RATE** (Training Speed)
How fast model learns

```python
LEARNING_RATE = 1e-3  # Typical starting point
```

| Model | Default | Range |
|-------|---------|-------|
| Simple U-Net | 1e-3 | 1e-4 to 1e-2 |
| cGAN | 2e-4 | 1e-5 to 1e-3 |
| YOLOv8 | 1e-3 | 1e-4 to 1e-2 |
| FPN+Swin | 1e-3 | 1e-4 to 1e-2 |
| SAM-VMNet | 1e-4 | 1e-5 to 1e-3 |

### 5. **IMAGE_SIZE** (Resolution)
Input image size (square)

```python
IMAGE_SIZE = 256  # Default for most models
```

| Size | Training Speed | Quality | Memory |
|------|---|---|---|
| 128 | Fast ⚡ | Lower | 2GB |
| 256 | Medium ⚡⚡ | Good | 4GB |
| 384 | Slow ⚡⚡⚡ | Better | 6GB |
| 512 | Very slow | Best | 8GB+ |

---

## 📍 Where to Find Configuration Cells

### Notebook: `1_simple_unet_learning/simple_unet_colab.ipynb`

**Cell:** "Part 10: Train the Model"

```python
# ===== TRAINING CONFIGURATION (Change these values) =====
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
IMAGE_SIZE = 256
# =====================================================
```

### Notebook: `2_cgan/colab_train.ipynb`

**Cell:** Top of notebook

```python
# ===== CONFIGURATION =====
BASE_URL = "/content/drive/MyDrive/experiments"
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
# ========================
```

### Notebook: `3_yolov8/colab_train.ipynb`

**Cell:** Top of notebook

```python
# ===== CONFIGURATION =====
BASE_URL = "/content/drive/MyDrive/experiments"
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
# ========================
```

### Notebook: `4_fpn_unet_swin/02_train_colab.ipynb`

**Cell:** Top of notebook (after imports)

```python
# ===== CONFIGURATION =====
BASE_URL = "/content/drive/MyDrive/experiments"
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
IMAGE_SIZE = 384
# ========================
```

### Notebook: `5_sam_vmnet/02_sam_vmnet_finetune.ipynb`

**Cell:** First cell

```python
# ===== CONFIGURATION =====
BASE_URL = "/content/drive/MyDrive/experiments"
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
IMAGE_SIZE = 512
# ========================
```

---

## 🎯 Quick Adjustment Guide

### 🚀 Want Faster Training?
```python
EPOCHS = 10       # Reduce epochs
BATCH_SIZE = 8    # Larger batches
IMAGE_SIZE = 128  # Smaller images
```

### 🎓 Want Better Quality?
```python
EPOCHS = 100      # More training
BATCH_SIZE = 2    # Smaller batches (more updates)
IMAGE_SIZE = 512  # Larger images
LEARNING_RATE = 1e-4  # Slower, more careful learning
```

### 💾 Out of Memory?
```python
BATCH_SIZE = 2    # Reduce batch size first!
IMAGE_SIZE = 256  # Then reduce image size
EPOCHS = 30       # Can still train well
```

### ⚡ GPU Limitations?
```python
# If GPU < 8GB:
BATCH_SIZE = 2
IMAGE_SIZE = 256
EPOCHS = 20

# If GPU 8-12GB:
BATCH_SIZE = 4
IMAGE_SIZE = 384
EPOCHS = 50

# If GPU > 12GB:
BATCH_SIZE = 8
IMAGE_SIZE = 512
EPOCHS = 100
```

---

## 📊 Expected Training Times

### Simple U-Net
- **Default:** ~1 hour (20 epochs, 256×256)
- **Fast:** ~30 min (10 epochs, 128×128)
- **Quality:** ~3 hours (50 epochs, 384×384)

### cGAN
- **Default:** ~2-3 hours (50 epochs, 256×256)
- **Fast:** ~1 hour (20 epochs, 256×256)
- **Quality:** ~5 hours (100 epochs, 512×512)

### YOLOv8
- **Default:** ~1.5-2 hours (50 epochs)
- **Fast:** ~1 hour (30 epochs)
- **Quality:** ~3 hours (100 epochs)

### FPN + U-Net + Swin
- **Default:** ~3-5 hours (50 epochs, 384×384)
- **Fast:** ~2 hours (30 epochs, 256×256)
- **Quality:** ~8+ hours (100 epochs, 512×512)

### SAM-VMNet
- **Default:** ~3-4 hours (50 epochs)
- **Fast:** ~2 hours (30 epochs)
- **Quality:** ~6 hours (100 epochs)

---

## ✅ Recommended Configurations for Different Goals

### 🎓 Learning (First Time)
```python
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
IMAGE_SIZE = 256
# Time: 30 min - 1 hour
# Good enough to understand how it works
```

### 📊 Research (Paper)
```python
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
IMAGE_SIZE = 256
# Time: 1-3 hours depending on model
# Good performance for comparison
```

### 🏆 Best Quality (Publication)
```python
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
IMAGE_SIZE = 384
# Time: 4-8 hours depending on model
# Best possible results
```

### ⚡ Quick Test (Colab Demo)
```python
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
IMAGE_SIZE = 128
# Time: 10-30 min
# Just to verify it runs
```

---

## 🔧 How to Use

1. **Open notebook in Colab**
2. **Scroll to configuration cell** (usually at top)
3. **Change these variables:**
   ```python
   BASE_URL = "/content/drive/MyDrive/experiments"  # Your location
   EPOCHS = 20        # How many times to train
   BATCH_SIZE = 4     # Images per step
   LEARNING_RATE = 1e-3  # Training speed
   IMAGE_SIZE = 256   # Resolution
   ```
4. **Run the notebook** - uses your values!

---

## 💡 Tips

- ✅ **Always set BASE_URL first** - required for data loading
- ✅ **Start with recommended values** - they're pre-tuned
- ✅ **Increase EPOCHS for better quality** - most impactful change
- ✅ **Reduce BATCH_SIZE if out of memory** - not IMAGE_SIZE
- ✅ **Lower LEARNING_RATE if loss becomes erratic** - more stable
- ✅ **Increase IMAGE_SIZE only if needed** - impacts memory most

---

## 🚀 Ready to Train?

1. Edit the **CONFIGURATION** cell
2. Set your values
3. Run all cells
4. Models auto-save best checkpoint

**Happy training!** 🎓
