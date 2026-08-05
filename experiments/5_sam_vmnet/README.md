# SAM-VMNet: Foundation Model + Vessel Segmentation

**Hybrid approach combining Meta's Segment Anything Model (SAM) with vessel-specialized architecture for coronary artery segmentation.**

Based on: [qimingfan10/SAM-VMNet](https://github.com/qimingfan10/SAM-VMNet)

---

## 🎯 What is SAM-VMNet?

SAM-VMNet = SAM (Foundation Model) + VMNet (Vessel Segmentation)

A hybrid approach that:
- Leverages SAM's foundation knowledge (trained on 1B images)
- Adds vessel-specific expertise for medical imaging
- Combines general segmentation with specialized architecture
- Better for thin coronary arteries than vanilla SAM

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Expected Dice | 0.85-0.92 |
| Training Time | 2-3 hours |
| GPU Memory | 12GB (Colab free) |
| Best For | Foundation model comparison |

---

## 📁 Folder Structure

```
5_sam_vmnet/
├── README.md                       (this file)
├── 01_sam_vmnet_setup.ipynb        (clone & setup)
├── 02_sam_vmnet_finetune.ipynb     (fine-tune on ARCADE)
├── 03_sam_vmnet_inference.ipynb    (evaluate & visualize)
└── models.py                       (utility functions)
```

---

## 🚀 Quick Start

### Step 1: Setup (01_sam_vmnet_setup.ipynb)
- Clone SAM-VMNet repository
- Install dependencies
- Load pre-trained weights
- Test on sample ARCADE images

**Time:** 15-20 minutes

### Step 2: Fine-tune (02_sam_vmnet_finetune.ipynb)
- Fine-tune on ARCADE training set
- Monitor training progress
- Validate on validation set
- Save best checkpoint

**Time:** 2-3 hours (GPU dependent)

### Step 3: Inference (03_sam_vmnet_inference.ipynb)
- Evaluate on test set
- Compute Dice, IoU, metrics
- Visualize predictions
- Compare with other 4 approaches

**Time:** 20-30 minutes

---

## 💻 Colab Setup

All notebooks are **Colab-eligible** with BASE_URL configuration:

```python
# At top of every notebook:

BASE_URL = "/content/drive/MyDrive/experiments"  # Colab

# OR for local:
# BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"

# Auto-detection + Drive mount handled automatically
```

---

## 📋 Notebook Details

### 01_sam_vmnet_setup.ipynb
**Purpose:** Clone, install, verify SAM-VMNet works

**What it does:**
1. Set BASE_URL and mount Drive
2. Clone SAM-VMNet GitHub repo
3. Install requirements
4. Load pre-trained SAM-VMNet model
5. Test on 2-3 ARCADE images
6. Display sample predictions

**Outputs:**
- Confirms model loads
- Shows sample segmentations
- GPU memory usage info

---

### 02_sam_vmnet_finetune.ipynb
**Purpose:** Fine-tune SAM-VMNet on ARCADE dataset

**What it does:**
1. Load training/validation data
2. Set up training configuration
3. Fine-tune model (50 epochs)
4. Track loss and metrics
5. Save best checkpoint
6. Plot training history

**Configuration:**
```python
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
IMAGE_SIZE = 512
```

**Outputs:**
- best_sam_vmnet.pth (model checkpoint)
- training_history.png
- metrics.csv

---

### 03_sam_vmnet_inference.ipynb
**Purpose:** Evaluate and compare with other 4 approaches

**What it does:**
1. Load best fine-tuned model
2. Evaluate on test set
3. Compute metrics (Dice, IoU, Sensitivity, etc.)
4. Visualize predictions
5. Compare performance with other 4 approaches
6. Generate comparison table

**Outputs:**
- predictions.png (visual comparisons)
- metrics.csv (quantitative results)
- comparison_table.png
- performance_summary.txt

---

## ⚙️ Key Features

### Colab-Optimized
- ✓ Automatic Drive mounting
- ✓ GPU detection
- ✓ Memory-efficient fine-tuning
- ✓ Progress bars and logging
- ✓ Auto-save checkpoints

### BASE_URL Configuration
```python
# Single variable, works everywhere:
BASE_URL = "/content/drive/MyDrive/experiments"  # Colab
# OR
BASE_URL = "D:/MPHIL_CODES/MPHIL_MAIN_REPO/experiments"  # Windows
```

### Dataset Compatibility
- ✓ COCO format (like ARCADE)
- ✓ Automatic data loading
- ✓ Polygon-to-mask conversion
- ✓ 5-fold cross-validation support

---

## 🎓 Learning Outcomes

After running these notebooks, you'll understand:

✅ Foundation models for segmentation
✅ SAM architecture and capabilities
✅ Hybrid approaches (foundation + specialized)
✅ Fine-tuning on medical imaging
✅ Colab-based training workflows
✅ Model comparison methodology

---

## 📊 Comparison with Other 4 Approaches

| Approach | Dice | Type | Best For |
|----------|------|------|----------|
| Simple U-Net | 0.75 | CNN Baseline | Learning |
| cGAN | 0.98 | Generative | Generation |
| YOLOv8 | 0.60 | Detection | Speed |
| FPN+Swin | 0.91 | CNN+ViT | Best accuracy |
| **SAM-VMNet** | **0.85-0.92** | **Foundation+Medical** | **Novel hybrid** |

---

## 🔧 GPU Requirements

| Environment | GPU | Memory | Suitable |
|------------|-----|--------|----------|
| Google Colab | T4/A100 | 12-40GB | ✅ YES |
| Local GPU | RTX 3080+ | 10GB+ | ✅ YES |
| CPU only | - | - | ❌ No (too slow) |

---

## 📚 References

**SAM-VMNet Paper/Code:**
- GitHub: https://github.com/qimingfan10/SAM-VMNet
- Citation: Follow original repository guidelines

**SAM (Segment Anything):**
- Paper: Kirillov et al., 2023
- GitHub: https://github.com/facebookresearch/segment-anything

**Coronary Segmentation:**
- ARCADE Challenge: https://arcade.grand-challenge.org/
- Dataset: Popov et al., Scientific Data, 2024

---

## 💡 Tips & Tricks

1. **First Run:**
   - Start with setup notebook (quick)
   - Test on 2-3 images
   - Verify GPU works

2. **Fine-tuning:**
   - Start with smaller lr (1e-4)
   - Monitor validation Dice
   - Stop if no improvement for 5 epochs

3. **Memory Issues:**
   - Reduce batch_size to 2
   - Reduce image_size to 384
   - Use gradient accumulation

4. **Better Results:**
   - Increase epochs to 100
   - Use data augmentation
   - Ensemble with FPN+Swin predictions

---

## ⏱️ Timeline

```
Setup:        15-20 min
Fine-tuning:  2-3 hours (with GPU)
Inference:    20-30 min
Total:        ~3-4 hours
```

---

## 🎉 Ready to Start?

1. **Open notebook 1:** `01_sam_vmnet_setup.ipynb`
2. **Run in Colab:** Right-click → Open with Colaboratory
3. **Follow the cells:** Each cell has detailed comments
4. **Move to notebook 2:** When setup complete
5. **Compare results:** In notebook 3

All notebooks follow same pattern as other 4 approaches!

---

**Happy training! 🚀**
