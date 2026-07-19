# cGAN - UCNet Implementation Usage Guide

## 📁 Project Structure

```
cGAN/
├── README.md                    # Paper info & implementation details
├── requirements.txt             # Dependencies
├── USAGE.md                     # This file
├── notebooks/                   # Jupyter notebooks for analysis
│
└── ucnet/                       # Main implementation (organized)
    ├── train.py                 # Training script
    ├── inference.py             # Inference & visualization
    ├── smoke_test.py            # Test without data
    │
    ├── models/                  # Model definitions
    │   ├── __init__.py
    │   ├── generator.py         # U-Net generator
    │   ├── discriminator.py     # PatchGAN discriminator
    │   ├── criss_cross_attention.py  # Attention module
    │   └── losses.py            # Loss functions
    │
    ├── data/                    # Data handling
    │   ├── __init__.py
    │   └── arcade_dataset.py    # COCO dataset loader
    │
    └── utils/                   # Utilities
        ├── __init__.py
        └── metrics.py           # Evaluation metrics
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd experiments/paper_implementations/cGAN
pip install -r requirements.txt
# Or using UV:
uv pip install -r requirements.txt
```

### 2. Test Installation (No Data Required)
```bash
cd ucnet
python smoke_test.py
```
This runs a synthetic test to verify the environment is set up correctly.

### 3. Train on ARCADE Dataset
```bash
cd ucnet
python train.py \
  --train-images ../../datasets/ARCADE/train/images \
  --train-ann ../../datasets/ARCADE/train/annotations/train.json \
  --val-images ../../datasets/ARCADE/val/images \
  --val-ann ../../datasets/ARCADE/val/annotations/val.json \
  --epochs 400 --batch-size 4 --lr 2e-4
```

**Optional parameters:**
```bash
--epochs 400              # Training epochs
--batch-size 4            # Batch size
--lr 2e-4                # Learning rate
--alpha 1.0              # Segment loss weight
--beta 1.0               # Focal loss weight
--lambda 100.0           # L1 loss weight
--gamma 1.0              # Dice loss weight
--device cuda            # Device (cuda or cpu)
```

### 4. Run Inference
```bash
cd ucnet
python inference.py \
  --checkpoint checkpoints/ucnet_best.pth \
  --images ../../datasets/ARCADE/val/images \
  --ann ../../datasets/ARCADE/val/annotations/val.json \
  --out ../../results/paper_tests/cGAN/predictions
```

Outputs:
- Per-segment metrics (F1, IoU) printed to console
- Colored overlay images saved to `--out` directory

## 📊 Results Location

Training results are saved to:
```
results/paper_tests/cGAN/
├── checkpoints/
│   ├── ucnet_best.pth       # Best model by validation F1
│   └── ...
├── logs/
│   └── training.log         # Training logs
└── metrics/
    └── results.json         # Final metrics
```

## 🧪 Testing Without Data

Test the training pipeline without a dataset:
```bash
cd ucnet
python smoke_test.py
```

This:
- Creates synthetic COCO data
- Tests data loading
- Runs generator & discriminator forward passes
- Tests loss computation
- Tests backward pass & metrics
- Takes ~30 seconds

## 📚 Code Overview

### Training Loop (`train.py`)
- Loads ARCADE dataset in COCO format
- Trains cGAN with adversarial + auxiliary losses
- Saves best checkpoint based on validation F1
- Logs training progress

### Model Architecture (`models/`)
- **generator.py**: Deep U-Net (8 down, 8 up) with Criss-Cross attention
- **discriminator.py**: PatchGAN discriminator
- **criss_cross_attention.py**: Recurrent Criss-Cross Attention module
- **losses.py**: Segment Loss (novel), Focal, L1, Dice

### Data Pipeline (`data/arcade_dataset.py`)
- Loads ARCADE dataset in COCO polygon format
- Converts polygons to class masks
- Provides 3-channel input (vessel mask + original image)
- Applies augmentations (rotation, etc.)

### Evaluation (`utils/metrics.py`)
- Accuracy, Sensitivity, Specificity, Precision
- IoU (Intersection over Union)
- F1 score
- Per-segment and overall metrics

## 🔧 Configuration

Edit training parameters in `train.py` or pass via CLI:

```python
# Default values in train.py
config = {
    'epochs': 400,
    'batch_size': 4,
    'learning_rate': 2e-4,
    'device': 'cuda',
    'loss_weights': {
        'alpha': 1.0,    # Segment loss
        'beta': 1.0,     # Focal loss
        'lambda': 100.0, # L1 loss
        'gamma': 1.0,    # Dice loss
    }
}
```

## 📝 Paper Reference

**Paper**: "Accurate segmentation and labeling of coronary artery segments in X-ray angiography with an improved UNet-based cGAN architecture"

**Authors**: Yang et al.

**Published**: 2026

**Key contributions implemented**:
- Improved U-Net generator (8 down/up layers)
- Criss-Cross attention at encoder bottleneck
- PatchGAN discriminator
- Novel Segment Loss (Eq. 1) + auxiliary losses (focal, L1, dice)
- Evaluation on MICCAI-2023 ARCADE dataset (20 coronary segments)

See `README.md` for complete mapping between paper and code.

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 2
```

### Slow Training
```bash
# Use GPU
python train.py --device cuda

# Reduce image resolution (modify in arcade_dataset.py)
# resize to 256x256 instead of 512x512
```

### Dataset Not Found
```bash
# Make sure ARCADE dataset is in correct location:
datasets/ARCADE/
├── train/images/ *.png
├── train/annotations/train.json
├── val/images/ *.png
├── val/annotations/val.json
```

## 📊 Next Steps

1. **Run smoke test**: Verify environment works
2. **Train on ARCADE**: Start training
3. **Monitor results**: Check metrics in `results/paper_tests/cGAN/`
4. **Analyze results**: Use Jupyter notebooks for analysis
5. **When validated**: Move to `src/main_research/models/ucnet/`

## 📞 Questions?

Refer to:
- `README.md` - Paper implementation details
- `ucnet/smoke_test.py` - Example of complete pipeline
- `ucnet/models/` - Model definitions
- `ucnet/data/` - Data loading
- `ucnet/utils/` - Evaluation metrics
