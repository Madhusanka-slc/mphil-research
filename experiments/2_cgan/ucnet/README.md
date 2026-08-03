# UCNet — Coronary Artery Segment Segmentation & Labeling (ARCADE)

A from-scratch PyTorch reimplementation of:

> Yang et al., *"Accurate segmentation and labeling of coronary artery
> segments in X-ray angiography with an improved UNet-based cGAN
> architecture,"* Biomedical Signal Processing and Control 112 (2026) 108812.

UCNet is a conditional GAN whose generator is a deepened U-Net (8 down / 8
up) with a Criss-Cross attention bottleneck, trained with a novel **Segment
Loss** plus focal, L1, and multi-class dice losses to do *instance*
segmentation of 20 coronary artery segments on the MICCAI-2023 **ARCADE**
dataset.

## Project layout

```
ucnet/
├── models/
│   ├── criss_cross_attention.py   # RCCA module (CCNet), Sec. 3.1 / Fig. 2
│   ├── generator.py               # improved 8-down/8-up U-Net generator
│   ├── discriminator.py           # PatchGAN discriminator, Sec. 3.2
│   └── losses.py                  # Segment/Focal/L1/Dice, Eqs. 1–9
├── data/
│   └── arcade_dataset.py          # COCO-polygon → class-mask loader
├── utils/
│   └── metrics.py                 # Acc/Sen/Spe/Prec/IoU/F1, Eqs. 10–15
├── train.py                       # cGAN training loop (Sec. 4.3 settings)
├── inference.py                   # predict + coloured overlays
└── smoke_test.py                  # end-to-end test on synthetic data
```

## How the code maps to the paper

| Paper component | Where |
|---|---|
| Improved U-Net generator, 8 down + 8 up | `models/generator.py` |
| Criss-Cross attention at encoder bottom (Fig. 2) | `models/criss_cross_attention.py` |
| PatchGAN discriminator, N×N patch map (Sec. 3.2) | `models/discriminator.py` |
| Segment Loss `L_seg = M/N` (Eq. 1) | `losses.SegmentLoss` (soft surrogate + exact `hard_ratio`) |
| Focal Loss (Eq. 6) | `losses.FocalLoss` |
| L1 Loss (Eq. 7) | inside `losses.UCNetLoss` |
| Multi-class Dice (Eq. 8) | `losses.MultiClassDiceLoss` |
| Combined objective (Eq. 9) | `losses.UCNetLoss.generator_loss` |
| Six metrics (Eqs. 10–15) | `utils/metrics.py` |
| Binary vessel mask as cGAN condition + overlaid original image | `arcade_dataset.__getitem__` (3-channel input) |
| lr 2e-4, batch 4, 400 epochs, Adam, random-rotation aug | `train.py` |

## Data

Uses the **ARCADE** segmentation task (Popov et al., *Sci. Data* 2024),
distributed in COCO format. Point the loader at each split's image folder
and annotation json:

```
ARCADE/
  train/images/*.png      train/annotations/train.json
  val/images/*.png        val/annotations/val.json
```

Category ids are read from the json and mapped to contiguous class indices
automatically, so the loader works whether your split has the full 25
segments or the 20 the paper evaluates. To reproduce the paper's exact
20-segment evaluation, drop the rare segments (see Table 1: *second diagonal
a*, and the four test segments with <10 occurrences) from the annotation
file before training.

## Install

```bash
pip install -r requirements.txt   # torch, torchvision, numpy, pillow
```

## Train

```bash
python train.py \
  --train-images ARCADE/train/images \
  --train-ann    ARCADE/train/annotations/train.json \
  --val-images   ARCADE/val/images \
  --val-ann      ARCADE/val/annotations/val.json \
  --epochs 400 --batch-size 4 --lr 2e-4
```

Loss weights α (segment), β (focal), λ (L1), γ (dice) are CLI flags; tune
them on a validation split (Eq. 9 leaves them as hyperparameters). Best
checkpoint by validation F1 is saved to `checkpoints/ucnet_best.pth`.

## Inference / visualisation

```bash
python inference.py \
  --checkpoint checkpoints/ucnet_best.pth \
  --images ARCADE/val/images \
  --ann    ARCADE/val/annotations/val.json \
  --out    predictions
```

Prints overall + per-segment F1/IoU and writes one colour-coded overlay per
image (one colour per segment).

## Verify it runs (no dataset needed)

```bash
python smoke_test.py
```

Builds tiny synthetic COCO data and runs data → G/D → losses → backward →
metrics.

## Notes / deviations

- **Segment Loss** in Eq. 1 is piecewise-constant (a hard 90 % overlap
  count), so it has no gradient. `SegmentLoss` optimises a soft coverage
  surrogate with identical semantics, and `SegmentLoss.hard_ratio` computes
  the exact `M/N` for logging.
- Generator is the pix2pix-style deep U-Net (matching "eight down / eight
  up" and the UENet reference [36]); it requires 512×512 inputs (eight
  halvings). ~54.8M params at `base_filters=64`.
- The paper doesn't publish every loss weight / the exact dice formulation
  constant-for-constant; those are exposed as configurable knobs rather than
  hard-coded.
- Discriminator conditions on the 3-channel input image and receives the
  softmax segmentation (fake) or one-hot GT (real).
