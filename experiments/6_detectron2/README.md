# Detectron2: Instance Segmentation (Future Implementation)

**Placeholder for Detectron2 implementation**

## 📋 Status

🔄 **PLANNED** - Not yet implemented

This folder is reserved for future Detectron2 implementation for instance segmentation on ARCADE dataset.

---

## 📚 About Detectron2

**Detectron2** = Meta's object detection & instance segmentation framework

### Key Features:
- ✓ Pre-trained models available
- ✓ Mask R-CNN architecture
- ✓ Instance-level segmentation
- ✓ Detection + segmentation combined

### For ARCADE:
- Each coronary segment = one instance
- Can detect and segment 25 segments simultaneously
- Bounding box + mask output per segment

---

## 🗂️ Expected Structure (When Implemented)

```
6_detectron2/
├── README.md
├── 01_detectron2_setup.ipynb
├── 02_detectron2_finetune.ipynb
├── 03_detectron2_inference.ipynb
└── models.py (utilities)
```

---

## 📊 Expected Performance

- **Dice Score:** ~0.70-0.75 (with fine-tuning)
- **Training Time:** 4-6 hours
- **GPU Memory:** 16GB+ recommended

---

## 🎯 Implementation Plan

1. Clone Detectron2 from Meta
2. Convert ARCADE to Detectron2 format
3. Configure Mask R-CNN for coronary data
4. Fine-tune on 300 ARCADE images
5. Evaluate and compare with other 5 approaches

---

## 💡 Why Later?

Currently have 5 complete approaches:
1. ✅ Simple U-Net (Dice: 0.75)
2. ✅ cGAN (Dice: 0.98) ← Better for instance seg
3. ✅ YOLOv8 (Detection: mAP50 0.60)
4. ✅ FPN+Swin (Dice: 0.91)
5. ✅ SAM-VMNet (Dice: 0.85-0.92)

**cGAN and YOLOv8 already cover instance segmentation,**
**so Detectron2 can be added as future work/extension.**

---

## 📌 Notes

- Detectron2 is SOTA but complex to configure
- Better for natural images, less ideal for thin vessels
- Heavier than current approaches
- Good for research, not critical for MPhil

---

**Status: Reserved for future implementation ⏳**

When ready to implement:
1. Create setup, fine-tuning, inference notebooks
2. Follow same BASE_URL pattern as other experiments
3. Add to comparison tables
4. Update experiments/README.md

---

Created: 2026-08-05
Target: Post-MPhil research extension
