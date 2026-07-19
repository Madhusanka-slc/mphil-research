# Paper Implementations

This directory contains implementations of research papers you're testing and learning from.

## Structure

```
paper_implementations/
├── paper_1_name/
│   ├── __init__.py
│   ├── model.py              # Paper model implementation
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── README.md             # Paper notes & reference
│   └── paper_reference.pdf   # Link to paper (if stored locally)
├── paper_2_name/
└── ...
```

## Workflow

### 1. **Testing a Paper Implementation**

```bash
# Create folder for paper
mkdir experiments/paper_implementations/yolo_v8_custom

# Copy/create scripts
# - Implement model in model.py
# - Write training script in train.py

# Run experiment
python experiments/paper_implementations/yolo_v8_custom/train.py

# Results saved to:
# results/paper_tests/yolo_v8_custom/
```

### 2. **Document Your Implementation**

Create `README.md` in each paper folder:
```markdown
# YOLO V8 Custom Implementation

**Paper**: YOLOv8: A Faster and Better Real-Time Object Detection Algorithm
**Authors**: Ultralytics
**Year**: 2023
**Link**: https://arxiv.org/abs/...

## Key Implementation Details
- Model architecture
- Training parameters
- Dataset used

## Results
- Accuracy: 95.2%
- Inference time: 23ms

## Notes
- What worked well
- What was challenging
- Differences from original paper
```

### 3. **When to Move to Main Research**

Once validated, move good implementations to:
```
src/main_research/
```

**Checklist before moving:**
- ✓ Achieves paper results or better
- ✓ Well-tested on your datasets
- ✓ Code is clean and documented
- ✓ Can be integrated with other components
