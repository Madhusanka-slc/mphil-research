# MPhil Research Workflow Guide

## Overview

Your repository is now set up for a **3-phase research workflow**:

```
Phase 1: EXPLORE PAPERS
 │
 └─→ experiments/paper_implementations/
     • Test different paper implementations
     • Track results in results/paper_tests/
     
Phase 2: VALIDATE & INTEGRATE
 │
 └─→ Move validated code → src/main_research/
     • Follow integration checklist
     • Clean up and document
     
Phase 3: BUILD YOUR MPHIL RESEARCH
 │
 └─→ src/main_research/
     • Combine tested implementations
     • Create your final research
     • Save results to results/main_research/
```

---

## Phase 1: Testing Paper Implementations

### Step 1: Create Paper Implementation Folder

```bash
# Create folder
mkdir experiments/paper_implementations/paper_name

# Create necessary files
touch experiments/paper_implementations/paper_name/__init__.py
touch experiments/paper_implementations/paper_name/README.md
touch experiments/paper_implementations/paper_name/model.py
touch experiments/paper_implementations/paper_name/train.py
touch experiments/paper_implementations/paper_name/evaluate.py
touch experiments/paper_implementations/paper_name/config.yaml
```

### Step 2: Document the Paper

Fill in `README.md` using template from `experiments/paper_implementations/TEMPLATE_paper_implementation.md`

**Example structure:**
```markdown
# Paper Title

## Metadata
- Paper Name: YOLOv8...
- Authors: Ultralytics
- Published: 2023
- Status: ⏳ In Progress

## Implementation Details
- Architecture
- Training setup
- Results

## Key Learnings
- What worked
- Challenges
```

### Step 3: Implement & Test

**Example train.py:**
```python
import torch
from model import YOLOv8
from data import DataLoader
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create model
model = YOLOv8(config)

# Load data
dataloader = DataLoader(...)

# Train
for epoch in range(config['epochs']):
    # Training loop
    pass

# Save results
torch.save({
    'model': model.state_dict(),
    'accuracy': accuracy,
    'speed': inference_time,
}, 'checkpoint.pth')
```

### Step 4: Track Results

Save outputs to: `results/paper_tests/paper_name/`

Structure:
```
results/paper_tests/paper_name/
├── checkpoints/          # Model weights
│   └── best_model.pth
├── logs/                 # Training logs
│   └── training.log
├── metrics/              # Accuracy, speed, etc.
│   └── results.json
└── visualizations/       # Plots, sample outputs
    └── inference_samples.jpg
```

### Step 5: Update Tracking

Update `PAPERS_TRACKING.md`:
```markdown
| # | Paper Name | Status | Folder | mAP | Notes |
|---|-----------|--------|--------|-----|-------|
| 1 | YOLOv8 | 🧪 Testing | paper_impl/ | 94.2% | Good baseline |
```

---

## Phase 2: Validate & Integrate

### Integration Checklist

Before moving code to `src/main_research/`, verify:

**Code Quality**
- [ ] Follows PEP 8 style guide
- [ ] Has docstrings on all functions
- [ ] Type hints where applicable
- [ ] No hardcoded paths or values
- [ ] No debug print statements

**Testing**
- [ ] Unit tests pass
- [ ] Works with different datasets
- [ ] Performance documented and matches paper claims
- [ ] Edge cases handled

**Documentation**
- [ ] README.md with usage instructions
- [ ] Configuration fully documented
- [ ] Paper reference included
- [ ] Code comments for complex sections

**Integration**
- [ ] Imports work cleanly
- [ ] Compatible with other modules in src/
- [ ] Uses project config system
- [ ] Logging implemented

### How to Integrate

1. **Create directory in src/main_research/**
   ```bash
   mkdir src/main_research/models/yolo
   cp -r experiments/paper_implementations/yolo_v8/* src/main_research/models/yolo/
   ```

2. **Update imports**
   ```python
   # Before (experiment)
   from model import YOLOv8
   
   # After (production)
   from src.main_research.models.yolo import YOLOv8
   ```

3. **Update PAPERS_TRACKING.md**
   ```markdown
   | 1 | YOLOv8 | 🔗 Integrated | yolo/ | 94.2% | Moved to main_research |
   ```

---

## Phase 3: Build Your MPhil Research

### Structure

```
src/main_research/
├── models/
│   ├── yolo/                    # From paper_impl
│   ├── faster_rcnn/             # From paper_impl
│   ├── custom_model.py          # Your new research
│   └── __init__.py
├── data/
│   ├── dataset.py               # Main dataset class
│   ├── transforms.py            # Data augmentation
│   └── loaders.py               # DataLoaders
├── training/
│   ├── trainer.py               # Main training loop
│   ├── validators.py            # Validation/testing
│   └── callbacks.py             # Logging, checkpointing
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   ├── visualization.py         # Plotting
│   └── helpers.py               # Common utilities
└── config.py                    # Configuration management
```

### Example: Using Integrated Code

```python
# experiments/run_main_research.py

from src.main_research.models import YOLOv8, FasterRCNN
from src.main_research.data import ARCADEDataset
from src.main_research.training import Trainer
import yaml

# Load config
with open('configs/research.yaml') as f:
    config = yaml.safe_load(f)

# Combine multiple paper implementations
model = YOLOv8(config['model1'])
backbone = FasterRCNN(config['model2']).backbone

# Create your custom architecture
from src.main_research.models import MyCustomModel
research_model = MyCustomModel(model, backbone, config)

# Load your dataset (ARCADE)
dataset = ARCADEDataset('datasets/ARCADE', config)

# Train
trainer = Trainer(research_model, config)
trainer.train(dataset)

# Save results
import json
with open('results/main_research/results.json', 'w') as f:
    json.dump(trainer.metrics, f)
```

---

## Practical Workflow Example

### Day 1: Start Testing YOLOv8 Paper

```bash
# 1. Create folder
mkdir experiments/paper_implementations/yolo_v8

# 2. Copy template
cp experiments/paper_implementations/TEMPLATE_paper_implementation.md \
   experiments/paper_implementations/yolo_v8/README.md

# 3. Edit README with paper info
nano experiments/paper_implementations/yolo_v8/README.md

# 4. Create implementation
touch experiments/paper_implementations/yolo_v8/model.py
touch experiments/paper_implementations/yolo_v8/train.py
touch experiments/paper_implementations/yolo_v8/config.yaml

# 5. Run experiment
cd experiments/paper_implementations/yolo_v8/
python train.py

# 6. Update tracking
# Edit PAPERS_TRACKING.md with: "YOLOv8 | 🧪 Testing | yolo_v8/ | - | Started implementation"
```

### Day 7: Paper Implementation Achieves Goals

```bash
# 1. Verify results
# - Performance matches paper? ✅
# - Well documented? ✅
# - Ready to integrate? ✅

# 2. Integrate to main_research
mkdir -p src/main_research/models/yolo
cp -r experiments/paper_implementations/yolo_v8/* src/main_research/models/yolo/

# 3. Test integration
python -c "from src.main_research.models.yolo import YOLOv8; print('Import successful')"

# 4. Update tracking
# Change status to: "🔗 Integrated"
```

### Week 3: Combine Implementations into Main Research

```bash
# 1. Create your research model combining YOLOv8 + FasterRCNN
# Edit: src/main_research/models/custom_model.py

# 2. Train on ARCADE dataset
python experiments/run_main_research.py --config configs/research.yaml

# 3. Results saved automatically to: results/main_research/
```

---

## Useful Commands

```bash
# Activate environment
source .venv/Scripts/activate  # Windows
source .venv/bin/activate     # Linux/Mac

# Run paper implementation
python experiments/paper_implementations/yolo_v8/train.py

# Run tests
pytest tests/ -v

# Run main research
python experiments/run_main_research.py --config configs/research.yaml

# View results
ls results/paper_tests/        # Paper implementation results
ls results/main_research/      # Final research results

# Update tracking
nano PAPERS_TRACKING.md
```

---

## File Organization Quick Reference

| Task | Directory |
|------|-----------|
| Test paper implementation | `experiments/paper_implementations/` |
| Write notebooks | `notebooks/` |
| Store configs | `configs/` |
| Write unit tests | `tests/` |
| Save paper results | `results/paper_tests/` |
| Finalized code | `src/main_research/` |
| Final results | `results/main_research/` |
| Dataset | `datasets/` (gitignored) |
| Model checkpoints | `models/` (gitignored) |
| Papers | `papers/` (gitignored) |

---

## Tips for Success

✅ **Do:**
- Keep experiments isolated in `experiments/`
- Document everything in README files
- Track progress in `PAPERS_TRACKING.md`
- Test before moving to `src/main_research/`
- Use configuration files, not hardcoded values

❌ **Don't:**
- Put all code directly in `src/main_research/`
- Mix paper implementations with final code
- Skip documentation
- Commit large files (they're gitignored for a reason)
- Forget to update PAPERS_TRACKING.md

---

## Questions?

Refer to:
- [`experiments/paper_implementations/README.md`](experiments/paper_implementations/README.md) - Paper implementation guide
- [`src/main_research/README.md`](src/main_research/README.md) - Integration requirements
- [`PAPERS_TRACKING.md`](PAPERS_TRACKING.md) - Track all papers
- [`TEMPLATE_paper_implementation.md`](experiments/paper_implementations/TEMPLATE_paper_implementation.md) - Template for new papers
