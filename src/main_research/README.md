# Main Research Code

This directory contains your **final MPhil research implementation** - code that's been tested, validated, and integrated.

## Structure

```
main_research/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── backbone.py           # Base model
│   ├── heads.py              # Detection/classification heads
│   └── losses.py             # Loss functions
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Dataset classes
│   ├── transforms.py         # Data augmentation
│   └── loaders.py            # DataLoader utilities
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training loop
│   ├── validators.py         # Validation/testing
│   └── callbacks.py          # Logging, checkpointing
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Plotting utilities
│   └── helpers.py            # Common utilities
└── config.py                 # Configuration management
```

## When to Move Code Here

Only move code from `experiments/paper_implementations/` when:

✅ **DO MOVE IF:**
- Code achieves target performance
- Thoroughly tested on multiple datasets
- Integrates with other components
- Documentation is complete
- No known bugs or major TODOs

❌ **DON'T MOVE IF:**
- Still experimenting/iterating
- Hardcoded parameters for one dataset
- Missing error handling
- Incomplete documentation
- Still comparing with multiple paper versions

## Integration Checklist

Before moving paper implementation here:

```python
# Code quality
[ ] Follows PEP 8 style guide
[ ] Has docstrings on functions/classes
[ ] Type hints where applicable
[ ] No hardcoded paths/values

# Testing
[ ] Unit tests pass
[ ] Works with different datasets
[ ] Performance documented
[ ] Edge cases handled

# Documentation
[ ] README.md with usage
[ ] Configuration documented
[ ] Paper reference included
[ ] Code comments for complex logic

# Integration
[ ] Imports work cleanly
[ ] Compatible with other modules
[ ] Uses project configs
[ ] Logging implemented
```

## Usage Example

```python
from src.main_research.models import MyModel
from src.main_research.data import MyDataset
from src.main_research.training import Trainer

# Load model
model = MyModel(config)

# Load data
dataset = MyDataset(path='datasets/mydata')
dataloader = DataLoader(dataset, batch_size=32)

# Train
trainer = Trainer(model, config)
trainer.train(dataloader)
```
