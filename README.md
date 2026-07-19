# MPhil Research Repository

Computer Vision and Machine Learning Research - PyTorch & Ultralytics

## 📊 Research Workflow

```
PHASE 1: EXPLORE PAPERS
   ↓
experiments/paper_implementations/ → Test & implement key papers
   ↓
results/paper_tests/ → Track results
   ↓
PHASE 2: VALIDATE & INTEGRATE
   ↓
Move validated code → src/main_research/
   ↓
PHASE 3: PRODUCTION RESEARCH
   ↓
src/main_research/ → Your final MPhil codebase
   ↓
results/main_research/ → Final results
```

## Repository Structure

```
.
├── experiments/
│   ├── paper_implementations/   # Test different paper implementations
│   │   ├── sample_yolo_v8/     # Example: Paper 1
│   │   ├── paper_2_name/       # Example: Paper 2
│   │   └── README.md           # How to add papers
│   └── ablations/              # Ablation studies
│
├── src/
│   ├── main_research/          # VALIDATED & INTEGRATED CODE
│   │   ├── models/             # Final model definitions
│   │   ├── data/               # Data pipeline
│   │   ├── training/           # Training infrastructure
│   │   └── utils/              # Utilities
│   └── paper_implementations/  # [Shared utilities if needed]
│
├── notebooks/                   # Jupyter exploration & analysis
├── configs/                     # Configuration files (YAML)
├── tests/                       # Unit tests
│
├── datasets/                    # Data (gitignored)
├── models/                      # Checkpoints (gitignored)
├── papers/                      # Research papers (gitignored)
│
├── results/
│   ├── paper_tests/            # Results from paper implementations
│   └── main_research/          # Final research results
│
├── docs/                        # Documentation
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project config & UV
├── .gitignore                  # Git rules
└── README.md                   # This file
```

## 🔬 Research Workflow

### Phase 1: Testing Paper Implementations

1. **Create a folder for the paper:**
   ```bash
   mkdir experiments/paper_implementations/paper_name
   ```

2. **Document the paper:** Copy template from `TEMPLATE_paper_implementation.md` to create `README.md`

3. **Implement & test:**
   ```bash
   # Create your scripts
   - model.py (model definition)
   - train.py (training script)
   - evaluate.py (testing)
   - config.yaml (configuration)
   ```

4. **Run experiments:**
   ```bash
   python experiments/paper_implementations/paper_name/train.py
   ```

5. **Save results to:**
   ```
   results/paper_tests/paper_name/
   ```

### Phase 2: Validate & Move to Main Research

Once a paper implementation is:
- ✅ Achieving target performance
- ✅ Thoroughly tested
- ✅ Well documented
- ✅ Ready to integrate

Move the validated code to `src/main_research/` using the integration checklist in `src/main_research/README.md`

### Phase 3: Build Your MPhil Research

In `src/main_research/`, combine tested implementations into your final research project:
```bash
python experiments/run_main_research.py --config configs/research.yaml
```

Results saved to `results/main_research/`

## Setup

### 1. Install UV (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create Virtual Environment with UV
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
# or use pyproject.toml
uv pip install -e .
```

## Development Workflow

### Running Experiments
```bash
# Add your experiment scripts to experiments/
python experiments/train.py --config configs/model_config.yaml
```

### Working with Notebooks
```bash
jupyter lab notebooks/
```

### Running Tests
```bash
pytest tests/ -v
```

## Important Notes
- `datasets/`, `models/`, `papers/`, and `results/` are gitignored to avoid committing large files
- Add your data to `datasets/` locally
- Save model checkpoints to `models/`
- Store results and outputs in `results/`
- Use `configs/` for experiment configuration files
