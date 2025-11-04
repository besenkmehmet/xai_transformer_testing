# Transformer Based XAI Adversarial Testing - Core Pipeline

## Prerequisites

This project requires [UV](https://docs.astral.sh/uv/) to run.

Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Create a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:
```bash
uv pip install -e .
```

## Usage

### Step 1: Setup Foundation

```bash
python foundation_setup.py --tasks sst2 --models distilbert-base-uncased
```

### Step 2: Run Experiment

```bash
python run_experiment.py \
  --task sst2 \
  --model distilbert-base-uncased \
  --explainer ig \
  --strategy lm \
  --threshold 0.9
```

## Options

### foundation_setup.py
- `--tasks`: Task name(s) (e.g., sst2, qnli)
- `--models`: Model name(s) (e.g., distilbert-base-uncased)
- `--mining-split-ratio`: Mining split ratio (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--max-items`: Max items for debugging
- `--force`: Force re-creation

### run_experiment.py
- `--task`: Task name
- `--model`: Model name
- `--explainer`: Explainer type (ig, attn)
- `--strategy`: Placement strategy (lm, random, prefix)
- `--threshold`: NLI threshold
- `--steps`: Run specific steps (mine, probe, aft)
- `--max-probe-items`: Max items to probe
- `--candidates-per-item`: Candidates per item (default: 3)
- `--batch-size`: Batch size (default: 64)
