# bot-or-not

Our approach: fine-tune OpenAI models (GPT-4.1 Mini) on labeled bot/human datasets using both supervised and reinforcement fine-tuning, then run inference on unlabeled competition datasets to produce final bot detection submissions.

## Repo structure

```
datasets/
  prev/          # Earlier rounds (datasets 30-33)
  current/       # Later rounds (datasets 1-6)
  final/         # Unlabeled competition datasets for submission
fine_tuning/     # Python package — data prep, training, evaluation, inference
artifacts/       # Generated training JSONL (gitignored)
runs/            # Inference run logs
raw/             # Raw inference outputs
final_results/   # Submission files (one per language)
```

## Setup

```bash
uv sync
```

## Usage

```bash
uv run fine-tuning --help
```

### 1. Prepare training data

```bash
uv run fine-tuning openai prepare-data
```

### 2. Submit a fine-tuning job

```bash
uv run fine-tuning openai train \
  --method supervised \
  --split both \
  --base-model gpt-4.1-mini-2025-04-14
```

### 3. Run inference and generate submissions

```bash
uv run fine-tuning openai run --help
```
