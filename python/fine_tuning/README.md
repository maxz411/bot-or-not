# v5 Fine-Tuning CLI

Run from `/Users/max/code/bot-or-not/python`.

## Prepare data

```bash
python3 -m fine_tuning prepare-data
```

This writes strict pair-aware splits and JSONL artifacts under:

- `/Users/max/code/bot-or-not/python/artifacts/fine_tuning/prepared`

## Run pair-holdout CV tuning

```bash
python3 -m fine_tuning run-cv --base-model gpt-4.1-mini-2025-04-14
```

Dry run:

```bash
python3 -m fine_tuning run-cv --dry-run
```

## Train final model

```bash
python3 -m fine_tuning train-final --base-model gpt-4.1-mini-2025-04-14
```

Dry run:

```bash
python3 -m fine_tuning train-final --dry-run
```

## Evaluate model and write run file

```bash
python3 -m fine_tuning eval-model --model ft:YOUR_MODEL_ID --write-run-file --run-tag v5
```

## Runtime integration env var

Set `OPENAI_FT_MODEL_V5` for the JS detector runtime. Example:

```bash
export OPENAI_FT_MODEL_V5=ft:gpt-4.1-mini-2025-04-14:org:bot-or-not-v5
```

