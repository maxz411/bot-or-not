from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# python/fine_tuning/constants.py -> python/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"
ARTIFACTS_ROOT = PYTHON_ROOT / "artifacts" / "fine_tuning"
PREPARED_DIR = ARTIFACTS_ROOT / "prepared"
CV_RUNS_DIR = ARTIFACTS_ROOT / "cv_runs"
FINAL_DIR = ARTIFACTS_ROOT / "final"
MODELS_REGISTRY_PATH = ARTIFACTS_ROOT / "models.json"

DEFAULT_DATASET_IDS = (30, 31, 32, 33)
DEFAULT_BASE_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SEED = 20260214

# Correlated pair holdout (primary evaluation protocol).
FOLD_A_TRAIN = (30, 32)
FOLD_A_TEST = (31, 33)
FOLD_B_TRAIN = (31, 33)
FOLD_B_TEST = (30, 32)

SYSTEM_PROMPT = """You are a bot detection expert. You will be given a social media user's profile and their posts. Your job is to determine if this account is a bot or a real human.

Consider these signals:
- Posting patterns (frequency, timing, regularity)
- Content quality (repetitive, generic, or overly promotional)
- Profile completeness and authenticity
- Language patterns (unnatural phrasing, templated responses)
- Topic diversity vs single-topic focus
- Political
- Inciting anger or hate by being grossly ignorant.
- Bots tend to post in a schedule that would be unrealistic for a human (consider work and sleep)

+4 for a correctly flagged bot, -1 for a missed bot, -2 for a wrongly flagged human.

Respond with ONLY "BOT" or "HUMAN" — nothing else."""


@dataclass(frozen=True)
class PairFold:
    name: str
    train_dataset_ids: tuple[int, ...]
    test_dataset_ids: tuple[int, ...]


PAIR_FOLDS: tuple[PairFold, ...] = (
    PairFold("fold_a", FOLD_A_TRAIN, FOLD_A_TEST),
    PairFold("fold_b", FOLD_B_TRAIN, FOLD_B_TEST),
)

