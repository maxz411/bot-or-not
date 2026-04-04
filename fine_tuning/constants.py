from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
PREV_DATASETS_DIR = DATASETS_DIR / "prev"
CURRENT_DATASETS_DIR = DATASETS_DIR / "current"
FINAL_DATASETS_DIR = DATASETS_DIR / "final"

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "fine_tuning"
PREPARED_DIR = ARTIFACTS_ROOT / "prepared"
EVALS_DIR = ARTIFACTS_ROOT / "evals"
RAW_DIR = PROJECT_ROOT / "raw"
RUNS_DIR = PROJECT_ROOT / "runs"
FINAL_RESULTS_DIR = PROJECT_ROOT / "final_results"
OPENAI_PREPARED_DIR = PREPARED_DIR / "openai"
GEMINI_PREPARED_DIR = PREPARED_DIR / "gemini"
JOBS_DIR = ARTIFACTS_ROOT / "jobs"
REGISTRY_PATH = ARTIFACTS_ROOT / "registry.json"

OPENAI_SUPERVISED_PREPARED_DIR = OPENAI_PREPARED_DIR / "supervised"
OPENAI_REINFORCEMENT_PREPARED_DIR = OPENAI_PREPARED_DIR / "reinforcement"
OPENAI_TRAINING_METHODS = ("supervised", "reinforcement")
OPENAI_PREPARE_METHODS = (*OPENAI_TRAINING_METHODS, "all")

FOLD_DATASET_IDS = {
    "fold_a": (30, 32, 1, 3, 5),
    "fold_b": (31, 33, 2, 4, 6),
}

DATASET_COLLECTIONS = ("prev", "current", "fold_a", "fold_b", "both")
PREPARED_SPLITS = ("prev", "current", "fold_a", "fold_b", "both", "full")

DEFAULT_PREPARED_SPLIT = "both"
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SEED = 20260214
DEFAULT_OPENAI_SUPERVISED_EPOCHS = 1
DEFAULT_OPENAI_SUPERVISED_BATCH_SIZE = 8
DEFAULT_OPENAI_SUPERVISED_LEARNING_RATE_MULTIPLIER = 1.0
DEFAULT_OPENAI_RFT_REASONING_EFFORT = "medium"
DEFAULT_OPENAI_RFT_COMPUTE_MULTIPLIER = 1.0
DEFAULT_OPENAI_RFT_GRADER_IMAGE_TAG = "2025-05-08"

OPENAI_RFT_TRUE_POSITIVE_REWARD = 2
OPENAI_RFT_FALSE_NEGATIVE_REWARD = -2
OPENAI_RFT_FALSE_POSITIVE_REWARD = -6
OPENAI_RFT_TRUE_NEGATIVE_REWARD = 0

DEFAULT_GEMINI_LOCATION = "us-central1"
DEFAULT_GEMINI_HTTP_API_VERSION = "v1beta1"
DEFAULT_GEMINI_GCS_PREFIX = "bot-or-not"
DEFAULT_TEAM_NAME = "maxilillian"

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

Respond with ONLY "BOT" or "HUMAN" - nothing else."""
