"""Fine-tuning pipeline for bot-or-not v5."""

from .cli import eval_model, prepare_data, run_cv, train_final

__all__ = ["prepare_data", "run_cv", "train_final", "eval_model"]

from .cli import (
    eval_model,
    evaluate_records,
    prepare_data,
    print_metrics,
    run_cv,
    train_final,
)

__all__ = [
    "eval_model",
    "evaluate_records",
    "prepare_data",
    "print_metrics",
    "run_cv",
    "train_final",
]
