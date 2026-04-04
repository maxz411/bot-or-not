from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .env import load_project_env
from .serialization import to_jsonable


def get_client(api_key: str | None = None) -> OpenAI:
    load_project_env()
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return OpenAI(api_key=resolved_api_key)


def upload_file(client: OpenAI, path: Path, *, purpose: str = "fine-tune") -> str:
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose=purpose)
    return uploaded.id


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != {}
    }


def build_supervised_hyperparameters(
    *,
    n_epochs: int | str | None,
    batch_size: int | str | None,
    learning_rate_multiplier: float | str | None,
) -> dict[str, int | float | str]:
    hyperparameters: dict[str, int | float | str] = {}
    if n_epochs is not None:
        hyperparameters["n_epochs"] = n_epochs
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier
    return hyperparameters


def build_reinforcement_hyperparameters(
    *,
    n_epochs: int | str | None,
    batch_size: int | str | None,
    learning_rate_multiplier: float | str | None,
    reasoning_effort: str | None,
    compute_multiplier: float | None,
    eval_interval: int | None,
    eval_samples: int | None,
) -> dict[str, int | float | str]:
    hyperparameters = build_supervised_hyperparameters(
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
    )
    if reasoning_effort is not None:
        hyperparameters["reasoning_effort"] = reasoning_effort
    if compute_multiplier is not None:
        hyperparameters["compute_multiplier"] = compute_multiplier
    if eval_interval is not None:
        hyperparameters["eval_interval"] = eval_interval
    if eval_samples is not None:
        hyperparameters["eval_samples"] = eval_samples
    return hyperparameters


def build_supervised_fine_tuning_request(
    *,
    model: str,
    training_file_id: str,
    validation_file_id: str | None,
    n_epochs: int | str | None,
    batch_size: int | str | None,
    learning_rate_multiplier: float | str | None,
    metadata: dict[str, str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    hyperparameters = build_supervised_hyperparameters(
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
    )
    return _drop_none_values(
        {
            "model": model,
            "training_file": training_file_id,
            "validation_file": validation_file_id,
            "metadata": metadata,
            "seed": seed,
            "method": {"type": "supervised"},
            "hyperparameters": hyperparameters or None,
        }
    )


def build_reinforcement_fine_tuning_request(
    *,
    model: str,
    training_file_id: str,
    validation_file_id: str | None,
    grader: dict[str, Any],
    n_epochs: int | str | None,
    batch_size: int | str | None,
    learning_rate_multiplier: float | str | None,
    reasoning_effort: str | None,
    compute_multiplier: float | None,
    eval_interval: int | None,
    eval_samples: int | None,
    metadata: dict[str, str] | None = None,
    seed: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hyperparameters = build_reinforcement_hyperparameters(
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
        reasoning_effort=reasoning_effort,
        compute_multiplier=compute_multiplier,
        eval_interval=eval_interval,
        eval_samples=eval_samples,
    )
    return _drop_none_values(
        {
            "model": model,
            "training_file": training_file_id,
            "validation_file": validation_file_id,
            "metadata": metadata,
            "seed": seed,
            "method": {
                "type": "reinforcement",
                "reinforcement": _drop_none_values(
                    {
                        "grader": grader,
                        "hyperparameters": hyperparameters or None,
                        "response_format": response_format,
                    }
                ),
            },
        }
    )


def create_fine_tuning_job(
    client: OpenAI,
    *,
    request: dict[str, Any],
) -> Any:
    return client.fine_tuning.jobs.create(**request)


def validate_grader(
    client: OpenAI,
    *,
    grader: dict[str, Any],
) -> dict[str, Any]:
    response = client.fine_tuning.alpha.graders.validate(grader=grader)
    return to_jsonable(response)


def retrieve_fine_tuning_job(client: OpenAI, *, job_id: str) -> Any:
    return client.fine_tuning.jobs.retrieve(job_id)


def list_fine_tuning_job_events(
    client: OpenAI,
    *,
    job_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    page = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
    return [to_jsonable(event) for event in getattr(page, "data", [])]


def wait_for_fine_tuning_job(
    client: OpenAI,
    *,
    job_id: str,
    poll_seconds: float,
    max_wait_minutes: float,
) -> Any:
    started_at = time.monotonic()

    while True:
        job = retrieve_fine_tuning_job(client, job_id=job_id)
        status = getattr(job, "status", "")
        if status in {"succeeded", "failed", "cancelled"}:
            return job

        if max_wait_minutes > 0:
            elapsed_minutes = (time.monotonic() - started_at) / 60.0
            if elapsed_minutes >= max_wait_minutes:
                raise TimeoutError(
                    f"Timed out waiting for job {job_id} after "
                    f"{max_wait_minutes} minutes."
                )

        time.sleep(poll_seconds)
