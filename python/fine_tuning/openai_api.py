from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def get_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=resolved_api_key)


def upload_file(client: OpenAI, path: Path, purpose: str = "fine-tune") -> str:
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose=purpose)
    return uploaded.id


def create_fine_tuning_job(
    client: OpenAI,
    *,
    model: str,
    training_file_id: str,
    validation_file_id: str,
    n_epochs: int | str | None = "auto",
    batch_size: int | str | None = None,
    learning_rate_multiplier: float | str | None = None,
) -> Any:
    hyperparameters: dict[str, int | float | str] = {}
    if n_epochs is not None:
        hyperparameters["n_epochs"] = n_epochs
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

    request: dict[str, Any] = {
        "model": model,
        "training_file": training_file_id,
        "validation_file": validation_file_id,
    }
    if hyperparameters:
        request["hyperparameters"] = hyperparameters

    return client.fine_tuning.jobs.create(
        **request,
    )


def wait_for_fine_tuning_job(
    client: OpenAI,
    *,
    job_id: str,
    poll_seconds: float = 30.0,
    max_wait_minutes: float = 0.0,
) -> Any:
    started = time.monotonic()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = getattr(job, "status", "")
        if status in {"succeeded", "failed", "cancelled"}:
            return job

        if max_wait_minutes > 0:
            elapsed_minutes = (time.monotonic() - started) / 60.0
            if elapsed_minutes >= max_wait_minutes:
                raise TimeoutError(
                    f"Timed out waiting for job {job_id} after {max_wait_minutes} minutes"
                )
        time.sleep(poll_seconds)


def normalize_label(raw_text: str) -> str | None:
    text = raw_text.strip().upper()
    if not text:
        return None
    if text.startswith("BOT"):
        return "BOT"
    if text.startswith("HUMAN"):
        return "HUMAN"
    has_bot = "BOT" in text
    has_human = "HUMAN" in text
    if has_bot and not has_human:
        return "BOT"
    if has_human and not has_bot:
        return "HUMAN"
    return None


def classify_messages(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=8,
    )
    raw = response.choices[0].message.content or ""
    normalized = normalize_label(raw) or "HUMAN"
    return normalized, raw
