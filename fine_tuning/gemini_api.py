from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from google import genai
from google.auth import default as google_auth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from google.genai import types

from .constants import (
    DEFAULT_GEMINI_HTTP_API_VERSION,
    DEFAULT_GEMINI_LOCATION,
)
from .env import load_project_env
from .serialization import to_jsonable


@dataclass(frozen=True)
class VertexContext:
    project: str
    location: str


_STATUS_MAP = {
    "JOB_STATE_QUEUED": "queued",
    "JOB_STATE_PENDING": "pending",
    "JOB_STATE_RUNNING": "running",
    "JOB_STATE_SUCCEEDED": "succeeded",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLING": "cancelling",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_PAUSED": "paused",
    "JOB_STATE_EXPIRED": "expired",
    "JOB_STATE_UPDATING": "updating",
    "JOB_STATE_PARTIALLY_SUCCEEDED": "partially_succeeded",
}

_ADAPTER_SIZE_MAP = {
    "1": types.AdapterSize.ADAPTER_SIZE_ONE,
    "2": types.AdapterSize.ADAPTER_SIZE_TWO,
    "4": types.AdapterSize.ADAPTER_SIZE_FOUR,
    "8": types.AdapterSize.ADAPTER_SIZE_EIGHT,
    "16": types.AdapterSize.ADAPTER_SIZE_SIXTEEN,
    "32": types.AdapterSize.ADAPTER_SIZE_THIRTY_TWO,
}

_TUNING_MODE_MAP = {
    "full": types.TuningMode.TUNING_MODE_FULL,
    "adapter": types.TuningMode.TUNING_MODE_PEFT_ADAPTER,
}


def resolve_vertex_context(
    *,
    project: str | None,
    location: str | None,
) -> VertexContext:
    load_project_env()
    resolved_project = (
        project
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    if not resolved_project:
        raise RuntimeError(
            "A Google Cloud project is required. Pass --project or set "
            "GOOGLE_CLOUD_PROJECT."
        )

    resolved_location = (
        location
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or DEFAULT_GEMINI_LOCATION
    )
    return VertexContext(project=resolved_project, location=resolved_location)


def _ensure_adc() -> None:
    try:
        google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "Google Application Default Credentials were not found. Run "
            "`gcloud auth application-default login` or configure "
            "GOOGLE_APPLICATION_CREDENTIALS."
        ) from exc


def get_client(
    *,
    project: str | None,
    location: str | None,
) -> tuple[genai.Client, VertexContext]:
    context = resolve_vertex_context(project=project, location=location)
    _ensure_adc()
    client = genai.Client(
        vertexai=True,
        project=context.project,
        location=context.location,
        http_options=types.HttpOptions(api_version=DEFAULT_GEMINI_HTTP_API_VERSION),
    )
    return client, context


def normalize_gcs_uri(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("A non-empty GCS bucket or URI is required.")
    normalized = raw if raw.startswith("gs://") else f"gs://{raw}"
    return normalized.rstrip("/")


def split_gcs_uri(value: str) -> tuple[str, str]:
    normalized = normalize_gcs_uri(value)
    bucket_and_path = normalized.removeprefix("gs://")
    bucket_name, _, object_path = bucket_and_path.partition("/")
    if not bucket_name:
        raise ValueError(f"Invalid GCS URI: {value}")
    return bucket_name, object_path


def _join_gcs_path(bucket_uri: str, relative_path: str) -> str:
    bucket_name, bucket_prefix = split_gcs_uri(bucket_uri)
    base_path = PurePosixPath(bucket_prefix) if bucket_prefix else PurePosixPath()
    object_path = str(base_path / relative_path)
    return f"gs://{bucket_name}/{object_path}"


def upload_file_to_gcs(
    *,
    project: str,
    local_path: Path,
    destination_uri: str,
) -> str:
    bucket_name, object_path = split_gcs_uri(destination_uri)
    if not object_path:
        raise ValueError(
            "Destination GCS URI must include an object path, not just a bucket."
        )

    storage_client = storage.Client(project=project)
    blob = storage_client.bucket(bucket_name).blob(object_path)
    blob.upload_from_filename(str(local_path))
    return destination_uri


def stage_training_files(
    *,
    project: str,
    bucket_uri: str,
    object_prefix: str,
    train_path: Path,
    validation_path: Path | None,
) -> dict[str, str | None]:
    normalized_bucket = normalize_gcs_uri(bucket_uri)
    clean_prefix = re.sub(r"/+", "/", object_prefix.strip("/"))
    if not clean_prefix:
        raise ValueError("object_prefix must not be empty.")

    training_gcs_uri = _join_gcs_path(
        normalized_bucket,
        f"{clean_prefix}/train.jsonl",
    )
    upload_file_to_gcs(
        project=project,
        local_path=train_path,
        destination_uri=training_gcs_uri,
    )

    validation_gcs_uri: str | None = None
    if validation_path is not None:
        validation_gcs_uri = _join_gcs_path(
            normalized_bucket,
            f"{clean_prefix}/val.jsonl",
        )
        upload_file_to_gcs(
            project=project,
            local_path=validation_path,
            destination_uri=validation_gcs_uri,
        )

    return {
        "bucket_uri": normalized_bucket,
        "training_gcs_uri": training_gcs_uri,
        "validation_gcs_uri": validation_gcs_uri,
    }


def _parse_tuning_mode(value: str | None) -> types.TuningMode | None:
    if value is None:
        return None
    try:
        return _TUNING_MODE_MAP[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported tuning mode: {value}") from exc


def _parse_adapter_size(value: str | None) -> types.AdapterSize | None:
    if value is None:
        return None
    try:
        return _ADAPTER_SIZE_MAP[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported adapter size: {value}") from exc


def create_tuning_job(
    client: genai.Client,
    *,
    base_model: str,
    training_gcs_uri: str,
    validation_gcs_uri: str | None,
    tuned_model_display_name: str,
    description: str | None,
    epoch_count: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    learning_rate_multiplier: float | None,
    tuning_mode: str | None,
    adapter_size: str | None,
    labels: dict[str, str] | None,
) -> Any:
    config_kwargs: dict[str, Any] = {
        "method": types.TuningMethod.SUPERVISED_FINE_TUNING,
        "tuned_model_display_name": tuned_model_display_name,
    }
    if description:
        config_kwargs["description"] = description
    if epoch_count is not None:
        config_kwargs["epoch_count"] = epoch_count
    if batch_size is not None:
        config_kwargs["batch_size"] = batch_size
    if learning_rate is not None:
        config_kwargs["learning_rate"] = learning_rate
    if learning_rate_multiplier is not None:
        config_kwargs["learning_rate_multiplier"] = learning_rate_multiplier

    parsed_tuning_mode = _parse_tuning_mode(tuning_mode)
    if parsed_tuning_mode is not None:
        config_kwargs["tuning_mode"] = parsed_tuning_mode

    parsed_adapter_size = _parse_adapter_size(adapter_size)
    if parsed_adapter_size is not None:
        config_kwargs["adapter_size"] = parsed_adapter_size

    if labels:
        config_kwargs["labels"] = labels
    if validation_gcs_uri is not None:
        config_kwargs["validation_dataset"] = types.TuningValidationDataset(
            gcs_uri=validation_gcs_uri,
        )

    return client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(gcs_uri=training_gcs_uri),
        config=types.CreateTuningJobConfig(**config_kwargs),
    )


def retrieve_tuning_job(client: genai.Client, *, job_name: str) -> Any:
    return client.tunings.get(name=job_name)


def list_tuning_jobs(
    client: genai.Client,
    *,
    filter_text: str | None = None,
    page_size: int = 20,
) -> list[Any]:
    config_kwargs: dict[str, Any] = {"page_size": page_size}
    if filter_text:
        config_kwargs["filter"] = filter_text
    pager = client.tunings.list(config=types.ListTuningJobsConfig(**config_kwargs))
    return list(pager)


def cancel_tuning_job(client: genai.Client, *, job_name: str) -> Any:
    return client.tunings.cancel(name=job_name)


def wait_for_tuning_job(
    client: genai.Client,
    *,
    job_name: str,
    poll_seconds: float,
    max_wait_minutes: float,
) -> Any:
    started_at = time.monotonic()

    while True:
        job = retrieve_tuning_job(client, job_name=job_name)
        if getattr(job, "has_ended", False):
            return job

        if max_wait_minutes > 0:
            elapsed_minutes = (time.monotonic() - started_at) / 60.0
            if elapsed_minutes >= max_wait_minutes:
                raise TimeoutError(
                    f"Timed out waiting for Gemini tuning job {job_name} after "
                    f"{max_wait_minutes} minutes."
                )

        time.sleep(poll_seconds)


def status_name(job: Any) -> str:
    raw_state = getattr(job, "state", None)
    if raw_state is None:
        return "unknown"

    raw_name = getattr(raw_state, "name", None)
    if raw_name is None:
        raw_name = str(raw_state).split(".")[-1]
    return _STATUS_MAP.get(raw_name, raw_name.lower())


def tuned_model_name(job: Any) -> str | None:
    tuned_model = getattr(job, "tuned_model", None)
    return getattr(tuned_model, "model", None)


def tuned_model_endpoint(job: Any) -> str | None:
    tuned_model = getattr(job, "tuned_model", None)
    return getattr(tuned_model, "endpoint", None)


def job_to_record_updates(job: Any) -> dict[str, Any]:
    return {
        "job_id": getattr(job, "name", None),
        "submitted_at": to_jsonable(getattr(job, "create_time", None)),
        "status": status_name(job),
        "raw_status": getattr(getattr(job, "state", None), "name", None),
        "base_model": getattr(job, "base_model", None),
        "fine_tuned_model": tuned_model_name(job),
        "tuned_model_display_name": getattr(job, "tuned_model_display_name", None),
        "tuned_model_endpoint": tuned_model_endpoint(job),
        "job": to_jsonable(job),
        "events": [],
    }
