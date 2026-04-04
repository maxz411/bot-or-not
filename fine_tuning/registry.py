from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .constants import JOBS_DIR, REGISTRY_PATH
from .storage import load_json, now_iso, save_json


def _empty_registry() -> dict[str, Any]:
    return {
        "updated_at": now_iso(),
        "latest_job_id": None,
        "latest_job_ids": {},
        "latest_model": None,
        "latest_models": {},
        "jobs": [],
    }


def _load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        return load_json(REGISTRY_PATH)
    return _empty_registry()


def _save_registry(registry: dict[str, Any]) -> None:
    registry["updated_at"] = now_iso()
    save_json(REGISTRY_PATH, registry)


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if not normalized:
        raise ValueError("provider is required.")
    return normalized


def _snapshot_filename(provider: str, job_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", job_id).strip("._-") or "job"
    digest = hashlib.sha256(f"{provider}:{job_id}".encode("utf-8")).hexdigest()[:12]
    return f"{slug[:80]}-{digest}.json"


def _snapshot_path(provider: str, job_id: str) -> Path:
    normalized_provider = _normalize_provider(provider)
    return JOBS_DIR / normalized_provider / _snapshot_filename(normalized_provider, job_id)


def _job_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": record["provider"],
        "job_id": record["job_id"],
        "status": record.get("status"),
        "method": record.get("method"),
        "base_model": record.get("base_model"),
        "split": record.get("split"),
        "tag": record.get("tag"),
        "submitted_at": record.get("submitted_at"),
        "last_synced_at": record.get("last_synced_at"),
        "fine_tuned_model": record.get("fine_tuned_model"),
        "snapshot_path": record["snapshot_path"],
        "project": record.get("project"),
        "location": record.get("location"),
        "tuned_model_endpoint": record.get("tuned_model_endpoint"),
    }


def _matches_provider(summary: dict[str, Any], provider: str | None) -> bool:
    if provider is None:
        return True
    return summary.get("provider") == _normalize_provider(provider)


def _persist_record(record: dict[str, Any]) -> dict[str, Any]:
    provider = _normalize_provider(str(record["provider"]))
    job_id_raw = record.get("job_id")
    if not job_id_raw:
        raise RuntimeError("Cannot persist a fine-tuning job record without a job_id.")
    job_id = str(job_id_raw)

    snapshot_path = _snapshot_path(provider, job_id)
    record["provider"] = provider
    record["snapshot_path"] = str(snapshot_path)

    save_json(snapshot_path, record)

    registry = _load_registry()
    summary = _job_summary(record)
    jobs = registry.setdefault("jobs", [])

    for index, existing in enumerate(jobs):
        if (
            existing.get("provider") == provider
            and existing.get("job_id") == job_id
        ):
            jobs[index] = summary
            break
    else:
        jobs.append(summary)

    jobs.sort(
        key=lambda item: (
            item.get("submitted_at") or "",
            item.get("job_id") or "",
        ),
        reverse=True,
    )

    registry["latest_job_id"] = job_id
    registry.setdefault("latest_job_ids", {})[provider] = job_id

    fine_tuned_model = record.get("fine_tuned_model")
    if fine_tuned_model:
        registry["latest_model"] = fine_tuned_model
        registry.setdefault("latest_models", {})[provider] = fine_tuned_model

    _save_registry(registry)
    return record


def register_job_record(record: dict[str, Any]) -> dict[str, Any]:
    persisted = dict(record)
    timestamp = now_iso()
    persisted.setdefault("submitted_at", timestamp)
    persisted.setdefault("last_synced_at", timestamp)
    persisted.setdefault("events", [])
    return _persist_record(persisted)


def upsert_job_record(
    existing_record: dict[str, Any] | None,
    *,
    updates: dict[str, Any],
) -> dict[str, Any]:
    record = dict(existing_record or {})
    record.update(updates)
    record.setdefault("submitted_at", now_iso())
    record["last_synced_at"] = now_iso()
    record.setdefault("events", [])
    return _persist_record(record)


def get_latest_job_id(provider: str | None = None) -> str | None:
    registry = _load_registry()
    if provider is None:
        return registry.get("latest_job_id")
    latest_job_ids = registry.get("latest_job_ids", {})
    return latest_job_ids.get(_normalize_provider(provider))


def list_registered_jobs(provider: str | None = None) -> list[dict[str, Any]]:
    jobs = _load_registry().get("jobs", [])
    return [job for job in jobs if _matches_provider(job, provider)]


def list_model_records(provider: str | None = None) -> list[dict[str, Any]]:
    return [
        job
        for job in list_registered_jobs(provider=provider)
        if job.get("fine_tuned_model")
    ]


def get_job_record(
    job_id: str,
    *,
    provider: str | None = None,
) -> dict[str, Any] | None:
    summaries = [
        summary
        for summary in list_registered_jobs(provider=provider)
        if summary.get("job_id") == job_id
    ]
    if summaries:
        snapshot_path = Path(summaries[0]["snapshot_path"])
        if snapshot_path.exists():
            return load_json(snapshot_path)

    if provider is None:
        return None

    snapshot_path = _snapshot_path(provider, job_id)
    if not snapshot_path.exists():
        return None
    return load_json(snapshot_path)
