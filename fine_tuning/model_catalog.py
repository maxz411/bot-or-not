from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .gemini_api import (
    get_client as get_gemini_client,
    list_tuning_jobs,
    status_name as gemini_status_name,
)
from .openai_api import get_client as get_openai_client
from .registry import list_model_records

ModelProvider = Literal["openai", "gemini"]
ModelKind = Literal["final", "latest_checkpoint"]


@dataclass(frozen=True)
class RunnableModel:
    provider: ModelProvider
    runnable_id: str
    kind: ModelKind
    title: str
    lineage_id: str | None
    base_model: str | None
    source: str
    project: str | None = None
    location: str | None = None
    checkpoint_id: str | None = None
    checkpoint_epoch: int | None = None
    checkpoint_step: int | None = None


@dataclass(frozen=True)
class RunnableModelCatalog:
    models: tuple[RunnableModel, ...]
    openai_error: str | None = None
    gemini_error: str | None = None


def _dedupe_models(models: list[RunnableModel]) -> tuple[RunnableModel, ...]:
    def merge(existing: RunnableModel, incoming: RunnableModel) -> RunnableModel:
        preferred = existing
        alternate = incoming
        if existing.source != "live" and incoming.source == "live":
            preferred = incoming
            alternate = existing
        elif existing.source == incoming.source:
            if existing.kind != "final" and incoming.kind == "final":
                preferred = incoming
                alternate = existing

        has_final = existing.kind == "final" or incoming.kind == "final"
        merged_kind: ModelKind = "final" if has_final else "latest_checkpoint"
        checkpoint_id = preferred.checkpoint_id or alternate.checkpoint_id
        checkpoint_epoch = preferred.checkpoint_epoch or alternate.checkpoint_epoch
        checkpoint_step = preferred.checkpoint_step or alternate.checkpoint_step
        merged_title = preferred.title

        has_latest_checkpoint_details = (
            existing.kind == "latest_checkpoint"
            or incoming.kind == "latest_checkpoint"
            or checkpoint_id is not None
            or checkpoint_epoch is not None
            or checkpoint_step is not None
        )
        if has_final and has_latest_checkpoint_details:
            detail_parts = []
            if checkpoint_id is not None:
                detail_parts.append(f"checkpoint_id={checkpoint_id}")
            if checkpoint_epoch is not None:
                detail_parts.append(f"epoch={checkpoint_epoch}")
            if checkpoint_step is not None:
                detail_parts.append(f"step={checkpoint_step}")
            detail_suffix = f" {' '.join(detail_parts)}" if detail_parts else ""
            merged_title = (
                f"[{preferred.provider} final/latest checkpoint] "
                f"{preferred.runnable_id} "
                f"(job={preferred.lineage_id or alternate.lineage_id or '-'} "
                f"base={preferred.base_model or alternate.base_model or '-'}"
                f"{detail_suffix}"
            )
            if preferred.project or alternate.project:
                merged_title += (
                    f" project={preferred.project or alternate.project or '-'}"
                )
            if preferred.location or alternate.location:
                merged_title += (
                    f" location={preferred.location or alternate.location or '-'}"
                )
            merged_title += ")"

        return RunnableModel(
            provider=preferred.provider,
            runnable_id=preferred.runnable_id,
            kind=merged_kind,
            title=merged_title,
            lineage_id=preferred.lineage_id or alternate.lineage_id,
            base_model=preferred.base_model or alternate.base_model,
            source=preferred.source,
            project=preferred.project or alternate.project,
            location=preferred.location or alternate.location,
            checkpoint_id=checkpoint_id,
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_step=checkpoint_step,
        )

    deduped: dict[tuple[str, str], RunnableModel] = {}
    for model in models:
        key = (model.provider, model.runnable_id)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = model
            continue
        deduped[key] = merge(existing, model)

    return tuple(
        sorted(
            deduped.values(),
            key=lambda model: (
                model.provider,
                model.base_model or "",
                model.lineage_id or "",
                0 if model.kind == "final" else 1,
                -(model.checkpoint_step or -1),
                model.runnable_id,
            ),
        )
    )


def _iter_openai_jobs(client: Any, *, page_limit: int = 100) -> list[Any]:
    jobs: list[Any] = []
    after: str | None = None

    while True:
        page = client.fine_tuning.jobs.list(limit=page_limit, after=after)
        page_items = list(getattr(page, "data", []))
        if not page_items:
            break
        jobs.extend(page_items)
        if len(page_items) < page_limit:
            break
        last_job_id = getattr(page_items[-1], "id", None)
        if not isinstance(last_job_id, str) or not last_job_id:
            break
        after = last_job_id

    return jobs


def _iter_openai_checkpoints(
    client: Any,
    *,
    fine_tuning_job_id: str,
    page_limit: int = 100,
) -> list[Any]:
    checkpoints: list[Any] = []
    after: str | None = None

    while True:
        page = client.fine_tuning.jobs.checkpoints.list(
            fine_tuning_job_id=fine_tuning_job_id,
            limit=page_limit,
            after=after,
        )
        page_items = list(getattr(page, "data", []))
        if not page_items:
            break
        checkpoints.extend(page_items)
        if len(page_items) < page_limit:
            break
        last_checkpoint_id = getattr(page_items[-1], "id", None)
        if not isinstance(last_checkpoint_id, str) or not last_checkpoint_id:
            break
        after = last_checkpoint_id

    return checkpoints


def _local_openai_models() -> list[RunnableModel]:
    local_models = list_model_records(provider="openai")
    discovered: list[RunnableModel] = []
    for model in local_models:
        model_id = model.get("fine_tuned_model")
        if not isinstance(model_id, str) or not model_id:
            continue
        discovered.append(
            RunnableModel(
                provider="openai",
                runnable_id=model_id,
                kind="final",
                title=(
                    f"[openai final] {model_id} "
                    f"(local job={model.get('job_id') or '-'} "
                    f"base={model.get('base_model') or '-'} "
                    f"tag={model.get('tag') or '-'} "
                    f"split={model.get('split') or '-'})"
                ),
                lineage_id=str(model.get("job_id")) if model.get("job_id") else None,
                base_model=(
                    str(model.get("base_model"))
                    if model.get("base_model") is not None
                    else None
                ),
                source="local",
            )
        )
    return discovered


def _live_openai_models(*, api_key: str | None) -> list[RunnableModel]:
    client = get_openai_client(api_key=api_key)
    discovered: list[RunnableModel] = []

    for job in _iter_openai_jobs(client):
        job_id = getattr(job, "id", None)
        if not isinstance(job_id, str) or not job_id:
            continue
        base_model = getattr(job, "model", None)
        base_model_text = str(base_model) if isinstance(base_model, str) else None
        fine_tuned_model = getattr(job, "fine_tuned_model", None)
        if isinstance(fine_tuned_model, str) and fine_tuned_model:
            discovered.append(
                RunnableModel(
                    provider="openai",
                    runnable_id=fine_tuned_model,
                    kind="final",
                    title=(
                        f"[openai final] {fine_tuned_model} "
                        f"(job={job_id} base={base_model_text or '-'} "
                        f"status={getattr(job, 'status', '-') or '-'})"
                    ),
                    lineage_id=job_id,
                    base_model=base_model_text,
                    source="live",
                )
            )

        checkpoints = _iter_openai_checkpoints(
            client,
            fine_tuning_job_id=job_id,
        )
        if not checkpoints:
            continue

        latest_checkpoint = max(
            checkpoints,
            key=lambda checkpoint: (
                int(getattr(checkpoint, "step_number", 0) or 0),
                int(getattr(checkpoint, "created_at", 0) or 0),
            ),
        )
        checkpoint_model_id = getattr(
            latest_checkpoint,
            "fine_tuned_model_checkpoint",
            None,
        )
        if not isinstance(checkpoint_model_id, str) or not checkpoint_model_id:
            continue

        discovered.append(
            RunnableModel(
                provider="openai",
                runnable_id=checkpoint_model_id,
                kind="latest_checkpoint",
                title=(
                    f"[openai latest checkpoint] {checkpoint_model_id} "
                    f"(job={job_id} base={base_model_text or '-'} "
                    f"step={getattr(latest_checkpoint, 'step_number', '-')})"
                ),
                lineage_id=job_id,
                base_model=base_model_text,
                source="live",
                checkpoint_step=(
                    int(getattr(latest_checkpoint, "step_number", 0) or 0) or None
                ),
            )
        )

    return discovered


def _local_gemini_models() -> list[RunnableModel]:
    local_models = list_model_records(provider="gemini")
    discovered: list[RunnableModel] = []
    for model in local_models:
        runnable_id = model.get("tuned_model_endpoint") or model.get("fine_tuned_model")
        if not isinstance(runnable_id, str) or not runnable_id:
            continue
        discovered.append(
            RunnableModel(
                provider="gemini",
                runnable_id=runnable_id,
                kind="final",
                title=(
                    f"[gemini final] {runnable_id} "
                    f"(local job={model.get('job_id') or '-'} "
                    f"base={model.get('base_model') or '-'} "
                    f"project={model.get('project') or '-'} "
                    f"location={model.get('location') or '-'} "
                    f"tag={model.get('tag') or '-'} "
                    f"split={model.get('split') or '-'})"
                ),
                lineage_id=str(model.get("job_id")) if model.get("job_id") else None,
                base_model=(
                    str(model.get("base_model"))
                    if model.get("base_model") is not None
                    else None
                ),
                source="local",
                project=str(model.get("project")) if model.get("project") else None,
                location=(
                    str(model.get("location")) if model.get("location") else None
                ),
            )
        )
    return discovered


def _live_gemini_models(
    *,
    project: str | None,
    location: str | None,
) -> list[RunnableModel]:
    client, context = get_gemini_client(project=project, location=location)
    discovered: list[RunnableModel] = []

    for job in list_tuning_jobs(client, page_size=100):
        job_id = getattr(job, "name", None)
        if not isinstance(job_id, str) or not job_id:
            continue

        tuned_model = getattr(job, "tuned_model", None)
        if tuned_model is None:
            continue

        base_model = getattr(job, "base_model", None)
        base_model_text = str(base_model) if isinstance(base_model, str) else None
        status = gemini_status_name(job)

        final_runnable_id = getattr(tuned_model, "endpoint", None) or getattr(
            tuned_model,
            "model",
            None,
        )
        if (
            status == "succeeded"
            and isinstance(final_runnable_id, str)
            and final_runnable_id
        ):
            discovered.append(
                RunnableModel(
                    provider="gemini",
                    runnable_id=final_runnable_id,
                    kind="final",
                    title=(
                        f"[gemini final] {final_runnable_id} "
                        f"(job={job_id} base={base_model_text or '-'} "
                        f"project={context.project} location={context.location})"
                    ),
                    lineage_id=job_id,
                    base_model=base_model_text,
                    source="live",
                    project=context.project,
                    location=context.location,
                )
            )

        checkpoints = list(getattr(tuned_model, "checkpoints", None) or [])
        runnable_checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if isinstance(getattr(checkpoint, "endpoint", None), str)
            and getattr(checkpoint, "endpoint")
        ]
        if not runnable_checkpoints:
            continue

        latest_checkpoint = max(
            runnable_checkpoints,
            key=lambda checkpoint: (
                int(getattr(checkpoint, "step", 0) or 0),
                int(getattr(checkpoint, "epoch", 0) or 0),
            ),
        )
        checkpoint_endpoint = getattr(latest_checkpoint, "endpoint", None)
        if not isinstance(checkpoint_endpoint, str) or not checkpoint_endpoint:
            continue
        discovered.append(
            RunnableModel(
                provider="gemini",
                runnable_id=checkpoint_endpoint,
                kind="latest_checkpoint",
                title=(
                    f"[gemini latest checkpoint] {checkpoint_endpoint} "
                    f"(job={job_id} base={base_model_text or '-'} "
                    f"epoch={getattr(latest_checkpoint, 'epoch', '-')} "
                    f"step={getattr(latest_checkpoint, 'step', '-')} "
                    f"project={context.project} location={context.location})"
                ),
                lineage_id=job_id,
                base_model=base_model_text,
                source="live",
                project=context.project,
                location=context.location,
                checkpoint_id=(
                    str(getattr(latest_checkpoint, "checkpoint_id"))
                    if getattr(latest_checkpoint, "checkpoint_id", None)
                    else None
                ),
                checkpoint_epoch=(
                    int(getattr(latest_checkpoint, "epoch", 0) or 0) or None
                ),
                checkpoint_step=(
                    int(getattr(latest_checkpoint, "step", 0) or 0) or None
                ),
            )
        )

    return discovered


def list_runnable_models(
    *,
    openai_api_key: str | None,
    gemini_project: str | None,
    gemini_location: str | None,
) -> RunnableModelCatalog:
    discovered: list[RunnableModel] = []
    openai_error: str | None = None
    gemini_error: str | None = None

    discovered.extend(_local_openai_models())
    discovered.extend(_local_gemini_models())

    try:
        discovered.extend(_live_openai_models(api_key=openai_api_key))
    except Exception as exc:  # noqa: BLE001
        openai_error = str(exc)

    try:
        discovered.extend(
            _live_gemini_models(
                project=gemini_project,
                location=gemini_location,
            )
        )
    except Exception as exc:  # noqa: BLE001
        gemini_error = str(exc)

    return RunnableModelCatalog(
        models=_dedupe_models(discovered),
        openai_error=openai_error,
        gemini_error=gemini_error,
    )
