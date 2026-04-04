from __future__ import annotations

import argparse
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google.genai import types as gemini_types

from .constants import (
    DATASET_COLLECTIONS,
    DEFAULT_TEAM_NAME,
    DEFAULT_GEMINI_GCS_PREFIX,
    DEFAULT_GEMINI_LOCATION,
    DEFAULT_OPENAI_RFT_COMPUTE_MULTIPLIER,
    DEFAULT_OPENAI_RFT_REASONING_EFFORT,
    DEFAULT_OPENAI_SUPERVISED_BATCH_SIZE,
    DEFAULT_OPENAI_SUPERVISED_EPOCHS,
    DEFAULT_OPENAI_SUPERVISED_LEARNING_RATE_MULTIPLIER,
    DEFAULT_PREPARED_SPLIT,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    GEMINI_PREPARED_DIR,
    FINAL_RESULTS_DIR,
    OPENAI_PREPARED_DIR,
    OPENAI_PREPARE_METHODS,
    OPENAI_REINFORCEMENT_PREPARED_DIR,
    OPENAI_SUPERVISED_PREPARED_DIR,
    OPENAI_TRAINING_METHODS,
    PREPARED_SPLITS,
    RAW_DIR,
    RUNS_DIR,
    SYSTEM_PROMPT,
)
from .data import load_examples, load_final_examples, prepare_data, resolve_training_files
from .dataset_inventory import (
    discover_dataset_assets,
    discover_final_dataset_assets,
    group_assets_by_collection,
)
from .evaluation import (
    RawPrediction,
    RunningEvaluation,
    add_prediction_to_running_evaluation,
    build_raw_results_path,
    build_run_report_path,
    build_run_report_path_for_raw,
    build_running_evaluation_from_raw_results,
    initialize_raw_evaluation_artifact,
    load_and_score_raw_results,
    load_raw_results,
    save_raw_results,
    save_run_report,
    write_submission_files,
)
from .gemini_api import (
    create_tuning_job,
    get_client as get_gemini_client,
    job_to_record_updates as gemini_job_to_record_updates,
    normalize_gcs_uri,
    retrieve_tuning_job,
    stage_training_files,
    status_name as gemini_status_name,
    wait_for_tuning_job as wait_for_gemini_job,
)
from .openai_api import (
    build_reinforcement_fine_tuning_request,
    build_supervised_fine_tuning_request,
    create_fine_tuning_job,
    get_client as get_openai_client,
    list_fine_tuning_job_events,
    retrieve_fine_tuning_job,
    upload_file,
    validate_grader as validate_openai_grader,
    wait_for_fine_tuning_job,
)
from .openai_graders import (
    BOT_LABEL,
    HUMAN_LABEL,
    build_bot_detection_reinforcement_grader_schema,
    bot_detection_reward_table,
    normalize_prediction_label,
)
from .registry import (
    get_job_record,
    get_latest_job_id,
    list_model_records,
    list_registered_jobs,
    register_job_record,
    upsert_job_record,
)
from .serialization import to_jsonable
from .storage import now_slug


def _parse_optional_int_or_auto(
    value: str | int | None,
    *,
    field: str,
) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field} must be > 0 or 'auto'.")
        return value
    text = value.strip()
    if text == "":
        return None
    if text.lower() == "auto":
        return "auto"
    parsed = int(text)
    if parsed <= 0:
        raise ValueError(f"{field} must be > 0 or 'auto'.")
    return parsed


def _parse_optional_float_or_auto(
    value: str | float | int | None,
    *,
    field: str,
) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"{field} must be > 0 or 'auto'.")
        return parsed
    text = value.strip()
    if text == "":
        return None
    if text.lower() == "auto":
        return "auto"
    parsed = float(text)
    if parsed <= 0:
        raise ValueError(f"{field} must be > 0 or 'auto'.")
    return parsed


def _parse_optional_positive_int(value: str | None, *, field: str) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = int(text)
    if parsed <= 0:
        raise ValueError(f"{field} must be > 0.")
    return parsed


def _parse_optional_positive_float(value: str | None, *, field: str) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = float(text)
    if parsed <= 0:
        raise ValueError(f"{field} must be > 0.")
    return parsed


def _resolve_job_id(raw_job_id: str | None, *, provider: str) -> str:
    if raw_job_id:
        return raw_job_id

    latest_job_id = get_latest_job_id(provider=provider)
    if latest_job_id:
        return latest_job_id

    raise ValueError(
        f"No {provider} job id was provided and no locally registered {provider} "
        "jobs were found. Submit a training job first or pass --job-id explicitly."
    )


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "run"


def _label_value(value: str) -> str:
    return _slugify(value)[:63]


def _parse_dataset_ids(raw: str) -> tuple[int, ...]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("No dataset ids provided.")
    parsed = tuple(int(part) for part in parts)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate dataset ids are not allowed: {raw}")
    return parsed


def _default_openai_prepared_dir(method: str) -> Path:
    if method == "supervised":
        return OPENAI_SUPERVISED_PREPARED_DIR
    if method == "reinforcement":
        return OPENAI_REINFORCEMENT_PREPARED_DIR
    raise ValueError(
        f"Unsupported OpenAI training method: {method}. "
        f"Expected one of {', '.join(OPENAI_TRAINING_METHODS)}."
    )


def _resolve_openai_prepared_dir(
    *,
    method: str,
    prepared_dir: str | None,
) -> Path:
    if prepared_dir:
        return Path(prepared_dir)
    return _default_openai_prepared_dir(method)


def _openai_metadata(
    *,
    method: str,
    split_name: str,
    tag: str | None,
) -> dict[str, str]:
    metadata = {
        "workflow": "bot-or-not",
        "provider": "openai",
        "method": _label_value(method),
        "split": _label_value(split_name),
    }
    if tag:
        metadata["tag"] = _label_value(tag)
    return metadata


def _openai_method_from_job_payload(job_payload: dict[str, Any]) -> str | None:
    method_payload = job_payload.get("method")
    if not isinstance(method_payload, dict):
        return None

    method = method_payload.get("type")
    if isinstance(method, str) and method:
        return method
    return None


def _openai_train_hyperparameters(
    args: argparse.Namespace,
) -> dict[str, int | float | str | None]:
    method = args.method
    if method == "supervised":
        default_epochs = DEFAULT_OPENAI_SUPERVISED_EPOCHS
        default_batch_size = DEFAULT_OPENAI_SUPERVISED_BATCH_SIZE
        default_learning_rate_multiplier = (
            DEFAULT_OPENAI_SUPERVISED_LEARNING_RATE_MULTIPLIER
        )
    elif method == "reinforcement":
        default_epochs = None
        default_batch_size = None
        default_learning_rate_multiplier = None
    else:
        raise ValueError(
            f"Unsupported OpenAI training method: {method}. "
            f"Expected one of {', '.join(OPENAI_TRAINING_METHODS)}."
        )

    return {
        "n_epochs": _parse_optional_int_or_auto(
            args.epochs if args.epochs is not None else default_epochs,
            field="epochs",
        ),
        "batch_size": _parse_optional_int_or_auto(
            args.batch_size if args.batch_size is not None else default_batch_size,
            field="batch_size",
        ),
        "learning_rate_multiplier": _parse_optional_float_or_auto(
            (
                args.learning_rate_multiplier
                if args.learning_rate_multiplier is not None
                else default_learning_rate_multiplier
            ),
            field="learning_rate_multiplier",
        ),
    }


def _default_gemini_display_name(
    *,
    base_model: str,
    split_name: str,
    tag: str | None,
) -> str:
    suffix = f" {tag}" if tag else ""
    return f"bot-or-not {split_name} {base_model}{suffix} {now_slug()}"


def _default_gemini_gcs_prefix(
    *,
    split_name: str,
    base_model: str,
    tag: str | None,
) -> str:
    parts = [
        DEFAULT_GEMINI_GCS_PREFIX,
        "gemini",
        _slugify(split_name),
        _slugify(base_model),
    ]
    if tag:
        parts.append(_slugify(tag))
    parts.append(now_slug())
    return "/".join(parts)


def _gemini_labels(*, split_name: str, tag: str | None) -> dict[str, str]:
    labels = {
        "workflow": "bot-or-not",
        "provider": "gemini",
        "split": _label_value(split_name),
    }
    if tag:
        labels["tag"] = _label_value(tag)
    return labels


def _print_preparation_summary(summary: dict[str, Any]) -> None:
    inventory = summary["inventory"]
    integrity = summary["data_integrity"]
    provider = summary.get("provider", "unknown")
    method = summary.get("method")

    if method:
        print(
            f"Prepared {provider}/{method} fine-tuning data in "
            f"{summary['prepared_dir']}"
        )
    else:
        print(f"Prepared {provider} fine-tuning data in {summary['prepared_dir']}")
    print(
        "Inventory: "
        f"prev={inventory['prev']['dataset_count']} datasets "
        f"current={inventory['current']['dataset_count']} datasets"
    )
    print(
        "Integrity: "
        f"total={integrity['total_examples']} "
        f"BOT={integrity['labels'].get('BOT', 0)} "
        f"HUMAN={integrity['labels'].get('HUMAN', 0)}"
    )

    for split_name in PREPARED_SPLITS:
        split = summary["splits"].get(split_name)
        if split is None:
            continue
        print(
            f"{split_name}: train={split['train_count']} "
            f"val={split['validation_count']} "
            f"train_path={split['train_path']}"
        )


def _print_job_record(record: dict[str, Any], *, show_events: bool) -> None:
    print(
        f"provider={record.get('provider', 'unknown')} "
        f"job_id={record['job_id']} "
        f"status={record.get('status', 'unknown')} "
        f"method={record.get('method', '(unknown)')} "
        f"base_model={record.get('base_model', '(unknown)')} "
        f"split={record.get('split', '(unknown)')}"
    )

    if record.get("project") or record.get("location"):
        print(
            f"project={record.get('project', '(unknown)')} "
            f"location={record.get('location', '(unknown)')}"
        )

    tag = record.get("tag")
    if tag:
        print(f"tag={tag}")

    fine_tuned_model = record.get("fine_tuned_model")
    if fine_tuned_model:
        print(f"fine_tuned_model={fine_tuned_model}")

    tuned_model_endpoint = record.get("tuned_model_endpoint")
    if tuned_model_endpoint:
        print(f"tuned_model_endpoint={tuned_model_endpoint}")

    print(f"submitted_at={record.get('submitted_at', '(unknown)')}")
    print(f"last_synced_at={record.get('last_synced_at', '(never)')}")

    train_path = record.get("train_path")
    if train_path:
        print(f"train_path={train_path}")

    validation_path = record.get("validation_path")
    if validation_path:
        print(f"validation_path={validation_path}")

    training_file_id = record.get("training_file_id")
    if training_file_id:
        print(f"training_file_id={training_file_id}")

    validation_file_id = record.get("validation_file_id")
    if validation_file_id:
        print(f"validation_file_id={validation_file_id}")

    training_gcs_uri = record.get("training_gcs_uri")
    if training_gcs_uri:
        print(f"training_gcs_uri={training_gcs_uri}")

    validation_gcs_uri = record.get("validation_gcs_uri")
    if validation_gcs_uri:
        print(f"validation_gcs_uri={validation_gcs_uri}")

    if show_events:
        events = record.get("events", [])
        print("")
        print("Recent events:")
        if not events:
            print("  (none)")
        else:
            for event in events:
                message = event.get("message", "").strip() or "(no message)"
                print(
                    "  "
                    f"[{event.get('created_at', '?')}] "
                    f"{event.get('level', 'info')}: {message}"
                )


def cmd_openai_prepare_data(args: argparse.Namespace) -> int:
    if args.method == "all" and args.prepared_dir is not None:
        raise ValueError(
            "--prepared-dir cannot be used with --method all. "
            "Use the default OpenAI prepared root or prepare one method at a time."
        )

    methods = OPENAI_TRAINING_METHODS if args.method == "all" else (args.method,)
    for index, method in enumerate(methods):
        summary = prepare_data(
            prepared_dir=_resolve_openai_prepared_dir(
                method=method,
                prepared_dir=args.prepared_dir,
            ),
            val_fraction=args.val_fraction,
            seed=args.seed,
            provider="openai",
            openai_method=method,
        )
        if index > 0:
            print("")
        _print_preparation_summary(summary)
    return 0


def cmd_openai_train(args: argparse.Namespace) -> int:
    if args.eval_interval is not None and args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0.")
    if args.eval_samples is not None and args.eval_samples <= 0:
        raise ValueError("--eval-samples must be > 0.")
    if args.compute_multiplier is not None and args.compute_multiplier <= 0:
        raise ValueError("--compute-multiplier must be > 0.")

    hyperparameters = _openai_train_hyperparameters(args)
    prepared_dir = _resolve_openai_prepared_dir(
        method=args.method,
        prepared_dir=args.prepared_dir,
    )
    train_path, validation_path, split_name = resolve_training_files(
        prepared_dir=prepared_dir,
        split_name=args.split,
    )
    metadata = _openai_metadata(
        method=args.method,
        split_name=split_name,
        tag=args.tag,
    )

    print(
        f"Submitting OpenAI {args.method} fine-tuning job for split={split_name} "
        f"base_model={args.base_model} train={train_path}"
    )
    if validation_path is not None:
        print(f"Using validation file: {validation_path}")

    client = get_openai_client(api_key=args.api_key)
    training_file_id = upload_file(client, train_path)
    validation_file_id = (
        upload_file(client, validation_path) if validation_path is not None else None
    )

    record_hyperparameters = dict(hyperparameters)
    grader: dict[str, Any] | None = None
    grader_validation: dict[str, Any] | None = None
    reward_table: dict[str, Any] | None = None

    if args.method == "supervised":
        request = build_supervised_fine_tuning_request(
            model=args.base_model,
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            n_epochs=hyperparameters["n_epochs"],
            batch_size=hyperparameters["batch_size"],
            learning_rate_multiplier=hyperparameters["learning_rate_multiplier"],
            metadata=metadata,
            seed=args.seed,
        )
    elif args.method == "reinforcement":
        grader = build_bot_detection_reinforcement_grader_schema()
        reward_table = bot_detection_reward_table()
        grader_validation = validate_openai_grader(client, grader=grader)
        record_hyperparameters.update(
            {
                "reasoning_effort": args.reasoning_effort,
                "compute_multiplier": args.compute_multiplier,
                "eval_interval": args.eval_interval,
                "eval_samples": args.eval_samples,
            }
        )
        print("Validated OpenAI reinforcement grader.")
        request = build_reinforcement_fine_tuning_request(
            model=args.base_model,
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            grader=grader,
            n_epochs=hyperparameters["n_epochs"],
            batch_size=hyperparameters["batch_size"],
            learning_rate_multiplier=hyperparameters["learning_rate_multiplier"],
            reasoning_effort=args.reasoning_effort,
            compute_multiplier=args.compute_multiplier,
            eval_interval=args.eval_interval,
            eval_samples=args.eval_samples,
            metadata=metadata,
            seed=args.seed,
            response_format=None,
        )
    else:
        raise ValueError(
            f"Unsupported OpenAI training method: {args.method}. "
            f"Expected one of {', '.join(OPENAI_TRAINING_METHODS)}."
        )

    job = create_fine_tuning_job(client, request=request)

    record = register_job_record(
        {
            "provider": "openai",
            "job_id": getattr(job, "id", None),
            "status": getattr(job, "status", "submitted"),
            "method": args.method,
            "base_model": args.base_model,
            "split": split_name,
            "tag": args.tag,
            "train_path": str(train_path),
            "validation_path": str(validation_path) if validation_path else None,
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
            "hyperparameters": record_hyperparameters,
            "metadata": metadata,
            "seed": args.seed,
            "request": request,
            "grader": grader,
            "grader_validation": grader_validation,
            "reward_table": reward_table,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "job": to_jsonable(job),
            "events": [],
        }
    )

    print(f"Submitted job: {record['job_id']}")
    print(f"Snapshot: {record['snapshot_path']}")

    if not args.wait:
        return 0

    final_job = wait_for_fine_tuning_job(
        client,
        job_id=record["job_id"],
        poll_seconds=args.poll_seconds,
        max_wait_minutes=args.max_wait_minutes,
    )
    events = list_fine_tuning_job_events(
        client,
        job_id=record["job_id"],
        limit=args.events_limit,
    )
    final_record = upsert_job_record(
        record,
        updates={
            "provider": "openai",
            "job_id": getattr(final_job, "id", None),
            "status": getattr(final_job, "status", "unknown"),
            "base_model": record.get("base_model") or getattr(final_job, "model", None),
            "fine_tuned_model": getattr(final_job, "fine_tuned_model", None),
            "job": to_jsonable(final_job),
            "events": events,
        },
    )
    _print_job_record(final_record, show_events=args.events_limit > 0)

    if final_record.get("status") != "succeeded":
        raise RuntimeError(
            f"OpenAI fine-tuning job {record['job_id']} finished with status "
            f"{final_record.get('status', 'unknown')}."
        )

    return 0


def cmd_openai_job(args: argparse.Namespace) -> int:
    job_id = _resolve_job_id(args.job_id, provider="openai")
    existing_record = get_job_record(job_id, provider="openai")

    if args.refresh or args.wait:
        client = get_openai_client(api_key=args.api_key)
        job = (
            wait_for_fine_tuning_job(
                client,
                job_id=job_id,
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
            )
            if args.wait
            else retrieve_fine_tuning_job(client, job_id=job_id)
        )
        job_payload = to_jsonable(job)
        events = list_fine_tuning_job_events(
            client,
            job_id=job_id,
            limit=args.events_limit,
        )
        record = upsert_job_record(
            existing_record,
            updates={
                "provider": "openai",
                "job_id": getattr(job, "id", None),
                "status": getattr(job, "status", "unknown"),
                "method": _openai_method_from_job_payload(job_payload)
                or (existing_record or {}).get("method"),
                "base_model": (
                    (existing_record or {}).get("base_model")
                    or getattr(job, "model", None)
                ),
                "fine_tuned_model": getattr(job, "fine_tuned_model", None),
                "job": job_payload,
                "events": events,
            },
        )
    else:
        if existing_record is None:
            raise ValueError(
                f"No local snapshot exists for OpenAI job {job_id}. "
                "Use --refresh with a valid OPENAI_API_KEY to fetch it."
            )
        record = existing_record

    _print_job_record(record, show_events=args.events_limit > 0)
    return 0


def cmd_openai_models(_: argparse.Namespace) -> int:
    models = list_model_records(provider="openai")
    if not models:
        print("No OpenAI fine-tuned models have been registered locally yet.")
        return 0

    for model in models:
        line = (
            f"{model['fine_tuned_model']} "
            f"(job={model['job_id']} base={model.get('base_model', '(unknown)')} "
            f"method={model.get('method', '(unknown)')} "
            f"split={model.get('split', '(unknown)')} "
            f"synced={model.get('last_synced_at', '(never)')})"
        )
        if model.get("tag"):
            line += f" tag={model['tag']}"
        print(line)

    return 0


def cmd_openai_jobs(_: argparse.Namespace) -> int:
    jobs = list_registered_jobs(provider="openai")
    if not jobs:
        print("No OpenAI fine-tuning jobs have been registered locally yet.")
        return 0

    for job in jobs:
        line = (
            f"{job['job_id']} status={job.get('status', 'unknown')} "
            f"base={job.get('base_model', '(unknown)')} "
            f"method={job.get('method', '(unknown)')} "
            f"split={job.get('split', '(unknown)')}"
        )
        if job.get("fine_tuned_model"):
            line += f" model={job['fine_tuned_model']}"
        if job.get("tag"):
            line += f" tag={job['tag']}"
        print(line)

    return 0


def _resolve_evaluation_examples(
    *,
    collection_name: str | None,
    dataset_ids: tuple[int, ...] | None = None,
) -> tuple[str, list[Any]]:
    assets = discover_dataset_assets()
    if dataset_ids is not None:
        assets_by_id = {asset.dataset_id: asset for asset in assets}
        missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in assets_by_id]
        if missing_ids:
            raise ValueError(
                "Some requested dataset ids do not exist on disk: "
                + ", ".join(str(dataset_id) for dataset_id in missing_ids)
            )
        selection_name = "datasets_" + "_".join(str(dataset_id) for dataset_id in dataset_ids)
        selected_assets = tuple(assets_by_id[dataset_id] for dataset_id in dataset_ids)
        return selection_name, load_examples(selected_assets)

    if collection_name is None:
        raise ValueError("Either collection_name or dataset_ids must be provided.")

    collections = group_assets_by_collection(assets)
    resolved_collection = "both" if collection_name == "full" else collection_name
    if resolved_collection not in collections:
        raise ValueError(
            f"Unsupported evaluation collection: {collection_name}. "
            f"Expected one of {', '.join((*DATASET_COLLECTIONS, 'full'))}."
        )
    return resolved_collection, load_examples(collections[resolved_collection])


def _resolve_final_submission_examples(
    *,
    dataset_ids: tuple[int, ...] | None = None,
) -> tuple[str, list[Any]]:
    assets = discover_final_dataset_assets()
    assets_by_id = {asset.dataset_id: asset for asset in assets}

    if dataset_ids is None:
        selection_name = "final"
        selected_assets = assets
    else:
        missing_ids = [
            dataset_id for dataset_id in dataset_ids if dataset_id not in assets_by_id
        ]
        if missing_ids:
            raise ValueError(
                "Some requested final dataset ids do not exist on disk: "
                + ", ".join(str(dataset_id) for dataset_id in missing_ids)
            )
        selected_assets = tuple(assets_by_id[dataset_id] for dataset_id in dataset_ids)
        all_dataset_ids = tuple(asset.dataset_id for asset in assets)
        selection_name = (
            "final"
            if tuple(dataset_ids) == all_dataset_ids
            else "final_datasets_" + "_".join(str(dataset_id) for dataset_id in dataset_ids)
        )

    return selection_name, load_final_examples(selected_assets)


def _print_incremental_score(
    *,
    running: Any,
    completed: int,
    total_examples: int,
    elapsed_seconds: float,
) -> None:
    print(
        "progress "
        f"completed={completed}/{total_examples} "
        f"score={running.raw_score}/{running.max_possible_score} "
        f"pct={running.pct_of_max:.4f} "
        f"tp={running.tp} fp={running.fp} fn={running.fn} "
        f"invalid={running.invalid_output_count} "
        f"elapsed={elapsed_seconds:.1f}s"
    )


def _print_final_score(*, running: Any, total_examples: int) -> None:
    print(
        "final "
        f"completed={running.completed_examples}/{total_examples} "
        f"score={running.raw_score}/{running.max_possible_score} "
        f"pct={running.pct_of_max:.4f} "
        f"tp={running.tp} fp={running.fp} fn={running.fn} "
        f"invalid={running.invalid_output_count}"
    )


def _print_submission_progress(
    *,
    completed: int,
    total_examples: int,
    detected_bots: int,
    invalid_count: int,
    elapsed_seconds: float,
) -> None:
    print(
        "progress "
        f"completed={completed}/{total_examples} "
        f"detected_bots={detected_bots} "
        f"invalid={invalid_count} "
        f"elapsed={elapsed_seconds:.1f}s"
    )


def _print_submission_final(
    *,
    completed: int,
    total_examples: int,
    detected_bots: int,
    invalid_count: int,
) -> None:
    print(
        "final "
        f"completed={completed}/{total_examples} "
        f"detected_bots={detected_bots} "
        f"invalid={invalid_count}"
    )


def _dataset_ids_for_examples(examples: list[Any]) -> tuple[int, ...]:
    return tuple(sorted({int(example.dataset_id) for example in examples}))


def _load_or_initialize_raw_artifact_for_examples(
    *,
    provider: str,
    model: str,
    collection: str,
    raw_results_path: Path,
    examples: list[Any],
):
    artifact = (
        load_raw_results(raw_results_path)
        if raw_results_path.exists()
        else initialize_raw_evaluation_artifact(
            provider=provider,
            model=model,
            collection=collection,
            dataset_ids=_dataset_ids_for_examples(examples),
            total_examples=len(examples),
        )
    )
    if raw_results_path.exists():
        if artifact.provider != provider:
            raise ValueError(
                f"Raw results provider mismatch: expected {provider!r}, got {artifact.provider!r}"
            )
        if artifact.model != model:
            raise ValueError(
                "Raw results model mismatch: "
                f"raw={artifact.model!r} requested={model!r}"
            )
        if artifact.collection != collection:
            raise ValueError(
                "Raw results collection mismatch: "
                f"raw={artifact.collection!r} requested={collection!r}"
            )
        if artifact.total_examples != len(examples):
            raise ValueError(
                "Raw results total_examples mismatch: "
                f"raw={artifact.total_examples} expected={len(examples)}"
            )
        expected_dataset_ids = _dataset_ids_for_examples(examples)
        if artifact.dataset_ids != expected_dataset_ids:
            raise ValueError(
                "Raw results dataset_ids mismatch: "
                f"raw={artifact.dataset_ids} expected={expected_dataset_ids}"
            )
    return artifact


def _classify_submission_examples_to_raw(
    *,
    provider: str,
    model: str,
    collection: str,
    raw_results_path: Path,
    examples: list[Any],
    classify_one: Any,
    max_workers: int,
    report_every: int,
    save_every: int,
) -> tuple[Path, Any]:
    artifact = _load_or_initialize_raw_artifact_for_examples(
        provider=provider,
        model=model,
        collection=collection,
        raw_results_path=raw_results_path,
        examples=examples,
    )

    print(f"loaded_examples={len(examples)}")
    print(f"raw_results_path={raw_results_path}")

    invalid_count = sum(
        1
        for prediction in artifact.predictions_by_user.values()
        if prediction.predicted_label is None
    )
    detected_bots = len(artifact.detected_bot_ids)

    if artifact.completed_examples:
        print(
            f"resuming_predictions={artifact.completed_examples} path={raw_results_path}"
        )
        _print_submission_progress(
            completed=artifact.completed_examples,
            total_examples=len(examples),
            detected_bots=detected_bots,
            invalid_count=invalid_count,
            elapsed_seconds=0.0,
        )

    remaining_examples = [
        example
        for example in examples
        if example.user_id not in artifact.predictions_by_user
    ]
    print(
        f"remaining_examples={len(remaining_examples)} "
        f"workers={max_workers} report_every={report_every}"
    )

    user_id_order = [example.user_id for example in examples]
    completed = artifact.completed_examples
    last_saved_completed = completed
    started_at = time.monotonic()

    if remaining_examples:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(classify_one, example): example
                for example in remaining_examples
            }
            for future in as_completed(futures):
                prediction = future.result()
                artifact.with_prediction(prediction)
                completed += 1
                if prediction.predicted_label is None:
                    invalid_count += 1
                elif prediction.predicted_label == BOT_LABEL:
                    detected_bots += 1

                should_save = (
                    completed == len(examples)
                    or completed % report_every == 0
                    or completed - last_saved_completed >= save_every
                )
                if should_save:
                    save_raw_results(
                        raw_results_path,
                        artifact,
                        user_id_order=user_id_order,
                    )
                    last_saved_completed = completed
                if completed == len(examples) or completed % report_every == 0:
                    _print_submission_progress(
                        completed=completed,
                        total_examples=len(examples),
                        detected_bots=detected_bots,
                        invalid_count=invalid_count,
                        elapsed_seconds=time.monotonic() - started_at,
                    )

    if completed != last_saved_completed or not raw_results_path.exists():
        save_raw_results(
            raw_results_path,
            artifact,
            user_id_order=user_id_order,
        )

    return raw_results_path, artifact

def cmd_interactive(args: argparse.Namespace) -> int:
    from .interactive import run_interactive_wizard

    return run_interactive_wizard(
        openai_api_key=args.api_key,
        google_project=args.project,
        google_location=args.location,
        workflow=args.workflow,
        team_name=args.team_name,
        raw_path=args.raw_path,
        raw_dir=args.raw_dir,
        runs_dir=args.runs_dir,
        final_results_dir=args.final_results_dir,
        report_path=args.report_path,
        run_slug=args.run_slug,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
    )


def cmd_submit(args: argparse.Namespace) -> int:
    from .interactive import run_interactive_wizard

    return run_interactive_wizard(
        openai_api_key=args.api_key,
        google_project=args.project,
        google_location=args.location,
        workflow="submit-final",
        team_name=args.team_name,
        raw_path=args.raw_path,
        raw_dir=args.raw_dir,
        runs_dir=str(RUNS_DIR),
        final_results_dir=args.final_results_dir,
        report_path=None,
        run_slug=args.run_slug,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
    )


def _score_raw_results_to_report(
    *,
    raw_results_path: Path,
    examples: list[Any],
    report_path: Path,
) -> Any:
    scored = load_and_score_raw_results(
        raw_results_path=raw_results_path,
        examples=examples,
    )
    save_run_report(report_path=report_path, scored=scored)
    return scored


def _classify_openai_examples_to_raw(
    *,
    model: str,
    collection: str,
    raw_results_path: Path,
    examples: list[Any],
    max_workers: int,
    max_retries: int,
    max_output_tokens: int,
    report_every: int,
    save_every: int,
    api_key: str | None,
) -> tuple[Path, Any]:
    user_id_to_example = {example.user_id: example for example in examples}
    artifact = (
        load_raw_results(raw_results_path)
        if raw_results_path.exists()
        else initialize_raw_evaluation_artifact(
            provider="openai",
            model=model,
            collection=collection,
            dataset_ids=tuple(sorted({example.dataset_id for example in examples})),
            total_examples=len(examples),
        )
    )
    if raw_results_path.exists():
        if artifact.provider != "openai":
            raise ValueError(
                f"Raw results provider mismatch: expected 'openai', got {artifact.provider!r}"
            )
        if artifact.model != model:
            raise ValueError(
                "Raw results model mismatch: "
                f"raw={artifact.model!r} requested={model!r}"
            )
        if artifact.collection != collection:
            raise ValueError(
                "Raw results collection mismatch: "
                f"raw={artifact.collection!r} requested={collection!r}"
            )
        if artifact.total_examples != len(examples):
            raise ValueError(
                "Raw results total_examples mismatch: "
                f"raw={artifact.total_examples} expected={len(examples)}"
            )
        expected_dataset_ids = tuple(sorted({example.dataset_id for example in examples}))
        if artifact.dataset_ids != expected_dataset_ids:
            raise ValueError(
                "Raw results dataset_ids mismatch: "
                f"raw={artifact.dataset_ids} expected={expected_dataset_ids}"
            )

    running = (
        build_running_evaluation_from_raw_results(
            artifact=artifact,
            examples_by_user_id=user_id_to_example,
        )
        if artifact.completed_examples
        else RunningEvaluation()
    )
    print(f"loaded_examples={len(examples)}")
    print(f"raw_results_path={raw_results_path}")
    if artifact.completed_examples:
        print(
            f"resuming_predictions={artifact.completed_examples} path={raw_results_path}"
        )
        _print_incremental_score(
            running=running,
            completed=artifact.completed_examples,
            total_examples=len(examples),
            elapsed_seconds=0.0,
        )

    remaining_examples = [
        example for example in examples if example.user_id not in artifact.predictions_by_user
    ]
    print(
        f"remaining_examples={len(remaining_examples)} "
        f"workers={max_workers} report_every={report_every}"
    )

    thread_local = threading.local()

    def get_thread_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = get_openai_client(api_key=api_key)
            thread_local.client = client
        return client

    def classify(example: Any) -> RawPrediction:
        client = get_thread_client()
        last_error: Exception | None = None
        messages = example.to_openai_reinforcement_record()["messages"]
        for attempt in range(max_retries):
            try:
                response = client.responses.create(
                    model=model,
                    input=messages,
                    temperature=0,
                    max_output_tokens=max_output_tokens,
                )
                raw_output = getattr(response, "output_text", "") or ""
                predicted_label = normalize_prediction_label(raw_output)
                return RawPrediction(
                    user_id=example.user_id,
                    predicted_label=predicted_label,
                    raw_output=raw_output,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait_seconds = min(60.0, (2**attempt) + random.random())
                error_text = str(exc)
                if "429" in error_text or "rate limit" in error_text.lower():
                    print(
                        "rate_limit "
                        f"user_id={example.user_id} "
                        f"attempt={attempt + 1}/{max_retries} "
                        f"wait={wait_seconds:.1f}s "
                        f"error={exc}",
                        flush=True,
                    )
                else:
                    print(
                        "retry "
                        f"user_id={example.user_id} "
                        f"attempt={attempt + 1}/{max_retries} "
                        f"wait={wait_seconds:.1f}s "
                        f"error={exc}",
                        flush=True,
                    )
                time.sleep(wait_seconds)
        raise RuntimeError(
            f"Failed to classify user {example.user_id} after "
            f"{max_retries} attempts: {last_error}"
        )

    user_id_order = [example.user_id for example in examples]
    completed = artifact.completed_examples
    last_saved_completed = completed
    started_at = time.monotonic()

    if remaining_examples:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(classify, example): example
                for example in remaining_examples
            }
            for future in as_completed(futures):
                prediction = future.result()
                example = futures[future]
                artifact.with_prediction(prediction)
                add_prediction_to_running_evaluation(
                    running,
                    truth_label=example.label,
                    predicted_label=prediction.predicted_label,
                    raw_output=prediction.raw_output,
                    user_id=prediction.user_id,
                )
                completed += 1
                should_save = (
                    completed == len(examples)
                    or completed % report_every == 0
                    or completed - last_saved_completed >= save_every
                )
                if should_save:
                    save_raw_results(
                        raw_results_path,
                        artifact,
                        user_id_order=user_id_order,
                    )
                    last_saved_completed = completed
                if completed == len(examples) or completed % report_every == 0:
                    _print_incremental_score(
                        running=running,
                        completed=completed,
                        total_examples=len(examples),
                        elapsed_seconds=time.monotonic() - started_at,
                    )

    if completed != last_saved_completed or not raw_results_path.exists():
        save_raw_results(
            raw_results_path,
            artifact,
            user_id_order=user_id_order,
        )

    final_scored = load_and_score_raw_results(
        raw_results_path=raw_results_path,
        examples=examples,
    )
    return raw_results_path, final_scored


def cmd_openai_infer_raw(args: argparse.Namespace) -> int:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be > 0.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be > 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0.")
    if args.dataset_ids and args.collection:
        raise ValueError("--dataset-ids and --collection cannot be used together.")

    get_openai_client(api_key=args.api_key)

    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    selection_name, examples = _resolve_evaluation_examples(
        collection_name=args.collection or "both",
        dataset_ids=requested_dataset_ids,
    )
    raw_results_path = (
        Path(args.raw_path).resolve()
        if args.raw_path
        else build_raw_results_path(
            provider="openai",
            model=args.model,
            collection=selection_name,
            run_slug=args.run_slug,
            output_dir=Path(args.raw_dir),
        ).resolve()
    )
    raw_results_path, scored = _classify_openai_examples_to_raw(
        model=args.model,
        collection=selection_name,
        raw_results_path=raw_results_path,
        examples=examples,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
        api_key=args.api_key,
    )
    print("RAW_RESULTS")
    print(raw_results_path)
    _print_final_score(running=scored.running, total_examples=len(examples))
    return 0


def cmd_openai_score_raw(args: argparse.Namespace) -> int:
    raw_results_path = Path(args.raw_path).resolve()
    artifact = load_raw_results(raw_results_path)
    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    if requested_dataset_ids is not None and args.collection:
        raise ValueError("--dataset-ids and --collection cannot be used together.")
    _, examples = _resolve_evaluation_examples(
        collection_name=(args.collection or artifact.collection or "both"),
        dataset_ids=requested_dataset_ids or artifact.dataset_ids,
    )
    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else build_run_report_path_for_raw(
            raw_results_path=raw_results_path,
            output_dir=Path(args.runs_dir),
        ).resolve()
    )
    scored = _score_raw_results_to_report(
        raw_results_path=raw_results_path,
        examples=examples,
        report_path=report_path,
    )
    print("RUN_REPORT")
    print(report_path)
    _print_final_score(running=scored.running, total_examples=scored.total_examples)
    return 0


def cmd_openai_evaluate(args: argparse.Namespace) -> int:
    run_slug = args.run_slug or now_slug()
    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    if requested_dataset_ids is not None and args.collection:
        raise ValueError("--dataset-ids and --collection cannot be used together.")
    selection_name, _ = _resolve_evaluation_examples(
        collection_name=args.collection or "both",
        dataset_ids=requested_dataset_ids,
    )
    raw_results_path = (
        Path(args.raw_path).resolve()
        if args.raw_path
        else build_raw_results_path(
            provider="openai",
            model=args.model,
            collection=selection_name,
            run_slug=run_slug,
            output_dir=Path(args.raw_dir),
        ).resolve()
    )
    infer_args = argparse.Namespace(
        model=args.model,
        collection=args.collection,
        dataset_ids=args.dataset_ids,
        raw_path=str(raw_results_path),
        raw_dir=args.raw_dir,
        run_slug=run_slug,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
        api_key=args.api_key,
    )
    infer_result = cmd_openai_infer_raw(infer_args)
    if infer_result != 0:
        return infer_result

    score_args = argparse.Namespace(
        raw_path=str(raw_results_path),
        collection=args.collection,
        dataset_ids=args.dataset_ids,
        report_path=(
            args.report_path
            if args.report_path
            else str(
                build_run_report_path(
                    provider="openai",
                    model=args.model,
                    collection=selection_name,
                    run_slug=run_slug,
                    output_dir=Path(args.runs_dir),
                ).resolve()
            )
        ),
        runs_dir=args.runs_dir,
    )
    return cmd_openai_score_raw(score_args)


def cmd_openai_submit_final(args: argparse.Namespace) -> int:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be > 0.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be > 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0.")

    get_openai_client(api_key=args.api_key)

    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    selection_name, examples = _resolve_final_submission_examples(
        dataset_ids=requested_dataset_ids,
    )
    raw_results_path = (
        Path(args.raw_path).resolve()
        if args.raw_path
        else build_raw_results_path(
            provider="openai",
            model=args.model,
            collection=selection_name,
            run_slug=args.run_slug,
            output_dir=Path(args.raw_dir),
        ).resolve()
    )

    thread_local = threading.local()

    def get_thread_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = get_openai_client(api_key=args.api_key)
            thread_local.client = client
        return client

    def classify(example: Any) -> RawPrediction:
        client = get_thread_client()
        last_error: Exception | None = None
        messages = example.to_openai_messages()
        for attempt in range(args.max_retries):
            try:
                response = client.responses.create(
                    model=args.model,
                    input=messages,
                    temperature=0,
                    max_output_tokens=args.max_output_tokens,
                )
                raw_output = getattr(response, "output_text", "") or ""
                predicted_label = normalize_prediction_label(raw_output)
                return RawPrediction(
                    user_id=example.user_id,
                    predicted_label=predicted_label,
                    raw_output=raw_output,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait_seconds = min(60.0, (2**attempt) + random.random())
                error_text = str(exc)
                if "429" in error_text or "rate limit" in error_text.lower():
                    print(
                        "rate_limit "
                        f"user_id={example.user_id} "
                        f"attempt={attempt + 1}/{args.max_retries} "
                        f"wait={wait_seconds:.1f}s "
                        f"error={exc}",
                        flush=True,
                    )
                else:
                    print(
                        "retry "
                        f"user_id={example.user_id} "
                        f"attempt={attempt + 1}/{args.max_retries} "
                        f"wait={wait_seconds:.1f}s "
                        f"error={exc}",
                        flush=True,
                    )
                time.sleep(wait_seconds)
        raise RuntimeError(
            f"Failed to classify user {example.user_id} after "
            f"{args.max_retries} attempts: {last_error}"
        )

    raw_results_path, artifact = _classify_submission_examples_to_raw(
        provider="openai",
        model=args.model,
        collection=selection_name,
        raw_results_path=raw_results_path,
        examples=examples,
        classify_one=classify,
        max_workers=args.max_workers,
        report_every=args.report_every,
        save_every=args.save_every,
    )
    submission_paths = write_submission_files(
        raw_results_path=raw_results_path,
        examples=examples,
        team_name=args.team_name,
        output_dir=Path(args.output_dir),
    )
    print("RAW_RESULTS")
    print(raw_results_path)
    print("SUBMISSION_FILES")
    for path in submission_paths:
        print(path)
    _print_submission_final(
        completed=artifact.completed_examples,
        total_examples=len(examples),
        detected_bots=len(artifact.detected_bot_ids),
        invalid_count=sum(
            1
            for prediction in artifact.predictions_by_user.values()
            if prediction.predicted_label is None
        ),
    )
    return 0


def _gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    candidates = getattr(response, "candidates", None) or []
    text_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                text_parts.append(part_text)
    return "\n".join(text_parts)


def _build_gemini_bot_detection_config(
    *,
    max_output_tokens: int,
) -> gemini_types.GenerateContentConfig:
    return gemini_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        candidate_count=1,
        max_output_tokens=max(max_output_tokens, 8),
        response_mime_type="text/x.enum",
        response_schema=gemini_types.Schema(
            type=gemini_types.Type.STRING,
            enum=[BOT_LABEL, HUMAN_LABEL],
        ),
        thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
    )


def _classify_gemini_examples_to_raw(
    *,
    model: str,
    collection: str,
    raw_results_path: Path,
    examples: list[Any],
    max_workers: int,
    max_retries: int,
    max_output_tokens: int,
    report_every: int,
    save_every: int,
    project: str | None,
    location: str | None,
) -> tuple[Path, Any]:
    user_id_to_example = {example.user_id: example for example in examples}
    artifact = (
        load_raw_results(raw_results_path)
        if raw_results_path.exists()
        else initialize_raw_evaluation_artifact(
            provider="gemini",
            model=model,
            collection=collection,
            dataset_ids=tuple(sorted({example.dataset_id for example in examples})),
            total_examples=len(examples),
        )
    )
    if raw_results_path.exists():
        if artifact.provider != "gemini":
            raise ValueError(
                f"Raw results provider mismatch: expected 'gemini', got {artifact.provider!r}"
            )
        if artifact.model != model:
            raise ValueError(
                "Raw results model mismatch: "
                f"raw={artifact.model!r} requested={model!r}"
            )
        if artifact.collection != collection:
            raise ValueError(
                "Raw results collection mismatch: "
                f"raw={artifact.collection!r} requested={collection!r}"
            )
        if artifact.total_examples != len(examples):
            raise ValueError(
                "Raw results total_examples mismatch: "
                f"raw={artifact.total_examples} expected={len(examples)}"
            )
        expected_dataset_ids = tuple(sorted({example.dataset_id for example in examples}))
        if artifact.dataset_ids != expected_dataset_ids:
            raise ValueError(
                "Raw results dataset_ids mismatch: "
                f"raw={artifact.dataset_ids} expected={expected_dataset_ids}"
            )

    running = (
        build_running_evaluation_from_raw_results(
            artifact=artifact,
            examples_by_user_id=user_id_to_example,
        )
        if artifact.completed_examples
        else RunningEvaluation()
    )
    print(f"loaded_examples={len(examples)}")
    print(f"raw_results_path={raw_results_path}")
    if artifact.completed_examples:
        print(
            f"resuming_predictions={artifact.completed_examples} path={raw_results_path}"
        )
        _print_incremental_score(
            running=running,
            completed=artifact.completed_examples,
            total_examples=len(examples),
            elapsed_seconds=0.0,
        )

    remaining_examples = [
        example for example in examples if example.user_id not in artifact.predictions_by_user
    ]
    print(
        f"remaining_examples={len(remaining_examples)} "
        f"workers={max_workers} report_every={report_every}"
    )

    thread_local = threading.local()

    def get_thread_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client, _ = get_gemini_client(project=project, location=location)
            thread_local.client = client
        return client

    def classify(example: Any) -> RawPrediction:
        client = get_thread_client()
        last_error: Exception | None = None
        config = _build_gemini_bot_detection_config(
            max_output_tokens=max_output_tokens,
        )
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=example.user_prompt,
                    config=config,
                )
                raw_output = _gemini_response_text(response)
                predicted_label = normalize_prediction_label(raw_output)
                return RawPrediction(
                    user_id=example.user_id,
                    predicted_label=predicted_label,
                    raw_output=raw_output,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait_seconds = min(60.0, (2**attempt) + random.random())
                print(
                    "retry "
                    f"user_id={example.user_id} "
                    f"attempt={attempt + 1}/{max_retries} "
                    f"wait={wait_seconds:.1f}s "
                    f"error={exc}",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise RuntimeError(
            f"Failed to classify user {example.user_id} after "
            f"{max_retries} attempts: {last_error}"
        )

    user_id_order = [example.user_id for example in examples]
    completed = artifact.completed_examples
    last_saved_completed = completed
    started_at = time.monotonic()

    if remaining_examples:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(classify, example): example
                for example in remaining_examples
            }
            for future in as_completed(futures):
                prediction = future.result()
                example = futures[future]
                artifact.with_prediction(prediction)
                add_prediction_to_running_evaluation(
                    running,
                    truth_label=example.label,
                    predicted_label=prediction.predicted_label,
                    raw_output=prediction.raw_output,
                    user_id=prediction.user_id,
                )
                completed += 1
                should_save = (
                    completed == len(examples)
                    or completed % report_every == 0
                    or completed - last_saved_completed >= save_every
                )
                if should_save:
                    save_raw_results(
                        raw_results_path,
                        artifact,
                        user_id_order=user_id_order,
                    )
                    last_saved_completed = completed
                if completed == len(examples) or completed % report_every == 0:
                    _print_incremental_score(
                        running=running,
                        completed=completed,
                        total_examples=len(examples),
                        elapsed_seconds=time.monotonic() - started_at,
                    )

    if completed != last_saved_completed or not raw_results_path.exists():
        save_raw_results(
            raw_results_path,
            artifact,
            user_id_order=user_id_order,
        )

    final_scored = load_and_score_raw_results(
        raw_results_path=raw_results_path,
        examples=examples,
    )
    return raw_results_path, final_scored


def cmd_gemini_infer_raw(args: argparse.Namespace) -> int:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be > 0.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be > 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0.")
    if args.dataset_ids and args.collection:
        raise ValueError("--dataset-ids and --collection cannot be used together.")

    get_gemini_client(project=args.project, location=args.location)

    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    selection_name, examples = _resolve_evaluation_examples(
        collection_name=args.collection or "both",
        dataset_ids=requested_dataset_ids,
    )
    raw_results_path = (
        Path(args.raw_path).resolve()
        if args.raw_path
        else build_raw_results_path(
            provider="gemini",
            model=args.model,
            collection=selection_name,
            run_slug=args.run_slug,
            output_dir=Path(args.raw_dir),
        ).resolve()
    )
    raw_results_path, scored = _classify_gemini_examples_to_raw(
        model=args.model,
        collection=selection_name,
        raw_results_path=raw_results_path,
        examples=examples,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
        project=args.project,
        location=args.location,
    )
    print("RAW_RESULTS")
    print(raw_results_path)
    _print_final_score(running=scored.running, total_examples=len(examples))
    return 0


def cmd_gemini_score_raw(args: argparse.Namespace) -> int:
    raw_results_path = Path(args.raw_path).resolve()
    artifact = load_raw_results(raw_results_path)
    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    if requested_dataset_ids is not None and args.collection:
        raise ValueError("--dataset-ids and --collection cannot be used together.")
    _, examples = _resolve_evaluation_examples(
        collection_name=(args.collection or artifact.collection or "both"),
        dataset_ids=requested_dataset_ids or artifact.dataset_ids,
    )
    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else build_run_report_path_for_raw(
            raw_results_path=raw_results_path,
            output_dir=Path(args.runs_dir),
        ).resolve()
    )
    scored = _score_raw_results_to_report(
        raw_results_path=raw_results_path,
        examples=examples,
        report_path=report_path,
    )
    print("RUN_REPORT")
    print(report_path)
    _print_final_score(running=scored.running, total_examples=scored.total_examples)
    return 0


def cmd_gemini_evaluate(args: argparse.Namespace) -> int:
    run_slug = args.run_slug or now_slug()
    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    if requested_dataset_ids is not None and args.collection:
        raise ValueError("--dataset-ids and --collection cannot be used together.")
    selection_name, _ = _resolve_evaluation_examples(
        collection_name=args.collection or "both",
        dataset_ids=requested_dataset_ids,
    )
    raw_results_path = (
        Path(args.raw_path).resolve()
        if args.raw_path
        else build_raw_results_path(
            provider="gemini",
            model=args.model,
            collection=selection_name,
            run_slug=run_slug,
            output_dir=Path(args.raw_dir),
        ).resolve()
    )
    infer_args = argparse.Namespace(
        model=args.model,
        collection=args.collection,
        dataset_ids=args.dataset_ids,
        raw_path=str(raw_results_path),
        raw_dir=args.raw_dir,
        run_slug=run_slug,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
        project=args.project,
        location=args.location,
    )
    infer_result = cmd_gemini_infer_raw(infer_args)
    if infer_result != 0:
        return infer_result

    score_args = argparse.Namespace(
        raw_path=str(raw_results_path),
        collection=args.collection,
        dataset_ids=args.dataset_ids,
        report_path=(
            args.report_path
            if args.report_path
            else str(
                build_run_report_path(
                    provider="gemini",
                    model=args.model,
                    collection=selection_name,
                    run_slug=run_slug,
                    output_dir=Path(args.runs_dir),
                ).resolve()
            )
        ),
        runs_dir=args.runs_dir,
    )
    return cmd_gemini_score_raw(score_args)


def cmd_gemini_submit_final(args: argparse.Namespace) -> int:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be > 0.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be > 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0.")

    get_gemini_client(project=args.project, location=args.location)

    requested_dataset_ids = (
        _parse_dataset_ids(args.dataset_ids) if args.dataset_ids else None
    )
    selection_name, examples = _resolve_final_submission_examples(
        dataset_ids=requested_dataset_ids,
    )
    raw_results_path = (
        Path(args.raw_path).resolve()
        if args.raw_path
        else build_raw_results_path(
            provider="gemini",
            model=args.model,
            collection=selection_name,
            run_slug=args.run_slug,
            output_dir=Path(args.raw_dir),
        ).resolve()
    )

    thread_local = threading.local()

    def get_thread_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client, _ = get_gemini_client(
                project=args.project,
                location=args.location,
            )
            thread_local.client = client
        return client

    def classify(example: Any) -> RawPrediction:
        client = get_thread_client()
        last_error: Exception | None = None
        config = _build_gemini_bot_detection_config(
            max_output_tokens=args.max_output_tokens,
        )
        for attempt in range(args.max_retries):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=example.user_prompt,
                    config=config,
                )
                raw_output = _gemini_response_text(response)
                predicted_label = normalize_prediction_label(raw_output)
                return RawPrediction(
                    user_id=example.user_id,
                    predicted_label=predicted_label,
                    raw_output=raw_output,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait_seconds = min(60.0, (2**attempt) + random.random())
                print(
                    "retry "
                    f"user_id={example.user_id} "
                    f"attempt={attempt + 1}/{args.max_retries} "
                    f"wait={wait_seconds:.1f}s "
                    f"error={exc}",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise RuntimeError(
            f"Failed to classify user {example.user_id} after "
            f"{args.max_retries} attempts: {last_error}"
        )

    raw_results_path, artifact = _classify_submission_examples_to_raw(
        provider="gemini",
        model=args.model,
        collection=selection_name,
        raw_results_path=raw_results_path,
        examples=examples,
        classify_one=classify,
        max_workers=args.max_workers,
        report_every=args.report_every,
        save_every=args.save_every,
    )
    submission_paths = write_submission_files(
        raw_results_path=raw_results_path,
        examples=examples,
        team_name=args.team_name,
        output_dir=Path(args.output_dir),
    )
    print("RAW_RESULTS")
    print(raw_results_path)
    print("SUBMISSION_FILES")
    for path in submission_paths:
        print(path)
    _print_submission_final(
        completed=artifact.completed_examples,
        total_examples=len(examples),
        detected_bots=len(artifact.detected_bot_ids),
        invalid_count=sum(
            1
            for prediction in artifact.predictions_by_user.values()
            if prediction.predicted_label is None
        ),
    )
    return 0


def cmd_gemini_prepare_data(args: argparse.Namespace) -> int:
    summary = prepare_data(
        prepared_dir=Path(args.prepared_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
        provider="gemini",
    )
    _print_preparation_summary(summary)
    return 0


def cmd_gemini_train(args: argparse.Namespace) -> int:
    if args.adapter_size and args.tuning_mode != "adapter":
        raise ValueError("--adapter-size requires --tuning-mode adapter.")

    train_path, validation_path, split_name = resolve_training_files(
        prepared_dir=Path(args.prepared_dir),
        split_name=args.split,
    )

    display_name = args.display_name or _default_gemini_display_name(
        base_model=args.base_model,
        split_name=split_name,
        tag=args.tag,
    )
    gcs_prefix = args.gcs_prefix or _default_gemini_gcs_prefix(
        split_name=split_name,
        base_model=args.base_model,
        tag=args.tag,
    )

    client, context = get_gemini_client(
        project=args.project,
        location=args.location,
    )
    bucket_uri = normalize_gcs_uri(args.bucket)

    print(
        f"Submitting Gemini tuning job for split={split_name} "
        f"base_model={args.base_model} project={context.project} "
        f"location={context.location}"
    )
    print(f"Staging training data to {bucket_uri}/{gcs_prefix}")

    staged_paths = stage_training_files(
        project=context.project,
        bucket_uri=bucket_uri,
        object_prefix=gcs_prefix,
        train_path=train_path,
        validation_path=validation_path,
    )

    job = create_tuning_job(
        client,
        base_model=args.base_model,
        training_gcs_uri=str(staged_paths["training_gcs_uri"]),
        validation_gcs_uri=(
            str(staged_paths["validation_gcs_uri"])
            if staged_paths["validation_gcs_uri"] is not None
            else None
        ),
        tuned_model_display_name=display_name,
        description=args.description,
        epoch_count=_parse_optional_positive_int(args.epochs, field="epochs"),
        batch_size=_parse_optional_positive_int(args.batch_size, field="batch_size"),
        learning_rate=_parse_optional_positive_float(
            args.learning_rate,
            field="learning_rate",
        ),
        learning_rate_multiplier=_parse_optional_positive_float(
            args.learning_rate_multiplier,
            field="learning_rate_multiplier",
        ),
        tuning_mode=args.tuning_mode,
        adapter_size=args.adapter_size,
        labels=_gemini_labels(split_name=split_name, tag=args.tag),
    )

    record = register_job_record(
        {
            "provider": "gemini",
            "job_id": getattr(job, "name", None),
            "status": gemini_status_name(job),
            "method": "supervised",
            "raw_status": getattr(getattr(job, "state", None), "name", None),
            "base_model": args.base_model,
            "split": split_name,
            "tag": args.tag,
            "project": context.project,
            "location": context.location,
            "bucket_uri": bucket_uri,
            "train_path": str(train_path),
            "validation_path": str(validation_path) if validation_path else None,
            "training_gcs_uri": staged_paths["training_gcs_uri"],
            "validation_gcs_uri": staged_paths["validation_gcs_uri"],
            "gcs_prefix": gcs_prefix,
            "tuned_model_display_name": display_name,
            "description": args.description,
            "hyperparameters": {
                "epoch_count": _parse_optional_positive_int(args.epochs, field="epochs"),
                "batch_size": _parse_optional_positive_int(
                    args.batch_size,
                    field="batch_size",
                ),
                "learning_rate": _parse_optional_positive_float(
                    args.learning_rate,
                    field="learning_rate",
                ),
                "learning_rate_multiplier": _parse_optional_positive_float(
                    args.learning_rate_multiplier,
                    field="learning_rate_multiplier",
                ),
                "tuning_mode": args.tuning_mode,
                "adapter_size": args.adapter_size,
            },
            "fine_tuned_model": getattr(getattr(job, "tuned_model", None), "model", None),
            "tuned_model_endpoint": getattr(
                getattr(job, "tuned_model", None),
                "endpoint",
                None,
            ),
            "job": to_jsonable(job),
            "events": [],
        }
    )

    print(f"Submitted job: {record['job_id']}")
    print(f"Snapshot: {record['snapshot_path']}")

    if not args.wait:
        return 0

    final_job = wait_for_gemini_job(
        client,
        job_name=record["job_id"],
        poll_seconds=args.poll_seconds,
        max_wait_minutes=args.max_wait_minutes,
    )
    final_record = upsert_job_record(
        record,
        updates={
            "provider": "gemini",
            "method": "supervised",
            **gemini_job_to_record_updates(final_job),
        },
    )
    _print_job_record(final_record, show_events=False)

    if final_record.get("status") != "succeeded":
        raise RuntimeError(
            f"Gemini tuning job {record['job_id']} finished with status "
            f"{final_record.get('status', 'unknown')}."
        )

    return 0


def cmd_gemini_job(args: argparse.Namespace) -> int:
    job_id = _resolve_job_id(args.job_id, provider="gemini")
    existing_record = get_job_record(job_id, provider="gemini")

    if args.refresh or args.wait:
        project = args.project or (existing_record or {}).get("project")
        location = args.location or (existing_record or {}).get("location")
        client, context = get_gemini_client(project=project, location=location)
        job = (
            wait_for_gemini_job(
                client,
                job_name=job_id,
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
            )
            if args.wait
            else retrieve_tuning_job(client, job_name=job_id)
        )
        record = upsert_job_record(
            existing_record,
            updates={
                "provider": "gemini",
                "method": (existing_record or {}).get("method") or "supervised",
                "project": context.project,
                "location": context.location,
                **gemini_job_to_record_updates(job),
            },
        )
    else:
        if existing_record is None:
            raise ValueError(
                f"No local snapshot exists for Gemini job {job_id}. "
                "Use --refresh and pass --project/--location if needed."
            )
        record = existing_record

    _print_job_record(record, show_events=False)
    return 0


def cmd_gemini_models(_: argparse.Namespace) -> int:
    models = list_model_records(provider="gemini")
    if not models:
        print("No Gemini tuned models have been registered locally yet.")
        return 0

    for model in models:
        line = (
            f"{model['fine_tuned_model']} "
            f"(job={model['job_id']} base={model.get('base_model', '(unknown)')} "
            f"split={model.get('split', '(unknown)')} "
            f"project={model.get('project', '(unknown)')} "
            f"location={model.get('location', '(unknown)')})"
        )
        endpoint = model.get("tuned_model_endpoint")
        if endpoint:
            line += f" endpoint={endpoint}"
        if model.get("tag"):
            line += f" tag={model['tag']}"
        print(line)

    return 0


def cmd_gemini_jobs(_: argparse.Namespace) -> int:
    jobs = list_registered_jobs(provider="gemini")
    if not jobs:
        print("No Gemini tuning jobs have been registered locally yet.")
        return 0

    for job in jobs:
        line = (
            f"{job['job_id']} status={job.get('status', 'unknown')} "
            f"base={job.get('base_model', '(unknown)')} "
            f"split={job.get('split', '(unknown)')} "
            f"project={job.get('project', '(unknown)')} "
            f"location={job.get('location', '(unknown)')}"
        )
        if job.get("fine_tuned_model"):
            line += f" model={job['fine_tuned_model']}"
        if job.get("tag"):
            line += f" tag={job['tag']}"
        print(line)

    return 0


def _add_prepare_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    default_prepared_dir: Path,
    handler: Any,
    add_prepared_dir: bool = True,
) -> argparse.ArgumentParser:
    prepare = subparsers.add_parser(
        "prepare-data",
        help="Stage 1: build provider-compatible JSONL artifacts from prev and current datasets.",
    )
    if add_prepared_dir:
        prepare.add_argument(
            "--prepared-dir",
            default=str(default_prepared_dir),
            help="Output directory for prepared training artifacts.",
        )
    prepare.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.set_defaults(handler=handler)
    return prepare


def _add_submit_final_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handler: Any,
    provider_help: str,
    add_api_key: bool = False,
    add_google_context: bool = False,
) -> argparse.ArgumentParser:
    submit = subparsers.add_parser(
        "submit-final",
        help=(
            f"Run a {provider_help} model on datasets/final/, persist a reusable raw "
            "artifact under raw/, and write competition-formatted detections files."
        ),
    )
    submit.add_argument(
        "--model",
        required=True,
        help="Runnable fine-tuned model id to use for the final submission set.",
    )
    submit.add_argument(
        "--dataset-ids",
        default=None,
        help="Optional comma-separated final dataset ids. Defaults to all files under datasets/final/.",
    )
    if add_google_context:
        submit.add_argument(
            "--project",
            default=None,
            help="Google Cloud project id. Defaults to GOOGLE_CLOUD_PROJECT.",
        )
        submit.add_argument(
            "--location",
            default=None,
            help=(
                "Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or "
                f"{DEFAULT_GEMINI_LOCATION}."
            ),
        )
    submit.add_argument(
        "--team-name",
        default=DEFAULT_TEAM_NAME,
        help="Team name used to build submission file names.",
    )
    submit.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    submit.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw submission artifacts.",
    )
    submit.add_argument(
        "--output-dir",
        default=str(FINAL_RESULTS_DIR),
        help="Directory for final competition-formatted detection files.",
    )
    submit.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating the raw artifact path.",
    )
    submit.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    submit.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the submission run.",
    )
    submit.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    submit.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print submission progress every N completed examples.",
    )
    submit.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    if add_api_key:
        submit.add_argument(
            "--api-key",
            default=None,
            help="Optional OPENAI_API_KEY override.",
        )
    submit.set_defaults(handler=handler)
    return submit


def _build_openai_parser(provider_parser: argparse.ArgumentParser) -> None:
    subparsers = provider_parser.add_subparsers(dest="command", required=True)
    prepare = _add_prepare_data_parser(
        subparsers,
        default_prepared_dir=OPENAI_PREPARED_DIR,
        handler=cmd_openai_prepare_data,
        add_prepared_dir=False,
    )
    prepare.add_argument(
        "--method",
        default="all",
        choices=OPENAI_PREPARE_METHODS,
        help=(
            "Prepared artifact type to generate. "
            "'all' writes both supervised and reinforcement trees."
        ),
    )
    prepare.add_argument(
        "--prepared-dir",
        default=None,
        help=(
            "Optional output directory for a single OpenAI method. "
            "Cannot be combined with --method all."
        ),
    )

    train = subparsers.add_parser(
        "train",
        help="Stage 2: submit an OpenAI fine-tuning job from prepared artifacts.",
    )
    train.add_argument(
        "--method",
        default="supervised",
        choices=OPENAI_TRAINING_METHODS,
        help="OpenAI fine-tuning method to submit.",
    )
    train.add_argument(
        "--prepared-dir",
        default=None,
        help=(
            "Optional prepared artifact directory for the selected method. "
            "Defaults to the method-specific OpenAI prepared path."
        ),
    )
    train.add_argument(
        "--split",
        default=DEFAULT_PREPARED_SPLIT,
        choices=PREPARED_SPLITS,
        help="Prepared split to train from.",
    )
    train.add_argument(
        "--base-model",
        required=True,
        help="Base model to fine-tune. Can be any supported OpenAI fine-tunable model.",
    )
    train.add_argument("--tag", default=None, help="Optional local label for this run.")
    train.add_argument(
        "--epochs",
        default=None,
        help=(
            "Optional epoch count (integer or 'auto'). "
            f"Supervised defaults to {DEFAULT_OPENAI_SUPERVISED_EPOCHS} when omitted."
        ),
    )
    train.add_argument(
        "--batch-size",
        default=None,
        help=(
            "Fine-tune batch size (integer or 'auto'). "
            f"Supervised defaults to {DEFAULT_OPENAI_SUPERVISED_BATCH_SIZE} "
            "when omitted."
        ),
    )
    train.add_argument(
        "--learning-rate-multiplier",
        default=None,
        help=(
            "Learning rate multiplier (float or 'auto'). "
            "Supervised defaults to "
            f"{DEFAULT_OPENAI_SUPERVISED_LEARNING_RATE_MULTIPLIER:g} "
            "when omitted."
        ),
    )
    train.add_argument(
        "--reasoning-effort",
        default=DEFAULT_OPENAI_RFT_REASONING_EFFORT,
        choices=("low", "medium", "high"),
        help=(
            "Reinforcement reasoning effort. Ignored for supervised jobs. "
            f"Defaults to {DEFAULT_OPENAI_RFT_REASONING_EFFORT}."
        ),
    )
    train.add_argument(
        "--compute-multiplier",
        type=float,
        default=DEFAULT_OPENAI_RFT_COMPUTE_MULTIPLIER,
        help=(
            "Reinforcement compute multiplier. Ignored for supervised jobs. "
            f"Defaults to {DEFAULT_OPENAI_RFT_COMPUTE_MULTIPLIER:g}."
        ),
    )
    train.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Optional reinforcement evaluation interval.",
    )
    train.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Optional reinforcement evaluation sample count.",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="OpenAI fine-tuning seed.",
    )
    train.add_argument("--api-key", default=None, help="Optional OPENAI_API_KEY override.")
    train.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job reaches a terminal state.",
    )
    train.add_argument("--poll-seconds", type=float, default=30.0)
    train.add_argument("--max-wait-minutes", type=float, default=0.0)
    train.add_argument("--events-limit", type=int, default=10)
    train.set_defaults(handler=cmd_openai_train)

    job = subparsers.add_parser(
        "job",
        help="Inspect an OpenAI fine-tuning job. Defaults to the latest registered job.",
    )
    job.add_argument("--job-id", default=None)
    job.add_argument("--api-key", default=None, help="Optional OPENAI_API_KEY override.")
    job.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch the current job status from OpenAI before printing.",
    )
    job.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the job reaches a terminal state before printing.",
    )
    job.add_argument("--poll-seconds", type=float, default=30.0)
    job.add_argument("--max-wait-minutes", type=float, default=0.0)
    job.add_argument("--events-limit", type=int, default=10)
    job.set_defaults(handler=cmd_openai_job)

    models = subparsers.add_parser(
        "models",
        help="List OpenAI fine-tuned models that have succeeded locally.",
    )
    models.set_defaults(handler=cmd_openai_models)

    jobs = subparsers.add_parser(
        "jobs",
        help="List locally registered OpenAI fine-tuning jobs.",
    )
    jobs.set_defaults(handler=cmd_openai_jobs)

    infer_raw = subparsers.add_parser(
        "infer-raw",
        help=(
            "Run a model against a full labeled dataset collection and persist "
            "raw per-user verdicts plus detected bot ids under raw/."
        ),
    )
    infer_raw.add_argument(
        "--model",
        required=True,
        help="Model id to evaluate, including fine-tuned model ids.",
    )
    infer_target = infer_raw.add_mutually_exclusive_group()
    infer_target.add_argument(
        "--collection",
        default=None,
        choices=(*DATASET_COLLECTIONS, "full"),
        help="Dataset collection to evaluate. Defaults to 'both'. 'full' aliases to 'both'.",
    )
    infer_target.add_argument(
        "--dataset-ids",
        default=None,
        help="Comma-separated full dataset ids to evaluate, for example 30,31,32,33.",
    )
    infer_raw.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    infer_raw.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw evaluation artifacts.",
    )
    infer_raw.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating the raw artifact path.",
    )
    infer_raw.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    infer_raw.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the evaluation.",
    )
    infer_raw.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    infer_raw.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print the current score every N completed examples.",
    )
    infer_raw.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    infer_raw.add_argument(
        "--api-key",
        default=None,
        help="Optional OPENAI_API_KEY override.",
    )
    infer_raw.set_defaults(handler=cmd_openai_infer_raw)

    score_raw = subparsers.add_parser(
        "score-raw",
        help=(
            "Generate a run report from an existing raw artifact and the full "
            "ground-truth dataset without rerunning inference."
        ),
    )
    score_raw.add_argument(
        "--raw-path",
        required=True,
        help="Path to a raw evaluation artifact under raw/.",
    )
    score_target = score_raw.add_mutually_exclusive_group()
    score_target.add_argument(
        "--collection",
        default=None,
        choices=(*DATASET_COLLECTIONS, "full"),
        help="Optional dataset collection override. Defaults to the raw file metadata.",
    )
    score_target.add_argument(
        "--dataset-ids",
        default=None,
        help="Comma-separated full dataset ids to score against.",
    )
    score_raw.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the generated run report.",
    )
    score_raw.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory for generated human-readable run reports.",
    )
    score_raw.set_defaults(handler=cmd_openai_score_raw)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Run inference to a raw artifact, then score that raw artifact into "
            "a human-readable run report."
        ),
    )
    evaluate.add_argument(
        "--model",
        required=True,
        help="Model id to evaluate, including fine-tuned model ids.",
    )
    evaluate_target = evaluate.add_mutually_exclusive_group()
    evaluate_target.add_argument(
        "--collection",
        default=None,
        choices=(*DATASET_COLLECTIONS, "full"),
        help="Dataset collection to evaluate. Defaults to 'both'. 'full' aliases to 'both'.",
    )
    evaluate_target.add_argument(
        "--dataset-ids",
        default=None,
        help="Comma-separated full dataset ids to evaluate, for example 30,31,32,33.",
    )
    evaluate.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    evaluate.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw evaluation artifacts.",
    )
    evaluate.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory for generated human-readable run reports.",
    )
    evaluate.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the generated run report.",
    )
    evaluate.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating raw/report paths.",
    )
    evaluate.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    evaluate.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the evaluation.",
    )
    evaluate.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    evaluate.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print the current score every N completed examples.",
    )
    evaluate.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    evaluate.add_argument(
        "--api-key",
        default=None,
        help="Optional OPENAI_API_KEY override.",
    )
    evaluate.set_defaults(handler=cmd_openai_evaluate)

    _add_submit_final_parser(
        subparsers,
        handler=cmd_openai_submit_final,
        provider_help="OpenAI",
        add_api_key=True,
    )


def _build_gemini_parser(provider_parser: argparse.ArgumentParser) -> None:
    subparsers = provider_parser.add_subparsers(dest="command", required=True)
    _add_prepare_data_parser(
        subparsers,
        default_prepared_dir=GEMINI_PREPARED_DIR,
        handler=cmd_gemini_prepare_data,
    )

    train = subparsers.add_parser(
        "train",
        help="Stage 2: stage prepared JSONL to GCS and submit a Gemini tuning job.",
    )
    train.add_argument("--prepared-dir", default=str(GEMINI_PREPARED_DIR))
    train.add_argument(
        "--split",
        default=DEFAULT_PREPARED_SPLIT,
        choices=PREPARED_SPLITS,
        help="Prepared split to train from.",
    )
    train.add_argument(
        "--base-model",
        required=True,
        help="Base Gemini model to fine-tune on Vertex AI.",
    )
    train.add_argument(
        "--project",
        default=None,
        help="Google Cloud project id. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    train.add_argument(
        "--location",
        default=None,
        help=(
            "Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or "
            f"{DEFAULT_GEMINI_LOCATION}."
        ),
    )
    train.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket or URI prefix used to stage training files, for example gs://my-bucket.",
    )
    train.add_argument(
        "--gcs-prefix",
        default=None,
        help="Optional path prefix inside the bucket. Defaults to a timestamped bot-or-not path.",
    )
    train.add_argument(
        "--display-name",
        default=None,
        help="Vertex tuned model display name. Defaults to an auto-generated name.",
    )
    train.add_argument(
        "--description",
        default=None,
        help="Optional Vertex tuning job description.",
    )
    train.add_argument("--tag", default=None, help="Optional local label for this run.")
    train.add_argument("--epochs", default=None, help="Optional epoch count.")
    train.add_argument("--batch-size", default=None, help="Optional batch size.")
    learning_rate_group = train.add_mutually_exclusive_group()
    learning_rate_group.add_argument(
        "--learning-rate",
        default=None,
        help="Optional explicit learning rate.",
    )
    learning_rate_group.add_argument(
        "--learning-rate-multiplier",
        default=None,
        help="Optional learning rate multiplier.",
    )
    train.add_argument(
        "--tuning-mode",
        default=None,
        choices=("full", "adapter"),
        help="Optional tuning mode. Use 'adapter' for PEFT adapter tuning.",
    )
    train.add_argument(
        "--adapter-size",
        default=None,
        choices=("1", "2", "4", "8", "16", "32"),
        help="Adapter size for PEFT adapter tuning.",
    )
    train.add_argument(
        "--wait",
        action="store_true",
        help="Block until the Vertex tuning job reaches a terminal state.",
    )
    train.add_argument("--poll-seconds", type=float, default=30.0)
    train.add_argument("--max-wait-minutes", type=float, default=0.0)
    train.set_defaults(handler=cmd_gemini_train)

    job = subparsers.add_parser(
        "job",
        help="Inspect a Gemini tuning job. Defaults to the latest registered job.",
    )
    job.add_argument("--job-id", default=None)
    job.add_argument(
        "--project",
        default=None,
        help="Google Cloud project id. Needed when refreshing a job that has no local snapshot.",
    )
    job.add_argument(
        "--location",
        default=None,
        help="Vertex AI region. Needed when refreshing a job that has no local snapshot.",
    )
    job.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch the current job status from Vertex AI before printing.",
    )
    job.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the job reaches a terminal state before printing.",
    )
    job.add_argument("--poll-seconds", type=float, default=30.0)
    job.add_argument("--max-wait-minutes", type=float, default=0.0)
    job.set_defaults(handler=cmd_gemini_job)

    models = subparsers.add_parser(
        "models",
        help="List Gemini tuned models that have succeeded locally.",
    )
    models.set_defaults(handler=cmd_gemini_models)

    jobs = subparsers.add_parser(
        "jobs",
        help="List locally registered Gemini tuning jobs.",
    )
    jobs.set_defaults(handler=cmd_gemini_jobs)

    infer_raw = subparsers.add_parser(
        "infer-raw",
        help=(
            "Run a Gemini tuned model against a full labeled dataset collection "
            "and persist raw per-user verdicts plus detected bot ids under raw/."
        ),
    )
    infer_raw.add_argument(
        "--model",
        required=True,
        help=(
            "Runnable Gemini model id. Prefer a tuned model endpoint resource "
            "name or tuned model resource name."
        ),
    )
    infer_target = infer_raw.add_mutually_exclusive_group()
    infer_target.add_argument(
        "--collection",
        default=None,
        choices=(*DATASET_COLLECTIONS, "full"),
        help="Dataset collection to evaluate. Defaults to 'both'. 'full' aliases to 'both'.",
    )
    infer_target.add_argument(
        "--dataset-ids",
        default=None,
        help="Comma-separated full dataset ids to evaluate, for example 30,31,32,33.",
    )
    infer_raw.add_argument(
        "--project",
        default=None,
        help="Google Cloud project id. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    infer_raw.add_argument(
        "--location",
        default=None,
        help=(
            "Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or "
            f"{DEFAULT_GEMINI_LOCATION}."
        ),
    )
    infer_raw.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    infer_raw.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw evaluation artifacts.",
    )
    infer_raw.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating the raw artifact path.",
    )
    infer_raw.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    infer_raw.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the evaluation.",
    )
    infer_raw.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    infer_raw.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print the current score every N completed examples.",
    )
    infer_raw.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    infer_raw.set_defaults(handler=cmd_gemini_infer_raw)

    score_raw = subparsers.add_parser(
        "score-raw",
        help=(
            "Generate a run report from an existing Gemini raw artifact and the "
            "full ground-truth dataset without rerunning inference."
        ),
    )
    score_raw.add_argument(
        "--raw-path",
        required=True,
        help="Path to a raw evaluation artifact under raw/.",
    )
    score_target = score_raw.add_mutually_exclusive_group()
    score_target.add_argument(
        "--collection",
        default=None,
        choices=(*DATASET_COLLECTIONS, "full"),
        help="Optional dataset collection override. Defaults to the raw file metadata.",
    )
    score_target.add_argument(
        "--dataset-ids",
        default=None,
        help="Comma-separated full dataset ids to score against.",
    )
    score_raw.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the generated run report.",
    )
    score_raw.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory for generated human-readable run reports.",
    )
    score_raw.set_defaults(handler=cmd_gemini_score_raw)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Run Gemini inference to a raw artifact, then score that raw "
            "artifact into a human-readable run report."
        ),
    )
    evaluate.add_argument(
        "--model",
        required=True,
        help=(
            "Runnable Gemini model id. Prefer a tuned model endpoint resource "
            "name or tuned model resource name."
        ),
    )
    evaluate_target = evaluate.add_mutually_exclusive_group()
    evaluate_target.add_argument(
        "--collection",
        default=None,
        choices=(*DATASET_COLLECTIONS, "full"),
        help="Dataset collection to evaluate. Defaults to 'both'. 'full' aliases to 'both'.",
    )
    evaluate_target.add_argument(
        "--dataset-ids",
        default=None,
        help="Comma-separated full dataset ids to evaluate, for example 30,31,32,33.",
    )
    evaluate.add_argument(
        "--project",
        default=None,
        help="Google Cloud project id. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    evaluate.add_argument(
        "--location",
        default=None,
        help=(
            "Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or "
            f"{DEFAULT_GEMINI_LOCATION}."
        ),
    )
    evaluate.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    evaluate.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw evaluation artifacts.",
    )
    evaluate.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory for generated human-readable run reports.",
    )
    evaluate.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the generated run report.",
    )
    evaluate.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating raw/report paths.",
    )
    evaluate.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    evaluate.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the evaluation.",
    )
    evaluate.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    evaluate.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print the current score every N completed examples.",
    )
    evaluate.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    evaluate.set_defaults(handler=cmd_gemini_evaluate)

    _add_submit_final_parser(
        subparsers,
        handler=cmd_gemini_submit_final,
        provider_help="Gemini / Vertex AI",
        add_google_context=True,
    )


def _add_common_wizard_args(
    command_parser: argparse.ArgumentParser,
    *,
    include_workflow: bool,
    include_report_paths: bool,
) -> None:
    if include_workflow:
        command_parser.add_argument(
            "--workflow",
            default=None,
            choices=("evaluate", "submit-final"),
            help="Optional workflow override. When omitted, the wizard asks.",
        )
    command_parser.add_argument(
        "--team-name",
        default=DEFAULT_TEAM_NAME,
        help="Team name used for final submission file names.",
    )
    command_parser.add_argument(
        "--project",
        default=None,
        help="Optional Google Cloud project id for Vertex model refresh/evaluation.",
    )
    command_parser.add_argument(
        "--location",
        default=None,
        help=(
            "Optional Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or "
            f"{DEFAULT_GEMINI_LOCATION}."
        ),
    )
    command_parser.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    command_parser.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw evaluation artifacts.",
    )
    command_parser.add_argument(
        "--final-results-dir",
        default=str(FINAL_RESULTS_DIR),
        help="Directory for final competition-formatted detection files.",
    )
    if include_report_paths:
        command_parser.add_argument(
            "--runs-dir",
            default=str(RUNS_DIR),
            help="Directory for generated human-readable run reports.",
        )
        command_parser.add_argument(
            "--report-path",
            default=None,
            help="Optional explicit path for the generated run report.",
        )
    command_parser.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating raw/report paths.",
    )
    command_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    command_parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the evaluation.",
    )
    command_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    command_parser.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print the current score every N completed examples.",
    )
    command_parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    command_parser.add_argument(
        "--api-key",
        default=None,
        help="Optional OPENAI_API_KEY override.",
    )


def _build_interactive_parser(command_parser: argparse.ArgumentParser) -> None:
    _add_common_wizard_args(
        command_parser,
        include_workflow=True,
        include_report_paths=True,
    )
    command_parser.set_defaults(handler=cmd_interactive)


def _build_submit_parser(command_parser: argparse.ArgumentParser) -> None:
    _add_common_wizard_args(
        command_parser,
        include_workflow=False,
        include_report_paths=False,
    )
    command_parser.set_defaults(handler=cmd_submit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fine-tuning",
        description=(
            "Prepare provider-specific datasets from the prev/current corpora "
            "and manage OpenAI or Gemini fine-tuning jobs for bot-or-not."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    interactive = commands.add_parser(
        "interactive",
        help=(
            "Open the provider-neutral interactive evaluator for OpenAI and "
            "Gemini fine-tuned models."
        ),
    )
    _build_interactive_parser(interactive)

    submit = commands.add_parser(
        "submit",
        help=(
            "Open the provider-neutral interactive wizard for generating final "
            "competition detections from datasets/final/."
        ),
    )
    _build_submit_parser(submit)

    openai = commands.add_parser(
        "openai",
        help="Prepare data and manage OpenAI fine-tuning jobs.",
    )
    _build_openai_parser(openai)

    gemini = commands.add_parser(
        "gemini",
        help="Prepare data and manage Gemini supervised tuning on Vertex AI.",
    )
    _build_gemini_parser(gemini)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
