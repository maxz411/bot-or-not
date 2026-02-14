from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    ARTIFACTS_ROOT,
    CV_RUNS_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET_IDS,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    FINAL_DIR,
    MODELS_REGISTRY_PATH,
    PAIR_FOLDS,
    PREPARED_DIR,
    RUNS_DIR,
)
from .data import (
    EvalJSONLRecord,
    examples_summary,
    filter_examples_by_dataset,
    load_examples,
    overlap_size,
    read_eval_examples,
    stratified_split,
    write_eval_examples,
    write_metadata,
    write_sft_examples,
)
from .metrics import ClassificationPoint, Metrics, compute_metrics


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _parse_dataset_ids(raw: str) -> tuple[int, ...]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("No dataset ids provided")
    return tuple(int(part) for part in parts)


def _parse_optional_int_or_auto(value: str | None, *, field: str) -> int | str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if text.lower() == "auto":
        return "auto"
    parsed = int(text)
    if parsed <= 0:
        raise ValueError(f"{field} must be > 0 or 'auto'")
    return parsed


def _parse_optional_float_or_auto(
    value: str | None, *, field: str
) -> float | str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if text.lower() == "auto":
        return "auto"
    parsed = float(text)
    if parsed <= 0:
        raise ValueError(f"{field} must be > 0 or 'auto'")
    return parsed


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_registry() -> dict[str, Any]:
    if MODELS_REGISTRY_PATH.exists():
        return _load_json(MODELS_REGISTRY_PATH)
    return {
        "updated_at": _now_iso(),
        "cv_runs": [],
        "final_jobs": [],
        "latest_final_model": None,
    }


def _save_registry(registry: dict[str, Any]) -> None:
    registry["updated_at"] = _now_iso()
    _save_json(MODELS_REGISTRY_PATH, registry)


def print_metrics(prefix: str, metrics: Metrics) -> None:
    print(
        (
            f"{prefix}: total={metrics.total} bots={metrics.bots} humans={metrics.humans} "
            f"| TP={metrics.tp} TN={metrics.tn} FP={metrics.fp} FN={metrics.fn} "
            f"| acc={metrics.accuracy:.2f}% score={metrics.score}/{metrics.max_score} ({metrics.pct_max:.1f}%)"
        )
    )


def _split_write(
    *,
    split_dir: Path,
    train_examples: list[Any],
    val_examples: list[Any],
    test_examples: list[Any] | None = None,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.jsonl"
    train_meta_path = split_dir / "train.meta.jsonl"
    val_path = split_dir / "val.jsonl"
    val_meta_path = split_dir / "val.meta.jsonl"

    write_sft_examples(train_path, train_examples)
    write_metadata(train_meta_path, train_examples)
    write_sft_examples(val_path, val_examples)
    write_metadata(val_meta_path, val_examples)

    out: dict[str, Any] = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "train_val_overlap": overlap_size(train_examples, val_examples),
    }

    if test_examples is not None:
        test_eval_path = split_dir / "test.eval.jsonl"
        test_meta_path = split_dir / "test.meta.jsonl"
        write_eval_examples(test_eval_path, test_examples)
        write_metadata(test_meta_path, test_examples)
        out.update(
            {
                "test_eval_path": str(test_eval_path),
                "test_count": len(test_examples),
                "train_test_overlap": overlap_size(train_examples, test_examples),
                "val_test_overlap": overlap_size(val_examples, test_examples),
            }
        )

    return out


def prepare_data(
    datasets: tuple[int, ...],
    prepared_dir: Path,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Prepare SFT/eval JSONL files and pair-aware splits. Returns the summary dict."""
    prepared_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(datasets)
    summary = examples_summary(examples)

    # Challenge sanity check for the default 4 datasets.
    if set(datasets) == set(DEFAULT_DATASET_IDS):
        expected_total = 889
        expected_bots = 184
        expected_humans = 705
        if summary["total_examples"] != expected_total:
            raise ValueError(
                f"Unexpected total examples: {summary['total_examples']} (expected {expected_total})"
            )
        if summary["labels"].get("BOT", 0) != expected_bots:
            raise ValueError(
                f"Unexpected BOT count: {summary['labels'].get('BOT', 0)} (expected {expected_bots})"
            )
        if summary["labels"].get("HUMAN", 0) != expected_humans:
            raise ValueError(
                f"Unexpected HUMAN count: {summary['labels'].get('HUMAN', 0)} (expected {expected_humans})"
            )

    all_dir = prepared_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    write_sft_examples(all_dir / "all.jsonl", examples)
    write_metadata(all_dir / "all.meta.jsonl", examples)
    write_eval_examples(all_dir / "all.eval.jsonl", examples)

    folds_out: list[dict[str, Any]] = []
    for index, fold in enumerate(PAIR_FOLDS):
        fold_dir = prepared_dir / fold.name
        fold_train_pool = filter_examples_by_dataset(examples, fold.train_dataset_ids)
        fold_test = filter_examples_by_dataset(examples, fold.test_dataset_ids)
        fold_train, fold_val = stratified_split(
            fold_train_pool, val_fraction=val_fraction, seed=seed + index
        )
        written = _split_write(
            split_dir=fold_dir,
            train_examples=fold_train,
            val_examples=fold_val,
            test_examples=fold_test,
        )
        written.update(
            {
                "name": fold.name,
                "train_dataset_ids": list(fold.train_dataset_ids),
                "test_dataset_ids": list(fold.test_dataset_ids),
            }
        )
        folds_out.append(written)

    final_train, final_val = stratified_split(
        examples, val_fraction=val_fraction, seed=seed + 999
    )
    final_written = _split_write(
        split_dir=prepared_dir / "final",
        train_examples=final_train,
        val_examples=final_val,
        test_examples=None,
    )

    summary_payload = {
        "generated_at": _now_iso(),
        "datasets": list(datasets),
        "seed": seed,
        "val_fraction": val_fraction,
        "data_integrity": summary,
        "pair_folds": folds_out,
        "final_split": final_written,
    }
    _save_json(prepared_dir / "summary.json", summary_payload)

    print(f"Prepared data written to: {prepared_dir}")
    print(
        f"Data integrity: total={summary['total_examples']} BOT={summary['labels'].get('BOT', 0)} HUMAN={summary['labels'].get('HUMAN', 0)}"
    )
    print(f"No-truncation mismatches: {summary['no_truncation_mismatches']}")
    for fold in folds_out:
        print(
            f"{fold['name']}: train={fold['train_count']} val={fold['val_count']} test={fold['test_count']} "
            f"| overlaps train/val={fold['train_val_overlap']} train/test={fold['train_test_overlap']} val/test={fold['val_test_overlap']}"
        )
    print(
        f"final: train={final_written['train_count']} val={final_written['val_count']} overlap={final_written['train_val_overlap']}"
    )
    return summary_payload


def cmd_prepare_data(args: argparse.Namespace) -> int:
    prepare_data(
        datasets=_parse_dataset_ids(args.datasets),
        prepared_dir=Path(args.prepared_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    return 0


def evaluate_records(
    *,
    client: Any,
    model: str,
    records: list[EvalJSONLRecord],
    temperature: float,
    max_samples: int,
) -> dict[str, Any]:
    from .openai_api import classify_messages, normalize_label

    if max_samples > 0:
        records = records[:max_samples]

    points: list[ClassificationPoint] = []
    per_dataset_points: dict[int, list[ClassificationPoint]] = defaultdict(list)
    predicted_bot_ids: list[str] = []
    invalid_output_count = 0
    invalid_examples: list[dict[str, str]] = []

    for record in records:
        prediction, raw_output = classify_messages(
            client,
            model=model,
            messages=[message.to_dict() for message in record.messages],
            temperature=temperature,
        )
        if normalize_label(raw_output) is None:
            invalid_output_count += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(
                    {"user_id": record.user_id, "raw_output": raw_output}
                )

        if prediction == "BOT":
            predicted_bot_ids.append(record.user_id)

        point = ClassificationPoint(
            user_id=record.user_id,
            dataset_id=record.dataset_id,
            truth_label=record.label,
            predicted_label=prediction,
        )
        points.append(point)
        per_dataset_points[record.dataset_id].append(point)

    combined_metrics = compute_metrics(points)
    dataset_metrics = {
        str(dataset_id): compute_metrics(dataset_points).to_dict()
        for dataset_id, dataset_points in sorted(per_dataset_points.items())
    }
    return {
        "evaluated_at": _now_iso(),
        "model": model,
        "total_examples": len(records),
        "invalid_output_count": invalid_output_count,
        "invalid_examples": invalid_examples,
        "metrics": combined_metrics.to_dict(),
        "per_dataset_metrics": dataset_metrics,
        "predicted_bot_ids": predicted_bot_ids,
    }


def run_cv(
    prepared_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    *,
    api_key: str | None = None,
    n_epochs: int | str | None = "auto",
    batch_size: int | str | None = None,
    learning_rate_multiplier: float | str | None = None,
    temperature: float = 0.0,
    poll_seconds: float = 30.0,
    max_wait_minutes: float = 0.0,
    max_samples: int = 0,
    skip_baseline: bool = False,
    dry_run: bool = False,
    no_wait: bool = False,
) -> dict[str, Any]:
    """Run strict pair-holdout CV fine-tuning and evaluate. Returns the CV summary dict."""
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared dir not found: {prepared_dir}")

    if dry_run:
        summary_path = prepared_dir / "summary.json"
        if summary_path.exists():
            summary = _load_json(summary_path)
            print(json.dumps(summary, indent=2))
            return summary
        raise FileNotFoundError(
            f"Missing {summary_path}; run prepare-data first."
        )

    from .openai_api import (
        create_fine_tuning_job,
        get_client,
        upload_file,
        wait_for_fine_tuning_job,
    )

    client = get_client(api_key=api_key)

    run_slug = f"cv-{_timestamp_slug()}"
    run_dir = CV_RUNS_DIR / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[dict[str, Any]] = []
    for fold in PAIR_FOLDS:
        fold_dir = prepared_dir / fold.name
        train_path = fold_dir / "train.jsonl"
        val_path = fold_dir / "val.jsonl"
        test_eval_path = fold_dir / "test.eval.jsonl"
        for required in (train_path, val_path, test_eval_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required file: {required}")

        print(f"Submitting {fold.name} fine-tuning job...")
        training_file_id = upload_file(client, train_path)
        validation_file_id = upload_file(client, val_path)
        job = create_fine_tuning_job(
            client,
            model=base_model,
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
        )

        fold_report: dict[str, Any] = {
            "fold": fold.name,
            "train_dataset_ids": list(fold.train_dataset_ids),
            "test_dataset_ids": list(fold.test_dataset_ids),
            "base_model": base_model,
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
            "job_initial": _to_jsonable(job),
        }

        if no_wait:
            fold_report["status"] = "submitted"
            fold_reports.append(fold_report)
            _save_json(run_dir / f"{fold.name}.json", fold_report)
            continue

        final_job = wait_for_fine_tuning_job(
            client,
            job_id=job.id,
            poll_seconds=poll_seconds,
            max_wait_minutes=max_wait_minutes,
        )
        fold_report["job_final"] = _to_jsonable(final_job)
        fold_report["status"] = getattr(final_job, "status", "")

        if fold_report["status"] != "succeeded":
            fold_reports.append(fold_report)
            _save_json(run_dir / f"{fold.name}.json", fold_report)
            raise RuntimeError(
                f"{fold.name} fine-tuning job failed with status {fold_report['status']}"
            )

        tuned_model = getattr(final_job, "fine_tuned_model", None)
        if not tuned_model:
            raise RuntimeError(f"{fold.name} job succeeded but no fine_tuned_model was returned")

        eval_records = read_eval_examples(test_eval_path)
        tuned_eval = evaluate_records(
            client=client,
            model=tuned_model,
            records=eval_records,
            temperature=temperature,
            max_samples=max_samples,
        )
        tuned_metrics = Metrics(**tuned_eval["metrics"])
        print_metrics(f"{fold.name} tuned", tuned_metrics)

        baseline_eval = None
        if not skip_baseline:
            baseline_eval = evaluate_records(
                client=client,
                model=base_model,
                records=eval_records,
                temperature=temperature,
                max_samples=max_samples,
            )
            baseline_metrics = Metrics(**baseline_eval["metrics"])
            print_metrics(f"{fold.name} baseline", baseline_metrics)

        non_regression_pass = (
            True
            if baseline_eval is None
            else tuned_eval["metrics"]["score"] >= baseline_eval["metrics"]["score"]
        )

        fold_report.update(
            {
                "tuned_model": tuned_model,
                "tuned_eval": tuned_eval,
                "baseline_eval": baseline_eval,
                "non_regression_pass": non_regression_pass,
            }
        )
        fold_reports.append(fold_report)
        _save_json(run_dir / f"{fold.name}.json", fold_report)

    summary: dict[str, Any] = {
        "run_id": run_slug,
        "created_at": _now_iso(),
        "base_model": base_model,
        "no_wait": no_wait,
        "fold_reports": fold_reports,
    }

    completed = [
        report for report in fold_reports if report.get("tuned_eval") is not None
    ]
    if completed:
        tuned_scores = [report["tuned_eval"]["metrics"]["score"] for report in completed]
        baseline_scores = [
            report["baseline_eval"]["metrics"]["score"]
            for report in completed
            if report.get("baseline_eval") is not None
        ]
        summary["selection"] = {
            "mean_tuned_score": sum(tuned_scores) / len(tuned_scores),
            "mean_baseline_score": (
                sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
            ),
            "all_non_regression_pass": all(
                bool(report.get("non_regression_pass", True)) for report in completed
            ),
            "selected_configuration": {
                "base_model": base_model,
                "hyperparameters": {
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": learning_rate_multiplier,
                },
            },
        }

    _save_json(run_dir / "summary.json", summary)

    registry = _load_registry()
    registry.setdefault("cv_runs", []).append(
        {
            "run_id": run_slug,
            "created_at": summary["created_at"],
            "base_model": base_model,
            "run_dir": str(run_dir),
            "selection": summary.get("selection"),
            "folds": [
                {
                    "name": report["fold"],
                    "status": report.get("status"),
                    "tuned_model": report.get("tuned_model"),
                    "non_regression_pass": report.get("non_regression_pass"),
                }
                for report in fold_reports
            ],
        }
    )
    _save_registry(registry)

    print(f"CV summary written: {run_dir / 'summary.json'}")
    return summary


def cmd_run_cv(args: argparse.Namespace) -> int:
    run_cv(
        prepared_dir=Path(args.prepared_dir),
        base_model=args.base_model,
        api_key=args.api_key,
        n_epochs=_parse_optional_int_or_auto(args.epochs, field="epochs"),
        batch_size=_parse_optional_int_or_auto(args.batch_size, field="batch_size"),
        learning_rate_multiplier=_parse_optional_float_or_auto(
            args.learning_rate_multiplier, field="learning_rate_multiplier"
        ),
        temperature=args.temperature,
        poll_seconds=args.poll_seconds,
        max_wait_minutes=args.max_wait_minutes,
        max_samples=args.max_samples,
        skip_baseline=args.skip_baseline,
        dry_run=args.dry_run,
        no_wait=args.no_wait,
    )
    return 0


def train_final(
    prepared_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    *,
    api_key: str | None = None,
    n_epochs: int | str | None = "auto",
    batch_size: int | str | None = None,
    learning_rate_multiplier: float | str | None = None,
    poll_seconds: float = 30.0,
    max_wait_minutes: float = 0.0,
    dry_run: bool = False,
    no_wait: bool = False,
) -> dict[str, Any]:
    """Train final model on all datasets. Returns the job payload dict."""
    final_split_dir = prepared_dir / "final"
    train_path = final_split_dir / "train.jsonl"
    val_path = final_split_dir / "val.jsonl"
    for required in (train_path, val_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    if dry_run:
        print(f"Would submit final train job using: {train_path} + {val_path}")
        return {"status": "dry_run", "train_path": str(train_path), "val_path": str(val_path)}

    from .openai_api import (
        create_fine_tuning_job,
        get_client,
        upload_file,
        wait_for_fine_tuning_job,
    )

    client = get_client(api_key=api_key)
    training_file_id = upload_file(client, train_path)
    validation_file_id = upload_file(client, val_path)
    job = create_fine_tuning_job(
        client,
        model=base_model,
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
    )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    job_slug = f"final-{_timestamp_slug()}"
    job_path = FINAL_DIR / f"{job_slug}.json"

    payload: dict[str, Any] = {
        "created_at": _now_iso(),
        "base_model": base_model,
        "hyperparameters": {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate_multiplier": learning_rate_multiplier,
        },
        "training_file_id": training_file_id,
        "validation_file_id": validation_file_id,
        "job_initial": _to_jsonable(job),
    }

    if no_wait:
        payload["status"] = "submitted"
        _save_json(job_path, payload)
        print(f"Submitted final training job {job.id}")
        print(f"Saved metadata: {job_path}")
        return payload

    final_job = wait_for_fine_tuning_job(
        client,
        job_id=job.id,
        poll_seconds=poll_seconds,
        max_wait_minutes=max_wait_minutes,
    )
    payload["job_final"] = _to_jsonable(final_job)
    payload["status"] = getattr(final_job, "status", "")

    if payload["status"] != "succeeded":
        _save_json(job_path, payload)
        raise RuntimeError(f"Final job failed with status {payload['status']}")

    fine_tuned_model = getattr(final_job, "fine_tuned_model", None)
    if not fine_tuned_model:
        _save_json(job_path, payload)
        raise RuntimeError("Final job succeeded but no fine_tuned_model returned")

    payload["fine_tuned_model"] = fine_tuned_model
    _save_json(job_path, payload)

    final_model_paths = [ARTIFACTS_ROOT / "final_model.txt", FINAL_DIR / "final_model.txt"]
    for path in final_model_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fine_tuned_model + "\n", encoding="utf-8")

    registry = _load_registry()
    registry.setdefault("final_jobs", []).append(
        {
            "created_at": _now_iso(),
            "base_model": base_model,
            "job_id": getattr(final_job, "id", None),
            "status": payload["status"],
            "fine_tuned_model": fine_tuned_model,
            "metadata_path": str(job_path),
        }
    )
    registry["latest_final_model"] = fine_tuned_model
    _save_registry(registry)

    print(f"Final model: {fine_tuned_model}")
    print(f"Saved: {ARTIFACTS_ROOT / 'final_model.txt'}")
    return payload


def cmd_train_final(args: argparse.Namespace) -> int:
    train_final(
        prepared_dir=Path(args.prepared_dir),
        base_model=args.base_model,
        api_key=args.api_key,
        n_epochs=_parse_optional_int_or_auto(args.epochs, field="epochs"),
        batch_size=_parse_optional_int_or_auto(args.batch_size, field="batch_size"),
        learning_rate_multiplier=_parse_optional_float_or_auto(
            args.learning_rate_multiplier, field="learning_rate_multiplier"
        ),
        poll_seconds=args.poll_seconds,
        max_wait_minutes=args.max_wait_minutes,
        dry_run=args.dry_run,
        no_wait=args.no_wait,
    )
    return 0


def _model_for_run_file(model: str) -> str:
    return model if model.startswith("openai/") else f"openai/{model}"


def _write_run_file(
    *,
    detector: str,
    model: str,
    dataset_ids: list[int],
    predicted_bot_ids: list[str],
    run_tag: str,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{run_tag}-{ '_'.join(str(x) for x in dataset_ids)}-{_timestamp_slug()}.txt"
    run_path = RUNS_DIR / file_name
    lines = [
        f"Datasets: {', '.join(str(x) for x in dataset_ids)}",
        f"Detector: {detector}",
        f"Model: {_model_for_run_file(model)}",
        "Batch size: 1",
        "",
        *predicted_bot_ids,
    ]
    run_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_path


def _resolve_model(model: str | None) -> str:
    if model:
        return model
    final_model_path = ARTIFACTS_ROOT / "final_model.txt"
    if final_model_path.exists():
        return final_model_path.read_text(encoding="utf-8").strip()
    raise ValueError("model is required (or create artifacts/fine_tuning/final_model.txt)")


def eval_model(
    model: str,
    datasets: tuple[int, ...] = DEFAULT_DATASET_IDS,
    *,
    eval_file: Path | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_samples: int = 0,
    output: Path | None = None,
    write_run_file: bool = False,
    run_tag: str = "v5",
    detector_name: str = "v5",
) -> dict[str, Any]:
    """Evaluate a model and optionally emit a run file. Returns the evaluation report dict."""
    from .openai_api import get_client

    client = get_client(api_key=api_key)

    if eval_file:
        eval_records = read_eval_examples(eval_file)
    else:
        examples = load_examples(datasets)
        eval_records = [example.to_eval_record() for example in examples]

    report = evaluate_records(
        client=client,
        model=model,
        records=eval_records,
        temperature=temperature,
        max_samples=max_samples,
    )
    metrics = Metrics(**report["metrics"])
    print_metrics("combined", metrics)
    print(f"Invalid outputs: {report['invalid_output_count']}")

    output_path = output or ARTIFACTS_ROOT / "last_eval.json"
    _save_json(output_path, report)
    print(f"Saved eval report: {output_path}")

    if write_run_file:
        run_path = _write_run_file(
            detector=detector_name,
            model=model,
            dataset_ids=list(datasets),
            predicted_bot_ids=report["predicted_bot_ids"],
            run_tag=run_tag,
        )
        print(f"Saved run file: {run_path}")

    return report


def cmd_eval_model(args: argparse.Namespace) -> int:
    model = _resolve_model(args.model)
    eval_model(
        model=model,
        datasets=_parse_dataset_ids(args.datasets),
        eval_file=Path(args.eval_file) if args.eval_file else None,
        api_key=args.api_key,
        temperature=args.temperature,
        max_samples=args.max_samples,
        output=Path(args.output) if args.output else None,
        write_run_file=args.write_run_file,
        run_tag=args.run_tag,
        detector_name=args.detector_name,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fine-tuning",
        description="v5 fine-tuning pipeline (strict pair holdout + full-post inputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data", help="Build SFT/eval JSONL files and pair-aware splits."
    )
    prepare.add_argument(
        "--datasets",
        default="30,31,32,33",
        help="Comma-separated dataset ids.",
    )
    prepare.add_argument(
        "--prepared-dir",
        default=str(PREPARED_DIR),
        help="Output directory for prepared artifacts.",
    )
    prepare.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.set_defaults(handler=cmd_prepare_data)

    run_cv = subparsers.add_parser(
        "run-cv",
        help="Run strict pair-holdout fine-tuning jobs and evaluate tuned models.",
    )
    run_cv.add_argument("--prepared-dir", default=str(PREPARED_DIR))
    run_cv.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    run_cv.add_argument(
        "--epochs",
        default="auto",
        help="Fine-tune epochs (integer or 'auto').",
    )
    run_cv.add_argument(
        "--batch-size",
        default=None,
        help="Fine-tune batch size (integer or 'auto'). Default: provider auto.",
    )
    run_cv.add_argument(
        "--learning-rate-multiplier",
        default=None,
        help="Learning rate multiplier (float or 'auto'). Default: provider auto.",
    )
    run_cv.add_argument("--api-key", default=None, help="Optional OPENAI API key override.")
    run_cv.add_argument("--temperature", type=float, default=0.0)
    run_cv.add_argument("--poll-seconds", type=float, default=30.0)
    run_cv.add_argument("--max-wait-minutes", type=float, default=0.0)
    run_cv.add_argument("--max-samples", type=int, default=0)
    run_cv.add_argument("--skip-baseline", action="store_true")
    run_cv.add_argument("--dry-run", action="store_true")
    run_cv.add_argument("--no-wait", action="store_true")
    run_cv.set_defaults(handler=cmd_run_cv)

    train_final = subparsers.add_parser(
        "train-final", help="Train final model on all datasets."
    )
    train_final.add_argument("--prepared-dir", default=str(PREPARED_DIR))
    train_final.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train_final.add_argument(
        "--epochs",
        default="auto",
        help="Fine-tune epochs (integer or 'auto').",
    )
    train_final.add_argument(
        "--batch-size",
        default=None,
        help="Fine-tune batch size (integer or 'auto'). Default: provider auto.",
    )
    train_final.add_argument(
        "--learning-rate-multiplier",
        default=None,
        help="Learning rate multiplier (float or 'auto'). Default: provider auto.",
    )
    train_final.add_argument("--api-key", default=None, help="Optional OPENAI API key override.")
    train_final.add_argument("--poll-seconds", type=float, default=30.0)
    train_final.add_argument("--max-wait-minutes", type=float, default=0.0)
    train_final.add_argument("--dry-run", action="store_true")
    train_final.add_argument("--no-wait", action="store_true")
    train_final.set_defaults(handler=cmd_train_final)

    eval_model = subparsers.add_parser(
        "eval-model",
        help="Evaluate a model and optionally emit a run file for JS analysis.",
    )
    eval_model.add_argument("--model", default=None, help="Model id (e.g. ft:...).")
    eval_model.add_argument("--datasets", default="30,31,32,33")
    eval_model.add_argument("--eval-file", default=None, help="Optional eval JSONL file.")
    eval_model.add_argument("--api-key", default=None, help="Optional OPENAI API key override.")
    eval_model.add_argument("--temperature", type=float, default=0.0)
    eval_model.add_argument("--max-samples", type=int, default=0)
    eval_model.add_argument("--output", default=None, help="JSON output path.")
    eval_model.add_argument("--write-run-file", action="store_true")
    eval_model.add_argument("--run-tag", default="v5")
    eval_model.add_argument("--detector-name", default="v5")
    eval_model.set_defaults(handler=cmd_eval_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
