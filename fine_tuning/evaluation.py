from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .constants import FINAL_RESULTS_DIR, RAW_DIR, RUNS_DIR
from .data import SubmissionExample, UserExample
from .openai_graders import (
    BOT_LABEL,
    HUMAN_LABEL,
    normalize_prediction_label,
    raw_bot_detection_reward,
)
from .storage import now_iso, now_slug, save_text

EvaluationCollection = str
PredictionLabel = Literal["BOT", "HUMAN"] | None
ErrorKind = Literal["false_positive", "false_negative", "invalid_output"]

PREDICTIONS_SECTION = "[predictions]"
DETECTED_BOT_IDS_SECTION = "[detected_bot_ids]"
INVALID_PREDICTION_LABEL = "INVALID"


def safe_model_id(model: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model)


def build_artifact_stem(
    *,
    provider: str,
    model: str,
    collection: EvaluationCollection,
    run_slug: str | None = None,
) -> str:
    return f"{provider}_{collection}_{safe_model_id(model)}_{run_slug or now_slug()}"


def build_raw_results_path(
    *,
    provider: str,
    model: str,
    collection: EvaluationCollection,
    run_slug: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    base_dir = output_dir or RAW_DIR
    stem = build_artifact_stem(
        provider=provider,
        model=model,
        collection=collection,
        run_slug=run_slug,
    )
    return base_dir / f"{stem}.txt"


def build_run_report_path(
    *,
    provider: str,
    model: str,
    collection: EvaluationCollection,
    run_slug: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    base_dir = output_dir or RUNS_DIR
    stem = build_artifact_stem(
        provider=provider,
        model=model,
        collection=collection,
        run_slug=run_slug,
    )
    return base_dir / f"{stem}.txt"


def build_run_report_path_for_raw(
    *,
    raw_results_path: Path,
    output_dir: Path | None = None,
) -> Path:
    base_dir = output_dir or RUNS_DIR
    return base_dir / raw_results_path.name


def safe_team_name(team_name: str) -> str:
    normalized = "".join(
        character.lower() for character in team_name.strip() if character.isalnum()
    )
    if not normalized:
        raise ValueError(f"Team name must contain at least one alphanumeric character: {team_name!r}")
    return normalized


@dataclass(frozen=True)
class RawPrediction:
    user_id: str
    predicted_label: PredictionLabel
    raw_output: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "predicted_label": self.predicted_label or INVALID_PREDICTION_LABEL,
            "raw_output": self.raw_output,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "RawPrediction":
        user_id = str(payload["user_id"])
        predicted_label = normalize_prediction_label(payload.get("predicted_label"))
        raw_output = str(payload.get("raw_output", ""))
        return cls(
            user_id=user_id,
            predicted_label=predicted_label,
            raw_output=raw_output,
        )


@dataclass
class RawEvaluationArtifact:
    provider: str
    model: str
    collection: EvaluationCollection
    dataset_ids: tuple[int, ...]
    total_examples: int
    created_at: str
    updated_at: str
    predictions_by_user: dict[str, RawPrediction] = field(default_factory=dict)

    @property
    def completed_examples(self) -> int:
        return len(self.predictions_by_user)

    @property
    def status(self) -> str:
        return "complete" if self.completed_examples == self.total_examples else "partial"

    @property
    def detected_bot_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    prediction.user_id
                    for prediction in self.predictions_by_user.values()
                    if prediction.predicted_label == BOT_LABEL
                ),
                key=_user_id_sort_key,
            )
        )

    def with_prediction(self, prediction: RawPrediction) -> None:
        self.predictions_by_user[prediction.user_id] = prediction
        self.updated_at = now_iso()


@dataclass
class RunningEvaluation:
    completed_examples: int = 0
    bots_seen: int = 0
    humans_seen: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    invalid_output_count: int = 0
    invalid_examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def raw_score(self) -> int:
        return (
            (self.tp * raw_bot_detection_reward(
                actual_label=BOT_LABEL,
                predicted_label=BOT_LABEL,
            ))
            + (self.fn * raw_bot_detection_reward(
                actual_label=BOT_LABEL,
                predicted_label=HUMAN_LABEL,
            ))
            + (self.fp * raw_bot_detection_reward(
                actual_label=HUMAN_LABEL,
                predicted_label=BOT_LABEL,
            ))
            + (self.tn * raw_bot_detection_reward(
                actual_label=HUMAN_LABEL,
                predicted_label=HUMAN_LABEL,
            ))
        )

    @property
    def max_possible_score(self) -> int:
        return self.bots_seen * raw_bot_detection_reward(
            actual_label=BOT_LABEL,
            predicted_label=BOT_LABEL,
        )

    @property
    def pct_of_max(self) -> float:
        if self.max_possible_score == 0:
            return 0.0
        return self.raw_score / self.max_possible_score

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "total": self.completed_examples,
            "bots": self.bots_seen,
            "humans": self.humans_seen,
            "asymmetric_score": self.raw_score,
            "asymmetric_max_score": self.max_possible_score,
            "asymmetric_pct_max": self.pct_of_max,
        }


@dataclass(frozen=True)
class EvaluationError:
    kind: ErrorKind
    user_id: str
    truth_label: str
    predicted_label: str
    raw_output: str
    dataset_id: int
    source: str
    lang: str
    post_count_used: int
    full_post_count: int
    user_prompt: str


@dataclass(frozen=True)
class ScoredEvaluation:
    provider: str
    model: str
    collection: EvaluationCollection
    dataset_ids: tuple[int, ...]
    total_examples: int
    raw_results_path: Path
    running: RunningEvaluation
    errors: tuple[EvaluationError, ...]


def _user_id_sort_key(user_id: str) -> tuple[int, str]:
    return (0, str(int(user_id))) if user_id.isdigit() else (1, user_id)


def _record_invalid_output(
    running: RunningEvaluation,
    *,
    user_id: str,
    raw_output: str,
) -> None:
    running.invalid_output_count += 1
    if len(running.invalid_examples) < 10:
        running.invalid_examples.append(
            {
                "user_id": user_id,
                "raw_output": raw_output,
            }
        )


def add_prediction_to_running_evaluation(
    running: RunningEvaluation,
    *,
    truth_label: str,
    predicted_label: PredictionLabel,
    raw_output: str,
    user_id: str,
) -> None:
    running.completed_examples += 1

    if truth_label == BOT_LABEL:
        running.bots_seen += 1
    elif truth_label == HUMAN_LABEL:
        running.humans_seen += 1
    else:
        raise ValueError(f"Unsupported truth label: {truth_label!r}")

    if predicted_label is None:
        _record_invalid_output(running, user_id=user_id, raw_output=raw_output)
        if truth_label == BOT_LABEL:
            running.fn += 1
            return
        if truth_label == HUMAN_LABEL:
            running.tn += 1
            return
        return

    if truth_label == BOT_LABEL and predicted_label == BOT_LABEL:
        running.tp += 1
        return
    if truth_label == BOT_LABEL and predicted_label == HUMAN_LABEL:
        running.fn += 1
        return
    if truth_label == HUMAN_LABEL and predicted_label == BOT_LABEL:
        running.fp += 1
        return
    if truth_label == HUMAN_LABEL and predicted_label == HUMAN_LABEL:
        running.tn += 1
        return

    raise ValueError(
        "Unsupported label combination: "
        f"truth={truth_label!r} predicted={predicted_label!r}."
    )


def initialize_raw_evaluation_artifact(
    *,
    provider: str,
    model: str,
    collection: EvaluationCollection,
    dataset_ids: tuple[int, ...],
    total_examples: int,
) -> RawEvaluationArtifact:
    timestamp = now_iso()
    return RawEvaluationArtifact(
        provider=provider,
        model=model,
        collection=collection,
        dataset_ids=dataset_ids,
        total_examples=total_examples,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _serialize_raw_results(
    artifact: RawEvaluationArtifact,
    *,
    user_id_order: list[str],
) -> str:
    ordered_predictions: list[RawPrediction] = []
    seen_user_ids: set[str] = set()
    for user_id in user_id_order:
        prediction = artifact.predictions_by_user.get(user_id)
        if prediction is None:
            continue
        ordered_predictions.append(prediction)
        seen_user_ids.add(user_id)

    remaining_predictions = sorted(
        (
            prediction
            for user_id, prediction in artifact.predictions_by_user.items()
            if user_id not in seen_user_ids
        ),
        key=lambda prediction: _user_id_sort_key(prediction.user_id),
    )
    ordered_predictions.extend(remaining_predictions)

    lines = [
        f"model: {artifact.model}",
        f"provider: {artifact.provider}",
        f"collection: {artifact.collection}",
        "dataset_ids: " + ",".join(str(dataset_id) for dataset_id in artifact.dataset_ids),
        f"total_examples: {artifact.total_examples}",
        f"completed_examples: {artifact.completed_examples}",
        f"status: {artifact.status}",
        f"created_at: {artifact.created_at}",
        f"updated_at: {artifact.updated_at}",
        "",
        PREDICTIONS_SECTION,
    ]
    lines.extend(
        json.dumps(prediction.to_json_dict(), ensure_ascii=False)
        for prediction in ordered_predictions
    )
    lines.extend(["", DETECTED_BOT_IDS_SECTION, *artifact.detected_bot_ids])
    return "\n".join(lines).rstrip() + "\n"


def save_raw_results(
    path: Path,
    artifact: RawEvaluationArtifact,
    *,
    user_id_order: list[str],
) -> None:
    serialized = _serialize_raw_results(artifact, user_id_order=user_id_order)
    save_text(path, serialized)


def _parse_header_value(
    header: dict[str, str],
    *,
    key: str,
) -> str:
    try:
        return header[key]
    except KeyError as exc:
        raise ValueError(f"Missing required raw results header: {key}") from exc


def load_raw_results(path: Path) -> RawEvaluationArtifact:
    if not path.exists():
        raise FileNotFoundError(f"Raw results file not found: {path}")

    header: dict[str, str] = {}
    predictions_by_user: dict[str, RawPrediction] = {}
    detected_bot_ids_from_file: set[str] = set()
    current_section: str | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line == PREDICTIONS_SECTION:
            current_section = PREDICTIONS_SECTION
            continue
        if line == DETECTED_BOT_IDS_SECTION:
            current_section = DETECTED_BOT_IDS_SECTION
            continue

        if current_section is None:
            if ":" not in raw_line:
                raise ValueError(
                    f"Invalid raw results header line {line_number} in {path}: {raw_line!r}"
                )
            key, value = raw_line.split(":", 1)
            header[key.strip()] = value.strip()
            continue

        if current_section == PREDICTIONS_SECTION:
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Prediction payload must be a JSON object on line {line_number}: {raw_line!r}"
                )
            prediction = RawPrediction.from_json_dict(payload)
            predictions_by_user[prediction.user_id] = prediction
            continue

        if current_section == DETECTED_BOT_IDS_SECTION:
            detected_bot_ids_from_file.add(line)
            continue

        raise ValueError(
            f"Unsupported raw results section {current_section!r} while reading {path}."
        )

    dataset_ids_text = _parse_header_value(header, key="dataset_ids")
    dataset_ids = tuple(
        int(dataset_id)
        for dataset_id in dataset_ids_text.split(",")
        if dataset_id.strip()
    )
    artifact = RawEvaluationArtifact(
        provider=_parse_header_value(header, key="provider"),
        model=_parse_header_value(header, key="model"),
        collection=_parse_header_value(header, key="collection"),
        dataset_ids=dataset_ids,
        total_examples=int(_parse_header_value(header, key="total_examples")),
        created_at=_parse_header_value(header, key="created_at"),
        updated_at=_parse_header_value(header, key="updated_at"),
        predictions_by_user=predictions_by_user,
    )

    completed_examples_text = header.get("completed_examples")
    if completed_examples_text is not None and int(completed_examples_text) != artifact.completed_examples:
        raise ValueError(
            "Raw results header completed_examples does not match parsed predictions: "
            f"header={completed_examples_text} parsed={artifact.completed_examples} path={path}"
        )

    status_text = header.get("status")
    if status_text is not None and status_text != artifact.status:
        raise ValueError(
            "Raw results header status does not match parsed predictions: "
            f"header={status_text!r} parsed={artifact.status!r} path={path}"
        )

    computed_detected_bot_ids = set(artifact.detected_bot_ids)
    if detected_bot_ids_from_file and detected_bot_ids_from_file != computed_detected_bot_ids:
        raise ValueError(
            "Raw results detected_bot_ids section does not match parsed predictions: "
            f"section={sorted(detected_bot_ids_from_file)} "
            f"computed={sorted(computed_detected_bot_ids)} path={path}"
        )

    return artifact


def validate_raw_results_against_examples(
    *,
    artifact: RawEvaluationArtifact,
    examples: list[UserExample],
) -> None:
    dataset_ids = tuple(sorted({example.dataset_id for example in examples}))
    if artifact.dataset_ids != dataset_ids:
        raise ValueError(
            "Raw results dataset ids do not match the dataset collection used for scoring: "
            f"raw={artifact.dataset_ids} expected={dataset_ids}"
        )
    if artifact.total_examples != len(examples):
        raise ValueError(
            "Raw results total_examples does not match dataset collection size: "
            f"raw={artifact.total_examples} expected={len(examples)}"
        )

    valid_user_ids = {example.user_id for example in examples}
    unknown_user_ids = sorted(set(artifact.predictions_by_user) - valid_user_ids, key=_user_id_sort_key)
    if unknown_user_ids:
        raise ValueError(
            "Raw results contain predictions for users not present in the scoring dataset: "
            + ", ".join(unknown_user_ids[:20])
        )


def validate_raw_results_against_submission_examples(
    *,
    artifact: RawEvaluationArtifact,
    examples: list[SubmissionExample],
) -> None:
    dataset_ids = tuple(sorted({example.dataset_id for example in examples}))
    if artifact.dataset_ids != dataset_ids:
        raise ValueError(
            "Raw results dataset ids do not match the submission dataset selection: "
            f"raw={artifact.dataset_ids} expected={dataset_ids}"
        )
    if artifact.total_examples != len(examples):
        raise ValueError(
            "Raw results total_examples does not match submission dataset size: "
            f"raw={artifact.total_examples} expected={len(examples)}"
        )

    valid_user_ids = {example.user_id for example in examples}
    unknown_user_ids = sorted(
        set(artifact.predictions_by_user) - valid_user_ids,
        key=_user_id_sort_key,
    )
    if unknown_user_ids:
        raise ValueError(
            "Raw results contain predictions for users not present in the submission dataset: "
            + ", ".join(unknown_user_ids[:20])
        )


def build_running_evaluation_from_raw_results(
    *,
    artifact: RawEvaluationArtifact,
    examples_by_user_id: dict[str, UserExample],
) -> RunningEvaluation:
    running = RunningEvaluation()
    for user_id in sorted(artifact.predictions_by_user, key=_user_id_sort_key):
        example = examples_by_user_id.get(user_id)
        if example is None:
            raise ValueError(f"Missing example for raw prediction user_id={user_id}")
        prediction = artifact.predictions_by_user[user_id]
        add_prediction_to_running_evaluation(
            running,
            truth_label=example.label,
            predicted_label=prediction.predicted_label,
            raw_output=prediction.raw_output,
            user_id=user_id,
        )
    return running


def _error_kind_for_prediction(
    *,
    truth_label: str,
    predicted_label: PredictionLabel,
) -> ErrorKind | None:
    if predicted_label is None:
        return "invalid_output"
    if truth_label == BOT_LABEL and predicted_label == HUMAN_LABEL:
        return "false_negative"
    if truth_label == HUMAN_LABEL and predicted_label == BOT_LABEL:
        return "false_positive"
    return None


def _error_rows(
    *,
    artifact: RawEvaluationArtifact,
    examples_by_user_id: dict[str, UserExample],
) -> tuple[EvaluationError, ...]:
    errors: list[EvaluationError] = []
    for user_id in sorted(artifact.predictions_by_user, key=_user_id_sort_key):
        example = examples_by_user_id.get(user_id)
        if example is None:
            raise ValueError(f"Missing example for raw prediction user_id={user_id}")
        prediction = artifact.predictions_by_user[user_id]
        error_kind = _error_kind_for_prediction(
            truth_label=example.label,
            predicted_label=prediction.predicted_label,
        )
        if error_kind is None:
            continue
        errors.append(
            EvaluationError(
                kind=error_kind,
                user_id=user_id,
                truth_label=example.label,
                predicted_label=prediction.predicted_label or INVALID_PREDICTION_LABEL,
                raw_output=prediction.raw_output,
                dataset_id=example.dataset_id,
                source=example.source,
                lang=example.lang,
                post_count_used=example.post_count_used,
                full_post_count=example.full_post_count,
                user_prompt=example.user_prompt,
            )
        )
    return tuple(errors)


def score_raw_results(
    *,
    artifact: RawEvaluationArtifact,
    raw_results_path: Path,
    examples: list[UserExample],
) -> ScoredEvaluation:
    validate_raw_results_against_examples(artifact=artifact, examples=examples)
    examples_by_user_id = {example.user_id: example for example in examples}
    running = build_running_evaluation_from_raw_results(
        artifact=artifact,
        examples_by_user_id=examples_by_user_id,
    )
    errors = _error_rows(
        artifact=artifact,
        examples_by_user_id=examples_by_user_id,
    )
    return ScoredEvaluation(
        provider=artifact.provider,
        model=artifact.model,
        collection=artifact.collection,
        dataset_ids=artifact.dataset_ids,
        total_examples=artifact.total_examples,
        raw_results_path=raw_results_path,
        running=running,
        errors=errors,
    )


def load_and_score_raw_results(
    *,
    raw_results_path: Path,
    examples: list[UserExample],
) -> ScoredEvaluation:
    artifact = load_raw_results(raw_results_path)
    return score_raw_results(
        artifact=artifact,
        raw_results_path=raw_results_path,
        examples=examples,
    )


def format_run_report(scored: ScoredEvaluation) -> str:
    header_lines = [
        f"model: {scored.model}",
        f"provider: {scored.provider}",
        f"collection: {scored.collection}",
        "dataset_ids: " + ",".join(str(dataset_id) for dataset_id in scored.dataset_ids),
        f"raw_results_path: {scored.raw_results_path}",
        f"completed: {scored.running.completed_examples}/{scored.total_examples}",
        f"points: {scored.running.raw_score}/{scored.running.max_possible_score}",
        f"tp: {scored.running.tp}",
        f"fp: {scored.running.fp}",
        f"fn: {scored.running.fn}",
        f"percentage: {scored.running.pct_of_max:.4f}",
        "",
        f"errors: {len(scored.errors)}",
        "",
    ]

    error_sections: list[str] = []
    for index, error in enumerate(scored.errors, start=1):
        error_sections.append(
            "\n".join(
                [
                    "=" * 80,
                    f"error #{index}",
                    f"kind: {error.kind}",
                    f"user_id: {error.user_id}",
                    f"truth_label: {error.truth_label}",
                    f"predicted_label: {error.predicted_label}",
                    f"raw_output: {error.raw_output}",
                    f"dataset_id: {error.dataset_id}",
                    f"source: {error.source}",
                    f"lang: {error.lang}",
                    f"post_count_used: {error.post_count_used}",
                    f"full_post_count: {error.full_post_count}",
                    "",
                    error.user_prompt,
                    "",
                ]
            )
        )

    if not error_sections:
        error_sections.append("No errors.\n")

    return "\n".join(header_lines + error_sections).rstrip() + "\n"


def save_run_report(
    *,
    report_path: Path,
    scored: ScoredEvaluation,
) -> None:
    save_text(report_path, format_run_report(scored))


def write_submission_files(
    *,
    raw_results_path: Path,
    examples: list[SubmissionExample],
    team_name: str,
    output_dir: Path | None = None,
) -> tuple[Path, ...]:
    artifact = load_raw_results(raw_results_path)
    validate_raw_results_against_submission_examples(
        artifact=artifact,
        examples=examples,
    )

    examples_by_user_id = {example.user_id: example for example in examples}
    output_root = output_dir or FINAL_RESULTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    team_slug = safe_team_name(team_name)

    user_ids_by_lang: dict[str, list[str]] = {}
    for example in examples:
        user_ids_by_lang.setdefault(example.lang.lower(), [])

    for user_id in artifact.detected_bot_ids:
        example = examples_by_user_id.get(user_id)
        if example is None:
            raise ValueError(f"Missing submission example for detected bot user_id={user_id}")
        user_ids_by_lang.setdefault(example.lang.lower(), []).append(user_id)

    paths: list[Path] = []
    for lang, user_ids in sorted(user_ids_by_lang.items()):
        path = output_root / f"{team_slug}.detections.{lang}.txt"
        sorted_user_ids = sorted(user_ids, key=_user_id_sort_key)
        payload = "\n".join(sorted_user_ids)
        if payload:
            payload += "\n"
        save_text(path, payload)
        paths.append(path)

    return tuple(paths)
