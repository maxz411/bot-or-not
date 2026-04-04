from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .constants import (
    DATASET_COLLECTIONS,
    PREPARED_SPLITS,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    OPENAI_TRAINING_METHODS,
    SYSTEM_PROMPT,
)
from .dataset_inventory import (
    DatasetAsset,
    FinalDatasetAsset,
    DatasetSource,
    discover_dataset_assets,
    group_assets_by_collection,
)
from .storage import now_iso, save_json

PreparedFormat = Literal["openai", "gemini"]
PreparedSplit = Literal["prev", "current", "fold_a", "fold_b", "both", "full"]
OpenAITrainingMethod = Literal["supervised", "reinforcement"]


@dataclass(frozen=True)
class OpenAIChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GeminiContent:
    role: Literal["system", "user", "model"]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "parts": [{"text": self.text}]}


@dataclass(frozen=True)
class UserExample:
    user_id: str
    dataset_id: int
    source: DatasetSource
    lang: str
    label: Literal["BOT", "HUMAN"]
    full_post_count: int
    post_count_used: int
    user_prompt: str

    def to_openai_messages(self) -> list[dict[str, str]]:
        return build_openai_classification_messages(self.user_prompt)

    def to_openai_sft_record(self) -> dict[str, Any]:
        return {
            "messages": [
                *self.to_openai_messages(),
                OpenAIChatMessage(role="assistant", content=self.label).to_dict(),
            ]
        }

    def to_openai_reinforcement_record(self) -> dict[str, Any]:
        return {
            "messages": self.to_openai_messages(),
            **self.to_metadata(),
        }

    def to_gemini_sft_record(self) -> dict[str, Any]:
        return {
            "systemInstruction": GeminiContent(
                role="system",
                text=SYSTEM_PROMPT,
            ).to_dict(),
            "contents": [
                GeminiContent(role="user", text=self.user_prompt).to_dict(),
                GeminiContent(role="model", text=self.label).to_dict(),
            ],
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "source": self.source,
            "lang": self.lang,
            "label": self.label,
            "full_post_count": self.full_post_count,
            "post_count_used": self.post_count_used,
        }


@dataclass(frozen=True)
class SubmissionExample:
    user_id: str
    dataset_id: int
    lang: str
    full_post_count: int
    post_count_used: int
    user_prompt: str

    def to_openai_messages(self) -> list[dict[str, str]]:
        return build_openai_classification_messages(self.user_prompt)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_bot_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_openai_classification_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        OpenAIChatMessage(role="system", content=SYSTEM_PROMPT).to_dict(),
        OpenAIChatMessage(role="user", content=user_prompt).to_dict(),
    ]


def _format_training_record(
    example: UserExample,
    *,
    provider: PreparedFormat,
    openai_method: OpenAITrainingMethod | None = None,
) -> dict[str, Any]:
    if provider == "openai":
        resolved_method = openai_method or "supervised"
        if resolved_method == "supervised":
            return example.to_openai_sft_record()
        if resolved_method == "reinforcement":
            return example.to_openai_reinforcement_record()
        raise ValueError(
            f"Unsupported OpenAI training method: {resolved_method}. "
            f"Expected one of {', '.join(OPENAI_TRAINING_METHODS)}."
        )
    if provider == "gemini":
        return example.to_gemini_sft_record()
    raise ValueError(f"Unsupported provider: {provider}")


def _write_training_examples(
    path: Path,
    examples: list[UserExample],
    *,
    provider: PreparedFormat,
    openai_method: OpenAITrainingMethod | None = None,
) -> None:
    _write_jsonl(
        path,
        (
            _format_training_record(
                example,
                provider=provider,
                openai_method=openai_method,
            )
            for example in examples
        ),
    )


def _write_metadata(path: Path, examples: list[UserExample]) -> None:
    _write_jsonl(path, (example.to_metadata() for example in examples))


def _user_id_set(examples: list[UserExample]) -> set[str]:
    return {example.user_id for example in examples}


def _overlap_size(a: list[UserExample], b: list[UserExample]) -> int:
    return len(_user_id_set(a) & _user_id_set(b))


def build_user_prompt(user: dict[str, Any], posts: list[dict[str, Any]]) -> str:
    profile_lines = [
        f"User ID: {user.get('id', '?')}",
        f"Username: {user.get('username', '?')}",
        f"Name: {user.get('name', '?')}",
        f"Description: {user.get('description') or '(none)'}",
        f"Location: {user.get('location') or '(none)'}",
        f"Tweet count: {user.get('tweet_count', '?')}",
        (
            "Z-score (posting activity deviation from average): "
            f"{user.get('z_score', '?')}"
        ),
    ]

    post_lines = [
        (
            f"[{post.get('created_at', '')}] "
            f"[id:{post.get('id', '')}] "
            f"[lang:{post.get('lang', '')}] "
            f"{post.get('text', '')}"
        )
        for post in posts
    ]
    if not post_lines:
        post_lines.append("(no posts)")

    return "\n".join(profile_lines) + "\n\nPosts:\n" + "\n".join(post_lines)


def load_examples(assets: Iterable[DatasetAsset]) -> list[UserExample]:
    examples: list[UserExample] = []
    seen_user_ids: set[str] = set()

    for asset in assets:
        dataset = _load_json(asset.dataset_path)
        lang = str(dataset.get("lang", "unknown"))
        bot_ids = _load_bot_ids(asset.bots_path)

        posts_by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for post in dataset.get("posts", []):
            author_id = str(post.get("author_id", ""))
            posts_by_author[author_id].append(post)

        for posts in posts_by_author.values():
            posts.sort(key=lambda post: str(post.get("created_at", "")))

        for user in dataset.get("users", []):
            user_id = str(user.get("id"))
            if user_id in seen_user_ids:
                raise ValueError(f"Duplicate user id across datasets: {user_id}")

            seen_user_ids.add(user_id)
            user_posts = posts_by_author.get(user_id, [])
            label: Literal["BOT", "HUMAN"] = "BOT" if user_id in bot_ids else "HUMAN"

            examples.append(
                UserExample(
                    user_id=user_id,
                    dataset_id=asset.dataset_id,
                    source=asset.source,
                    lang=lang,
                    label=label,
                    full_post_count=len(user_posts),
                    post_count_used=len(user_posts),
                    user_prompt=build_user_prompt(user, user_posts),
                )
            )

    examples.sort(key=lambda example: (example.source, example.dataset_id, example.user_id))
    return examples


def load_final_examples(assets: Iterable[FinalDatasetAsset]) -> list[SubmissionExample]:
    examples: list[SubmissionExample] = []
    seen_user_ids: set[str] = set()

    for asset in assets:
        dataset = _load_json(asset.dataset_path)
        lang = str(dataset.get("lang", "unknown"))

        posts_by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for post in dataset.get("posts", []):
            author_id = str(post.get("author_id", ""))
            posts_by_author[author_id].append(post)

        for posts in posts_by_author.values():
            posts.sort(key=lambda post: str(post.get("created_at", "")))

        for user in dataset.get("users", []):
            user_id = str(user.get("id"))
            if user_id in seen_user_ids:
                raise ValueError(f"Duplicate user id across final datasets: {user_id}")

            seen_user_ids.add(user_id)
            user_posts = posts_by_author.get(user_id, [])
            examples.append(
                SubmissionExample(
                    user_id=user_id,
                    dataset_id=asset.dataset_id,
                    lang=lang,
                    full_post_count=len(user_posts),
                    post_count_used=len(user_posts),
                    user_prompt=build_user_prompt(user, user_posts),
                )
            )

    examples.sort(key=lambda example: (example.dataset_id, example.user_id))
    return examples


def stratified_split(
    examples: list[UserExample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[UserExample], list[UserExample]]:
    if not examples:
        raise ValueError("Cannot split an empty example set.")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    by_bucket: dict[tuple[str, str, str], list[UserExample]] = defaultdict(list)
    for example in examples:
        by_bucket[(example.label, example.lang, example.source)].append(example)

    train: list[UserExample] = []
    validation: list[UserExample] = []

    for bucket_key in sorted(by_bucket):
        bucket = sorted(
            by_bucket[bucket_key],
            key=lambda example: (example.dataset_id, example.user_id),
        )
        rng.shuffle(bucket)

        if len(bucket) <= 1:
            validation_count = 0
        else:
            validation_count = max(1, round(len(bucket) * val_fraction))
            validation_count = min(validation_count, len(bucket) - 1)

        validation.extend(bucket[:validation_count])
        train.extend(bucket[validation_count:])

    train.sort(key=lambda example: (example.source, example.dataset_id, example.user_id))
    validation.sort(
        key=lambda example: (example.source, example.dataset_id, example.user_id)
    )
    return train, validation


def filter_examples_by_assets(
    examples: list[UserExample],
    *,
    assets: Iterable[DatasetAsset],
) -> list[UserExample]:
    allowed_pairs = {(asset.source, asset.dataset_id) for asset in assets}
    return [
        example
        for example in examples
        if (example.source, example.dataset_id) in allowed_pairs
    ]


def inventory_summary(assets: tuple[DatasetAsset, ...]) -> dict[str, Any]:
    grouped_assets = group_assets_by_collection(assets)
    return {
        name: {
            "dataset_ids": [asset.dataset_id for asset in grouped_assets[name]],
            "dataset_count": len(grouped_assets[name]),
        }
        for name in DATASET_COLLECTIONS
    }


def examples_summary(examples: list[UserExample]) -> dict[str, Any]:
    label_counts = Counter(example.label for example in examples)
    language_counts = Counter(example.lang for example in examples)
    dataset_counts = Counter(str(example.dataset_id) for example in examples)
    source_counts = Counter(example.source for example in examples)

    return {
        "total_examples": len(examples),
        "labels": dict(label_counts),
        "languages": dict(language_counts),
        "sources": dict(source_counts),
        "datasets": dict(dataset_counts),
        "no_truncation_mismatches": sum(
            1
            for example in examples
            if example.post_count_used != example.full_post_count
        ),
    }


def _write_split(
    *,
    split_dir: Path,
    train_examples: list[UserExample],
    validation_examples: list[UserExample] | None,
    source_collections: tuple[str, ...],
    source_datasets: tuple[int, ...],
    provider: PreparedFormat,
    openai_method: OpenAITrainingMethod | None = None,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.jsonl"
    train_meta_path = split_dir / "train.meta.jsonl"
    _write_training_examples(
        train_path,
        train_examples,
        provider=provider,
        openai_method=openai_method,
    )
    _write_metadata(train_meta_path, train_examples)

    summary: dict[str, Any] = {
        "train_path": str(train_path),
        "train_count": len(train_examples),
        "validation_path": None,
        "validation_count": 0,
        "train_validation_overlap": 0,
        "source_collections": list(source_collections),
        "source_datasets": list(source_datasets),
    }

    if validation_examples is not None:
        validation_path = split_dir / "val.jsonl"
        validation_meta_path = split_dir / "val.meta.jsonl"
        _write_training_examples(
            validation_path,
            validation_examples,
            provider=provider,
            openai_method=openai_method,
        )
        _write_metadata(validation_meta_path, validation_examples)
        summary.update(
            {
                "validation_path": str(validation_path),
                "validation_count": len(validation_examples),
                "train_validation_overlap": _overlap_size(
                    train_examples,
                    validation_examples,
                ),
            }
        )

    return summary


def prepare_data(
    *,
    prepared_dir: Path,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    provider: PreparedFormat = "openai",
    openai_method: OpenAITrainingMethod | None = None,
) -> dict[str, Any]:
    prepared_dir.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        resolved_openai_method = openai_method or "supervised"
        if resolved_openai_method not in OPENAI_TRAINING_METHODS:
            raise ValueError(
                f"Unsupported OpenAI training method: {resolved_openai_method}. "
                f"Expected one of {', '.join(OPENAI_TRAINING_METHODS)}."
            )
    else:
        resolved_openai_method = None

    assets = discover_dataset_assets()
    grouped_assets = group_assets_by_collection(assets)
    examples = load_examples(assets)

    splits: dict[str, dict[str, Any]] = {}
    for index, collection in enumerate(DATASET_COLLECTIONS):
        source_assets = grouped_assets[collection]
        collection_examples = filter_examples_by_assets(
            examples,
            assets=source_assets,
        )
        train_examples, validation_examples = stratified_split(
            collection_examples,
            val_fraction=val_fraction,
            seed=seed + index,
        )
        source_collections = tuple(
            source
            for source in ("prev", "current")
            if any(asset.source == source for asset in source_assets)
        )
        splits[collection] = _write_split(
            split_dir=prepared_dir / collection,
            train_examples=train_examples,
            validation_examples=validation_examples,
            source_collections=source_collections,
            source_datasets=tuple(asset.dataset_id for asset in source_assets),
            provider=provider,
            openai_method=resolved_openai_method,
        )

    splits["full"] = _write_split(
        split_dir=prepared_dir / "full",
        train_examples=examples,
        validation_examples=None,
        source_collections=("prev", "current"),
        source_datasets=tuple(asset.dataset_id for asset in assets),
        provider=provider,
        openai_method=resolved_openai_method,
    )

    payload = {
        "generated_at": now_iso(),
        "provider": provider,
        "method": resolved_openai_method,
        "prepared_dir": str(prepared_dir),
        "seed": seed,
        "val_fraction": val_fraction,
        "inventory": inventory_summary(assets),
        "data_integrity": examples_summary(examples),
        "splits": splits,
    }
    save_json(prepared_dir / "summary.json", payload)
    return payload


def resolve_training_files(
    *,
    prepared_dir: Path,
    split_name: str,
) -> tuple[Path, Path | None, str]:
    if split_name not in PREPARED_SPLITS:
        raise ValueError(
            f"Unsupported prepared split: {split_name}. "
            f"Expected one of {', '.join(PREPARED_SPLITS)}."
        )

    split_dir = prepared_dir / split_name
    train_path = split_dir / "train.jsonl"
    validation_path = split_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Prepared split is missing its training file: {train_path}. "
            "Run prepare-data first."
        )

    resolved_validation = validation_path if validation_path.exists() else None
    return train_path, resolved_validation, split_name
