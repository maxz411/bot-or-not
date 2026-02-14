from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .constants import DATASETS_DIR, SYSTEM_PROMPT


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class EvalJSONLRecord:
    user_id: str
    dataset_id: int
    lang: str
    label: Literal["BOT", "HUMAN"]
    full_post_count: int
    post_count_used: int
    messages: list[ChatMessage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "lang": self.lang,
            "label": self.label,
            "full_post_count": self.full_post_count,
            "post_count_used": self.post_count_used,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalJSONLRecord":
        raw_messages = data.get("messages", [])
        messages = [
            ChatMessage(
                role=str(message["role"]),
                content=str(message["content"]),
            )
            for message in raw_messages
        ]
        return cls(
            user_id=str(data["user_id"]),
            dataset_id=int(data["dataset_id"]),
            lang=str(data["lang"]),
            label=str(data["label"]),  # type: ignore[arg-type]
            full_post_count=int(data["full_post_count"]),
            post_count_used=int(data["post_count_used"]),
            messages=messages,
        )


@dataclass(frozen=True)
class UserExample:
    user_id: str
    dataset_id: int
    lang: str
    label: Literal["BOT", "HUMAN"]
    full_post_count: int
    post_count_used: int
    user_prompt: str

    @property
    def is_bot(self) -> bool:
        return self.label == "BOT"

    def to_sft_record(self) -> dict[str, Any]:
        return {
            "messages": [
                ChatMessage(role="system", content=SYSTEM_PROMPT).to_dict(),
                ChatMessage(role="user", content=self.user_prompt).to_dict(),
                ChatMessage(role="assistant", content=self.label).to_dict(),
            ]
        }

    def to_eval_record(self) -> EvalJSONLRecord:
        return EvalJSONLRecord(
            user_id=self.user_id,
            dataset_id=self.dataset_id,
            lang=self.lang,
            label=self.label,
            full_post_count=self.full_post_count,
            post_count_used=self.post_count_used,
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=self.user_prompt),
            ],
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "lang": self.lang,
            "label": self.label,
            "full_post_count": self.full_post_count,
            "post_count_used": self.post_count_used,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bot_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def build_user_prompt(user: dict[str, Any], posts: list[dict[str, Any]]) -> str:
    profile = [
        f"User ID: {user.get('id', '?')}",
        f"Username: {user.get('username', '?')}",
        f"Name: {user.get('name', '?')}",
        f"Description: {user.get('description') or '(none)'}",
        f"Location: {user.get('location') or '(none)'}",
        f"Tweet count: {user.get('tweet_count', '?')}",
        f"Z-score (posting activity deviation from average): {user.get('z_score', '?')}",
    ]

    post_lines = [
        f"[{post.get('created_at', '')}] [id:{post.get('id', '')}] [lang:{post.get('lang', '')}] {post.get('text', '')}"
        for post in posts
    ]

    if not post_lines:
        post_lines.append("(no posts)")

    return "\n".join(profile) + "\n\nPosts:\n" + "\n".join(post_lines)


def load_examples(dataset_ids: Iterable[int]) -> list[UserExample]:
    examples: list[UserExample] = []
    seen_user_ids: set[str] = set()

    for dataset_id in dataset_ids:
        dataset_path = DATASETS_DIR / f"dataset.posts&users.{dataset_id}.json"
        bots_path = DATASETS_DIR / f"dataset.bots.{dataset_id}.txt"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {dataset_path}")
        if not bots_path.exists():
            raise FileNotFoundError(f"Missing bot label file: {bots_path}")

        dataset = _load_json(dataset_path)
        lang = str(dataset.get("lang", "unknown"))
        bot_ids = _load_bot_ids(bots_path)

        posts_by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for post in dataset.get("posts", []):
            author_id = str(post.get("author_id", ""))
            posts_by_author[author_id].append(post)

        for posts in posts_by_author.values():
            posts.sort(key=lambda p: str(p.get("created_at", "")))

        for user in dataset.get("users", []):
            user_id = str(user.get("id"))
            if user_id in seen_user_ids:
                raise ValueError(f"Duplicate user id across datasets: {user_id}")
            seen_user_ids.add(user_id)

            user_posts = posts_by_author.get(user_id, [])
            full_post_count = len(user_posts)
            post_count_used = full_post_count
            label: Literal["BOT", "HUMAN"] = "BOT" if user_id in bot_ids else "HUMAN"

            examples.append(
                UserExample(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    lang=lang,
                    label=label,
                    full_post_count=full_post_count,
                    post_count_used=post_count_used,
                    user_prompt=build_user_prompt(user, user_posts),
                )
            )

    examples.sort(key=lambda ex: ex.user_id)
    return examples


def stratified_split(
    examples: list[UserExample],
    val_fraction: float,
    seed: int,
) -> tuple[list[UserExample], list[UserExample]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = random.Random(seed)
    by_bucket: dict[tuple[str, str], list[UserExample]] = defaultdict(list)
    for example in examples:
        by_bucket[(example.label, example.lang)].append(example)

    train: list[UserExample] = []
    val: list[UserExample] = []

    for bucket_key in sorted(by_bucket):
        bucket = sorted(by_bucket[bucket_key], key=lambda ex: ex.user_id)
        rng.shuffle(bucket)

        if len(bucket) <= 1:
            n_val = 0
        else:
            n_val = max(1, round(len(bucket) * val_fraction))
            n_val = min(n_val, len(bucket) - 1)

        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    train.sort(key=lambda ex: ex.user_id)
    val.sort(key=lambda ex: ex.user_id)
    return train, val


def filter_examples_by_dataset(
    examples: list[UserExample], dataset_ids: Iterable[int]
) -> list[UserExample]:
    allowed = set(dataset_ids)
    return [example for example in examples if example.dataset_id in allowed]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sft_examples(path: Path, examples: list[UserExample]) -> None:
    write_jsonl(path, (ex.to_sft_record() for ex in examples))


def write_eval_examples(path: Path, examples: list[UserExample]) -> None:
    write_jsonl(path, (ex.to_eval_record().to_dict() for ex in examples))


def write_metadata(path: Path, examples: list[UserExample]) -> None:
    write_jsonl(path, (ex.to_metadata() for ex in examples))


def read_eval_examples(path: Path) -> list[EvalJSONLRecord]:
    records: list[EvalJSONLRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(EvalJSONLRecord.from_dict(json.loads(line)))
    return records


def examples_summary(examples: list[UserExample]) -> dict[str, Any]:
    label_counts = Counter(example.label for example in examples)
    lang_counts = Counter(example.lang for example in examples)
    dataset_counts = Counter(str(example.dataset_id) for example in examples)
    no_truncation_mismatches = sum(
        1
        for example in examples
        if example.post_count_used != example.full_post_count
    )

    return {
        "total_examples": len(examples),
        "labels": dict(label_counts),
        "languages": dict(lang_counts),
        "datasets": dict(dataset_counts),
        "no_truncation_mismatches": no_truncation_mismatches,
    }


def user_id_set(examples: list[UserExample]) -> set[str]:
    return {example.user_id for example in examples}


def overlap_size(a: list[UserExample], b: list[UserExample]) -> int:
    return len(user_id_set(a) & user_id_set(b))

