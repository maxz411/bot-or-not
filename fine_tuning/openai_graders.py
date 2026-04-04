from __future__ import annotations

from textwrap import dedent
from typing import Any

from .constants import (
    DEFAULT_OPENAI_RFT_GRADER_IMAGE_TAG,
    OPENAI_RFT_FALSE_NEGATIVE_REWARD,
    OPENAI_RFT_FALSE_POSITIVE_REWARD,
    OPENAI_RFT_TRUE_NEGATIVE_REWARD,
    OPENAI_RFT_TRUE_POSITIVE_REWARD,
)

BOT_LABEL = "BOT"
HUMAN_LABEL = "HUMAN"

OUTCOME_RAW_REWARDS = {
    "true_positive": OPENAI_RFT_TRUE_POSITIVE_REWARD,
    "false_negative": OPENAI_RFT_FALSE_NEGATIVE_REWARD,
    "false_positive": OPENAI_RFT_FALSE_POSITIVE_REWARD,
    "true_negative": OPENAI_RFT_TRUE_NEGATIVE_REWARD,
}

THEORETICAL_MAX_RAW_REWARD = max(OUTCOME_RAW_REWARDS.values())
THEORETICAL_MIN_RAW_REWARD = min(OUTCOME_RAW_REWARDS.values())
RAW_REWARD_RANGE = THEORETICAL_MAX_RAW_REWARD - THEORETICAL_MIN_RAW_REWARD


def normalize_prediction_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().upper()
    if normalized in {BOT_LABEL, HUMAN_LABEL}:
        return normalized
    return None


def raw_bot_detection_reward(*, actual_label: str, predicted_label: str) -> int:
    if actual_label == BOT_LABEL and predicted_label == BOT_LABEL:
        return OPENAI_RFT_TRUE_POSITIVE_REWARD
    if actual_label == BOT_LABEL and predicted_label == HUMAN_LABEL:
        return OPENAI_RFT_FALSE_NEGATIVE_REWARD
    if actual_label == HUMAN_LABEL and predicted_label == BOT_LABEL:
        return OPENAI_RFT_FALSE_POSITIVE_REWARD
    if actual_label == HUMAN_LABEL and predicted_label == HUMAN_LABEL:
        return OPENAI_RFT_TRUE_NEGATIVE_REWARD

    raise ValueError(
        "Unsupported label combination for bot detection reward: "
        f"actual={actual_label!r} predicted={predicted_label!r}."
    )


def normalized_bot_detection_reward(raw_reward: int) -> float:
    if (
        raw_reward < THEORETICAL_MIN_RAW_REWARD
        or raw_reward > THEORETICAL_MAX_RAW_REWARD
    ):
        raise ValueError(
            "raw_reward must be between "
            f"{THEORETICAL_MIN_RAW_REWARD} and {THEORETICAL_MAX_RAW_REWARD}, "
            f"got {raw_reward}."
        )
    return float((raw_reward - THEORETICAL_MIN_RAW_REWARD) / RAW_REWARD_RANGE)


def bot_detection_reward_table() -> dict[str, dict[str, float | int]]:
    cases = {
        "true_positive": {
            "actual": BOT_LABEL,
            "predicted": BOT_LABEL,
        },
        "false_negative": {
            "actual": BOT_LABEL,
            "predicted": HUMAN_LABEL,
        },
        "false_positive": {
            "actual": HUMAN_LABEL,
            "predicted": BOT_LABEL,
        },
        "true_negative": {
            "actual": HUMAN_LABEL,
            "predicted": HUMAN_LABEL,
        },
    }

    return {
        name: {
            **labels,
            "raw_reward": raw_bot_detection_reward(
                actual_label=labels["actual"],
                predicted_label=labels["predicted"],
            ),
            "normalized_reward": normalized_bot_detection_reward(
                raw_bot_detection_reward(
                    actual_label=labels["actual"],
                    predicted_label=labels["predicted"],
                )
            ),
        }
        for name, labels in cases.items()
    }


def build_bot_detection_reinforcement_grader_source() -> str:
    return dedent(
        f"""
        BOT_LABEL = "{BOT_LABEL}"
        HUMAN_LABEL = "{HUMAN_LABEL}"

        OUTCOME_RAW_REWARDS = {{
            "true_positive": {OPENAI_RFT_TRUE_POSITIVE_REWARD},
            "false_negative": {OPENAI_RFT_FALSE_NEGATIVE_REWARD},
            "false_positive": {OPENAI_RFT_FALSE_POSITIVE_REWARD},
            "true_negative": {OPENAI_RFT_TRUE_NEGATIVE_REWARD},
        }}

        THEORETICAL_MAX_RAW_REWARD = max(OUTCOME_RAW_REWARDS.values())
        THEORETICAL_MIN_RAW_REWARD = min(OUTCOME_RAW_REWARDS.values())
        RAW_REWARD_RANGE = THEORETICAL_MAX_RAW_REWARD - THEORETICAL_MIN_RAW_REWARD


        def _normalize_label(value):
            if not isinstance(value, str):
                return None

            normalized = value.strip().upper()
            if normalized in {{BOT_LABEL, HUMAN_LABEL}}:
                return normalized
            return None


        def _raw_reward(actual_label, predicted_label):
            if actual_label == BOT_LABEL and predicted_label == BOT_LABEL:
                return OUTCOME_RAW_REWARDS["true_positive"]
            if actual_label == BOT_LABEL and predicted_label == HUMAN_LABEL:
                return OUTCOME_RAW_REWARDS["false_negative"]
            if actual_label == HUMAN_LABEL and predicted_label == BOT_LABEL:
                return OUTCOME_RAW_REWARDS["false_positive"]
            if actual_label == HUMAN_LABEL and predicted_label == HUMAN_LABEL:
                return OUTCOME_RAW_REWARDS["true_negative"]

            raise ValueError(
                "Unsupported label combination for bot detection reward: "
                f"actual={{actual_label!r}} predicted={{predicted_label!r}}."
            )


        def _normalize_reward(raw_reward):
            return float(
                (raw_reward - THEORETICAL_MIN_RAW_REWARD) / RAW_REWARD_RANGE
            )


        def grade(sample, item):
            actual_label = _normalize_label(item.get("label"))
            if actual_label is None:
                raise ValueError(
                    "Training item must include a 'label' field with value 'BOT' or 'HUMAN'."
                )

            predicted_label = _normalize_label(sample.get("output_text"))
            if predicted_label is None:
                # Invalid outputs should not receive partial credit.
                return 0.0

            raw_reward = _raw_reward(actual_label, predicted_label)
            return _normalize_reward(raw_reward)
        """
    ).strip() + "\n"


def build_bot_detection_reinforcement_grader_schema() -> dict[str, Any]:
    return {
        "type": "python",
        "source": build_bot_detection_reinforcement_grader_source(),
        "image_tag": DEFAULT_OPENAI_RFT_GRADER_IMAGE_TAG,
    }
