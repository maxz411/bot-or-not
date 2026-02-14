from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ClassificationPoint:
    user_id: str
    dataset_id: int
    truth_label: str
    predicted_label: str


@dataclass
class Metrics:
    total: int
    bots: int
    humans: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    score: int
    max_score: int
    pct_max: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "bots": self.bots,
            "humans": self.humans,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "score": self.score,
            "max_score": self.max_score,
            "pct_max": self.pct_max,
        }


def compute_metrics(points: Iterable[ClassificationPoint]) -> Metrics:
    tp = tn = fp = fn = bots = humans = total = 0

    for point in points:
        total += 1
        is_bot = point.truth_label == "BOT"
        predicted_bot = point.predicted_label == "BOT"

        if is_bot:
            bots += 1
            if predicted_bot:
                tp += 1
            else:
                fn += 1
        else:
            humans += 1
            if predicted_bot:
                fp += 1
            else:
                tn += 1

    accuracy = (100.0 * (tp + tn) / total) if total else 0.0
    score = tp * 4 - fp * 2 - fn
    max_score = bots * 4
    pct_max = (100.0 * score / max_score) if max_score else 0.0

    return Metrics(
        total=total,
        bots=bots,
        humans=humans,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        score=score,
        max_score=max_score,
        pct_max=pct_max,
    )

