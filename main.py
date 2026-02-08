"""
Check accuracy of a bot-detection run file against dataset ground truth.
Usage: python main.py runs/<runfile>.txt
"""

import json
import re
import sys
from pathlib import Path

# Project root (directory containing main.py)
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"


def parse_datasets_line(line: str) -> list[int]:
    """Parse first line like 'Datasets: 30, 31, 32' -> [30, 31, 32]."""
    match = re.search(r"Datasets:\s*([\d\s,]+)", line.strip(), re.IGNORECASE)
    if not match:
        return []
    return [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]


def load_users_and_bots(dataset_ids: list[int]) -> dict[str, bool]:
    """Build {user_id: is_bot} for all users in the given datasets."""
    truth = {}
    for n in dataset_ids:
        # Load all user IDs from dataset.posts&users.{n}.json
        json_path = DATASETS_DIR / f"dataset.posts&users.{n}.json"
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping dataset {n}")
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        user_ids = [u["id"] for u in data.get("users", [])]
        for uid in user_ids:
            truth[uid] = False  # default: not bot

        # Mark bots from dataset.bots.{n}.txt
        bots_path = DATASETS_DIR / f"dataset.bots.{n}.txt"
        if not bots_path.exists():
            print(f"Warning: {bots_path} not found, skipping bot labels for dataset {n}")
            continue
        with open(bots_path, encoding="utf-8") as f:
            for line in f:
                uid = line.strip()
                if uid:
                    if uid in truth:
                        truth[uid] = True
                    # else: bot file lists an ID not in users (shouldn't happen)
    return truth


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <runfile.txt>")
        print("Example: python main.py runs/my_run.txt")
        sys.exit(1)

    run_path = Path(sys.argv[1])
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    if not run_path.exists():
        print(f"Error: run file not found: {run_path}")
        sys.exit(1)

    with open(run_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    if not lines:
        print("Error: run file is empty")
        sys.exit(1)

    dataset_ids = parse_datasets_line(lines[0])
    if not dataset_ids:
        print("Error: could not parse 'Datasets: 30, 31, ...' from first line")
        sys.exit(1)

    # Ground truth: all users in those datasets with is_bot flag
    truth = load_users_and_bots(dataset_ids)
    if not truth:
        print("Error: no users loaded from datasets")
        sys.exit(1)

    # Predicted bot IDs (rest of file)
    predicted_bots = set()
    for line in lines[1:]:
        uid = line.strip()
        if uid:
            predicted_bots.add(uid)

    # Warn about predicted IDs not in any dataset
    unknown = predicted_bots - set(truth.keys())
    if unknown:
        print(f"Warning: {len(unknown)} predicted user ID(s) not in datasets (ignored):")
        for uid in sorted(unknown)[:10]:
            print(f"  {uid}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown) - 10} more")
        predicted_bots &= set(truth.keys())

    # Count TP, FP, FN, TN
    total = len(truth)
    actual_bots = {uid for uid, is_bot in truth.items() if is_bot}
    actual_humans = set(truth.keys()) - actual_bots

    tp = len(predicted_bots & actual_bots)
    fp = len(predicted_bots & actual_humans)
    fn = len(actual_bots - predicted_bots)
    tn = len(actual_humans - predicted_bots)

    # Percentages (of total users)
    pct_correct = 100.0 * (tp + tn) / total if total else 0
    pct_fp = 100.0 * fp / total if total else 0
    pct_fn = 100.0 * fn / total if total else 0

    # Score: +4 TP, -1 FN, -2 FP
    score = tp * 4 + fn * (-1) + fp * (-2)

    # Output
    print("--- Run accuracy ---")
    print(f"Run file: {run_path.name}")
    print(f"Datasets: {dataset_ids}")
    print(f"Total users: {total}")
    print()
    print("Counts:")
    print(f"  Correct (TP + TN): {tp + tn}  ({pct_correct:.2f}%)")
    print(f"  False Positives:   {fp}  ({pct_fp:.2f}%)")
    print(f"  False Negatives:   {fn}  ({pct_fn:.2f}%)")
    print()
    print(f"Score (+4 TP, -1 FN, -2 FP): {score}")


if __name__ == "__main__":
    main()
