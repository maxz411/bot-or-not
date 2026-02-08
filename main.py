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
RESULTS_DIR = ROOT / "results"


def parse_datasets_line(line: str) -> list[int]:
    """Parse first line like 'Datasets: 30, 31, 32' -> [30, 31, 32]."""
    match = re.search(r"Datasets:\s*([\d\s,]+)", line.strip(), re.IGNORECASE)
    if not match:
        return []
    return [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]


def load_users_and_bots(
    dataset_ids: list[int],
) -> tuple[dict[str, bool], dict[str, dict]]:
    """Build {user_id: is_bot} and {user_id: user_metadata} for all users."""
    truth: dict[str, bool] = {}
    users: dict[str, dict] = {}

    for n in dataset_ids:
        json_path = DATASETS_DIR / f"dataset.posts&users.{n}.json"
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping dataset {n}")
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for u in data.get("users", []):
            uid = u["id"]
            truth[uid] = False
            users[uid] = u

        bots_path = DATASETS_DIR / f"dataset.bots.{n}.txt"
        if not bots_path.exists():
            print(
                f"Warning: {bots_path} not found, skipping bot labels for dataset {n}"
            )
            continue
        with open(bots_path, encoding="utf-8") as f:
            for line in f:
                uid = line.strip()
                if uid and uid in truth:
                    truth[uid] = True

    return truth, users


def format_user(u: dict) -> str:
    """Format a user dict into a readable string."""
    return (
        f"  ID:          {u.get('id', '?')}\n"
        f"  Username:    {u.get('username', '?')}\n"
        f"  Name:        {u.get('name', '?')}\n"
        f"  Description: {u.get('description') or '(none)'}\n"
        f"  Location:    {u.get('location') or '(none)'}\n"
        f"  Tweets:      {u.get('tweet_count', '?')}\n"
        f"  Z-score:     {u.get('z_score', '?')}"
    )


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

    # Ground truth
    truth, users = load_users_and_bots(dataset_ids)
    if not truth:
        print("Error: no users loaded from datasets")
        sys.exit(1)

    # Predicted bot IDs
    predicted_bots = set()
    for line in lines[1:]:
        uid = line.strip()
        if uid:
            predicted_bots.add(uid)

    # Warn about predicted IDs not in any dataset
    unknown = predicted_bots - set(truth.keys())
    if unknown:
        print(
            f"Warning: {len(unknown)} predicted user ID(s) not in datasets (ignored):"
        )
        for uid in sorted(unknown)[:10]:
            print(f"  {uid}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown) - 10} more")
        predicted_bots &= set(truth.keys())

    # Compute sets
    actual_bots = {uid for uid, is_bot in truth.items() if is_bot}
    actual_humans = set(truth.keys()) - actual_bots
    total = len(truth)

    tp_ids = sorted(predicted_bots & actual_bots)
    fp_ids = sorted(predicted_bots & actual_humans)
    fn_ids = sorted(actual_bots - predicted_bots)
    tn_ids = sorted(actual_humans - predicted_bots)

    tp, fp, fn, tn = len(tp_ids), len(fp_ids), len(fn_ids), len(tn_ids)

    pct_correct = 100.0 * (tp + tn) / total if total else 0
    pct_fp = 100.0 * fp / total if total else 0
    pct_fn = 100.0 * fn / total if total else 0

    # Score: +4 TP, -1 FN, -2 FP
    score = tp * 4 + fn * (-1) + fp * (-2)

    # Build output
    out: list[str] = []
    out.append("=" * 60)
    out.append("RUN ACCURACY REPORT")
    out.append("=" * 60)
    out.append(f"Run file:    {run_path.name}")
    out.append(f"Datasets:    {dataset_ids}")
    out.append(
        f"Total users: {total}  (bots: {len(actual_bots)}, humans: {len(actual_humans)})"
    )
    out.append("")
    out.append("COUNTS:")
    out.append(f"  True Positives:    {tp}")
    out.append(f"  True Negatives:    {tn}")
    out.append(f"  False Positives:   {fp}  ({pct_fp:.2f}%)")
    out.append(f"  False Negatives:   {fn}  ({pct_fn:.2f}%)")
    out.append(f"  Correct:           {tp + tn}  ({pct_correct:.2f}%)")
    out.append("")
    out.append(f"SCORE (+4 TP, -1 FN, -2 FP): {score}")

    # False Positives detail
    out.append("")
    out.append("-" * 60)
    out.append(f"FALSE POSITIVES — {fp} human(s) wrongly flagged as bot")
    out.append("-" * 60)
    if fp_ids:
        for uid in fp_ids:
            out.append("")
            out.append(format_user(users.get(uid, {"id": uid})))
    else:
        out.append("  (none)")

    # False Negatives detail
    out.append("")
    out.append("-" * 60)
    out.append(f"FALSE NEGATIVES — {fn} bot(s) missed")
    out.append("-" * 60)
    if fn_ids:
        for uid in fn_ids:
            out.append("")
            out.append(format_user(users.get(uid, {"id": uid})))
    else:
        out.append("  (none)")

    report = "\n".join(out)

    # Print to terminal
    print(report)

    # Write to results/
    RESULTS_DIR.mkdir(exist_ok=True)
    result_name = run_path.stem + ".results.txt"
    result_path = RESULTS_DIR / result_name
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nResults saved to: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
