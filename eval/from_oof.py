"""Convert the included training script's OOF CSV into the comparison schema."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from compare import CLASSES, FIELDS


def convert(source: Path, output: Path):
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"train_relpath", "category", "predicted_category", "fold", "covered"}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError("Use oof_predictions.csv from the included training script, not submission.csv")
        all_rows = list(reader)
    if not all_rows or len({r["train_relpath"] for r in all_rows}) != len(all_rows):
        raise ValueError("Input is empty or contains duplicate sample IDs")
    for r in all_rows:
        if r["covered"].lower() not in {"true", "false", "1", "0"}:
            raise ValueError("Invalid covered flag")
        if not r["train_relpath"] or r["category"] not in CLASSES or not r["fold"]:
            raise ValueError("Invalid sample identity, true class or fold")
    rows = [r for r in all_rows if r["covered"].lower() in {"true", "1"}]
    if not rows or any(r["predicted_category"] not in CLASSES for r in rows):
        raise ValueError("No covered validation rows, or invalid predicted labels")
    # The full labeled ID/fold mapping is hashed; hyperparameters may differ between runs.
    identity = sorted((r["train_relpath"], r["category"], r["fold"]) for r in all_rows)
    split_id = "split-" + hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:20]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(ID=r["train_relpath"], true_label=r["category"],
                                 pred_label=r["predicted_category"], fold=r["fold"],
                                 split_id=split_id, source_kind="validation"))
    print(f"Converted {len(rows)}/{len(all_rows)} covered validation samples. split_id={split_id}")
    print("The split hash checks IDs, labels and folds, not image bytes or absence of training leakage.")
    return len(rows), split_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        convert(args.input, args.output)
    except (ValueError, KeyError, OSError, csv.Error) as exc:
        parser.exit(2, f"Cannot convert OOF: {exc}\n")


if __name__ == "__main__":
    main()
