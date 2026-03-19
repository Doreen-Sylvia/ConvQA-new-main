# coding: utf-8
"""Split comprehensive_merged_dialogues.json into train/valid/test.

This script is designed for large JSON files.

Input format (expected):
  A JSON list of conversations, where each item is a dict like:
    {"conv_id": "...", "questions": [...] , ...}

Output:
  - train.json
  - valid.json
  - test.json

We do a deterministic shuffle with a seed.

Usage (Windows cmd):
  python -m src.data_processing.split_merged_dialogues --input data/merged_dialogues/comprehensive_merged_dialogues.json --out_dir data/merged_dialogues --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1 --seed 42

Notes:
  - If the input is extremely large, this still loads the full JSON array in memory.
    For most datasets this is fine; if you need true streaming splitting, we can
    implement an incremental JSON parser later.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    s = float(train_ratio) + float(valid_ratio) + float(test_ratio)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {s}")


def split_conversations(
    conversations: List[Dict[str, Any]],
    *,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    _validate_ratios(train_ratio, valid_ratio, test_ratio)

    data = list(conversations)
    rnd = random.Random(int(seed))
    rnd.shuffle(data)

    n = len(data)
    n_train = int(n * float(train_ratio))
    n_valid = int(n * float(valid_ratio))
    # Put the remainder to test to ensure total matches.
    n_test = n - n_train - n_valid

    train = data[:n_train]
    valid = data[n_train : n_train + n_valid]
    test = data[n_train + n_valid :]

    assert len(train) + len(valid) + len(test) == n
    assert len(test) == n_test

    return {"train": train, "valid": valid, "test": test}


def main() -> None:
    # Resolve defaults relative to repository root so the script works regardless
    # of the current working directory (IDE/terminal differences).
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=str(repo_root / "data" / "merged_dialogues" / "comprehensive_merged_dialogues.json"),
        help="Path to comprehensive_merged_dialogues.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(repo_root / "data" / "merged_dialogues"),
        help="Output directory for train/valid/test JSON",
    )
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    root = _read_json(in_path)
    if not isinstance(root, list):
        raise ValueError("Input JSON must be a list of conversations")

    convs = [x for x in root if isinstance(x, dict)]
    if not convs:
        raise ValueError("No conversation dicts found in input JSON")

    splits = split_conversations(
        convs,
        train_ratio=float(args.train_ratio),
        valid_ratio=float(args.valid_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    _write_json(out_dir / "train.json", splits["train"])
    _write_json(out_dir / "valid.json", splits["valid"])
    _write_json(out_dir / "test.json", splits["test"])

    print(
        "[split] "
        f"n_total={len(convs)} n_train={len(splits['train'])} n_valid={len(splits['valid'])} n_test={len(splits['test'])} "
        f"ratios=({args.train_ratio},{args.valid_ratio},{args.test_ratio}) seed={args.seed}"
    )


if __name__ == "__main__":
    main()
