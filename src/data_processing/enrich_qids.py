# coding: utf-8
"""Enrich merged dialogue JSON with Wikidata QID fields.

Goal
----
For ranking metrics (Hits/MRR) and cleaner KG linking, we want each turn to
carry explicit QIDs when available.

This script:
  1) Extracts QID from turn["answer"] if it is a Wikidata entity URL.
     Writes: turn["answer_qid"].
  2) Optionally maps conversation-level seed_entities text -> QID using a local
     lookup table built from a (mention->QID) file.
     Writes: seed_entity["qid"].

Supported input
---------------
The repo's merged_dialogues/*.json typically contains a list of conversations:
  {
    "conv_id": ...,
    "seed_entities": [{"entity": "IT", "text": "IT"}, ...],
    "questions": [{"answer": "https://www.wikidata.org/wiki/Q...", ...}, ...]
  }

Usage (PowerShell)
------------------
python -m src.data_processing.enrich_qids \
  --in_json data/merged_dialogues/test.json \
  --out_json data/merged_dialogues/test.enriched.json \
  --mention_qid_tsv data/wikidata_convex/mention2qid.tsv

If you do not provide --mention_qid_tsv, only answer_qid extraction will run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_QID_RE = re.compile(r"(?:^|/)(Q\d+)(?:$|[?#/])", flags=re.IGNORECASE)


def _norm(x: Any) -> str:
    return "" if x is None else str(x).strip()


def extract_qid(x: Any) -> str:
    s = _norm(x)
    if not s:
        return ""
    m = _QID_RE.search(s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"Q\d+", s, flags=re.IGNORECASE):
        return s.upper()
    return ""


def load_mention2qid(path: Path) -> Dict[str, str]:
    """Load mention->qid mapping file.

Expected format per line:
  mention\tQ123
or
  mention Q123
"""

    m: Dict[str, str] = {}
    if not path or not path.exists():
        return m
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "\t" in s:
                a, b = s.split("\t", 1)
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
            a = _norm(a)
            b = extract_qid(b)
            if a and b and a.casefold() not in m:
                m[a.casefold()] = b
    return m


def iter_conversations(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return
    if isinstance(obj, dict):
        for k in ("dialogues", "conversations", "data"):
            v = obj.get(k)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x
                return
        if any(k in obj for k in ("conv_id", "questions", "turns")):
            yield obj


def get_turns(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("questions", "turns"):
        v = conv.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def enrich(
    data: Any,
    *,
    mention2qid: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Dict[str, int]]:
    stats = {
        "n_conversations": 0,
        "n_turns": 0,
        "n_answer_qid_added": 0,
        "n_seed_qid_added": 0,
    }
    m2q = mention2qid or {}

    # mutate in place
    for conv in iter_conversations(data):
        stats["n_conversations"] += 1

        # seed_entities enrichment
        seeds = conv.get("seed_entities")
        if isinstance(seeds, list):
            for se in seeds:
                if not isinstance(se, dict):
                    continue
                if _norm(se.get("qid")):
                    continue
                key = _norm(se.get("entity") or se.get("text")).casefold()
                if not key:
                    continue
                q = m2q.get(key, "")
                if q:
                    se["qid"] = q
                    stats["n_seed_qid_added"] += 1

        # turn enrichment
        for t in get_turns(conv):
            stats["n_turns"] += 1
            if not _norm(t.get("answer_qid")):
                q = extract_qid(t.get("answer"))
                if q:
                    t["answer_qid"] = q
                    stats["n_answer_qid_added"] += 1

    return data, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument(
        "--mention_qid_tsv",
        type=str,
        default="",
        help="Optional mention->QID mapping file (mention\\tQID).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_json)
    obj = json.loads(in_path.read_text(encoding="utf-8"))

    m2q = load_mention2qid(Path(args.mention_qid_tsv)) if _norm(args.mention_qid_tsv) else None
    enriched, stats = enrich(obj, mention2qid=m2q)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"in": str(in_path), "out": str(out_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

