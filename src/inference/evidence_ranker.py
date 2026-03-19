# coding: utf-8
"""Step 6: Evidence ranking across sources (MemoryKG + Wikidata).

This module unifies "in-KG retrieval strategy":
  - Input: top-k candidate evidence triples from MemoryKG and/or Wikidata
  - Output: ranked evidence (top-1 or top-m) plus a suggested pred_answer_value

Key heuristics (v1):
  1) Prefer Memory evidence over Wikidata evidence (dialogue consistency).
  2) Enforce lightweight type constraints by relation:
       - publication_year / award_year -> YEAR
       - num_books -> COUNT
     If mismatch, down-rank heavily.
  3) Prefer non-empty tail and typed literals when expected.

The output remains compatible with the existing JSONL schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.inference.kg_execute import EvidenceTriple


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


@dataclass(frozen=True)
class RankedEvidence:
    evidence: List[EvidenceTriple]
    pred_answer_value: str
    rank_debug: Dict[str, Any]


class EvidenceRanker:
    def __init__(
        self,
        *,
        memory_first: bool = True,
        type_mismatch_penalty: int = 100,
    ) -> None:
        self.memory_first = bool(memory_first)
        self.type_mismatch_penalty = int(type_mismatch_penalty)

    @staticmethod
    def _tail_type(tail: str) -> str:
        t = _norm(tail)
        if not t:
            return "EMPTY"
        if t.startswith("YEAR::"):
            return "YEAR"
        if t.startswith("COUNT::"):
            return "COUNT"
        if t.startswith("BOOL::"):
            return "BOOL"
        # also accept pure 4-digit year strings
        if len(t) == 4 and t.isdigit():
            return "YEAR"
        # accept integer-like
        if t.lstrip("-").isdigit():
            return "COUNT"
        return "ENTITY"

    @staticmethod
    def _expected_tail_type(relation: str) -> Optional[str]:
        r = _norm(relation)
        if r in {"publication_year", "award_year"}:
            return "YEAR"
        if r in {"num_books"}:
            return "COUNT"
        return None

    def _source_priority(self, scope: str) -> int:
        s = _norm(scope).casefold()
        if not self.memory_first:
            return 0
        # smaller is better
        if s in {"topic", "conv", "memory", "memorykg"}:
            return 0
        if s == "wikidata":
            return 1
        return 2

    def _score(self, ev: EvidenceTriple, *, relation: str) -> Tuple[int, int, int, int, int]:
        """Lower tuple is better (sort ascending)."""
        tail = _norm(getattr(ev, "tail", ""))
        scope = _norm(getattr(ev, "scope", ""))

        src = self._source_priority(scope)

        # type constraint
        exp = self._expected_tail_type(relation)
        tail_t = self._tail_type(tail)
        mismatch = 0
        if exp is not None and tail_t not in {exp}:
            mismatch = self.type_mismatch_penalty

        empty_pen = 1 if not tail else 0

        # prefer typed literals if expected
        typed_bonus = 0
        if exp == "YEAR" and tail.startswith("YEAR::"):
            typed_bonus = -1
        if exp == "COUNT" and tail.startswith("COUNT::"):
            typed_bonus = -1

        # for stability, keep more recent (turn_id larger) earlier
        try:
            turn_id = int(getattr(ev, "turn_id", -1))
        except (TypeError, ValueError):
            turn_id = -1

        # sort key: src, mismatch, empty, typed_bonus, -turn_id
        return (src, mismatch, empty_pen, typed_bonus, -turn_id)

    @staticmethod
    def _extract_answer_value_from_tail(tail: str) -> str:
        t = _norm(tail)
        for p in ("YEAR::", "COUNT::", "BOOL::"):
            if t.startswith(p):
                return t[len(p) :]
        return t

    def rank(
        self,
        *,
        candidates: Sequence[EvidenceTriple],
        relation: str,
        top_m: int = 1,
    ) -> RankedEvidence:
        items = [x for x in (candidates or []) if x is not None]
        if not items:
            return RankedEvidence(evidence=[], pred_answer_value="", rank_debug={"reason": "no_candidates"})

        scored: List[Tuple[Tuple[int, int, int, int, int], EvidenceTriple]] = []
        for ev in items:
            scored.append((self._score(ev, relation=relation), ev))

        scored.sort(key=lambda x: x[0])
        m = max(1, int(top_m))
        chosen = [ev for _, ev in scored[:m]]

        pred_answer_value = ""
        if chosen:
            pred_answer_value = self._extract_answer_value_from_tail(getattr(chosen[0], "tail", ""))

        dbg = {
            "relation": _norm(relation),
            "top_m": m,
            "scores": [
                {
                    "score": list(k),
                    "head": getattr(ev, "head", ""),
                    "relation": getattr(ev, "relation", ""),
                    "tail": getattr(ev, "tail", ""),
                    "turn_id": getattr(ev, "turn_id", None),
                    "scope": getattr(ev, "scope", ""),
                }
                for k, ev in scored[: min(20, len(scored))]
            ],
        }

        return RankedEvidence(evidence=chosen, pred_answer_value=pred_answer_value, rank_debug=dbg)


__all__ = ["EvidenceRanker", "RankedEvidence"]
