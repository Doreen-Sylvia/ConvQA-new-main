# coding: utf-8
"""Step 2: Dialogue-level retrieval decision (Controller/Critic).

This module makes an explicit 4-way decision for each turn:
  - USE_MEMORY
  - USE_WIKIDATA
  - ASK_CLARIFY
  - GENERATE

Design goals:
  - Rule-based, robust, low-cost.
  - Doesn't require network access.
  - Produces diagnostics-friendly outputs: decision, reason, and optional wikidata_query.

The controller can optionally run a *cheap* MemoryKG probe (topic/conv search
limited to top_k) to check answerability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.inference.kg_execute import EvidenceTriple, KGExecutor


Decision = str


@dataclass(frozen=True)
class ControlDecision:
    decision: Decision  # USE_MEMORY | USE_WIKIDATA | GENERATE | ASK_CLARIFY
    reason: str
    # reserved for future integration; keep schema stable
    wikidata_query: Optional[Dict[str, Any]] = None


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _looks_open_ended(q: str) -> bool:
    """Detect explanation/opinion/summarization questions that shouldn't hit KG."""
    t = _norm(q).casefold()
    if not t:
        return True

    triggers = (
        "why",
        "how do you",
        "how should",
        "explain",
        "opinion",
        "think",
        "feel",
        "summarize",
        "summary",
        "compare",
        "difference",
        # Chinese
        "为什么",
        "怎么",
        "如何",
        "解释",
        "总结",
        "概括",
        "评价",
        "看法",
        "对比",
        "区别",
    )
    return any(x in t for x in triggers)


def _too_short(q: str, *, max_tokens: int = 3, max_chars: int = 6) -> bool:
    s = _norm(q)
    if not s:
        return True
    if len(s) <= max_chars:
        return True
    toks = [x for x in s.split() if x]
    return 0 < len(toks) <= max_tokens


def _extract_from_router(router_result: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[int], str]:
    if not router_result:
        return None, ""
    seg_id = router_result.get("assigned_seg_id")
    try:
        seg_id_i = int(seg_id) if seg_id is not None else None
    except (TypeError, ValueError):
        seg_id_i = None
    topic = _norm(router_result.get("assigned_topic"))
    return seg_id_i, topic


def _extract_from_gating(gating_result: Optional[Mapping[str, Any]] = None) -> Tuple[bool, List[str], str]:
    if not gating_result:
        return False, [], ""
    use_history = bool(gating_result.get("use_history"))
    heads = gating_result.get("head_candidates") or []
    if not isinstance(heads, list):
        heads = []
    heads_n = [_norm(x) for x in heads if _norm(x)]
    rel = _norm(gating_result.get("relation_candidate"))
    return use_history, heads_n, rel


def _evidence_quality(evidence: Sequence[EvidenceTriple]) -> Tuple[bool, str]:
    """Very cheap quality check: non-empty tails, non-empty relation/head."""
    if not evidence:
        return False, "no_evidence"
    for ev in evidence:
        head = _norm(getattr(ev, "head", ""))
        rel = _norm(getattr(ev, "relation", ""))
        tail = _norm(getattr(ev, "tail", ""))
        if head and rel and tail:
            return True, "evidence_has_non_empty_triple"
    return False, "evidence_low_quality(empty_fields)"


class Controller:
    def __init__(
        self,
        *,
        open_ended_generate: bool = True,
        clarify_if_no_head: bool = True,
        clarify_if_related_to_and_short: bool = True,
    ) -> None:
        self.open_ended_generate = bool(open_ended_generate)
        self.clarify_if_no_head = bool(clarify_if_no_head)
        self.clarify_if_related_to_and_short = bool(clarify_if_related_to_and_short)

    def decide(
        self,
        *,
        question_text: str,
        router_result: Optional[Mapping[str, Any]] = None,
        gating_result: Optional[Mapping[str, Any]] = None,
        memory_executor: Optional[KGExecutor] = None,
        conv_id: Optional[str] = None,
        topic: Optional[str] = None,
        top_k: int = 3,
        current_turn: Optional[int] = None,
        allow_current_turn_evidence: bool = False,
    ) -> ControlDecision:
        q = _norm(question_text)

        # 1) open-ended => generate
        if self.open_ended_generate and _looks_open_ended(q):
            return ControlDecision("GENERATE", "open_ended_question")

        # 2) clarify when missing heads (avoid hallucinating)
        _seg_id, router_topic = _extract_from_router(router_result)
        _use_history, head_candidates, relation_candidate = _extract_from_gating(gating_result)
        topic_n = _norm(topic) or router_topic

        if self.clarify_if_no_head and not head_candidates:
            return ControlDecision("ASK_CLARIFY", "no_head_candidates")

        if (
            self.clarify_if_related_to_and_short
            and relation_candidate == "related_to"
            and _too_short(q)
        ):
            return ControlDecision("ASK_CLARIFY", "related_to_and_question_too_short")

        # 3) probe MemoryKG quickly (optional)
        if memory_executor is None or not conv_id:
            # can't probe -> default to memory first (conservative offline behavior)
            return ControlDecision("USE_MEMORY", "no_memory_probe_available")

        evidence = memory_executor.execute(
            conv_id=conv_id,
            topic=topic_n,
            head_candidates=head_candidates,
            relation_candidate=relation_candidate or None,
            top_k=max(1, int(top_k)),
            current_turn=current_turn,
            allow_current_turn=allow_current_turn_evidence,
        )

        ok, q_reason = _evidence_quality(evidence)
        if ok:
            return ControlDecision("USE_MEMORY", f"memory_probe_ok:{q_reason}")

        # 4) no/low-quality evidence => go wikidata (placeholder query structure)
        return ControlDecision(
            "USE_WIKIDATA",
            f"memory_probe_failed:{q_reason}",
            wikidata_query={
                "head_qids": [],
                "property_ids": [],
                "topic": topic_n or None,
                "relation_candidate": relation_candidate or None,
                "head_candidates": head_candidates[:10],
            },
        )


__all__ = [
    "ControlDecision",
    "Controller",
]
