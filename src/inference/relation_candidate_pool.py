# coding: utf-8
"""Hybrid relation candidate pool.

Implements the design:
  1) R_local: local (conv_id, topic) outgoing relations from MemoryKG (strong constraint)
  2) P_pool: global Wikidata property pool with text description for semantic match (coverage)
  3) Merge: start with mapped local relations, then add top-M global properties by similarity

The module is intentionally lightweight (no heavy ML deps) and uses a TF-IDF cosine
similarity baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.inference.memory_kg import MemoryKG
from src.inference.wikidata_properties import map_relation_to_properties


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


@dataclass(frozen=True)
class WikidataProperty:
    pid: str
    text: str  # label/aliases/description concatenated


class WikidataPropertyPool:
    """In-memory property pool with a fast semantic top-M query."""

    def __init__(self, properties: Sequence[WikidataProperty]):
        self.properties: List[WikidataProperty] = [p for p in properties if _norm(p.pid) and _norm(p.text)]
        self._token_cache: Optional[List[Set[str]]] = None

    @classmethod
    def from_dicts(cls, rows: Sequence[Dict[str, str]]) -> "WikidataPropertyPool":
        props: List[WikidataProperty] = []
        for r in rows:
            pid = _norm(r.get("pid") or r.get("id") or r.get("property_id"))
            text = _norm(r.get("text") or r.get("label") or "")
            if not text:
                # try common fields
                parts = [
                    _norm(r.get("label")),
                    _norm(r.get("aliases")),
                    _norm(r.get("description")),
                ]
                text = " ".join([p for p in parts if p])
            if pid and text:
                props.append(WikidataProperty(pid=pid, text=text))
        return cls(props)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        # Lightweight normalization + tokenization; avoids heavy ML deps.
        t = _norm(text).lower()
        if not t:
            return set()
        t = re.sub(r"[^a-z0-9]+", " ", t)
        toks = [x for x in t.split() if len(x) >= 2]
        return set(toks)

    def _ensure_index(self) -> None:
        if self._token_cache is not None:
            return
        self._token_cache = [self._tokenize(p.text) for p in self.properties]

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter <= 0:
            return 0.0
        union = len(a | b)
        return float(inter) / float(union) if union else 0.0

    def top_m(self, question_text: str, *, m: int = 30, exclude_pids: Optional[Set[str]] = None) -> List[str]:
        """Return top-M property ids by semantic similarity to question_text.

        Similarity is a lightweight token Jaccard score between question_text and
        property text (label/aliases/description).
        """

        q = _norm(question_text)
        if not q or not self.properties:
            return []
        self._ensure_index()
        token_cache = self._token_cache or []
        q_tokens = self._tokenize(q)

        excl = {(_norm(x)) for x in (exclude_pids or set()) if _norm(x)}
        scored: List[Tuple[float, str]] = []
        for i, p in enumerate(self.properties):
            pid = _norm(p.pid)
            if not pid or pid in excl:
                continue
            scored.append((self._jaccard(q_tokens, token_cache[i] if i < len(token_cache) else set()), pid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pid for _score, pid in scored[: max(0, int(m))]]


class RelationCandidatePool:
    """Build merged candidate property list C = R_local + topM(P_pool)."""

    def __init__(
        self,
        *,
        memory_kg: MemoryKG,
        property_pool: WikidataPropertyPool,
        top_m_global: int = 30,
        fallback_to_conv_if_topic_empty: bool = True,
        fallback_to_global_if_conv_empty: bool = False,
    ) -> None:
        self.memory_kg = memory_kg
        self.property_pool = property_pool
        self.top_m_global = max(0, int(top_m_global))
        self.fallback_to_conv_if_topic_empty = bool(fallback_to_conv_if_topic_empty)
        self.fallback_to_global_if_conv_empty = bool(fallback_to_global_if_conv_empty)

    def build(
        self,
        *,
        conv_id: str,
        topic: Optional[str],
        head_candidates: Sequence[str],
        question_text: str,
        max_local_relations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return merged candidate properties and debug info.

        Output:
          {
            "local_relations": [...],
            "local_property_ids": [...],
            "global_property_ids": [...],
            "merged_property_ids": [...],
          }
        """

        # 1) Local relation strings from MemoryKG
        local_rels = self.memory_kg.local_relation_pool(
            conv_id=_norm(conv_id),
            topic=_norm(topic) if topic is not None else None,
            head_candidates=[_norm(h) for h in head_candidates if _norm(h)],
            per_head_fallback_to_conv=self.fallback_to_conv_if_topic_empty,
            per_head_fallback_to_global=self.fallback_to_global_if_conv_empty,
            max_relations=max_local_relations,
        )

        # 2) Map local relation strings -> Wikidata PIDs
        local_pids: List[str] = []
        for r in local_rels:
            for pid in map_relation_to_properties(r):
                pid_n = _norm(pid)
                if pid_n and pid_n not in local_pids:
                    local_pids.append(pid_n)

        # 3) Global semantic expansion
        global_pids = self.property_pool.top_m(
            question_text=question_text,
            m=self.top_m_global,
            exclude_pids=set(local_pids),
        )

        merged = list(local_pids)
        for pid in global_pids:
            if pid not in merged:
                merged.append(pid)

        return {
            "local_relations": local_rels,
            "local_property_ids": local_pids,
            "global_property_ids": global_pids,
            "merged_property_ids": merged,
        }


__all__ = [
    "WikidataProperty",
    "WikidataPropertyPool",
    "RelationCandidatePool",
]



