# coding: utf-8
"""High-level Wikidata retriever that glues EL + property mapping + 1-hop KG.

Used when controller decision is USE_WIKIDATA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.inference.kg_execute import EvidenceTriple
from src.inference.wikidata_el import EntityLinker
from src.inference.wikidata_kg import WikidataKG
from src.inference.wikidata_properties import map_relation_to_properties


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


@dataclass(frozen=True)
class WikidataRetrieveResult:
    head_qids: List[str]
    head_labels: Dict[str, str]
    property_ids: List[str]
    evidence: List[EvidenceTriple]


class WikidataRetriever:
    def __init__(
        self,
        *,
        language: str = "en",
        timeout_s: float = 10.0,
        el_limit_per_mention: int = 3,
        max_head_qids: int = 3,
    ) -> None:
        self.entity_linker = EntityLinker(language=language, limit_per_mention=el_limit_per_mention, timeout_s=timeout_s)
        self.kg = WikidataKG(timeout_s=timeout_s, language=language)
        self.max_head_qids = max(1, int(max_head_qids))

    def retrieve(
        self,
        *,
        head_candidates: Sequence[str],
        relation_candidate: str,
        question_text: str,
        top_k: int = 3,
    ) -> Tuple[WikidataRetrieveResult, Dict[str, Any]]:
        rel = _norm(relation_candidate)
        props = map_relation_to_properties(rel)

        links = self.entity_linker.link(head_candidates=head_candidates, question_text=question_text)
        head_qids = [l.qid for l in links[: self.max_head_qids]]
        head_labels = {l.qid: l.label for l in links[: self.max_head_qids] if _norm(l.label)}

        evidence = self.kg.retrieve_1hop(head_qids=head_qids, property_ids=props, relation_name=rel, top_k=top_k)

        query_dbg = {
            "head_qids": head_qids,
            "head_labels": head_labels,
            "property_ids": props,
            "relation_candidate": rel or None,
            "head_candidates": list(head_candidates)[:10],
        }

        return (
            WikidataRetrieveResult(
                head_qids=head_qids,
                head_labels=head_labels,
                property_ids=props,
                evidence=evidence,
            ),
            query_dbg,
        )


__all__ = ["WikidataRetriever", "WikidataRetrieveResult"]
