# coding: utf-8
"""Wikidata 1-hop retrieval.

We fetch claims from the Wikidata entity data endpoint and extract
(head, property, tail) triples.

This is a minimal implementation:
  - Only supports ENTITY tail (QID) + common datavalue literals.
  - For year properties (P577), we convert to YEAR::YYYY.

Output triples are compatible with src.inference.kg_execute.EvidenceTriple
by setting:
  - turn_id = -1
  - scope = "wikidata"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.inference.kg_execute import EvidenceTriple
from src.inference.wikidata_http import build_url, get_json


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _extract_year(value: str) -> Optional[str]:
    # Wikidata time is often like '+1997-01-01T00:00:00Z'
    v = _norm(value)
    if not v:
        return None
    for ch in ("+", "-"):
        if v.startswith(ch):
            v2 = v[1:]
            if len(v2) >= 4 and v2[:4].isdigit():
                return v2[:4]
    if len(v) >= 4 and v[:4].isdigit():
        return v[:4]
    return None


class WikidataKG:
    def __init__(self, *, timeout_s: float = 10.0, language: str = "en") -> None:
        self.timeout_s = float(timeout_s)
        self.language = str(language or "en")

    def _get_entity_json(self, qid: str) -> Dict[str, Any]:
        url = build_url(
            "https://www.wikidata.org/wiki/Special:EntityData/{}.json".format(_norm(qid)),
            {},
        )
        return get_json(url, timeout_s=self.timeout_s) or {}

    def _get_label_for_qid(self, qid: str) -> str:
        """Resolve QID -> label (best-effort).

        This is used to turn raw QID tails into readable names so verbalizer/eval
        are more meaningful.
        """

        qid_n = _norm(qid)
        if not qid_n:
            return ""

        try:
            entity_json = self._get_entity_json(qid_n)
            _qid_used, ent = self._select_entity(entity_json, qid_n)
            labels = (ent.get("labels") or {}) if isinstance(ent, dict) else {}
            # prefer configured language, then English, then any label
            for lang in (self.language, "en"):
                v = ((labels.get(lang) or {}).get("value"))
                if _norm(v):
                    return _norm(v)
            if isinstance(labels, dict) and labels:
                any_lang = next(iter(labels.keys()))
                v = ((labels.get(any_lang) or {}).get("value"))
                return _norm(v)
        except Exception:
            return ""
        return ""

    def _select_entity(self, entity_json: Dict[str, Any], qid: str) -> Tuple[str, Dict[str, Any]]:
        """Select an entity dict from an EntityData response.

        Note: Special:EntityData sometimes returns entities keyed by numeric id (e.g. "42")
        rather than "Q42". We handle both and fall back to the first entity.
        """
        entities = (entity_json or {}).get("entities") or {}
        if not isinstance(entities, dict) or not entities:
            return _norm(qid), {}

        qid_n = _norm(qid)
        if qid_n in entities and isinstance(entities[qid_n], dict):
            return qid_n, entities[qid_n]

        # try numeric key: Q42 -> "42"
        if qid_n.upper().startswith("Q") and qid_n[1:].isdigit():
            k2 = qid_n[1:]
            if k2 in entities and isinstance(entities[k2], dict):
                return qid_n, entities[k2]

        # fallback to first entity
        k0 = next(iter(entities.keys()))
        ent0 = entities.get(k0)
        return (qid_n or str(k0), ent0 if isinstance(ent0, dict) else {})

    def _tail_to_text(self, snak: Dict[str, Any], *, property_id: str) -> Optional[str]:
        dv = (snak or {}).get("datavalue") or {}
        v = dv.get("value")
        if v is None:
            return None

        # entity
        if isinstance(v, dict) and v.get("entity-type") == "item":
            qid = _norm(v.get("id"))
            if not qid:
                return None
            # prefer readable label; fall back to qid
            label = self._get_label_for_qid(qid)
            return label or qid

        # time literal
        if isinstance(v, dict) and "time" in v:
            year = _extract_year(_norm(v.get("time")))
            if year and property_id in {"P577"}:
                return f"YEAR::{year}"
            if year:
                return year
            return None

        # quantity literal
        if isinstance(v, dict) and "amount" in v:
            amt = _norm(v.get("amount"))
            if amt.startswith("+"):
                amt = amt[1:]
            if amt:
                # keep integer-like quantities as COUNT
                if amt.replace("-", "").isdigit():
                    return f"COUNT::{amt}"
                return amt
            return None

        # string-like
        if isinstance(v, str):
            return v

        return _norm(v) or None

    def retrieve_1hop(
        self,
        *,
        head_qids: Sequence[str],
        property_ids: Sequence[str],
        relation_name: str,
        top_k: int = 3,
    ) -> List[EvidenceTriple]:
        qids = [_norm(x) for x in head_qids if _norm(x)]
        props = [_norm(x) for x in property_ids if _norm(x)]
        if not qids or not props:
            return []

        evidence: List[EvidenceTriple] = []

        for qid in qids:
            entity_json = self._get_entity_json(qid)
            _qid_used, ent = self._select_entity(entity_json, qid)
            head_label = (((ent.get("labels") or {}).get(self.language) or {}).get("value"))
            head_text = _norm(head_label) or qid

            claims = ent.get("claims") or {}
            for pid in props:
                for claim in (claims.get(pid) or []):
                    mainsnak = (claim or {}).get("mainsnak") or {}
                    tail = self._tail_to_text(mainsnak, property_id=pid)
                    if not tail:
                        continue
                    evidence.append(
                        EvidenceTriple(
                            head=head_text,
                            relation=_norm(relation_name) or pid,
                            tail=_norm(tail),
                            turn_id=-1,
                            scope="wikidata",
                        )
                    )

        # simple ranking: keep order of head_qids, but prefer non-empty literals over raw QIDs
        def _rank(ev: EvidenceTriple) -> Tuple[int, int]:
            tail = _norm(ev.tail)
            is_raw_qid = int(tail.startswith("Q") and tail[1:].isdigit())
            # prefer literals (not raw qid) -> is_raw_qid asc
            return (is_raw_qid, len(tail))

        evidence.sort(key=_rank)
        k = max(0, int(top_k))
        return evidence[:k] if k > 0 else evidence


__all__ = ["WikidataKG"]
