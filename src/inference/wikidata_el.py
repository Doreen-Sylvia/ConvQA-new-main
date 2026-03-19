# coding: utf-8
"""Wikidata Entity Linking.

Minimal implementation using the wbsearchentities API.

Input:
  - head_candidates: list[str]
  - question_text: str (optional, for future disambiguation)
Output:
  - list of dicts: {qid, label, description, match_text, score}

Note:
  This uses network access. If requests fail, callers should handle exceptions
  and fall back gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.inference.wikidata_http import build_url, get_json


@dataclass(frozen=True)
class EntityLink:
    qid: str
    label: str
    description: str = ""
    match_text: str = ""
    score: float = 0.0


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


class EntityLinker:
    def __init__(
        self,
        *,
        language: str = "en",
        limit_per_mention: int = 3,
        timeout_s: float = 10.0,
    ) -> None:
        self.language = str(language or "en")
        self.limit_per_mention = max(1, int(limit_per_mention))
        self.timeout_s = float(timeout_s)

    def link(
        self,
        *,
        head_candidates: Sequence[str],
        question_text: str = "",
        max_mentions: int = 5,
    ) -> List[EntityLink]:
        _ = _norm(question_text)

        mentions: List[str] = []
        for h in head_candidates or []:
            s = _norm(h)
            if not s:
                continue
            if s not in mentions:
                mentions.append(s)
            if len(mentions) >= max(1, int(max_mentions)):
                break

        out: List[EntityLink] = []
        for m in mentions:
            url = build_url(
                "https://www.wikidata.org/w/api.php",
                {
                    "action": "wbsearchentities",
                    "search": m,
                    "language": self.language,
                    "format": "json",
                    "limit": self.limit_per_mention,
                },
            )
            data = get_json(url, timeout_s=self.timeout_s)
            for item in (data or {}).get("search", []) or []:
                qid = _norm(item.get("id"))
                label = _norm(item.get("label"))
                if not qid:
                    continue
                out.append(
                    EntityLink(
                        qid=qid,
                        label=label,
                        description=_norm(item.get("description")),
                        match_text=m,
                        score=float(item.get("match", {}).get("score") or 0.0),
                    )
                )

        # global sort: (score desc, label length desc) to promote exact matches
        out.sort(key=lambda x: (float(x.score), len(x.label)), reverse=True)

        # de-dup by qid keep best
        dedup: List[EntityLink] = []
        seen = set()
        for x in out:
            if x.qid in seen:
                continue
            seen.add(x.qid)
            dedup.append(x)
        return dedup


__all__ = ["EntityLink", "EntityLinker"]
