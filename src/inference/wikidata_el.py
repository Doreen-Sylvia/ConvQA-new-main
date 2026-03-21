# coding: utf-8
"""Wikidata Entity Linking.
THIS MODULE IS DISABLED FOR OFFLINE MODE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class EntityLink:
    qid: str
    label: str
    description: str = ""
    match_text: str = ""
    score: float = 0.0


class EntityLinker:
    def __init__(self, *, language: str = "en", limit_per_mention: int = 3, timeout_s: float = 10.0) -> None:
        raise NotImplementedError("Online EntityLinker is disabled. Use local dictionary.")

    def link(self, *args, **kwargs) -> List[EntityLink]:
        raise NotImplementedError("Online EntityLinker is disabled. Use local dictionary.")


__all__ = ["EntityLink", "EntityLinker"]
