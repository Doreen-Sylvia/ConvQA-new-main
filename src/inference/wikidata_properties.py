# coding: utf-8
"""Relation -> Wikidata property mapping.

We keep a simple hand-maintained mapping for the first version.
"""

from __future__ import annotations

from typing import Dict, List


RELATION_TO_PROPERTIES: Dict[str, List[str]] = {
    "author": ["P50"],
    "writer": ["P50"],
    "nationality": ["P27"],
    "country": ["P17", "P27"],
    "genre": ["P136"],
    "publisher": ["P123"],
    "publication_year": ["P577"],
    "award": ["P166"],
    # award_year: not supported well in v1
    # "award_year": []
}


def map_relation_to_properties(relation: str) -> List[str]:
    r = (relation or "").strip()
    return list(RELATION_TO_PROPERTIES.get(r, []))
