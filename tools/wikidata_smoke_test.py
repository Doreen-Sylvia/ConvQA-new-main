# coding: utf-8
"""Minimal Wikidata connectivity smoke test.

Run:
  python tools\wikidata_smoke_test.py

It tests:
  1) wbsearchentities (entity linking)
  2) EntityData claims fetch (1-hop retrieval)

If your environment blocks wikidata.org, you'll see an exception.
"""

from __future__ import annotations

from src.inference.wikidata_el import EntityLinker
from src.inference.wikidata_kg import WikidataKG
from src.inference.wikidata_properties import map_relation_to_properties


def main() -> None:
    linker = EntityLinker(language="en", limit_per_mention=3, timeout_s=8.0)
    kg = WikidataKG(timeout_s=8.0, language="en")

    links = linker.link(head_candidates=["Douglas Adams"], question_text="Who is the author?")
    print("EL links (top-3):")
    for x in links[:3]:
        print(" ", x)

    if not links:
        print("No links returned. This usually means network issues or empty results.")
        return

    qid = links[0].qid
    props = map_relation_to_properties("author")
    ev = kg.retrieve_1hop(head_qids=[qid], property_ids=props, relation_name="author", top_k=3)
    print("\n1-hop evidence (top-3):")
    for e in ev:
        print(" ", e)


if __name__ == "__main__":
    main()
