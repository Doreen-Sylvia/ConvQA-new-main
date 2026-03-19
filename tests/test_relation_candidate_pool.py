import json
from pathlib import Path

from src.inference.memory_kg import MemoryKG
from src.inference.relation_candidate_pool import (
    RelationCandidatePool,
    WikidataPropertyPool,
)


def test_local_relation_pool_with_fallbacks(tmp_path: Path):
    tsv = tmp_path / "memory_triples.tsv"
    tsv.write_text(
        "conv_id\tturn_id\ttopic\thead\trelation\ttail\n"
        "c1\t1\tA\th1\tauthor\tt1\n"
        "c1\t2\tB\th1\tgenre\tt2\n",
        encoding="utf-8",
    )

    kg = MemoryKG(tsv, build_entity_vocab=False)
    kg.load()

    # topic scope hit
    rels_a = kg.local_relation_pool(conv_id="c1", topic="A", head_candidates=["h1"])
    assert rels_a == ["author"]

    # topic scope empty -> fallback to conv should recover both outgoing rels
    rels_missing = kg.local_relation_pool(conv_id="c1", topic="MISSING", head_candidates=["h1"], per_head_fallback_to_conv=True)
    assert set(rels_missing) == {"author", "genre"}


def test_merge_local_and_global_property_pool(tmp_path: Path):
    # tiny MemoryKG
    tsv = tmp_path / "memory_triples.tsv"
    tsv.write_text(
        "conv_id\tturn_id\ttopic\thead\trelation\ttail\n"
        "c1\t1\tA\th1\tauthor\tt1\n",
        encoding="utf-8",
    )
    kg = MemoryKG(tsv, build_entity_vocab=False)
    kg.load()

    # global property pool from repo data (json file)
    pool_path = Path(__file__).resolve().parents[1] / "data" / "wikidata_property_pool.json"
    rows = json.loads(pool_path.read_text(encoding="utf-8"))
    prop_pool = WikidataPropertyPool.from_dicts(rows)

    builder = RelationCandidatePool(memory_kg=kg, property_pool=prop_pool, top_m_global=3)

    out = builder.build(conv_id="c1", topic="A", head_candidates=["h1"], question_text="Who is the author?")

    # local relation -> mapped property id should come first
    assert out["local_property_ids"] == ["P50"]
    assert out["merged_property_ids"][0] == "P50"
    assert "P50" in out["merged_property_ids"]

