# coding: utf-8
"""Local (offline) Wikidata KG access.

This module loads the preprocessed Wikidata triples shipped under:
  data/data/wikidata/*.txt

File format:
  Each line: <head>\t<relation>\t<tail>
with integer IDs that map to readable strings via:
  entities.dict:  <entity>\t<id>
  relations.dict: <relation>\t<id>

We build a lightweight adjacency map:
  adj[head_str][relation_str] -> list[tail_str]

This is meant to be a *drop-in* replacement for the online 1-hop retriever
(src.inference.wikidata_kg.WikidataKG) when you want to avoid all network calls.

Design goals:
  - Pure Python, no extra dependencies.
  - Lazy loading + small memory overhead (store only strings in dict->list).
  - Safe defaults: if files missing, return empty evidence and keep pipeline running.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence

from collections import defaultdict

from src.inference.kg_execute import EvidenceTriple


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _load_dict_tsv(path: Path) -> Dict[str, str]:
    """Load <text>\t<id> mapping into a {id_str: text} dict.

    We keep ids as strings to avoid int conversion costs and potential leading zeros.
    """

    id2text: Dict[str, str] = {}
    if not path.exists():
        return id2text
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "\t" in s:
                a, b = s.split("\t", 1)
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
            text = _norm(a)
            idx = _norm(b)
            if idx and text and idx not in id2text:
                id2text[idx] = text
    return id2text


@dataclass
class LocalWikidataKGConfig:
    root_dir: Path
    split: str = "train"  # train | valid | test | train_pre
    add_reverse_relations: bool = False


class LocalWikidataKG:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        split: str = "train",
        add_reverse_relations: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = str(split or "train")
        self.add_reverse_relations = bool(add_reverse_relations)

        self._loaded = False
        self._id2entity: Dict[str, str] = {}
        self._id2relation: Dict[str, str] = {}
        self._adj: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    def _paths(self) -> Dict[str, Path]:
        return {
            "entities": self.root_dir / "entities.dict",
            "relations": self.root_dir / "relations.dict",
            "triples": self.root_dir / f"{self.split}.txt",
        }

    def load(self) -> None:
        if self._loaded:
            return
        p = self._paths()
        self._id2entity = _load_dict_tsv(p["entities"])
        self._id2relation = _load_dict_tsv(p["relations"])

        triples_path = p["triples"]
        if not triples_path.exists():
            # keep empty but mark loaded
            self._loaded = True
            return

        with triples_path.open("r", encoding="utf-8", newline="") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split("\t")
                if len(parts) != 3:
                    # tolerate space-separated
                    parts = s.split()
                if len(parts) != 3:
                    continue

                h_id, r_id, t_id = (_norm(parts[0]), _norm(parts[1]), _norm(parts[2]))
                if not h_id or not r_id or not t_id:
                    continue

                h = self._id2entity.get(h_id, h_id)
                r = self._id2relation.get(r_id, r_id)
                t = self._id2entity.get(t_id, t_id)

                self._adj[h][r].append(t)

                if self.add_reverse_relations:
                    r_rev = f"{r}_reverse"
                    self._adj[t][r_rev].append(h)

        self._loaded = True

    def retrieve_1hop(
        self,
        *,
        head_entities: Sequence[str],
        relation_names: Sequence[str],
        relation_name_for_evidence: str,
        top_k: int = 3,
        scope: str = "local_wikidata",
    ) -> List[EvidenceTriple]:
        """Offline 1-hop retrieval.

        Parameters
        ----------
        head_entities:
            A list of *entity strings* (labels as used in the dicts), not IDs/QIDs.
        relation_names:
            One or more KG relation strings to query (from relations.dict after decoding).
        relation_name_for_evidence:
            What to write into EvidenceTriple.relation (usually gating's relation_candidate).
        """

        self.load()

        heads = [_norm(x) for x in head_entities if _norm(x)]
        rels = [_norm(x) for x in relation_names if _norm(x)]
        if not heads or not rels:
            return []

        ev: List[EvidenceTriple] = []
        k = max(0, int(top_k))

        for h in heads:
            rel_map = self._adj.get(h)
            if not rel_map:
                continue
            for r in rels:
                tails = rel_map.get(r) or []
                if not tails:
                    continue
                for t in (tails[:k] if k > 0 else tails):
                    ev.append(
                        EvidenceTriple(
                            head=h,
                            relation=_norm(relation_name_for_evidence) or r,
                            tail=t,
                            turn_id=-1,
                            scope=scope,
                        )
                    )
        return ev


__all__ = ["LocalWikidataKG", "LocalWikidataKGConfig"]

