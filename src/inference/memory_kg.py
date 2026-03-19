# coding: utf-8

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict


@dataclass(frozen=True)
class MemoryTriple:
    conv_id: str
    turn_id: int
    topic: str
    head: str
    relation: str
    tail: str


class MemoryKG:
    """
    Step 5: KG 执行检索基础（加载 + 索引）

    输入文件: data/preprocessed/dialogue_kg/memory_triples.tsv
    需要表头字段:
      conv_id, turn_id, topic, head, relation, tail

    构建索引:
    - triples_by_conv[conv_id] -> List[MemoryTriple]
    - triples_by_conv_topic[(conv_id, topic)] -> List[MemoryTriple]
    - adj_by_conv_topic_head[(conv_id, topic, head)] -> List[(relation, tail, turn_id)]
    - (optional) entity_vocab -> Set[str]
    """

    REQUIRED_FIELDS = ("conv_id", "turn_id", "topic", "head", "relation", "tail")
    _TYPED_LITERAL_PREFIXES = ("YEAR::", "COUNT::", "BOOL::")

    def __init__(
        self,
        memory_tsv_path: str | Path,
        *,
        build_entity_vocab: bool = True,
        drop_typed_literals_from_vocab: bool = True,
    ) -> None:
        self.memory_tsv_path = Path(memory_tsv_path)
        self.build_entity_vocab = bool(build_entity_vocab)
        self.drop_typed_literals_from_vocab = bool(drop_typed_literals_from_vocab)

        self.triples: List[MemoryTriple] = []

        self.triples_by_conv: Dict[str, List[MemoryTriple]] = defaultdict(list)
        self.triples_by_conv_topic: Dict[Tuple[str, str], List[MemoryTriple]] = defaultdict(list)
        self.adj_by_conv_topic_head: Dict[
            Tuple[str, str, str], List[Tuple[str, str, int]]
        ] = defaultdict(list)

        self.entity_vocab: Set[str] = set()

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        if s is None:
            return ""
        return str(s).strip()

    @staticmethod
    def _safe_int(x: Optional[str]) -> Optional[int]:
        try:
            return int(str(x).strip())
        except (TypeError, ValueError):
            return None

    def _is_typed_literal(self, s: str) -> bool:
        if not s:
            return False
        return any(s.startswith(p) for p in self._TYPED_LITERAL_PREFIXES)

    def load(self) -> None:
        """
        读取 TSV 并构建索引。可重复调用：每次都会清空并重建。
        """
        if not self.memory_tsv_path.exists():
            raise FileNotFoundError(f"missing file: {self.memory_tsv_path}")

        self.triples.clear()
        self.triples_by_conv.clear()
        self.triples_by_conv_topic.clear()
        self.adj_by_conv_topic_head.clear()
        self.entity_vocab.clear()

        with self.memory_tsv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("memory_triples.tsv has no header")

            field_set = set(reader.fieldnames)
            missing = [k for k in self.REQUIRED_FIELDS if k not in field_set]
            if missing:
                raise ValueError(f"memory_triples.tsv missing fields: {missing}")

            for row in reader:
                conv_id = self._norm(row.get("conv_id"))
                topic = self._norm(row.get("topic"))
                head = self._norm(row.get("head"))
                relation = self._norm(row.get("relation"))
                tail = self._norm(row.get("tail"))
                turn_id_raw = row.get("turn_id")
                turn_id = self._safe_int(turn_id_raw)

                # 基础字段缺失就跳过（避免索引污染）
                if not conv_id or not topic or not head or not relation or not tail or turn_id is None:
                    continue

                t = MemoryTriple(
                    conv_id=conv_id,
                    turn_id=turn_id,
                    topic=topic,
                    head=head,
                    relation=relation,
                    tail=tail,
                )
                self.triples.append(t)

                self.triples_by_conv[conv_id].append(t)
                self.triples_by_conv_topic[(conv_id, topic)].append(t)
                self.adj_by_conv_topic_head[(conv_id, topic, head)].append((relation, tail, turn_id))

                if self.build_entity_vocab:
                    self._maybe_add_vocab(topic)
                    self._maybe_add_vocab(head)
                    self._maybe_add_vocab(tail)

    def _maybe_add_vocab(self, s: str) -> None:
        s = self._norm(s)
        if not s:
            return
        if self.drop_typed_literals_from_vocab and self._is_typed_literal(s):
            return
        self.entity_vocab.add(s)

    # -----------------
    # Query helpers
    # -----------------
    def get_triples_by_conv(self, conv_id: str) -> List[MemoryTriple]:
        return list(self.triples_by_conv.get(self._norm(conv_id), []))

    def get_triples_by_conv_topic(self, conv_id: str, topic: str) -> List[MemoryTriple]:
        key = (self._norm(conv_id), self._norm(topic))
        return list(self.triples_by_conv_topic.get(key, []))

    def neighbors(self, conv_id: str, topic: str, head: str) -> List[Tuple[str, str, int]]:
        key = (self._norm(conv_id), self._norm(topic), self._norm(head))
        return list(self.adj_by_conv_topic_head.get(key, []))

    def outgoing_relations(
        self,
        *,
        conv_id: str,
        topic: Optional[str],
        head: str,
        fallback_to_conv: bool = True,
        fallback_to_global: bool = False,
    ) -> Set[str]:
        """Return outgoing relation names seen for a head in MemoryKG.

        Primary scope: (conv_id, topic) when topic is provided.

        Fallbacks:
          - if topic scope yields empty and fallback_to_conv: scan all triples in conv
          - if still empty and fallback_to_global: scan all triples in the whole store

        Notes:
          - Relations are returned as normalized strings.
          - This is intended for candidate relation pooling (R_local).
        """

        conv_n = self._norm(conv_id)
        topic_n = self._norm(topic) if topic is not None else ""
        head_n = self._norm(head)
        out: Set[str] = set()

        # 1) topic scope
        if conv_n and topic_n and head_n:
            for rel, _tail, _turn in self.neighbors(conv_n, topic_n, head_n):
                if rel:
                    out.add(self._norm(rel))

        # 2) conv scope fallback
        if (not out) and fallback_to_conv and conv_n and head_n:
            for t in self.get_triples_by_conv(conv_n):
                if t.head == head_n and t.relation:
                    out.add(self._norm(t.relation))

        # 3) global fallback
        if (not out) and fallback_to_global and head_n:
            for t in self.triples:
                if t.head == head_n and t.relation:
                    out.add(self._norm(t.relation))

        return out

    def local_relation_pool(
        self,
        *,
        conv_id: str,
        topic: Optional[str],
        head_candidates: Iterable[str],
        per_head_fallback_to_conv: bool = True,
        per_head_fallback_to_global: bool = False,
        max_relations: Optional[int] = None,
    ) -> List[str]:
        """Union outgoing relations for a list of head candidates.

        Returns a deterministic list (sorted), which makes tests / caching stable.
        """

        rels: Set[str] = set()
        for h in head_candidates:
            rels |= self.outgoing_relations(
                conv_id=conv_id,
                topic=topic,
                head=h,
                fallback_to_conv=per_head_fallback_to_conv,
                fallback_to_global=per_head_fallback_to_global,
            )

        out = sorted(rels)
        if max_relations is not None:
            out = out[: max(0, int(max_relations))]
        return out

    def iter_entities(self) -> Iterable[str]:
        """
        返回 entity_vocab（若未开启 build_entity_vocab，则为空）。
        """
        return iter(self.entity_vocab)


__all__ = [
    "MemoryTriple",
    "MemoryKG",
]

