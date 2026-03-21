# coding: utf-8
"""
Step 4: Gating (C 的第二步)

修复点：
- 同步了 kg_extractor.py 中的 WIKIDATA_MAPPING 科学关系映射
- 确保推理时的 relation_candidate 与图谱中的 PID 完全一致
- 重构了 infer_relation 的正则优先级：优先匹配 year/when，防止被 written 劫持
- 更新了 needs_history 的判断集合，适配新的 PID
"""

from __future__ import annotations

import re
from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ==========================================
# 同步数据生成端的科学映射字典
# ==========================================
WIKIDATA_MAPPING = {
    'author': 'P50',
    'screenwriter': 'P58',
    'director': 'P57',
    'cast member': 'P161',
    'performer': 'P175',
    'sex or gender': 'P21',
    'date of birth': 'P569',
    'place of birth': 'P19',
    'date of death': 'P570',
    'place of death': 'P20',
    'publication date': 'P577',
    'genre': 'P136',
    'country of citizenship': 'P27',
    'employer': 'P108',
    'occupation': 'P106',
    'series': 'P179',
    'instance of': 'P31',
    'award received': 'P166',
    'composer': 'P86',
    'producer': 'P162',
    'main subject': 'P921',
    'father': 'P22',
    'mother': 'P25',
    'spouse': 'P26',
    'child': 'P40',
    'sibling': 'P3373',
    'influenced by': 'P737',
    'location': 'P276',

    'characters': 'P674',
    'followed_by': 'P156',
    'based_on': 'P128',
    'narrator': 'P2438',

    'first_book': 'P50_reverse_first',
    'final_book': 'P50_reverse_final',
    'num_books': 'num_books',
    'award_year': 'P166',
    'publication_year': 'P577',
    'book_title': 'P1476',
    'publisher': 'P123',
    'nationality': 'P27',
}


# -------------------------
# Shared relation inference
# -------------------------
def infer_relation(original_question: str) -> str:
    """
    与 extractor 侧完全保持一致的 relation 规则。
    输出的是映射后的 PID（如 P50, P31）
    """
    q = (original_question or "").strip().lower()
    inferred_raw = None

    # === 优先级 1：时间类问题（防止被 written, publish 劫持） ===
    if "year" in q and ("written" in q or "publish" in q):
        inferred_raw = "publication_year"
    elif "born" in q:
        if "where" in q or "city" in q or "country" in q or "place" in q:
            inferred_raw = "place of birth"
        else:
            inferred_raw = "date of birth"
    elif (q == "when") or ("when" in q):
        if any(k in q for k in ["award", "prize"]):
            inferred_raw = "award_year"
        else:
            inferred_raw = "publication_year"
    elif "year" in q:
        inferred_raw = "publication_year"

    # === 优先级 2：具体的实体角色 ===
    elif "protagonist" in q or "character" in q:
        inferred_raw = "characters"
    elif "sequel" in q or "followed by" in q or ("after" in q and "book" in q):
        inferred_raw = "followed_by"
    elif "movie" in q or "film" in q or "tv series" in q or "adaptation" in q:
        inferred_raw = "based_on"
    elif "narrator" in q or "narrating" in q:
        inferred_raw = "narrator"
    elif "wife" in q or "husband" in q or "spouse" in q or "married" in q:
        inferred_raw = "spouse"

    # === 优先级 3：书籍排序与数量 ===
    elif "first" in q and ("book" in q or "novel" in q):
        inferred_raw = "first_book"
    elif any(k in q for k in ["final", "ended", "concluded", "last book", "last novel"]):
        inferred_raw = "final_book"
    elif any(k in q for k in ["how many", "amount", "number of books", "number of", "how much"]):
        inferred_raw = "num_books"

    # === 优先级 4：通用属性 ===
    elif "nationality" in q or "country" in q:
        inferred_raw = "nationality"
    elif "author" in q or "wrote" in q or "written" in q:
        inferred_raw = "author"
    elif "title" in q:
        inferred_raw = "book_title"
    elif "award" in q or "prize" in q:
        inferred_raw = "award"
    elif "publish" in q:  # 包含 publisher, published, publishing
        inferred_raw = "publisher"
    elif "genre" in q:
        inferred_raw = "genre"
    else:
        inferred_raw = "related_to"

    # 映射到 PID
    if inferred_raw in WIKIDATA_MAPPING:
        return WIKIDATA_MAPPING[inferred_raw]
    if inferred_raw == 'award':
        return WIKIDATA_MAPPING.get('award received', 'P166')
    if inferred_raw == 'related_to':
        return "P31"

    return inferred_raw


# -------------------------
# Output schema
# -------------------------
@dataclass(frozen=True)
class GatingResult:
    use_history: bool
    history_turn_ids: List[int]
    head_candidates: List[str]
    relation_candidate: str
    gate_reason: str = ""


# -------------------------
# Gating core
# -------------------------
class Gating:
    _PRONOUN_PAT = re.compile(
        r"\b(when|where|who|whom|whose|which|it|he|she|they|them|this|that|these|those)\b",
        flags=re.IGNORECASE,
    )

    _ZH_TRIGGERS = (
        "他", "她", "它", "他们", "她们", "它们", "这", "那",
        "这个", "那个", "这些", "那些", "什么时候", "何时",
        "哪里", "在哪", "在哪儿", "谁",
    )

    _SHORT_Q_WORDS = {
        "when", "where", "who", "what", "which", "why", "how",
        "it", "he", "she", "they", "this", "that", "these", "those",
        "again", "then",
    }

    def __init__(
            self,
            *,
            recent_n_turns: int = 3,
            key_entities_top_m: int = 10,
            max_head_candidates: int = 30,
            min_entity_len: int = 2,
    ):
        self.recent_n_turns = max(0, int(recent_n_turns))
        self.key_entities_top_m = max(0, int(key_entities_top_m))
        self.max_head_candidates = max(1, int(max_head_candidates))
        self.min_entity_len = max(1, int(min_entity_len))

    @staticmethod
    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip()

    @staticmethod
    def _casefold(s: str) -> str:
        return s.casefold()

    @staticmethod
    def _safe_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    def _is_pronoun_or_short_question(self, q_t: str) -> bool:
        q = self._norm(q_t)
        if not q:
            return True
        q_cf = self._casefold(q)

        for z in self._ZH_TRIGGERS:
            if z in q:
                return True

        if self._PRONOUN_PAT.search(q):
            return True

        tokens = re.findall(r"[A-Za-z]+", q_cf)
        if 0 < len(tokens) <= 3 and all(t in self._SHORT_Q_WORDS for t in tokens):
            return True

        if len(q) <= 6:
            return True

        return False

    def _select_recent_turns_in_segment(
            self,
            assigned_seg: Optional[Dict[str, Any]],
            history_turns: Sequence[Dict[str, Any]],
    ) -> List[int]:
        if not assigned_seg or not history_turns:
            return []

        start_turn = self._safe_int(assigned_seg.get("start_turn"))
        end_turn = self._safe_int(assigned_seg.get("end_turn"))
        if start_turn is None or end_turn is None:
            return []

        in_seg: List[int] = []
        for t in history_turns:
            turn_id = self._safe_int(t.get("turn"))
            if turn_id is None:
                continue
            if start_turn <= turn_id <= end_turn:
                in_seg.append(turn_id)

        in_seg = sorted(set(in_seg))
        if self.recent_n_turns <= 0:
            return []
        return in_seg[-self.recent_n_turns:]

    def _collect_recent_answers_as_heads(
            self,
            history_turns: Sequence[Dict[str, Any]],
            selected_turn_ids: Sequence[int],
    ) -> List[str]:
        if not history_turns or not selected_turn_ids:
            return []

        sel = set(int(x) for x in selected_turn_ids)
        out: List[str] = []
        for t in sorted(history_turns, key=lambda x: (self._safe_int(x.get("turn")) or 10 ** 18)):
            turn_id = self._safe_int(t.get("turn"))
            if turn_id is None or turn_id not in sel:
                continue
            ans = self._norm(t.get("answer_text"))
            if ans and ans not in out:
                out.append(ans)
        return out

    def _match_known_entities_in_question(self, q_t: str, known_entities: Iterable[str]) -> List[str]:
        q = self._norm(q_t)
        q_cf = self._casefold(q)

        ents = []
        for e in known_entities:
            s = self._norm(e)
            if len(s) < self.min_entity_len:
                continue
            ents.append(s)

        ents.sort(key=lambda x: len(x), reverse=True)

        matched: List[str] = []
        seen_cf = set()
        for e in ents:
            e_cf = self._casefold(e)
            if not e_cf or e_cf in seen_cf:
                continue
            if e_cf in q_cf:
                seen_cf.add(e_cf)
                matched.append(e)
        return matched

    @staticmethod
    def _dedup_keep_order(items: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _triple_field(self, t: Any, key: str, default: Any = "") -> Any:
        if t is None:
            return default
        if isinstance(t, dict):
            return t.get(key, default)
        if is_dataclass(t) or hasattr(t, key):
            return getattr(t, key, default)
        return default

    def gate(
            self,
            *,
            q_t: str,
            assigned_seg: Optional[Dict[str, Any]],
            history_turns: Sequence[Dict[str, Any]],
            memory_store: Optional[Any] = None,
            conv_id: Optional[str] = None,
            known_entities: Optional[Iterable[str]] = None,
    ) -> GatingResult:

        q = self._norm(q_t)
        relation_candidate = infer_relation(q)

        history_turn_ids = self._select_recent_turns_in_segment(assigned_seg, history_turns)
        needs_history = self._is_pronoun_or_short_question(q)

        # ======== 重要修复：用新的 PID 更新需要历史的判定逻辑 ========
        if assigned_seg is not None and relation_candidate in {
            "P577",  # publication_year
            "P166",  # award received
            "P50",  # author
            "P27",  # nationality
            "P50_reverse_first",
            "P50_reverse_final",
            "num_books",
            "P19",  # place of birth
            "P569",  # date of birth
            "P26",  # spouse
        }:
            needs_history = True

        head_candidates: List[str] = []

        if known_entities is not None:
            head_candidates.extend(self._match_known_entities_in_question(q, known_entities))

        topic = self._norm(assigned_seg.get("topic")) if assigned_seg else ""
        if topic:
            head_candidates.append(topic)

        head_candidates.extend(self._collect_recent_answers_as_heads(history_turns, history_turn_ids))

        if assigned_seg:
            ke = assigned_seg.get("key_entities") or []
            if isinstance(ke, list) and ke:
                head_candidates.extend([self._norm(x) for x in ke[: self.key_entities_top_m]])

        if memory_store is not None and conv_id and topic:
            triples = []
            try:
                triples = memory_store.get_triples_by_conv_topic(str(conv_id), str(topic))
            except Exception:
                triples = []

            for t in triples or []:
                h = self._norm(self._triple_field(t, "head", ""))
                tail = self._norm(self._triple_field(t, "tail", ""))
                if h:
                    head_candidates.append(h)
                if tail:
                    head_candidates.append(tail)

        head_candidates = self._dedup_keep_order(head_candidates)[: self.max_head_candidates]

        if not head_candidates and topic:
            head_candidates = [topic]

        use_history = bool(needs_history and len(history_turn_ids) > 0)

        gate_reason = []
        if assigned_seg is None:
            gate_reason.append("no_assigned_seg")
        if needs_history:
            gate_reason.append("needs_history")
        if history_turn_ids:
            gate_reason.append(f"history_turns={len(history_turn_ids)}")
        gate_reason.append(f"rel={relation_candidate}")
        gate_reason.append(f"heads={len(head_candidates)}")

        return GatingResult(
            use_history=use_history,
            history_turn_ids=list(history_turn_ids),
            head_candidates=list(head_candidates),
            relation_candidate=relation_candidate,
            gate_reason=";".join(gate_reason),
        )


__all__ = [
    "GatingResult",
    "Gating",
    "infer_relation",
]