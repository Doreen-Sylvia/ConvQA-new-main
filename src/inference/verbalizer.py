# coding:utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    # 可选：若工程中已存在 EvidenceTriple（来自 kg_execute.py），则用于类型提示与 isinstance 判断
    from src.inference.kg_execute import EvidenceTriple as _EvidenceTriple
except Exception:  # pragma: no cover
    _EvidenceTriple = None  # type: ignore


@dataclass(frozen=True)
class VerbalizeResult:
    answer_text: str
    evidence_lines: List[str]


class Verbalizer:
    """
    Step 6: 自然语言回答生成（只带核心证据）
    输入:
      - q_t: 当前问题文本（可用于未来扩展；当前主要用于空证据时的提示）
      - best_evidence: 1~3 条 triples（EvidenceTriple 或 dict 形态）
    输出:
      - answer_text: 自然语言回答
      - evidence_lines: 极简证据行列表
    """

    _TYPED_LITERAL_PREFIXES: Tuple[str, ...] = ("YEAR::", "COUNT::", "BOOL::")

    def __init__(
        self,
        *,
        drop_typed_prefix_in_evidence: bool = True,
        max_evidence_lines: int = 3,
    ) -> None:
        self.drop_typed_prefix_in_evidence = bool(drop_typed_prefix_in_evidence)
        self.max_evidence_lines = max(0, int(max_evidence_lines))

    @staticmethod
    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip()

    def _strip_typed_prefix(self, v: str) -> str:
        s = self._norm(v)
        if not self.drop_typed_prefix_in_evidence:
            return s
        for p in self._TYPED_LITERAL_PREFIXES:
            if s.startswith(p):
                return s[len(p) :]
        return s

    def _coerce_evidence_item(self, ev: Any) -> Optional[Dict[str, Any]]:
        """
        统一 evidence item 到 dict:
          {head, relation, tail, turn_id, scope}
        """
        if ev is None:
            return None

        # EvidenceTriple dataclass
        if _EvidenceTriple is not None and isinstance(ev, _EvidenceTriple):
            return {
                "head": ev.head,
                "relation": ev.relation,
                "tail": ev.tail,
                "turn_id": ev.turn_id,
                "scope": getattr(ev, "scope", ""),
            }

        # dict like
        if isinstance(ev, dict):
            return {
                "head": ev.get("head"),
                "relation": ev.get("relation"),
                "tail": ev.get("tail"),
                "turn_id": ev.get("turn_id"),
                "scope": ev.get("scope", ""),
            }

        return None

    @staticmethod
    def _safe_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    def _format_evidence_line(self, ev: Dict[str, Any]) -> str:
        turn_id = self._safe_int(ev.get("turn_id"))
        head = self._norm(ev.get("head"))
        rel = self._norm(ev.get("relation"))
        tail_raw = self._norm(ev.get("tail"))
        tail = self._strip_typed_prefix(tail_raw)

        t = "?"
        if turn_id is not None:
            t = str(turn_id)

        return f"turn={t}: ({head}, {rel}, {tail})"

    def _relation_template(self, head: str, relation: str, tail: str) -> str:
        """
        极简模板：覆盖常见关系；未覆盖则用通用句式。
        """
        h = self._norm(head)
        r = self._norm(relation)
        t = self._norm(tail)

        mapping = {
            "author": f"{h} 的作者是 {t}。",
            "writer": f"{h} 的作者是 {t}。",
            "nationality": f"{h} 的国籍是 {t}。",
            "country": f"{h} 的国家\\/地区是 {t}。",
            "publication_year": f"{h} 的出版年份是 {t}。",
            "award_year": f"{h} 的获奖年份是 {t}。",
            "award": f"{h} 获得的奖项是 {t}。",
            "publisher": f"{h} 的出版社是 {t}。",
            "genre": f"{h} 的类型是 {t}。",
            "first_book": f"{h} 的第一本书是 {t}。",
            "final_book": f"{h} 的最后一本书是 {t}。",
            "num_books": f"{h} 的书籍数量是 {t}。",
            "book_title": f"{h} 的标题是 {t}。",
        }
        if r in mapping:
            return mapping[r]
        return f"根据对话记忆，{h} 的 {r} 是 {t}。"

    def verbalize_clarify(self, question_text: str, *, missing: str = "") -> VerbalizeResult:
        """Deterministic clarification prompt.

        missing can be a coarse tag, e.g. 'entity', 'relation', 'time'.
        """
        q = self._norm(question_text)
        m = (missing or "").strip().lower()

        # Heuristics just to make the fixed template slightly more specific.
        if not m:
            q_cf = q.casefold()
            if any(x in q_cf for x in ["it", "he", "she", "they", "this", "that"]):
                m = "entity"
            elif any(x in q_cf for x in ["when", "year", "date", "time"]):
                m = "time"
            elif any(x in q_cf for x in ["what", "which"]):
                m = "relation"

        if m == "time":
            answer_text = "你问的‘when’指的是哪件事的时间？例如：出版时间／获奖时间／上映时间等。"
        elif m == "entity":
            answer_text = "你这里的‘it’（或他/她/它）具体指代哪本书／哪个人／哪个对象？能否 уточ уточ（澄清）一下？"
        elif m == "relation":
            answer_text = "你想问的是哪种属性关系？例如：作者／国籍／出版年份／奖项等。能否 уточ уточ（澄清）一下？"
        else:
            answer_text = "当前信息不足以确定你指的实体或关系，能否 уточ уточ（澄清）一下你的意图？"

        return VerbalizeResult(answer_text=answer_text, evidence_lines=[])

    def verbalize_generate(self, question_text: str) -> VerbalizeResult:
        """Conservative generation fallback (no KG grounding in v1)."""
        _ = self._norm(question_text)
        answer_text = "这个问题更像是解释/主观类问题，当前版本无法从知识图谱中直接检索得到确定答案。能否 уточ уточ（澄清）你希望查询的具体事实点？"
        return VerbalizeResult(answer_text=answer_text, evidence_lines=[])

    def verbalize(self, q_t: str, best_evidence: Sequence[Any]) -> VerbalizeResult:
        evidence_items: List[Dict[str, Any]] = []
        for x in list(best_evidence or [])[: max(0, self.max_evidence_lines)]:
            d = self._coerce_evidence_item(x)
            if d is not None:
                evidence_items.append(d)

        # evidence_lines
        evidence_lines = [self._format_evidence_line(ev) for ev in evidence_items]

        # answer_text
        if evidence_items:
            e0 = evidence_items[0]
            head = self._norm(e0.get("head"))
            rel = self._norm(e0.get("relation"))
            tail = self._strip_typed_prefix(self._norm(e0.get("tail")))

            answer_text = self._relation_template(head=head, relation=rel, tail=tail)

            # 多条证据：追加补充 tail（仅做非常轻量的补充）
            if len(evidence_items) > 1:
                tails: List[str] = []
                for ev in evidence_items[1:]:
                    t2 = self._strip_typed_prefix(self._norm(ev.get("tail")))
                    if t2 and t2 != tail and t2 not in tails:
                        tails.append(t2)
                if tails:
                    answer_text = answer_text.rstrip("。") + f"（补充：{ '；'.join(tails) }）。"

            return VerbalizeResult(answer_text=answer_text, evidence_lines=evidence_lines)

        # 无证据：固定澄清文案（按需求保留“уточ уточ”提示）
        _ = self._norm(q_t)
        answer_text = "当前对话记忆中未找到相关事实，能否 уточ уточ（澄清）你指的实体\\/奖项\\/时间？"
        return VerbalizeResult(answer_text=answer_text, evidence_lines=[])


__all__ = [
    "VerbalizeResult",
    "Verbalizer",
]
