# coding: utf-8
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
from dateutil import parser as date_parser


_PREFIX_RE = re.compile(r"^(YEAR|COUNT|BOOL)::", flags=re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_answer(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = _PREFIX_RE.sub("", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    while t.endswith(".") or t.endswith("。"):
        t = t[:-1].rstrip()
    return t


def _match_date(pred: Any, gold: Any) -> bool:
    """归一化日期对比。如果能解析出相同日期（或同一年），则返回 True。"""
    s_pred = str(pred).strip()
    s_gold = str(gold).strip()
    if not s_pred or not s_gold:
        return False

    # 1. 快速检查：至少看起来像日期（含数字，或者月份名称，或者分隔符）
    has_year = re.search(r"\d{4}", s_pred) and re.search(r"\d{4}", s_gold)
    has_month = False
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    if any(m in s_pred.lower() for m in months) or any(m in s_gold.lower() for m in months):
        has_month = True
    
    if not (has_year or has_month):
        return False

    try:
        # 使用一个非常早的默认日期
        default_dt = datetime(1000, 1, 1)
        d1 = date_parser.parse(s_pred, default=default_dt, fuzzy=True)
        d2 = date_parser.parse(s_gold, default=default_dt, fuzzy=True)

        if d1.year == 1000 or d2.year == 1000:
             return False

        #精确匹配
        if d1.date() == d2.date():
            return True
        
        # 年份级匹配
        if d1.year == d2.year:
            if re.fullmatch(r"\d{4}", s_pred) or re.fullmatch(r"\d{4}", s_gold):
                return True

    except Exception:
        pass

    return False


def _is_match(pred: Any, gold: Any) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True

    # 防止把 'no' 匹配到 'novel' 里面去，如果是 yes/no 必须精确匹配
    if g in ['yes', 'no'] or p in ['yes', 'no']:
        return p == g

    # 规则1：包含匹配 (例如 Gold: "post-apocalyptic;horror", Pred: "post-apocalyptic novel")
    # 限制长度 >= 3，防止单个字母匹配成功
    if (len(g) >= 3 and g in p) or (len(p) >= 3 and p in g):
        return True

    # 规则2：日期归一化匹配
    if _match_date(pred, gold):
        return True

    return False

def _safe_get(d: Any, path: List[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _evidence_list(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = rec.get("pred_evidence", None)
    if ev is None:
        ev = rec.get("evidence", None)
    ev_list = _as_list(ev)
    out: List[Dict[str, Any]] = []
    for item in ev_list:
        if isinstance(item, dict):
            out.append(item)
    return out


def _extract_router_topic(rec: Dict[str, Any]) -> str:
    t = _safe_get(rec, ["router", "assigned_topic"], "")
    if t:
        return str(t)
    # fallback: meta.topic（main_infer 会写入）
    t = _safe_get(rec, ["meta", "topic"], "")
    return str(t) if t else ""


def _extract_relation_candidate(rec: Dict[str, Any]) -> str:
    rc = _safe_get(rec, ["gating", "relation_candidate"], "")
    return str(rc) if rc is not None else ""


def _extract_gold_fields(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    gold_answer_text / gold_topic / gold_relation 优先从 jsonl 顶层读取（main_infer 已写入）
    其次兼容 rec["gold"] 子结构
    """
    gold_answer = rec.get("gold_answer_text") or _safe_get(rec, ["gold", "answer_text"], "") or ""
    gold_topic = rec.get("gold_topic") or _safe_get(rec, ["gold", "topic"], "") or ""
    gold_relation = rec.get("gold_relation") or _safe_get(rec, ["gold", "relation"], "") or ""
    gold_tail = rec.get("gold_tail") or gold_answer or ""
    return str(gold_answer), str(gold_topic), str(gold_relation), str(gold_tail)


def _extract_qid(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"(?:^|/)(Q\d+)(?:$|[?#/])", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"Q\d+", s, flags=re.IGNORECASE):
        return s.upper()
    return ""


def _extract_gold_answer_entity(rec: Dict[str, Any]) -> str:
    """Best-effort gold entity id.

    Priority:
      1) gold_answer_entity / gold_answer_qid if present in JSONL
      2) gold_tail / gold_answer_text if it contains a QID or Wikidata URL
    """
    for k in ("gold_answer_qid", "gold_answer_entity", "gold_entity", "answer_entity"):
        q = _extract_qid(rec.get(k))
        if q:
            return q
    gold_answer, _, _, gold_tail = _extract_gold_fields(rec)
    q = _extract_qid(gold_tail)
    if q:
        return q
    return _extract_qid(gold_answer)


def _extract_ranked_entities(rec: Dict[str, Any]) -> List[str]:
    ranked = rec.get("pred_ranked_entities")
    if ranked is None:
        return []
    if isinstance(ranked, list):
        return [str(x) for x in ranked if str(x).strip()]
    return [str(ranked)]


def _to_qid_or_self(x: str) -> str:
    q = _extract_qid(x)
    return q if q else normalize_answer(x)


def _rank_of_gold(ranked: List[str], gold: str) -> Optional[int]:
    if not ranked or not gold:
        return None
    gold_key = _to_qid_or_self(gold)
    # match either by QID (preferred) or by normalized surface form
    for i, item in enumerate(ranked):
        if _to_qid_or_self(str(item)) == gold_key:
            return i + 1
    return None


def _has_non_empty_ranked(rec: Dict[str, Any]) -> bool:
    ranked = rec.get("pred_ranked_entities")
    return isinstance(ranked, list) and any(str(x).strip() for x in ranked)


def _extract_pred_answer_value(rec: Dict[str, Any]) -> str:
    # 优先用 pred_answer_value（main_infer 已写入）
    v = rec.get("pred_answer_value")
    if v is not None and str(v).strip():
        return str(v)
    # fallback: 用 pred_answer_text（不推荐）
    return str(rec.get("pred_answer_text") or rec.get("answer_text") or "")


def _evidence_contains_gold_tail(evidence: List[Dict[str, Any]], gold_tail: str) -> bool:
    g = normalize_answer(gold_tail)
    if not g:
        return False
    for ev in evidence:
        tail = normalize_answer(ev.get("tail", ""))
        if tail and tail == g:
            return True
    return False


@dataclass
class CounterBucket:
    n: int = 0
    em: int = 0
    ev_empty: int = 0
    routing_ok: int = 0
    ev_contains_gold: int = 0
    rel_ok: int = 0

    def add(
        self,
        *,
        em: bool,
        ev_empty: bool,
        routing_ok: Optional[bool],
        ev_contains_gold: Optional[bool],
        rel_ok: Optional[bool],
    ) -> None:
        self.n += 1
        self.em += int(bool(em))
        self.ev_empty += int(bool(ev_empty))
        if routing_ok is not None:
            self.routing_ok += int(bool(routing_ok))
        if ev_contains_gold is not None:
            self.ev_contains_gold += int(bool(ev_contains_gold))
        if rel_ok is not None:
            self.rel_ok += int(bool(rel_ok))

    def to_metrics(self) -> Dict[str, Any]:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "em": self.em / n,
            "evidence_empty_rate": self.ev_empty / n,
            "routing_accuracy": self.routing_ok / n,
            "evidence_contains_gold_rate": self.ev_contains_gold / n,
            "relation_accuracy": self.rel_ok / n,
        }


def evaluate_jsonl(preds_jsonl: Path) -> Dict[str, Any]:
    overall = CounterBucket()
    by_relation: Dict[str, CounterBucket] = defaultdict(CounterBucket)

    # Optional diagnostics (doesn't affect existing metrics):
    controller_counts: Dict[str, int] = defaultdict(int)
    evidence_scope_counts: Dict[str, int] = defaultdict(int)

    n_with_gold_answer = 0
    n_with_gold_topic = 0
    n_with_gold_relation = 0

    # ranking metrics (optional; only if gold entity and ranked list are present)
    rank_ranks: List[int] = []
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0}

    for rec in _read_jsonl(preds_jsonl):
        # controller (optional)
        c_dec = _safe_get(rec, ["controller", "decision"], "")
        if c_dec:
            controller_counts[str(c_dec)] += 1

        # evidence scope (optional)
        for ev in _evidence_list(rec):
            sc = ev.get("scope")
            if sc:
                evidence_scope_counts[str(sc)] += 1

        pred_answer_val = _extract_pred_answer_value(rec)
        gold_answer, gold_topic, gold_relation, gold_tail = _extract_gold_fields(rec)

        pred_topic = _extract_router_topic(rec)
        pred_relation = _extract_relation_candidate(rec)

        ev_list = _evidence_list(rec)

        em = False
        if gold_answer:
            # 使用我们新写的 _is_match 替代原本的 ==
            em = _is_match(pred_answer_val, gold_answer)
        ev_empty = len(ev_list) == 0

        routing_ok: Optional[bool] = None
        if gold_topic:
            n_with_gold_topic += 1
            routing_ok = normalize_answer(pred_topic) == normalize_answer(gold_topic)

        rel_ok: Optional[bool] = None
        if gold_relation:
            n_with_gold_relation += 1
            rel_ok = normalize_answer(pred_relation) == normalize_answer(gold_relation)

        ev_contains_gold: Optional[bool] = None
        if gold_tail or gold_answer:
            if gold_answer:
                n_with_gold_answer += 1
            ev_contains_gold = _evidence_contains_gold_tail(ev_list, gold_tail or gold_answer)

        overall.add(
            em=bool(em),
            ev_empty=bool(ev_empty),
            routing_ok=routing_ok,
            ev_contains_gold=ev_contains_gold,
            rel_ok=rel_ok,
        )

        bucket_key = pred_relation.strip() if pred_relation.strip() else "__NONE__"
        by_relation[bucket_key].add(
            em=bool(em),
            ev_empty=bool(ev_empty),
            routing_ok=routing_ok,
            ev_contains_gold=ev_contains_gold,
            rel_ok=rel_ok,
        )

        # ranking metrics: consume pred_ranked_entities + gold answer entity (QID)
        gold_ent = _extract_gold_answer_entity(rec)
        ranked = _extract_ranked_entities(rec)
        if gold_ent and ranked:
            r = _rank_of_gold(ranked, gold_ent)
            if r is not None:
                rank_ranks.append(int(r))
                for k in list(hits_at.keys()):
                    if r <= k:
                        hits_at[k] += 1

    metrics = {
        "preds_jsonl": str(preds_jsonl),
        "overall": overall.to_metrics(),
        "by_relation_candidate": {k: v.to_metrics() for k, v in sorted(by_relation.items(), key=lambda x: x[0])},
        "controller_decision_distribution": dict(sorted(controller_counts.items(), key=lambda x: x[0])),
        "evidence_scope_distribution": dict(sorted(evidence_scope_counts.items(), key=lambda x: x[0])),
        "coverage": {
            "n_total": overall.n,
            "n_with_gold_answer": n_with_gold_answer,
            "n_with_gold_topic": n_with_gold_topic,
            "n_with_gold_relation": n_with_gold_relation,
        },
        "ranking": {
            "n_ranked": len(rank_ranks),
            "hits@1": (hits_at[1] / max(len(rank_ranks), 1)) if rank_ranks else 0.0,
            "hits@3": (hits_at[3] / max(len(rank_ranks), 1)) if rank_ranks else 0.0,
            "hits@5": (hits_at[5] / max(len(rank_ranks), 1)) if rank_ranks else 0.0,
            "hits@10": (hits_at[10] / max(len(rank_ranks), 1)) if rank_ranks else 0.0,
            "mrr": (sum(1.0 / r for r in rank_ranks) / max(len(rank_ranks), 1)) if rank_ranks else 0.0,
            "mean_rank": (sum(rank_ranks) / max(len(rank_ranks), 1)) if rank_ranks else 0.0,
        },
    }
    return metrics


def _print_metrics(metrics: Dict[str, Any], *, top_relations: int = 50) -> None:
    overall = metrics.get("overall", {})
    print("=== Overall ===")
    print(f"n: {overall.get('n')}")
    print(f"EM: {overall.get('em'):.4f}")
    print(f"Evidence Empty Rate: {overall.get('evidence_empty_rate'):.4f}")
    print(f"Routing Accuracy: {overall.get('routing_accuracy'):.4f}")
    print(f"Evidence Contains Gold Rate: {overall.get('evidence_contains_gold_rate'):.4f}")
    print(f"Relation Accuracy: {overall.get('relation_accuracy'):.4f}")

    ranking = metrics.get("ranking", {})
    if isinstance(ranking, dict) and ranking.get("n_ranked", 0):
        print("\n=== Ranking (from pred_ranked_entities) ===")
        print(f"n_ranked: {ranking.get('n_ranked')}")
        print(f"Hits@1: {ranking.get('hits@1'):.4f}")
        print(f"Hits@3: {ranking.get('hits@3'):.4f}")
        print(f"Hits@5: {ranking.get('hits@5'):.4f}")
        print(f"Hits@10: {ranking.get('hits@10'):.4f}")
        print(f"Mean rank: {ranking.get('mean_rank'):.4f}")
        print(f"MRR: {ranking.get('mrr'):.6f}")
    else:
        print("\n=== Ranking (from pred_ranked_entities) ===")
        print("n_ranked: 0 (no gold entity id or no ranked list in JSONL)")

    print("\n=== By relation_candidate (show limited) ===")
    br: Dict[str, Any] = metrics.get("by_relation_candidate", {})
    keys = list(br.keys())[: max(int(top_relations), 0)]
    for k in keys:
        m = br[k]
        print(f"[{k}] n={m.get('n')} em={m.get('em'):.4f} empty_ev={m.get('evidence_empty_rate'):.4f}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds_jsonl",
        type=str,
        required=False,
        default=str(repo_root / "outputs" / "preds.jsonl"),
        help=r"推理输出 JSONL，例如 <repo_root>/outputs/preds.jsonl",
    )
    ap.add_argument("--metrics_json", type=str, default="", help="可选：保存 metrics.json 的路径")
    ap.add_argument("--top_relations", type=int, default=50, help="按 relation 分桶时打印前多少个桶")
    args = ap.parse_args()

    preds_path = Path(args.preds_jsonl)
    metrics = evaluate_jsonl(preds_path)
    _print_metrics(metrics, top_relations=int(args.top_relations))

    if str(args.metrics_json).strip():
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()