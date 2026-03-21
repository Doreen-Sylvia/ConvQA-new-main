# coding: utf-8
"""
推理入口脚本（main）：全链路跑通，并把每轮推理结果落盘为 JSONL（供 eval/diagnostics 读取）

修复点：
1) JSONL 写入 gold 字段：gold_answer_text / gold_topic / gold_relation
2) Segmenter 调用：优先传 turns list（Segmenter 常见签名），避免 no_segments
3) 输出 pred_answer_value：用于 EM（从 top-1 evidence.tail 提取）
4) KG 执行加入时间因果过滤：不允许使用未来 turn 的证据（<= current_turn 可配置）
5) turn id 统一：优先用数据字段 turn（通常从 0 开始），否则 fallback idx
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.inference.router import Router
from src.inference.gating import Gating
from src.inference.memory_kg import MemoryKG
from src.inference.kg_execute import KGExecutor, EvidenceTriple
from src.inference.verbalizer import Verbalizer
from src.inference.controller import Controller
from src.inference.wikidata_retriever import WikidataRetriever
from src.inference.evidence_ranker import EvidenceRanker


def _maybe_tqdm(it: Any, *, desc: str = "", total: Optional[int] = None, enable: bool = True) -> Any:
    """Best-effort tqdm wrapper.

    - If tqdm is installed and enable=True, return tqdm(it,...)
    - Otherwise, return the iterator unchanged.
    """
    if not enable:
        return it
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, desc=desc, total=total)
    except Exception:
        return it


def _fmt_seconds(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


class _Progress:
    def __init__(self, *, enable: bool = True) -> None:
        self.enable = bool(enable)
        self.t0 = time.time()
        self.last_print = 0.0
        self.turn_done = 0

    def maybe_print(
        self,
        *,
        conv_i: int,
        conv_total: int,
        turn_i: int,
        turn_total: int,
        conv_id: str,
        decision: str,
        every_s: float = 2.0,
    ) -> None:
        if not self.enable:
            return
        now = time.time()
        if (now - self.last_print) < float(every_s):
            return
        self.last_print = now
        elapsed = now - self.t0
        done = max(self.turn_done, 1)
        rate = elapsed / done
        # ETA computed if total turns known (>0)
        eta = ""
        if turn_total > 0 and conv_total > 0:
            # use within-conv eta as minimal reliable
            remain = max(turn_total - (turn_i + 1), 0)
            eta = f" ETA~{_fmt_seconds(remain * rate)}"
        print(
            f"[progress] conv {conv_i+1}/{conv_total} ({conv_id}) turn {turn_i+1}/{turn_total} "
            f"done={self.turn_done} elapsed={_fmt_seconds(elapsed)}{eta} decision={decision}",
            flush=True,
        )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _coerce_dialogues(root: Any) -> List[Dict[str, Any]]:
    if isinstance(root, list):
        return [x for x in root if isinstance(x, dict)]
    if isinstance(root, dict):
        for k in ("dialogues", "conversations", "data"):
            v = root.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if any(key in root for key in ("conv_id", "questions", "turns")):
            return [root]
    return []


def _get_conv_id(conv: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("conv_id", "conversation_id", "id", "convId"):
        v = _norm(conv.get(k))
        if v:
            return v
    return str(fallback_idx)


def _get_turns(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("questions", "turns"):
        v = conv.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _get_question_text(turn: Dict[str, Any]) -> str:
    for k in ("question", "question_text", "q", "q_t", "text"):
        s = _norm(turn.get(k))
        if s:
            return s
    return ""


def _get_original_question_text(turn: Dict[str, Any]) -> str:
    # 你的 merged 数据里通常是 original_question
    for k in ("original_question", "question_original", "orig_question"):
        s = _norm(turn.get(k))
        if s:
            return s
    return _get_question_text(turn)


def _turn_id(turn: Dict[str, Any], fallback_idx0: int) -> int:
    """
    turn id 优先取数据字段 turn（你的 merged 样例是从 0 开始）。
    如果缺失，用 fallback_idx0（也是从 0 开始）保证与“在线因果过滤”一致。
    """
    for k in ("turn", "turn_id", "tid"):
        v = _safe_int(turn.get(k))
        if v is not None:
            return v
    return int(fallback_idx0)


def _as_serializable_evidence(ev: EvidenceTriple) -> Dict[str, Any]:
    if is_dataclass(ev):
        d = asdict(ev)
    else:
        d = {
            "head": getattr(ev, "head", ""),
            "relation": getattr(ev, "relation", ""),
            "tail": getattr(ev, "tail", ""),
            "turn_id": getattr(ev, "turn_id", None),
            "scope": getattr(ev, "scope", ""),
        }

    # Best-effort: if tail looks like a Wikidata entity URL, also expose tail_qid.
    tail = _norm(d.get("tail"))
    m = re.search(r"(?:^|/)(Q\d+)(?:$|[?#/])", tail)
    if m:
        d.setdefault("tail_qid", m.group(1))
    return d


def _extract_qid(x: Any) -> str:
    """Extract Wikidata QID from url / raw string. Return "" if none."""
    s = _norm(x)
    if not s:
        return ""
    m = re.search(r"(?:^|/)(Q\d+)(?:$|[?#/])", s)
    if m:
        return m.group(1)
    # raw Q123
    if re.fullmatch(r"Q\d+", s, flags=re.IGNORECASE):
        return s.upper()
    return ""


def _extract_turn_answer_qid(turn: Dict[str, Any]) -> str:
    """Best-effort gold answer entity id (QID) from turn fields."""
    # preferred explicit field
    q = _extract_qid(turn.get("answer_qid"))
    if q:
        return q
    # sometimes answer is a Wikidata URL
    q = _extract_qid(turn.get("answer"))
    if q:
        return q
    return ""


def _build_ranked_entities(
    *,
    evidence_candidates: List[EvidenceTriple],
    relation: str,
    top_n: int,
) -> Dict[str, Any]:
    """Return {pred_ranked_entities, pred_ranked_entity_scores}.

    We reuse EvidenceRanker internal ordering by calling rank(top_m=len(items)),
    then take the tails in order, deduplicate, and keep top_n.
    """
    out_entities: List[str] = []
    out_scores: List[float] = []
    if not evidence_candidates:
        return {"pred_ranked_entities": [], "pred_ranked_entity_scores": []}

    try:
        rrk_all = EvidenceRanker().rank(candidates=evidence_candidates, relation=relation, top_m=len(evidence_candidates))
        ranked_evs: List[EvidenceTriple] = list(rrk_all.evidence)
        dbg_scores = rrk_all.rank_debug.get("scores", []) if isinstance(rrk_all.rank_debug, dict) else []
    except Exception:
        ranked_evs = list(evidence_candidates)
        dbg_scores = []

    score_by_key: Dict[tuple, float] = {}
    # Convert tuple score -> a float proxy (smaller tuple = better => higher float)
    for row in dbg_scores:
        if not isinstance(row, dict):
            continue
        try:
            tup = tuple(int(x) for x in (row.get("score") or []))
        except Exception:
            continue
        k = (
            _norm(row.get("head")),
            _norm(row.get("relation")),
            _norm(row.get("tail")),
            row.get("turn_id"),
            _norm(row.get("scope")),
        )
        # more negative is worse; we invert the sum to get a monotonic proxy
        score_by_key[k] = -float(sum(tup))

    seen: set[str] = set()
    for ev in ranked_evs:
        # Prefer QID if available, otherwise use tail string.
        tail_raw = _norm(getattr(ev, "tail", ""))
        tail_qid = _extract_qid(tail_raw)
        tail = tail_qid or tail_raw
        if not tail:
            continue
        if tail in seen:
            continue
        seen.add(tail)
        out_entities.append(tail)
        k = (tail,)
        ev_key = (
            _norm(getattr(ev, "head", "")),
            _norm(getattr(ev, "relation", "")),
            tail_raw,
            getattr(ev, "turn_id", None),
            _norm(getattr(ev, "scope", "")),
        )
        out_scores.append(score_by_key.get(ev_key, float("nan")))
        if len(out_entities) >= max(0, int(top_n)):
            break

    return {
        "pred_ranked_entities": out_entities,
        "pred_ranked_entity_scores": out_scores,
    }


def _predict_relation_from_question(q: str) -> str:
    """
    gold_relation 生成：用与你 extractor/预测端一致的轻量规则。
    先保证评测口径一致；后续你可以替换成共享函数或更完整的规则。
    """
    t = q.strip().lower()
    # order matters
    if "how many" in t or "amount" in t or "number of books" in t:
        return "num_books"
    if "first" in t and "book" in t:
        return "first_book"
    if ("final" in t or "ended" in t or "concluded" in t or "last book" in t) and "book" in t:
        return "final_book"
    if "author" in t or "writer" in t or "wrote" in t:
        return "author"
    if "nationality" in t or "country" in t:
        return "nationality"
    if "publisher" in t:
        return "publisher"
    if "genre" in t:
        return "genre"
    if "award" in t or "prize" in t:
        # when? + award -> award_year (如果你 KG 里有此关���)
        if "when" in t or "year" in t:
            return "award_year"
        return "award"
    if "year" in t or "when" in t:
        return "publication_year"
    if "title" in t:
        return "book_title"
    return "related_to"


def _extract_pred_answer_value(
    *,
    pred_answer_text: str,
    evidence: List[EvidenceTriple],
) -> str:
    """
    为 EM 评测提供一个“答案值”字段：
    - 优先用 top-1 evidence.tail（最可靠，且与 gold_answer_text 同空间）
    - 否��尝试从 pred_answer_text 里抽取（兜底）
    """
    if evidence:
        ev0 = evidence[0]
        tail = getattr(ev0, "tail", "")
        if tail:
            return str(tail)

    # fallback: 简单从句子里抽取最后一个“词/数字”，不保证完美
    s = _norm(pred_answer_text)
    if not s:
        return ""
    # 去掉中文句号
    s = s.rstrip("。.")
    # 常见模板 “X 的 Y 是 Z”
    m = re.search(r"是\s*([^\s，,]+)\s*$", s)
    if m:
        return m.group(1)
    # 否则取最后一个 token
    parts = re.split(r"\s+", s)
    return parts[-1] if parts else ""


# 1) 修改：_try_segment() 增加 memory_tsv_path 参数，并传给 Segmenter
def _try_segment(
    turns: List[Dict[str, Any]],
    conv: Dict[str, Any],
    *,
    memory_tsv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    让 Segmenter 读取 memory_triples.tsv，从而附加 key_entities/key_relations。
    """
    try:
        from src.inference.segmenter import Segmenter  # type: ignore
    except Exception:
        return []

    try:
        segger = Segmenter(memory_tsv_path=memory_tsv_path)  # <-- 新增：传 memory_tsv_path
        segs = None

        try:
            segs = segger.segment(turns)  # type: ignore
        except Exception:
            segs = None

        if segs is None:
            try:
                segs = segger.segment(conv)  # type: ignore
            except Exception:
                segs = None

        if isinstance(segs, list):
            return [s for s in segs if isinstance(s, dict)]
        return []
    except Exception:
        return []


class ResultWriter:
    """
    以 JSONL 方式落盘：每次 write_one 追加写一行 JSON。
    """

    def __init__(self, output_path: Path, *, append: bool = True) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        self._fp = self.output_path.open(self._mode, encoding="utf-8", newline="\n")

    def write_one(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_inference(
    *,
    merged_json_path: Path,
    memory_tsv_path: Path,
    out_path: Optional[Path],
    output_jsonl_path: Optional[Path],
    top_k: int,
    recent_n_turns: int,
    print_stdout: bool,
    allow_current_turn_evidence: bool,
    enable_progress: bool = True,
    local_wikidata_el_dict: str = "",
    local_wikidata_kg_dir: str = "",
    local_wikidata_kg_split: str = "train",
    write_topic_subgraph: bool = False,
    topic_subgraph_max_triples: int = 300,
) -> Dict[str, Any]:
    root = _read_json(merged_json_path)
    dialogues = _coerce_dialogues(root)

    kg = MemoryKG(memory_tsv_path, build_entity_vocab=True)
    kg.load()

    router = Router(allow_new=False)
    gating = Gating(recent_n_turns=recent_n_turns)
    executor = KGExecutor(kg)
    verbalizer = Verbalizer()
    controller_mod = Controller()
    wikidata_retriever = WikidataRetriever(
        local_el_dict_path=str(local_wikidata_el_dict) if _norm(local_wikidata_el_dict) else None,
        local_wikidata_kg_dir=str(local_wikidata_kg_dir) if _norm(local_wikidata_kg_dir) else None,
        local_wikidata_kg_split=str(local_wikidata_kg_split or "train"),
    )
    evidence_ranker = EvidenceRanker()

    results: Dict[str, Any] = {
        "merged_json": str(merged_json_path),
        "memory_tsv": str(memory_tsv_path),
        "num_dialogues": len(dialogues),
        "conversations": [],
    }

    known_entities = list(kg.iter_entities())

    writer_cm = ResultWriter(output_jsonl_path) if output_jsonl_path is not None else None
    writer = writer_cm.__enter__() if writer_cm is not None else None

    try:
        prog = _Progress(enable=enable_progress)
        conv_iter = _maybe_tqdm(
            list(enumerate(dialogues)),
            desc="Conversations",
            total=len(dialogues),
            enable=enable_progress,
        )
        for i, conv in conv_iter:
            conv_id = _get_conv_id(conv, i)
            turns_all = _get_turns(conv)

            segments = _try_segment(
                turns_all,
                conv,
                memory_tsv_path=memory_tsv_path,  # <-- 新增：把 memory_tsv_path 传进去
            )

            conv_out: Dict[str, Any] = {
                "conv_id": conv_id,
                "num_turns": len(turns_all),
                "turns": [],
            }

            history_turns: List[Dict[str, Any]] = []

            turn_iter = _maybe_tqdm(
                list(enumerate(turns_all)),
                desc=f"Turns[{conv_id}]",
                total=len(turns_all),
                enable=enable_progress,
            )
            for idx, turn in turn_iter:
                current_turn_id = _turn_id(turn, idx)  # 0-based
                q_t = _get_question_text(turn)

                # gold fields（仅用于评测/诊断，不参与推理决策）
                gold_answer_text = _norm(turn.get("answer_text"))
                gold_topic = _norm(turn.get("topic"))
                gold_relation = _predict_relation_from_question(_get_original_question_text(turn))
                gold_answer_qid = _extract_turn_answer_qid(turn)

                # optional entity ids (best-effort): propagate QID if present in seed_entities
                gold_seed_entity_qid = ""
                cand_seed_entities = conv.get("seed_entities") or conv.get("seed_entity") or []
                if isinstance(cand_seed_entities, list) and cand_seed_entities:
                    # try first seed entity
                    se0 = cand_seed_entities[0]
                    if isinstance(se0, dict):
                        gold_seed_entity_qid = _extract_qid(se0.get("entity") or se0.get("qid"))
                    else:
                        gold_seed_entity_qid = _extract_qid(se0)

                rr = router.route(q_t, segments)
                assigned_seg: Optional[Dict[str, Any]] = None
                if rr.assigned_seg_id is not None:
                    for s in segments:
                        if _safe_int(s.get("seg_id")) == int(rr.assigned_seg_id):
                            assigned_seg = s
                            break

                topic = _norm(rr.assigned_topic)
                if not topic and assigned_seg is not None:
                    topic = _norm(assigned_seg.get("topic"))

                gr = gating.gate(
                    q_t=q_t,
                    assigned_seg=assigned_seg,
                    history_turns=history_turns,
                    memory_store=kg,
                    conv_id=conv_id,
                    known_entities=known_entities,
                )

                # Step 2: Controller decision (may probe MemoryKG cheaply)
                ctrl = controller_mod.decide(
                    question_text=q_t,
                    router_result={
                        "assigned_seg_id": rr.assigned_seg_id,
                        "assigned_topic": rr.assigned_topic,
                        "route_reason": rr.route_reason,
                        "route_confidence": getattr(rr, "route_confidence", 0.0),
                    },
                    gating_result={
                        "use_history": gr.use_history,
                        "history_turn_ids": gr.history_turn_ids,
                        "head_candidates": gr.head_candidates,
                        "relation_candidate": gr.relation_candidate,
                        "gate_reason": getattr(gr, "gate_reason", ""),
                    },
                    memory_executor=executor,
                    conv_id=conv_id,
                    topic=topic,
                    top_k=top_k,
                    current_turn=current_turn_id,
                    allow_current_turn_evidence=allow_current_turn_evidence,
                )

                prog.turn_done += 1
                prog.maybe_print(
                    conv_i=i,
                    conv_total=len(dialogues),
                    turn_i=idx,
                    turn_total=len(turns_all),
                    conv_id=conv_id,
                    decision=str(getattr(ctrl, "decision", "")),
                )

                evidence: List[EvidenceTriple] = []
                wikidata_query = ctrl.wikidata_query
                final_evidence: List[EvidenceTriple] = []
                pred_answer_value = ""
                rank_debug: Dict[str, Any] = {}

                if ctrl.decision == "USE_MEMORY":
                    evidence = executor.execute(
                        conv_id=conv_id,
                        topic=topic,
                        head_candidates=gr.head_candidates,
                        relation_candidate=gr.relation_candidate,
                        top_k=top_k,
                        current_turn=current_turn_id,
                        allow_current_turn=allow_current_turn_evidence,
                    )
                    rrk = evidence_ranker.rank(candidates=evidence, relation=gr.relation_candidate, top_m=1)
                    final_evidence = rrk.evidence
                    pred_answer_value = rrk.pred_answer_value
                    rank_debug = rrk.rank_debug
                    ranked_pack = _build_ranked_entities(
                        evidence_candidates=evidence,
                        relation=gr.relation_candidate,
                        top_n=(int(top_k) if int(top_k) > 10 else 10),
                    )
                    vr = verbalizer.verbalize(q_t, final_evidence)
                elif ctrl.decision == "ASK_CLARIFY":
                    # keep it deterministic for eval: no evidence
                    evidence = []
                    final_evidence = []
                    pred_answer_value = ""
                    ranked_pack = {"pred_ranked_entities": [], "pred_ranked_entity_scores": []}
                    vr = verbalizer.verbalize_clarify(q_t)
                elif ctrl.decision == "GENERATE":
                    # fallback generation template (you can swap to an LLM later)
                    evidence = []
                    final_evidence = []
                    pred_answer_value = ""
                    ranked_pack = {"pred_ranked_entities": [], "pred_ranked_entity_scores": []}
                    vr = verbalizer.verbalize_generate(q_t)
                else:  # USE_WIKIDATA
                    try:
                        wd_res, wd_query_dbg = wikidata_retriever.retrieve(
                            head_candidates=gr.head_candidates,
                            relation_candidate=gr.relation_candidate,
                            question_text=q_t,
                            top_k=top_k,
                        )
                        evidence = list(wd_res.evidence)
                        wikidata_query = wd_query_dbg
                    except Exception as e:
                        evidence = []
                        wikidata_query = {
                            "error": str(e),
                            "relation_candidate": gr.relation_candidate,
                            "head_candidates": list(gr.head_candidates)[:10],
                        }

                    rrk = evidence_ranker.rank(candidates=evidence, relation=gr.relation_candidate, top_m=1)
                    final_evidence = rrk.evidence
                    pred_answer_value = rrk.pred_answer_value
                    rank_debug = rrk.rank_debug
                    ranked_pack = _build_ranked_entities(
                        evidence_candidates=evidence,
                        relation=gr.relation_candidate,
                        top_n=max(10, int(top_k)),
                    )
                    # attach rank debug into wikidata_query for easier diagnostics
                    if isinstance(wikidata_query, dict) and "rank_debug" not in wikidata_query:
                        wd_q: Dict[str, Any] = dict(wikidata_query)
                        wd_q["rank_debug"] = rrk.rank_debug
                        wikidata_query = wd_q

                    vr = verbalizer.verbalize(q_t, final_evidence)

                turn_out = {
                    "turn": current_turn_id,
                    "question": q_t,
                    "gold": {
                        "answer_text": gold_answer_text,
                        "topic": gold_topic,
                        "relation": gold_relation,
                    },
                    "controller": {
                        "decision": ctrl.decision,
                        "reason": ctrl.reason,
                    },
                    "wikidata": {"query": wikidata_query},
                    "router": {
                        "assigned_seg_id": rr.assigned_seg_id,
                        "assigned_topic": rr.assigned_topic,
                        "route_reason": rr.route_reason,
                        "route_confidence": getattr(rr, "route_confidence", 0.0),
                    },
                    "gating": {
                        "use_history": gr.use_history,
                        "history_turn_ids": gr.history_turn_ids,
                        "head_candidates": gr.head_candidates,
                        # diagnostics.py 会尝试读取 gating.head_used；这里提供一个稳定字段
                        "head_used": (gr.head_candidates[0] if gr.head_candidates else ""),
                        "relation_candidate": gr.relation_candidate,
                        "gate_reason": getattr(gr, "gate_reason", ""),
                    },
                    # keep both lists: evidence = raw top-k candidates; final_evidence = chosen top-1
                    "evidence": [_as_serializable_evidence(ev) for ev in final_evidence],
                    "candidate_evidence": [_as_serializable_evidence(ev) for ev in evidence],
                    "answer_text": vr.answer_text,
                    "pred_answer_value": pred_answer_value,
                    "pred_ranked_entities": ranked_pack.get("pred_ranked_entities", []),
                    "pred_ranked_entity_scores": ranked_pack.get("pred_ranked_entity_scores", []),
                    "evidence_lines": vr.evidence_lines,
                }
                conv_out["turns"].append(turn_out)

                # JSONL 记录（eval/diagnostics 用）
                if writer is not None:
                    topic_subgraph = None
                    if write_topic_subgraph and topic:
                        try:
                            triples = kg.get_triples_by_conv_topic(str(conv_id), str(topic))
                            # 控制体积：只写前 N 条即可用于 oracle 可答性
                            slim = [
                                {
                                    "head": getattr(t, "head", ""),
                                    "relation": getattr(t, "relation", ""),
                                    "tail": getattr(t, "tail", ""),
                                    "turn_id": getattr(t, "turn_id", None),
                                    "scope": "topic",
                                }
                                for t in (triples or [])[: max(0, int(topic_subgraph_max_triples))]
                            ]
                            topic_subgraph = {str(topic): slim}
                        except Exception:
                            topic_subgraph = None

                    jsonl_record = {
                        "conv_id": conv_id,
                        "turn": current_turn_id,
                        "question": q_t,
                        "question_original": _get_original_question_text(turn),
                        # pred
                        "pred_answer_text": vr.answer_text,
                        "pred_answer_value": pred_answer_value,
                        "pred_ranked_entities": ranked_pack.get("pred_ranked_entities", []),
                        "pred_ranked_entity_scores": ranked_pack.get("pred_ranked_entity_scores", []),
                        "pred_evidence": [_as_serializable_evidence(ev) for ev in final_evidence],
                        "pred_candidate_evidence": [_as_serializable_evidence(ev) for ev in evidence],
                        "rank_debug": rank_debug,
                        # NEW: controller diagnostics fields
                        "controller": {
                            "decision": ctrl.decision,
                            "reason": ctrl.reason,
                        },
                        "wikidata": {"query": wikidata_query},
                        "router": turn_out["router"],
                        "gating": turn_out["gating"],
                        # gold (for eval only)
                        "gold_answer_text": gold_answer_text,
                        "gold_answer_qid": gold_answer_qid,
                        "gold_topic": gold_topic,
                        "gold_relation": gold_relation,
                        "gold_seed_entity_qid": gold_seed_entity_qid,
                        # optional: for diagnostics --enable_oracle
                        "topic_subgraph": topic_subgraph,
                        # meta
                        "meta": {
                            "merged_json": str(merged_json_path),
                            "memory_tsv": str(memory_tsv_path),
                            "topic": topic,
                            "top_k": top_k,
                            "recent_n_turns": recent_n_turns,
                            "allow_current_turn_evidence": allow_current_turn_evidence,
                        },
                    }
                    writer.write_one(jsonl_record)

                if print_stdout:
                    print(f"[conv={conv_id} turn={current_turn_id}] Q: {q_t}")
                    print(f"  -> A: {vr.answer_text}")
                    if vr.evidence_lines:
                        for line in vr.evidence_lines:
                            print(f"     {line}")

                # 模拟在线：历史只包含 < current_turn 的 turns（我们按顺序遍历，所以 append 即可）
                history_turns.append(turn)

            results["conversations"].append(conv_out)

    finally:
        if writer_cm is not None:
            writer_cm.__exit__(None, None, None)

    if out_path is not None:
        _write_json(out_path, results)

    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged_json",
        type=str,
        required=False,
        default=str(repo_root / "data" / "merged_dialogues" / "comprehensive_merged_dialogues.json"),
        help="merged 对话 JSON 路径可选；不传则用 default）",
    )
    ap.add_argument(
        "--memory_tsv",
        type=str,
        required=False,
        default=str(repo_root / "data" / "preprocessed" / "dialogue_kg" / "memory_triples.tsv"),
        help="memory_triples.tsv 路径（可选；不传则用 default）",
    )
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument(
        "--output_jsonl",
        type=str,
        default=str(repo_root / "outputs" / "preds.jsonl"),
        help=r"逐 turn 写入 JSONL；默认 <repo_root>/outputs/preds.jsonl",
    )
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--recent_n_turns", type=int, default=3)
    ap.add_argument("--no_print", action="store_true")
    ap.add_argument(
        "--no_progress",
        action="store_true",
        help="若指定：禁用进度输出（tqdm/定期 progress 行）。默认开启。",
    )
    ap.add_argument(
        "--local_wikidata_el_dict",
        type=str,
        default="",
        help="本地实体链接字典路径（格式：mention\\tQID，每行一条）。Always used now.",
    )

    ap.add_argument(
        "--local_wikidata_kg_dir",
        type=str,
        default="",
        help="本地 Wikidata KG 目录（默认 <repo_root>/data/data/wikidata）。",
    )
    ap.add_argument(
        "--local_wikidata_kg_split",
        type=str,
        default="train",
        help="使用本地 KG 的哪个 split 文件：train/valid/test/train_pre（默认 train）。",
    )
    ap.add_argument(
        "--allow_current_turn_evidence",
        action="store_true",
        help="若指定：允许使用 turn_id == 当前 turn 的证据（离线评测更容易）。不指定则严格只用 < 当前 turn。",
    )
    ap.add_argument(
        "--write_topic_subgraph",
        action="store_true",
        help="写入 topic_subgraph 到 preds.jsonl（diagnostics --enable_oracle 需要；会增大文件体积）。",
    )
    ap.add_argument(
        "--topic_subgraph_max_triples",
        type=int,
        default=300,
        help="写入 topic_subgraph 时，每个 topic 最多写入多少条 triple（用于控制 JSONL 大小）。",
    )

    args = ap.parse_args()

    merged_json_path = Path(args.merged_json)
    memory_tsv_path = Path(args.memory_tsv)
    out_path = Path(args.out_json) if _norm(args.out_json) else None
    output_jsonl_path = Path(args.output_jsonl) if _norm(args.output_jsonl) else None

    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # 覆盖写，避免追加到旧文件
        output_jsonl_path.write_text("", encoding="utf-8")

    print(f"[main_infer] merged_json={merged_json_path}")
    print(f"[main_infer] memory_tsv={memory_tsv_path}")
    print(f"[main_infer] output_jsonl={output_jsonl_path}")

    run_inference(
        merged_json_path=merged_json_path,
        memory_tsv_path=memory_tsv_path,
        out_path=out_path,
        output_jsonl_path=output_jsonl_path,
        top_k=int(args.top_k),
        recent_n_turns=int(args.recent_n_turns),
        print_stdout=not bool(args.no_print),
        allow_current_turn_evidence=bool(args.allow_current_turn_evidence),
        enable_progress=not bool(args.no_progress),
        local_wikidata_el_dict=str(args.local_wikidata_el_dict),
        local_wikidata_kg_dir=str(args.local_wikidata_kg_dir),
        local_wikidata_kg_split=str(args.local_wikidata_kg_split),
        write_topic_subgraph=bool(args.write_topic_subgraph),
        topic_subgraph_max_triples=int(args.topic_subgraph_max_triples),
    )


if __name__ == "__main__":
    main()

