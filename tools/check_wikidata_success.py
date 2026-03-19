# coding:utf-8
import json
from pathlib import Path

p = Path(r"D:\Project\ConvQA-new\outputs\preds.jsonl")

n_use = 0
n_ok = 0
n_error = 0
n_has_wd_ev = 0
n_no_ev = 0

with p.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        decision = (rec.get("controller") or {}).get("decision")
        if decision != "USE_WIKIDATA":
            continue

        n_use += 1
        q = (rec.get("wikidata") or {}).get("query") or {}
        if isinstance(q, dict) and q.get("error"):
            n_error += 1

        head_qids = (q.get("head_qids") if isinstance(q, dict) else None) or []
        prop_ids = (q.get("property_ids") if isinstance(q, dict) else None) or []

        # evidence（final）+ candidate evidence（top-k）
        ev_final = rec.get("pred_evidence") or rec.get("evidence") or []
        ev_cand = rec.get("pred_candidate_evidence") or []
        ev_all = []
        if isinstance(ev_final, list): ev_all += ev_final
        if isinstance(ev_cand, list): ev_all += ev_cand

        has_wd_ev = any(isinstance(e, dict) and e.get("scope") == "wikidata" for e in ev_all)
        if has_wd_ev:
            n_has_wd_ev += 1

        if head_qids and prop_ids and has_wd_ev and not (isinstance(q, dict) and q.get("error")):
            n_ok += 1

        if not ev_all:
            n_no_ev += 1

print("USE_WIKIDATA turns:", n_use)
print("OK (head_qids+property_ids+wikidata evidence, no error):", n_ok)
print("Has wikidata evidence (scope=wikidata):", n_has_wd_ev)
print("Has error:", n_error)
print("No evidence at all:", n_no_ev)
