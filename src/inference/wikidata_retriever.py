# coding: utf-8
"""High-level Wikidata retriever that glues EL + property mapping + 1-hop KG.

Used when controller decision is USE_WIKIDATA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.inference.kg_execute import EvidenceTriple
from src.inference.wikidata_properties import map_relation_to_properties
from src.inference.local_wikidata_kg import LocalWikidataKG


class _DictEntityLinker:
    """A tiny local entity linker backed by a mention->qid mapping.

    This is intended as a fast/offline fallback. It only supports exact
    (case-insensitive) matches.
    """

    def __init__(self, *, mention2qid: Dict[str, str], language: str = "en", limit_per_mention: int = 3) -> None:
        self.mention2qid = {str(k).strip().casefold(): str(v).strip() for k, v in (mention2qid or {}).items() if str(k).strip()}
        self.language = language
        self.limit_per_mention = max(1, int(limit_per_mention))

    @dataclass(frozen=True)
    class _Link:
        qid: str
        label: str

    def link(self, *, head_candidates: Sequence[str], question_text: str) -> List[Any]:
        out: List[_DictEntityLinker._Link] = []
        seen: set[str] = set()
        for cand in list(head_candidates)[: max(20, self.limit_per_mention * 5)]:
            key = _norm(cand).casefold()
            if not key:
                continue
            q = self.mention2qid.get(key, "")
            if not q:
                continue
            q = q.upper()
            if q in seen:
                continue
            seen.add(q)
            out.append(_DictEntityLinker._Link(qid=q, label=_norm(cand)))
            if len(out) >= self.limit_per_mention:
                break
        return out


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


@dataclass(frozen=True)
class WikidataRetrieveResult:
    head_qids: List[str]
    head_labels: Dict[str, str]
    property_ids: List[str]
    evidence: List[EvidenceTriple]


class WikidataRetriever:
    def __init__(
        self,
        *,
        language: str = "en",
        timeout_s: float = 10.0,
        el_limit_per_mention: int = 3,
        max_head_qids: int = 3,
        use_local_el: bool = False,
        local_el_dict_path: Optional[str] = None,
        use_local_wikidata_kg: bool = False,
        local_wikidata_kg_dir: Optional[str] = None,
        local_wikidata_kg_split: str = "train",
    ) -> None:
        self._use_local_el = bool(use_local_el)
        self._local_dict_path = str(local_el_dict_path) if local_el_dict_path else ""
        self._local_mention2qid: Optional[Dict[str, str]] = None

        if self._use_local_el:
            # lazy-load on first retrieve to keep init fast
            self.entity_linker = None  # type: ignore
        else:
            # Import lazily so offline-only users can delete online modules.
            from src.inference.wikidata_el import EntityLinker  # local import

            self.entity_linker = EntityLinker(language=language, limit_per_mention=el_limit_per_mention, timeout_s=timeout_s)

        # KG backend: online (default) vs local preprocessed triples.
        self._use_local_wikidata_kg = bool(use_local_wikidata_kg)
        self._local_wikidata_kg_dir = str(local_wikidata_kg_dir) if local_wikidata_kg_dir else ""
        self._local_wikidata_kg_split = str(local_wikidata_kg_split or "train")
        self._local_kg: Optional[LocalWikidataKG] = None
        self._local_rel_map: Optional[Dict[str, List[str]]] = None
        # Online KG backend (optional). Only construct if we may use it.
        if self._use_local_wikidata_kg:
            self.kg = None  # type: ignore
        else:
            from src.inference.wikidata_kg import WikidataKG  # local import

            self.kg = WikidataKG(timeout_s=timeout_s, language=language)
        self.max_head_qids = max(1, int(max_head_qids))

        self._language = language
        self._el_limit_per_mention = el_limit_per_mention

    def _ensure_local_kg(self) -> None:
        if not self._use_local_wikidata_kg:
            return
        if self._local_kg is not None:
            return
        kg_dir = self._local_wikidata_kg_dir
        if not kg_dir:
            # default to <repo_root>/data/data/wikidata
            repo_root = Path(__file__).resolve().parents[2]
            kg_dir = str(repo_root / "data" / "data" / "wikidata")
        self._local_kg = LocalWikidataKG(root_dir=kg_dir, split=self._local_wikidata_kg_split, add_reverse_relations=False)
        # eager load once to front-load IO instead of incurring random latency mid-run
        self._local_kg.load()

        # Build and cache mapping from our small relation_candidate set -> local KG relation strings.
        # This reduces empty evidence when local relations.dict doesn't contain exactly the same names.
        self._local_rel_map = self._build_local_relation_map(Path(kg_dir) / "relations.dict")

    @staticmethod
    def _build_local_relation_map(relations_dict_path: Path) -> Dict[str, List[str]]:
        """Build mapping for local KG relation lookup.

        The local relations.dict contains many human-readable property names.
        We map our internal relation names to a small set of candidate strings.
        """

        want: Dict[str, List[str]] = {
            "author": ["author", "writer"],
            "genre": ["genre"],
            "publisher": ["publisher"],
            "publication_year": ["publication date", "publication year", "published"],
            "award": ["award", "prize"],
            "nationality": ["country of citizenship", "nationality", "country"],
            # less reliable, but keep a few keywords to avoid being completely empty
            "book_title": ["title"],
            "first_book": ["first", "debut"],
            "final_book": ["last", "final"],
            "num_books": ["number of", "count"],
            "related_to": ["related"],
        }

        out: Dict[str, List[str]] = {k: [] for k in want}
        if not relations_dict_path.exists():
            return out

        try:
            with relations_dict_path.open("r", encoding="utf-8", newline="") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if "\t" in s:
                        rel_text = s.split("\t", 1)[0]
                    else:
                        # tolerate space-separated: last token is id
                        rel_text = s.rsplit(" ", 1)[0]
                    rel_norm = rel_text.strip().casefold()
                    if not rel_norm or rel_norm.endswith("_reverse"):
                        continue

                    for k, kws in want.items():
                        if len(out[k]) >= 20:
                            continue
                        for w in kws:
                            ww = w.casefold()
                            if ww and ww in rel_norm:
                                out[k].append(rel_text.strip())
                                break
        except Exception:
            return out

        # de-dup keep order
        for k, v in list(out.items()):
            seen: set[str] = set()
            vv: List[str] = []
            for x in v:
                if x in seen:
                    continue
                seen.add(x)
                vv.append(x)
            out[k] = vv
        return out

    def _ensure_local_el(self) -> None:
        if not self._use_local_el:
            return
        if self._local_mention2qid is None:
            mention2qid: Dict[str, str] = {}
            if self._local_dict_path:
                try:
                    # file format: <mention>\t<qid> (or space-separated)
                    with open(self._local_dict_path, "r", encoding="utf-8") as f:
                        for line in f:
                            s = line.strip()
                            if not s or s.startswith("#"):
                                continue
                            if "\t" in s:
                                a, b = s.split("\t", 1)
                            else:
                                parts = s.split()
                                if len(parts) < 2:
                                    continue
                                a, b = parts[0], parts[1]
                            a = _norm(a)
                            b = _norm(b)
                            if a and b:
                                mention2qid[a] = b
                except Exception:
                    mention2qid = {}
            self._local_mention2qid = mention2qid

        if self.entity_linker is None:
            self.entity_linker = _DictEntityLinker(
                mention2qid=self._local_mention2qid or {},
                language=self._language,
                limit_per_mention=self._el_limit_per_mention,
            )

    def retrieve(
        self,
        *,
        head_candidates: Sequence[str],
        relation_candidate: str,
        question_text: str,
        property_ids: Optional[Sequence[str]] = None,
        top_k: int = 3,
    ) -> Tuple[WikidataRetrieveResult, Dict[str, Any]]:
        self._ensure_local_el()
        self._ensure_local_kg()
        rel = _norm(relation_candidate)
        rel_names: List[str] = []  # only used in local KG mode
        if property_ids is not None:
            props = [_norm(x) for x in property_ids if _norm(x)]
        else:
            props = map_relation_to_properties(rel)

        links = self.entity_linker.link(head_candidates=head_candidates, question_text=question_text)  # type: ignore
        head_qids = [l.qid for l in links[: self.max_head_qids]]
        head_labels = {l.qid: l.label for l in links[: self.max_head_qids] if _norm(l.label)}

        if self._use_local_wikidata_kg and self._local_kg is not None:
            # Offline mode: we cannot use QIDs/property IDs. We interpret:
            # - head_candidates as entity labels
            # - relation_candidate as relation label

            if rel and self._local_rel_map is not None:
                rel_names = list(self._local_rel_map.get(rel, []) or [])
            if not rel_names and rel:
                rel_names = [rel]

            evidence = self._local_kg.retrieve_1hop(
                head_entities=list(head_candidates),
                relation_names=rel_names if rel else [],
                relation_name_for_evidence=rel,
                top_k=top_k,
                scope="local_wikidata",
            )
        else:
            # Online mode
            evidence = self.kg.retrieve_1hop(head_qids=head_qids, property_ids=props, relation_name=rel, top_k=top_k)  # type: ignore

        query_dbg = {
            "head_qids": head_qids,
            "head_labels": head_labels,
            "property_ids": props,
            "relation_candidate": rel or None,
            "local_relation_names": (rel_names if (self._use_local_wikidata_kg and self._local_kg is not None) else None),
            "head_candidates": list(head_candidates)[:10],
            "kg_backend": ("local" if (self._use_local_wikidata_kg and self._local_kg is not None) else "online"),
        }

        return (
            WikidataRetrieveResult(
                head_qids=head_qids,
                head_labels=head_labels,
                property_ids=props,
                evidence=evidence,
            ),
            query_dbg,
        )


__all__ = ["WikidataRetriever", "WikidataRetrieveResult"]
