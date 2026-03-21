# coding: utf-8
"""High-level Wikidata retriever that glues EL + property mapping + 1-hop KG.

Used when controller decision is USE_WIKIDATA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

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
        local_el_dict_path: Optional[str] = None,
        local_wikidata_kg_dir: Optional[str] = None,
        local_wikidata_kg_split: str = "train",
    ) -> None:
        # Force offline mode parameters
        self._use_local_el = True
        self._local_dict_path = str(local_el_dict_path) if local_el_dict_path else ""
        self._local_mention2qid: Optional[Dict[str, str]] = None

        # lazy-load on first retrieve to keep init fast
        self.entity_linker = None  # type: ignore

        # KG backend: always local
        self._local_wikidata_kg_dir = str(local_wikidata_kg_dir) if local_wikidata_kg_dir else ""
        self._local_wikidata_kg_split = str(local_wikidata_kg_split or "train")
        self._local_kg: Optional[LocalWikidataKG] = None
        self._local_rel_map: Optional[Dict[str, List[str]]] = None
        
        self.max_head_qids = max(1, int(max_head_qids))

        self._language = language
        self._el_limit_per_mention = el_limit_per_mention

    def _ensure_local_kg(self) -> None:
        # Always run since we force _use_local_wikidata_kg = True
        if self._local_kg is not None:
            return
        kg_dir = self._local_wikidata_kg_dir
        if not kg_dir:
            # default to <repo_root>/data/data/wikidata
            repo_root = Path(__file__).resolve().parents[2]
            kg_dir = str(repo_root / "data" / "data" / "wikidata")
        # Enable reverse relations to handle "Who wrote X?" and "What did X write?"
        self._local_kg = LocalWikidataKG(root_dir=kg_dir, split=self._local_wikidata_kg_split, add_reverse_relations=True)
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
            "author": ["author", "writer", "creator", "written by", "screenwriter", "novelist", "author_reverse", "writer_reverse"],
            "genre": ["genre", "subclass", "instance of", "type"],
            "publisher": ["publisher", "published by"],
            "publication_year": ["publication date", "publication year", "published", "release date", "date", "year", "start time"],
            "award": ["award", "prize", "winner", "nominated for"],
            "nationality": ["country of citizenship", "nationality", "country", "citizenship"],
            # less reliable, but keep a few keywords to avoid being completely empty
            "book_title": ["title", "name", "original title"],
            "first_book": ["first", "debut", "start"],
            "final_book": ["last", "final", "end"],
            "num_books": ["number of", "count", "quantity", "author_reverse", "writer_reverse", "part of_reverse", "series_reverse"],
            "related_to": ["related", "connection", "part of", "series", "instance of", "subclass of", "has part", "based on", "preceded by", "followed by", "derivative work", "sequel", "prequel"],
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
                    
                    # BLOCKLIST: avoid picking up IDs, codes, classification properties
                    # e.g., "ISFDB author ID" should not map to "author"
                    block_words = {"id", "identifier", "code", "index", "classification", "url", 
                                   "viaf", "gnd", "oclc", "isbn", "doi", "lccn", "link", "website", 
                                   "category", "template", "image", "logo", "map", "flag", 
                                   "signature", "media", "video", "audio", "file", "commons",
                                   "described by source", "described at url", "subject", "topic"}
                    # check if rel_norm has ANY block words as separate tokens or endings
                    is_dirty = False
                    rel_tokens = set(re.split(r"[\s\(\)\-_]+", rel_norm))
                    if block_words & rel_tokens:
                        is_dirty = True
                    
                    # Exception: allow 'subject' or 'topic' if we are looking for 'related_to'
                    # but generally we want to exclude meta-relations.
                    
                    # Even safer: if the relation name ENDS with id/identifier
                    if rel_norm.endswith(" id") or rel_norm.endswith(" identifier") or rel_norm.endswith(" code"):
                        is_dirty = True
                        
                    if is_dirty:
                         # special case: if we are looking for 'related_to', maybe we allow some breadth,
                         # but for 'author', 'genre', etc. absolutely block IDs.
                         pass

                    for k, kws in want.items():
                        if len(out[k]) >= 20:
                            continue
                            
                        # If looking for author/publisher/genre etc, be strict about blocking IDs
                        if k != "related_to" and is_dirty:
                            continue
                            
                        for w in kws:
                            ww = w.casefold()
                            if ww and ww in rel_norm:
                                # Double check: if we matched "author" in "ISFDB author ID", skip it!
                                # Logic: if rel_norm has 'id' or 'identifier' and we matched inside it...
                                if is_dirty and k != "related_to":
                                    continue
                                
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
        # Always run since we force _use_local_el = True
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
        rel_names: List[str] = []
        
        # In offline mode, property_ids main use is limited since we rely on string matching in local KG
        if property_ids is not None:
            props = [_norm(x) for x in property_ids if _norm(x)]
        else:
            props = map_relation_to_properties(rel)

        links = self.entity_linker.link(head_candidates=head_candidates, question_text=question_text)  # type: ignore
        head_qids = [l.qid for l in links[: self.max_head_qids]]
        head_labels = {l.qid: l.label for l in links[: self.max_head_qids] if _norm(l.label)}

        # Offline mode: we cannot use QIDs/property IDs. We interpret:
        # - head_candidates as entity labels
        # - relation_candidate as relation label

        if rel and self._local_rel_map is not None:
            rel_names = list(self._local_rel_map.get(rel, []) or [])
        if not rel_names and rel:
            rel_names = [rel]

        if self._local_kg is not None:
             evidence = self._local_kg.retrieve_1hop(
                head_entities=list(head_candidates),
                relation_names=rel_names if rel else [],
                relation_name_for_evidence=rel,
                top_k=top_k,
                scope="local_wikidata",
            )
        else:
            evidence = []

        query_dbg = {
            "head_qids": head_qids,
            "head_labels": head_labels,
            "property_ids": props,
            "relation_candidate": rel or None,
            "local_relation_names": rel_names,
            "head_candidates": list(head_candidates)[:10],
            "kg_backend": "local",
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
