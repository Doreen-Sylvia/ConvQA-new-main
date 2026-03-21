# coding:utf-8
import json
import os
import re
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

# 扩展科学的关系映射字典
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

    # === 新增：细化 related_to 的具体关系 ===
    'characters': 'P674',  # 主角/角色
    'followed_by': 'P156',  # 续作
    'based_on': 'P128',  # 改编自/基于 (movie/film/tv)
    'narrator': 'P2438',  # 旁白/叙述者

    # === 修复自定义关系映射 ===
    'first_book': 'P50_reverse_first',  # 区分正反向
    'final_book': 'P50_reverse_final',
    'num_books': 'num_books',  # 绝对不能映射成 P50_reverse，保持独立属性
    'award_year': 'P166',
    'publication_year': 'P577',
    'book_title': 'P1476',
    'publisher': 'P123',
    'nationality': 'P27',
}


class DialogueToKGConverter:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.records: List[Dict[str, str]] = []

    def normalize_tail(self, answer_text: Optional[str]) -> Optional[str]:
        if answer_text is None:
            return None
        tail = str(answer_text).strip()
        if tail == "":
            return None

        if re.fullmatch(r"\d+", tail):
            if len(tail) == 4:
                return f"YEAR::{tail}"
            return f"COUNT::{tail}"

        low = tail.lower()
        if low in {"yes", "no"}:
            return f"BOOL::{tail[:1].upper()}{tail[1:].lower()}"

        return tail

    def _infer_relation(self, original_question: str) -> str:
        q = (original_question or "").strip().lower()

        inferred_raw = None

        # 1. 精确拦截细粒度关系，消灭 related_to
        if "protagonist" in q or "character" in q:
            inferred_raw = "characters"
        elif "sequel" in q or "followed by" in q or ("after" in q and "book" in q):
            inferred_raw = "followed_by"
        elif "movie" in q or "film" in q or "tv series" in q or "adaptation" in q:
            inferred_raw = "based_on"
        elif "narrator" in q or "narrating" in q:
            inferred_raw = "narrator"
        elif "wife" in q or "husband" in q or "spouse" in q or "married" in q:
            inferred_raw = "spouse"
        elif "born" in q:
            if "where" in q or "city" in q or "country" in q:
                inferred_raw = "place of birth"
            else:
                inferred_raw = "date of birth"

        # 2. 原本的逻辑（修复了 first book 的匹配缺陷）
        elif "first" in q and ("book" in q or "novel" in q):
            inferred_raw = "first_book"
        elif any(k in q for k in ["final", "ended", "concluded", "last book", "last novel"]):
            inferred_raw = "final_book"
        elif any(k in q for k in ["how many", "amount", "number of books", "number of", "how much"]):
            inferred_raw = "num_books"
        else:
            when_only = (q == "when") or ("when" in q)
            if when_only:
                if any(k in q for k in ["award", "prize"]):
                    inferred_raw = "award_year"
                else:
                    inferred_raw = "publication_year"
            elif "nationality" in q or "country" in q:
                inferred_raw = "nationality"
            elif "author" in q or "wrote" in q or "written" in q:
                inferred_raw = "author"
            elif "year" in q:
                inferred_raw = "publication_year"
            elif "title" in q:
                inferred_raw = "book_title"
            elif "award" in q or "prize" in q:
                inferred_raw = "award"
            elif "publisher" in q or "published by" in q:
                inferred_raw = "publisher"
            elif "genre" in q:
                inferred_raw = "genre"
            else:
                inferred_raw = "related_to"

        if inferred_raw in WIKIDATA_MAPPING:
            return WIKIDATA_MAPPING[inferred_raw]

        if inferred_raw == 'award':
            return WIKIDATA_MAPPING.get('award received', 'P166')

        if inferred_raw == 'related_to':
            return "P31"  # 泛化为 Instance of (P31) 比 <UNK> 更好

        return inferred_raw

    def extract_triples_from_dialogue(self, dialogue: Dict) -> List[Dict[str, str]]:
        conv_id = dialogue.get("conv_id")
        records: List[Dict[str, str]] = []

        questions = dialogue.get("questions", [])
        for question in questions:
            topic = (question.get("topic") or "").strip()
            if topic == "":
                continue

            turn_id = question.get("turn")
            original_question = question.get("original_question") or ""
            answer_text = question.get("answer_text")

            tail = self.normalize_tail(answer_text)
            if tail is None:
                continue

            head = topic
            relation = self._infer_relation(original_question)
            rev_relation = f"{relation}_reverse"  # 生成反向关系名称

            self.entities.add(head)
            self.entities.add(tail)
            self.relations.add(relation)
            self.relations.add(rev_relation)

            # ======== 核心修复：同时添加正向边和反向边 ========
            # 1. 正向边 (Topic -> Relation -> Answer)
            records.append({
                "conv_id": "" if conv_id is None else str(conv_id),
                "turn_id": "" if turn_id is None else str(turn_id),
                "topic": topic,
                "head": head,
                "relation": relation,
                "tail": tail,
            })

            # 2. 反向边 (Answer -> Relation_reverse -> Topic)
            # 这彻底解决了把书当成作者名字输出的串台问题！
            records.append({
                "conv_id": "" if conv_id is None else str(conv_id),
                "turn_id": "" if turn_id is None else str(turn_id),
                "topic": topic,
                "head": tail,  # 尾做头
                "relation": rev_relation,  # 关系加 reverse
                "tail": head,  # 头做尾
            })

        return records

    def convert(self) -> Tuple[List[Dict[str, str]], Set[str], Set[str]]:
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_records: List[Dict[str, str]] = []
        for dialogue in data:
            all_records.extend(self.extract_triples_from_dialogue(dialogue))

        seen: Set[Tuple[str, str, str, str, str]] = set()
        deduped: List[Dict[str, str]] = []
        for r in all_records:
            key = (r["conv_id"], r["turn_id"], r["head"], r["relation"], r["tail"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        self.records = deduped
        return self.records, self.entities, self.relations

    def save_dataset(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        records, entities, relations = self.convert()

        memory_path = os.path.join(output_dir, "memory_triples.tsv")
        self._save_memory_triples_tsv(memory_path, records)

        triples = [(r["head"], r["relation"], r["tail"]) for r in records]
        n = len(triples)
        train_end = int(0.8 * n)
        valid_end = int(0.9 * n)

        train_triples = triples[:train_end]
        valid_triples = triples[train_end:valid_end]
        test_triples = triples[valid_end:]

        self._save_triples(os.path.join(output_dir, "train.txt"), train_triples)
        self._save_triples(os.path.join(output_dir, "valid.txt"), valid_triples)
        self._save_triples(os.path.join(output_dir, "test.txt"), test_triples)

        self._save_dict(os.path.join(output_dir, "entities.dict"), entities)

        # 强制更新 relations.dict，因为我们新增了反向关系
        ref_rel_path = os.path.join(output_dir, "relations.dict")
        self._save_dict(ref_rel_path, relations)

        print(f"✅ 数据集已成功保存并注入反向边到 {output_dir}")
        print(f"memory_triples.tsv: {len(records)} records")

    def _save_memory_triples_tsv(self, filename: str, records: List[Dict[str, str]]):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("conv_id\tturn_id\ttopic\thead\trelation\ttail\n")
            for r in records:
                f.write(f"{r['conv_id']}\t{r['turn_id']}\t{r['topic']}\t{r['head']}\t{r['relation']}\t{r['tail']}\n")

    def _save_triples(self, filename: str, triples: List[Tuple[str, str, str]]):
        with open(filename, "w", encoding="utf-8") as f:
            for head, relation, tail in triples:
                f.write(f"{head}\t{relation}\t{tail}\n")

    def _save_dict(self, filename: str, items: Set[str]):
        # 加入控制符并排序
        special_tokens = ['<PAD>', '<UNK>']
        final_list = special_tokens + sorted(list(items))
        with open(filename, "w", encoding="utf-8") as f:
            for idx, item in enumerate(final_list):
                f.write(f"{item}\t{idx}\n")


def main():
    # 自动获取项目根目录 (ConvQA-new-main)
    repo_root = Path(__file__).resolve().parents[2]

    input_file = str(repo_root / "data" / "merged_dialogues" / "comprehensive_merged_dialogues.json")
    output_dir = str(repo_root / "data" / "preprocessed" / "dialogue_kg")

    print(f"正在读取: {input_file}")
    print(f"准备输出到: {output_dir}")

    converter = DialogueToKGConverter(input_file)
    converter.save_dataset(output_dir)


if __name__ == "__main__":
    main()