# coding:utf-8
import json
import os
import re
from typing import Dict, List, Set, Tuple, Optional


class DialogueToKGConverter:
    """
    将对话数据转换为知识图谱 triple record（带 metadata），并输出 memory_triples.tsv
    """

    def __init__(self, input_file: str):
        self.input_file = input_file
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.records: List[Dict[str, str]] = []

    def normalize_tail(self, answer_text: Optional[str]) -> Optional[str]:
        """
        tail 规范化:
        - 纯数字且长度==4 -> YEAR::2006
        - 纯数字但不是4位 -> COUNT::7
        - Yes/No -> BOOL::Yes
        - 否则原样
        """
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
        """
        relation 规则扩展（最小覆盖）
        """
        q = (original_question or "").strip().lower()

        if "first" in q and "book" in q:
            return "first_book"
        if any(k in q for k in ["final", "ended", "concluded", "last book"]):
            return "final_book"
        if any(k in q for k in ["how many", "amount", "number of books", "number of book", "number of"]):
            return "num_books"

        when_only = (q == "when") or ("when" in q)
        if when_only:
            if any(k in q for k in ["award", "prize"]):
                return "award_year"
            return "publication_year"

        if "author" in q:
            return "author"
        if "nationality" in q or "country" in q:
            return "nationality"
        if "year" in q:
            return "publication_year"
        if "title" in q:
            return "book_title"
        if "award" in q:
            return "award"
        if "publisher" in q:
            return "publisher"
        if "genre" in q:
            return "genre"

        return "related_to"

    def extract_triples_from_dialogue(self, dialogue: Dict) -> List[Dict[str, str]]:
        """
        从单个对话中提取 triple records（带 metadata）
        record 字段:
        conv_id, turn_id, topic, head, relation, tail
        """
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

            self.entities.add(topic)
            self.entities.add(tail)
            self.relations.add(relation)

            records.append(
                {
                    "conv_id": "" if conv_id is None else str(conv_id),
                    "turn_id": "" if turn_id is None else str(turn_id),
                    "topic": topic,
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                }
            )

        return records

    def convert(self) -> Tuple[List[Dict[str, str]], Set[str], Set[str]]:
        """
        转换整个数据集，并按 (conv_id, turn_id, head, relation, tail) 维度去重
        """
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
        """
        保存:
        - memory_triples.tsv (给记忆库推理用)
        - 仍保留 train/valid/test + dict（可用于 embedding 训练）
        """
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
        self._save_dict(os.path.join(output_dir, "relations.dict"), relations)

        print(f"数据集已保存到 {output_dir}")
        print(f"memory_triples.tsv: {len(records)} records")
        print(f"训练集: {len(train_triples)} 三元组")
        print(f"验证集: {len(valid_triples)} 三元组")
        print(f"测试集: {len(test_triples)} 三元组")
        print(f"实体数: {len(entities)}")
        print(f"关系数: {len(relations)}")

    def _save_memory_triples_tsv(self, filename: str, records: List[Dict[str, str]]):
        """
        保存带 metadata 的 triple records 到 TSV:
        conv_id \t turn_id \t topic \t head \t relation \t tail
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write("conv_id\tturn_id\ttopic\thead\trelation\ttail\n")
            for r in records:
                f.write(
                    f"{r['conv_id']}\t{r['turn_id']}\t{r['topic']}\t{r['head']}\t{r['relation']}\t{r['tail']}\n"
                )

    def _save_triples(self, filename: str, triples: List[Tuple[str, str, str]]):
        """
        保存三元组到文件
        """
        with open(filename, "w", encoding="utf-8") as f:
            for head, relation, tail in triples:
                f.write(f"{head}\t{relation}\t{tail}\n")

    def _save_dict(self, filename: str, items: Set[str]):
        """
        保存字典到文件
        """
        with open(filename, "w", encoding="utf-8") as f:
            for idx, item in enumerate(sorted(items)):
                f.write(f"{item}\t{idx}\n")


def main():
    """
    主函数：执行数据预处理
    """
    input_file = "../../data/merged_dialogues/comprehensive_merged_dialogues.json"
    output_dir = "../../data/preprocessed/dialogue_kg/"

    converter = DialogueToKGConverter(input_file)
    converter.save_dataset(output_dir)


if __name__ == "__main__":
    main()
