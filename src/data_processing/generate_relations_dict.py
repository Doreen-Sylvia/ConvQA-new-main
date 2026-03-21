import os
import json
import argparse
from collections import Counter
from pathlib import Path


def generate_relations_dict(data_dir, output_path, min_freq=1):
    relation_counter = Counter()
    processed_files = 0

    files_to_check = ['train.txt', 'valid.txt', 'test.txt']
    txt_files = [f for f in files_to_check if os.path.exists(os.path.join(data_dir, f))]

    if txt_files:
        print(f"Found TXT files: {txt_files}")
        for filename in txt_files:
            fpath = os.path.join(data_dir, filename)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        raw_rel = parts[1].strip()
                        relation_counter[raw_rel] += 1
            processed_files += 1

    if processed_files == 0:
        print(f"No valid data files found in {data_dir}!")
        return

    print(f"Found {len(relation_counter)} raw relation types in files.")

    # 提取有效的基础关系 (去掉尾缀 _reverse)
    base_relations = set()
    for rel, freq in relation_counter.items():
        if freq < min_freq:
            continue
        # 清洗掉 _reverse 以获取 Base
        if rel.endswith('_reverse'):
            base_relations.add(rel.replace('_reverse', ''))
        else:
            base_relations.add(rel)

    # 严格确保每一个 Base 关系都有一对 (正向 + 反向)
    full_relation_list = []
    for rel in sorted(list(base_relations)):
        full_relation_list.append(rel)
        full_relation_list.append(f"{rel}_reverse")

    special_tokens = ['<PAD>', '<UNK>']
    final_relations = special_tokens + full_relation_list

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, rel in enumerate(final_relations):
            f.write(f"{rel}\t{idx}\n")

    print(f"✅ Successfully generated strictly paired relations dictionary with {len(final_relations)} types.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 自动获取项目根目录
    repo_root = Path(__file__).resolve().parents[2]
    raw_data_path = str(repo_root / "data" / "preprocessed" / "dialogue_kg")
    output_dict_path = str(repo_root / "data" / "preprocessed" / "dialogue_kg" / "relations.dict")

    parser.add_argument('--data_dir', type=str, default=raw_data_path)
    parser.add_argument('--output_path', type=str, default=output_dict_path)
    parser.add_argument('--min_freq', type=int, default=1)

    args = parser.parse_args()

    print(f"正在扫描目录: {args.data_dir}")
    generate_relations_dict(args.data_dir, args.output_path, args.min_freq)