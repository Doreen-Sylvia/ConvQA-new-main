import os
import json
import argparse
from collections import Counter

# Step 2: Scientific relation mapping (partial example)
# Replace raw labels with Wikidata PIDs or cleaner labels
# This dictionary should be expanded based on rigorous analysis
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
    'date of publication': 'P577', 
    'genre': 'P136',
    'country of citizenship': 'P27',
    'member of sports team': 'P54',
    'educated at': 'P69',
    'employer': 'P108',
    'occupation': 'P106',
    'position (on team)': 'P413',
    'member of': 'P463',
    'part of': 'P361',
    'series': 'P179',
    'instance of': 'P31',
    'award received': 'P166',
    'record label': 'P264',
    'original channel': 'P449',
    'composer': 'P86',
    'lyrics by': 'P676',
    'producer': 'P162',
    'distributor': 'P750',
    'language used': 'P407',
    'main subject': 'P921',
    'conflict': 'P607',
    'participant of': 'P1344',
    'nominated for': 'P1411',
    'father': 'P22',
    'mother': 'P25',
    'spouse': 'P26',
    'child': 'P40',
    'sibling': 'P3373',
    'student of': 'P1066',
    'doctoral advisor': 'P184',
    'influenced by': 'P737',
    'location': 'P276',
    'capital': 'P36',
    'head of government': 'P6',
    'member of political party': 'P102',
    'religion': 'P140',
    'unmarried partner': 'P451',
    'field of work': 'P101',
    'movement': 'P135',
    'instrument': 'P1303',
}

def generate_relations_dict(data_dir, output_path, min_freq=5):
    """
    Generates a relations dictionary from raw data files.
    
    Args:
        data_dir (str): Directory containing train.txt/valid.txt/test.txt (triples) or JSON files.
        output_path (str): Path to save the relations.dict.
        min_freq (int): Minimum frequency to keep a relation.
    """
    relation_counter = Counter()
    processed_files = 0
    
    # 1. Loop through data files
    # Supports both TXT (triples) and JSON if available
    files_to_check = ['train.txt', 'valid.txt', 'test.txt']
    
    # Check for TXT files first
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
    
    # Also check for JSON files in the directory (if any)
    # This handles the user's hypothetical JSON case
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for filename in json_files:
        fpath = os.path.join(data_dir, filename)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Heuristic to find triples in JSON
                if isinstance(data, list):
                    for item in data:
                        if 'triples' in item:
                             for h, r, t in item['triples']:
                                relation_counter[r.strip()] += 1
                elif isinstance(data, dict):
                    pass
            processed_files += 1
        except Exception:
            pass # Skip invalid JSON

    if processed_files == 0:
        print(f"No valid data files found in {data_dir}!")
        return

    print(f"Found {len(relation_counter)} raw relation types.")

    # 2. Filter low frequency relations (Step 1)
    # Also apply Step 4: Generic/Garbage relations removal
    valid_relations_set = set()
    
    for rel, freq in relation_counter.items():
        # Step 4: Handle related_to or generic relations (skip generic)
        if rel == 'related_to':
            continue
            
        if freq < min_freq:
            continue
            
        # Step 3: Handle Qualifiers (Simplistic approach: filter extremely long-tail specific attributes)
        
        # Step 2: Align with Ontology
        mapped_rel = WIKIDATA_MAPPING.get(rel, rel)
        
        valid_relations_set.add(mapped_rel)

    # 3. Add special tokens
    special_tokens = ['<PAD>', '<UNK>'] 
    
    # Prepare full relation list with inverses
    base_relations = sorted(list(valid_relations_set))
    full_relation_list = []
    
    for rel in base_relations:
        full_relation_list.append(rel)
        full_relation_list.append(f"{rel}_reverse")
        
    # Sort alphabetically to reproduce similar ordering if desired, 
    # but strictly speaking user snippet said special tokens first.
    # We will do: Special Tokens -> Sorted(Others)
    
    full_relation_list.sort() # Sort alphabetically (e.g. 2017, 2017_reverse)
    final_relations = special_tokens + full_relation_list
    
    # 4. Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write format: relation \t index
        for idx, rel in enumerate(final_relations):
            f.write(f"{rel}\t{idx}\n")
            
    print(f"✅ Successfully generated relations dictionary with {len(final_relations)} types.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default path relative to project root execution
    # Assuming script run from project root
    raw_data_path = os.path.join('data', 'data', 'wikidata_convex')
    output_dict_path = os.path.join('data', 'data', 'wikidata_convex', 'relations.dict')
    
    parser.add_argument('--data_dir', type=str, default=raw_data_path, help='Directory containing data files')
    parser.add_argument('--output_path', type=str, default=output_dict_path, help='Output path')
    parser.add_argument('--min_freq', type=int, default=5, help='Minimum frequency threshold')
    
    args = parser.parse_args()
    
    generate_relations_dict(args.data_dir, args.output_path, args.min_freq)

