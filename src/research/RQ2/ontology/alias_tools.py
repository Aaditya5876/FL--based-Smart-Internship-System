import csv
import json
import os
from typing import Dict, List, Tuple


def load_aliases_review_csv(csv_path: str) -> Dict[str, str]:
    """
    Load a curated CSV of synonyms with columns: canonical_id,phrase
    Returns alias_map: phrase(lowercased) -> canonical_id
    Skips empty rows; trims whitespace.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    alias_map: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "canonical_id" not in reader.fieldnames or "phrase" not in reader.fieldnames:
            raise ValueError("CSV must have headers: canonical_id,phrase")
        for row in reader:
            cid = (row.get("canonical_id") or "").strip()
            phrase = (row.get("phrase") or "").strip()
            if not cid or not phrase:
                continue
            alias_map[phrase.lower()] = cid
    return alias_map


def write_alias_and_vocab(out_dir: str, alias_map: Dict[str, str]):
    """
    Write alias_map.json and canonical_vocab.json derived from curated alias map.
    canonical_vocab entries are flat, with each canonical_id assigned a name (first phrase seen).
    """
    os.makedirs(out_dir, exist_ok=True)
    # build canonical vocab from alias_map
    by_cid: Dict[str, List[str]] = {}
    for p, cid in alias_map.items():
        by_cid.setdefault(cid, []).append(p)
    canonical_vocab: Dict[str, Dict] = {}
    for cid, phrases in by_cid.items():
        name = phrases[0] if phrases else cid
        canonical_vocab[cid] = {"id": cid, "name": name, "parents": []}

    with open(os.path.join(out_dir, "alias_map.json"), "w", encoding="utf-8") as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "canonical_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(canonical_vocab, f, indent=2, ensure_ascii=False)

