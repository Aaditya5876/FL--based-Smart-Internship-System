import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RNG_SEED = 42


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    # split on non-alphanumeric, collapse spaces
    toks = re.split(r"[^a-z0-9]+", text)
    return [t for t in toks if t]


def _split_skills_field(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # common formats: comma-separated; possibly bracketed lists
    if text.strip().startswith("[") and text.strip().endswith("]"):
        inner = text.strip()[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
    else:
        parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def load_canonical_vocab(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_canonical_vocab(path: str, vocab: Dict[str, Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_alias_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_alias_map(path: str, alias_map: Dict[str, str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)


def bootstrap_from_raw(data_root: str, out_dir: str, top_k: int = 200) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    rng = np.random.default_rng(RNG_SEED)
    students_csv = os.path.join(data_root, "students.csv")
    jobs_csv = os.path.join(data_root, "jobs.csv")

    if not os.path.exists(students_csv) or not os.path.exists(jobs_csv):
        raise FileNotFoundError("Expected students.csv and jobs.csv under data_root")

    df_s = pd.read_csv(students_csv)
    df_j = pd.read_csv(jobs_csv)

    # Collect phrases from titles and skills
    title_terms = []
    if "title" in df_j.columns:
        for t in df_j["title"].fillna(""):
            toks = _tokenize(t)
            title_terms.extend(toks)

    skill_phrases = []
    if "skills_required" in df_j.columns:
        for s in df_j["skills_required"].fillna(""):
            skill_phrases.extend(_split_skills_field(s))
    if "skills" in df_s.columns:
        for s in df_s["skills"].fillna(""):
            skill_phrases.extend(_split_skills_field(s))

    # frequency-based bootstrap
    title_counts = Counter(title_terms)
    skill_counts = Counter([p.lower() for p in skill_phrases])

    # Choose top unique phrases across both sources
    top_titles = [w for w, _ in title_counts.most_common(top_k)]
    top_skills = [w for w, _ in skill_counts.most_common(top_k)]

    uniq_phrases = []
    seen = set()
    for p in top_skills + top_titles:
        if p not in seen:
            uniq_phrases.append(p)
            seen.add(p)

    # Build canonical vocab (flat)
    vocab: Dict[str, Dict] = {}
    alias_map: Dict[str, str] = {}
    for i, phrase in enumerate(uniq_phrases):
        cid = f"CANON_{i:04d}"
        vocab[cid] = {"id": cid, "name": phrase, "parents": []}
        alias_map[phrase] = cid

    # Save
    os.makedirs(out_dir, exist_ok=True)
    save_canonical_vocab(os.path.join(out_dir, "canonical_vocab.json"), vocab)
    save_alias_map(os.path.join(out_dir, "alias_map.json"), alias_map)

    return vocab, alias_map

