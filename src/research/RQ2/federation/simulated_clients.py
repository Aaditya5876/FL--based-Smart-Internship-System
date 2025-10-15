import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..encoders.bi_encoder import phrase_to_hashvec


def load_student_phrases_by_client(data_root: str = "data/raw") -> Dict[str, List[str]]:
    p = os.path.join(data_root, "students.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "university" not in df.columns:
        raise ValueError("students.csv missing 'university'")
    if "skills" not in df.columns:
        raise ValueError("students.csv missing 'skills'")

    def split_skills(s: str) -> List[str]:
        if not isinstance(s, str) or not s.strip():
            return []
        if s.strip().startswith("[") and s.strip().endswith("]"):
            inner = s.strip()[1:-1]
            parts = [p.strip().strip("'\"") for p in inner.split(",")]
        else:
            parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    by_client: Dict[str, List[str]] = {}
    for cid, grp in df.groupby("university"):
        phrases: List[str] = []
        for s in grp["skills"].fillna(""):
            phrases.extend(split_skills(s))
        # dedupe, lower
        phrases = list(dict.fromkeys([p.lower() for p in phrases if p]))
        by_client[str(cid)] = phrases
    return by_client


def build_labeled_dataset(phrases: List[str], alias_map: Dict[str, str], cid_to_idx: Dict[str, int], limit_per_class: int = 200):
    X_list = []
    y_list = []
    for ph in phrases:
        c = alias_map.get(ph.lower())
        if not c:
            continue
        y = cid_to_idx.get(c)
        if y is None:
            continue
        X_list.append(phrase_to_hashvec(ph))
        y_list.append(y)
        if len(y_list) >= limit_per_class * max(1, len(cid_to_idx)):
            break
    if not X_list:
        return np.zeros((0, 1024), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y

