import os
import re
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    toks = re.split(r"[^a-z0-9]+", text)
    return [t for t in toks if t]


def _split_skills_field(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if text.strip().startswith("[") and text.strip().endswith("]"):
        inner = text.strip()[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
    else:
        parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


class Normalizer:
    def __init__(self, alias_map: Dict[str, str], canonical_vocab: Dict[str, Dict],
                 substring_threshold: float = 0.8, jaccard_threshold: float = 0.6):
        self.alias_map = {k.lower(): v for k, v in (alias_map or {}).items()}
        self.canonical_vocab = canonical_vocab or {}
        self.substring_threshold = substring_threshold
        self.jaccard_threshold = jaccard_threshold
        self._canon_tokens = {cid: set(_tokenize(v.get("name", ""))) for cid, v in self.canonical_vocab.items()}

    def _best_match(self, phrase: str) -> str:
        p = phrase.lower().strip()
        if not p:
            return ""
        if p in self.alias_map:
            return self.alias_map[p]
        # substring heuristic
        for a, cid in self.alias_map.items():
            if len(a) >= 3 and a in p:
                frac = len(a) / max(1, len(p))
                if frac >= self.substring_threshold:
                    return cid
        # jaccard over canonical names
        ptoks = set(_tokenize(p))
        best = (0.0, "")
        for cid, ctoks in self._canon_tokens.items():
            if not ctoks:
                continue
            inter = len(ptoks & ctoks)
            union = len(ptoks | ctoks)
            j = inter / union if union > 0 else 0.0
            if j > best[0]:
                best = (j, cid)
        if best[0] >= self.jaccard_threshold:
            return best[1]
        return ""

    def normalize_title(self, text: str) -> Dict:
        toks = _tokenize(text)
        cids = []
        if text:
            cid = self._best_match(text)
            if cid:
                cids.append(cid)
        return {"ids": cids, "tokens": toks}

    def normalize_skills(self, text: str) -> Dict:
        phrases = _split_skills_field(text)
        cids = []
        toks = []
        for p in phrases:
            toks.extend(_tokenize(p))
            cid = self._best_match(p)
            if cid:
                cids.append(cid)
        # dedupe
        cids = list(dict.fromkeys(cids))
        return {"ids": cids, "tokens": toks}

