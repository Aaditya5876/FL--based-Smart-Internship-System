from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def synonym_match_f1(pred_pairs: Iterable[Tuple[int, int]], gold_pairs: Iterable[Tuple[int, int]]) -> Dict[str, float]:
    pred = set((min(i, j), max(i, j)) for i, j in pred_pairs)
    gold = set((min(i, j), max(i, j)) for i, j in gold_pairs)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def phrase_sim_auc(embeddings: np.ndarray, labels: Sequence[int], max_pairs: int = 20000) -> float:
    n = embeddings.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(42)
    pairs = set()
    scores = []
    ys = []
    pos, neg = 0, 0
    for _ in range(max_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in pairs:
            continue
        pairs.add(key)
        s = cosine_sim(embeddings[i], embeddings[j])
        y = 1 if labels[i] == labels[j] else 0
        scores.append(s)
        ys.append(y)
        if y == 1:
            pos += 1
        else:
            neg += 1
        if pos > 1000 and neg > 1000:
            break
    if not scores or pos == 0 or neg == 0:
        # Not enough diversity to compute AUC; return 0.5 as null baseline
        return 0.5
    return float(roc_auc_score(ys, scores))


def cluster_purity_nmi(embeddings: np.ndarray, true_ids: Sequence[int], k: int) -> Dict[str, float]:
    if embeddings.shape[0] == 0:
        return {"purity": 0.0, "nmi": 0.0}
    km = KMeans(n_clusters=max(2, k), random_state=42, n_init=10)
    pred = km.fit_predict(embeddings)
    # purity
    purity = 0.0
    for c in range(km.n_clusters):
        idx = np.where(pred == c)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(np.array(true_ids)[idx], return_counts=True)
        purity += counts.max()
    purity = purity / len(true_ids)
    # nmi
    nmi = float(normalized_mutual_info_score(true_ids, pred))
    return {"purity": float(purity), "nmi": nmi}
