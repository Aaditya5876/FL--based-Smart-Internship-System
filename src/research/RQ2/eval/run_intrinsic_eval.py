import argparse
import json
import os
from typing import Dict, List

import numpy as np

from ..encoders.bi_encoder import load_encoder, encode_phrases
from ..ontology.canonical_vocab import load_alias_map, load_canonical_vocab
from .intrinsic_metrics import synonym_match_f1, phrase_sim_auc, cluster_purity_nmi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--top_pairs", type=int, default=0, help="If 0, auto = number of gold synonym pairs")
    args = ap.parse_args()

    out_dir = os.path.join("data", "processed", f"{args.version}")
    enc_path = os.path.join(out_dir, "phrase_encoder.pt")
    alias_path = os.path.join(out_dir, "alias_map.json")
    vocab_path = os.path.join(out_dir, "canonical_vocab.json")

    alias_map = load_alias_map(alias_path)
    vocab = load_canonical_vocab(vocab_path)
    if not alias_map:
        raise RuntimeError("alias_map.json missing or empty")

    phrases = list(alias_map.keys())
    cids = [alias_map[p] for p in phrases]
    # map canonical id to small integers
    uniq_cids = sorted(set(cids))
    cid_to_idx = {c: i for i, c in enumerate(uniq_cids)}
    labels = [cid_to_idx[c] for c in cids]

    # Auto-detect classifier presence; embeddings are sufficient for intrinsic eval
    enc = load_encoder(enc_path, emb_dim=args.emb_dim)
    E = encode_phrases(enc, phrases)

    # predicted pairs by top cosine thresholding
    sims = E @ E.T
    np.fill_diagonal(sims, -1.0)
    flat_idx = np.dstack(np.unravel_index(np.argsort(sims.ravel())[::-1], sims.shape))[0]
    pred_pairs = []
    taken = set()
    # gold pairs from alias_map labels
    gold_pairs = []
    label_to_idx = {}
    for idx, lab in enumerate(labels):
        label_to_idx.setdefault(lab, []).append(idx)
    for lab, idxs in label_to_idx.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                gold_pairs.append((idxs[i], idxs[j]))

    # auto-top if not provided
    gold_pair_count = len(gold_pairs)
    top_k_pairs = args.top_pairs if args.top_pairs and args.top_pairs > 0 else max(1, gold_pair_count)

    for i, j in flat_idx:
        if len(pred_pairs) >= top_k_pairs:
            break
        a, b = int(i), int(j)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in taken:
            continue
        taken.add(key)
        pred_pairs.append(key)

    f1 = synonym_match_f1(pred_pairs, gold_pairs)
    auc = phrase_sim_auc(E, labels, max_pairs=20000)
    clu = cluster_purity_nmi(E, labels, k=min(50, len(uniq_cids)))

    report = {
        "synonym_f1": f1,
        "sim_auc": auc,
        "cluster": clu,
        "num_phrases": len(phrases),
        "top_pairs_used": int(top_k_pairs),
        "gold_pairs": int(gold_pair_count),
        "unique_labels": int(len(uniq_cids)),
    }
    with open(os.path.join(out_dir, "intrinsic_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
