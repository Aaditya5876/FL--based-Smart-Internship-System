import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .bi_encoder import HashingBiEncoder, set_seed, phrase_to_hashvec, save_encoder
from ..ontology.alias_tools import load_aliases_review_csv, write_alias_and_vocab
from ..ontology.canonical_vocab import load_alias_map, load_canonical_vocab
from ..eval.intrinsic_metrics import phrase_sim_auc, cluster_purity_nmi


class TripletDataset(Dataset):
    def __init__(self, by_cid: Dict[str, List[str]], hash_dim: int = 1024):
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        rng = np.random.default_rng(42)
        cids = [cid for cid, lst in by_cid.items() if len(lst) >= 2]
        neg_pool = []
        for cid, lst in by_cid.items():
            neg_pool.extend((cid, p) for p in lst)
        for cid in cids:
            phrases = by_cid[cid]
            for i in range(len(phrases) - 1):
                a = phrases[i]; p = phrases[i + 1]
                # pick a negative from a different cid
                for _ in range(3):
                    ncid, nphrase = neg_pool[int(rng.integers(0, len(neg_pool)))]
                    if ncid != cid:
                        break
                xa = phrase_to_hashvec(a, dim=hash_dim)
                xp = phrase_to_hashvec(p, dim=hash_dim)
                xn = phrase_to_hashvec(nphrase, dim=hash_dim)
                self.samples.append((xa, xp, xn))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xa, xp, xn = self.samples[idx]
        return torch.from_numpy(xa), torch.from_numpy(xp), torch.from_numpy(xn)


def build_by_cid(alias_map: Dict[str, str]) -> Dict[str, List[str]]:
    by_cid: Dict[str, List[str]] = {}
    for phrase, cid in alias_map.items():
        by_cid.setdefault(cid, []).append(phrase)
    # dedupe, sort for stability
    for cid in by_cid.keys():
        by_cid[cid] = sorted(list(dict.fromkeys(by_cid[cid])))
    return by_cid


def eval_embeddings(model: HashingBiEncoder, phrases: List[str], labels: List[int]) -> Dict[str, float]:
    from .bi_encoder import encode_phrases
    E = encode_phrases(model, phrases)
    auc = phrase_sim_auc(E, labels, max_pairs=10000)
    clu = cluster_purity_nmi(E, labels, k=len(set(labels)))
    return {"auc": float(auc), "purity": float(clu["purity"]), "nmi": float(clu["nmi"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hash_dim", type=int, default=1024)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--aliases_csv", type=str, default="src/research/RQ2/data/aliases_review.csv")
    args = ap.parse_args()

    set_seed(42)
    out_dir = os.path.join("data", "processed", f"{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    # Load curated aliases
    alias_map = load_aliases_review_csv(args.aliases_csv)
    # Persist curated alias/vocab to the versioned folder
    write_alias_and_vocab(out_dir, alias_map)
    by_cid = build_by_cid(alias_map)
    groups = [cid for cid, lst in by_cid.items() if len(lst) >= 2]
    if not groups:
        raise RuntimeError("Need at least one canonical group with >=2 phrases in aliases_review.csv")

    # Build dataset
    ds = TripletDataset(by_cid, hash_dim=args.hash_dim)
    if len(ds) == 0:
        raise RuntimeError("No triplets constructed; check aliases_review.csv")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = HashingBiEncoder(in_dim=args.hash_dim, emb_dim=args.emb_dim, num_classes=0)
    loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare eval labels
    phrases = list(alias_map.keys())
    uniq_cids = sorted(set(alias_map.values()))
    cid_to_idx = {c: i for i, c in enumerate(uniq_cids)}
    labels = [cid_to_idx[alias_map[p]] for p in phrases]

    history = []
    best = {"auc": -1.0, "state": None}
    for ep in range(args.epochs):
        model.train(); total=0.0; n=0
        for xa, xp, xn in loader:
            opt.zero_grad()
            za = model.encode_tensor(xa)
            zp = model.encode_tensor(xp)
            zn = model.encode_tensor(xn)
            loss = loss_fn(za, zp, zn)
            loss.backward(); opt.step()
            total += float(loss.item()) * len(xa)
            n += len(xa)
        tr_loss = total / max(1, n)
        stats = eval_embeddings(model, phrases, labels)
        stats.update({"epoch": ep, "train_loss": tr_loss})
        history.append(stats)
        print(f"[EP {ep}] train_loss={tr_loss:.4f} AUC={stats['auc']:.4f} purity={stats['purity']:.3f} nmi={stats['nmi']:.3f}")
        if stats["auc"] > best["auc"]:
            best = {"auc": stats["auc"], "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}

    if best["state"] is not None:
        model.load_state_dict(best["state"])  # type: ignore

    enc_path = os.path.join(out_dir, "phrase_encoder.pt")
    save_encoder(enc_path, model)
    with open(os.path.join(out_dir, "centralized_contrastive_report.json"), "w", encoding="utf-8") as f:
        json.dump({"best_auc": best["auc"], "history": history}, f, indent=2)
    print(f"[OK] Saved contrastive encoder to {enc_path}")


if __name__ == "__main__":
    main()
