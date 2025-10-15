import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..encoders.bi_encoder import HashingBiEncoder, set_seed, save_encoder
from ..ontology.alias_tools import load_aliases_review_csv, write_alias_and_vocab
from .simulated_clients import load_student_phrases_by_client
from ..encoders.bi_encoder import phrase_to_hashvec
from ..eval.intrinsic_metrics import phrase_sim_auc


class LocalTripletDataset(Dataset):
    def __init__(self, phrases: List[str], alias_map: Dict[str, str], hash_dim: int = 1024):
        by_cid: Dict[str, List[str]] = {}
        for p in phrases:
            cid = alias_map.get(p.lower())
            if cid:
                by_cid.setdefault(cid, []).append(p)
        # build triplets only from groups with >=2 phrases
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        rng = np.random.default_rng(42)
        neg_pool = []
        for cid, lst in by_cid.items():
            neg_pool.extend((cid, p) for p in lst)
        for cid, lst in by_cid.items():
            if len(lst) < 2:
                continue
            for i in range(len(lst) - 1):
                a, pos = lst[i], lst[i + 1]
                # pick negative from a different class
                for _ in range(3):
                    ncid, nphrase = neg_pool[int(rng.integers(0, len(neg_pool)))]
                    if ncid != cid:
                        break
                xa = phrase_to_hashvec(a, dim=hash_dim)
                xp = phrase_to_hashvec(pos, dim=hash_dim)
                xn = phrase_to_hashvec(nphrase, dim=hash_dim)
                self.samples.append((xa, xp, xn))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xa, xp, xn = self.samples[idx]
        return torch.from_numpy(xa), torch.from_numpy(xp), torch.from_numpy(xn)


def train_local_triplet(model: HashingBiEncoder, ds: Dataset, epochs: int, batch_size: int, lr: float, dp_clip: float, dp_noise: float):
    if len(ds) == 0:
        return {k: v.clone() for k, v in model.state_dict().items()}, 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    local = HashingBiEncoder(in_dim=model.in_dim, emb_dim=model.emb.out_features, num_classes=0)
    local.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    opt = torch.optim.Adam(local.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
    for _ in range(epochs):
        local.train()
        for xa, xp, xn in loader:
            opt.zero_grad()
            za = local.encode_tensor(xa)
            zp = local.encode_tensor(xp)
            zn = local.encode_tensor(xn)
            loss = loss_fn(za, zp, zn)
            loss.backward()
            # DP clipping + noise (simple, not per-sample)
            if dp_clip > 0:
                torch.nn.utils.clip_grad_norm_(local.parameters(), max_norm=dp_clip)
            if dp_noise > 0:
                for p in local.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * dp_noise)
            opt.step()
    return {k: v.clone() for k, v in local.state_dict().items()}, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hash_dim", type=int, default=1024)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--aliases_csv", type=str, default="src/research/RQ2/data/aliases_review.csv")
    ap.add_argument("--dp_clip", type=float, default=0.0)
    ap.add_argument("--dp_noise", type=float, default=0.0)
    args = ap.parse_args()

    set_seed(42)
    out_dir = os.path.join("data", "processed", f"{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    alias_map = load_aliases_review_csv(args.aliases_csv)
    # Persist curated alias/vocab to the versioned folder
    write_alias_and_vocab(out_dir, alias_map)
    clients = load_student_phrases_by_client("data/raw")
    print(f"[INFO] Simulating {len(clients)} clients (contrastive)")

    model = HashingBiEncoder(in_dim=args.hash_dim, emb_dim=args.emb_dim, num_classes=0)

    # Anchor evaluation set
    phrases = list(alias_map.keys())
    uniq_cids = sorted(set(alias_map.values()))
    cid_to_idx = {c: i for i, c in enumerate(uniq_cids)}
    labels = [cid_to_idx[alias_map[p]] for p in phrases]

    history = []
    for r in range(args.rounds):
        deltas = []
        weights = []
        for cid, plist in clients.items():
            ds = LocalTripletDataset(plist, alias_map, hash_dim=args.hash_dim)
            state_c, n_c = train_local_triplet(model, ds, epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr, dp_clip=args.dp_clip, dp_noise=args.dp_noise)
            if n_c <= 0:
                continue
            deltas.append(state_c)
            weights.append(float(n_c))
        if deltas:
            new_sd = {k: v.clone() for k, v in model.state_dict().items()}
            denom = max(1e-8, float(sum(weights)))
            keys = list(new_sd.keys())
            agg = {k: torch.zeros_like(new_sd[k]) for k in keys}
            for w, st in zip(weights, deltas):
                for k in keys:
                    agg[k] = agg[k] + (w / denom) * st[k]
            model.load_state_dict(agg)
        # Eval AUC on anchor
        from ..encoders.bi_encoder import encode_phrases
        E = encode_phrases(model, phrases)
        auc = phrase_sim_auc(E, labels, max_pairs=10000)
        history.append({"round": r, "auc": float(auc), "num_clients": len(clients)})
        print(f"[RD {r}] anchor AUC={auc:.4f}")

    enc_path = os.path.join(out_dir, "phrase_encoder.pt")
    save_encoder(enc_path, model)
    with open(os.path.join(out_dir, "fl_contrastive_report.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "params": sum(p.numel() for p in model.parameters())}, f, indent=2)
    print(f"[OK] Saved FL contrastive encoder to {enc_path}")


if __name__ == "__main__":
    main()
