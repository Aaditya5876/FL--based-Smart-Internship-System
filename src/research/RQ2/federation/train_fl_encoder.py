import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..encoders.bi_encoder import HashingBiEncoder, set_seed, save_encoder
from ..ontology.canonical_vocab import bootstrap_from_raw, load_alias_map, load_canonical_vocab
from .simulated_clients import load_student_phrases_by_client, build_labeled_dataset


def train_local(model: HashingBiEncoder, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 256, lr: float = 1e-3):
    if len(X) == 0:
        return {k: v.clone() for k, v in model.state_dict().items()}, 0
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=True)
    local = HashingBiEncoder(in_dim=model.in_dim, emb_dim=model.emb.out_features, num_classes=model.classifier.out_features if model.classifier is not None else 0)
    local.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    opt = torch.optim.Adam(local.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        local.train()
        for xb, yb in loader:
            opt.zero_grad(); logits = local(xb); loss = loss_fn(logits, yb); loss.backward(); opt.step()
    return {k: v.clone() for k, v in local.state_dict().items()}, len(X)


def eval_anchor(model: HashingBiEncoder, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(X) == 0:
        return {"loss": 0.0, "acc": 0.0}
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=512, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    model.eval(); total=0.0; n=0; correct=0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb); loss = loss_fn(logits, yb)
            total += float(loss.item()) * len(xb); n += len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
    return {"loss": total/max(1,n), "acc": correct/max(1,n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data_root", type=str, default="data/raw")
    ap.add_argument("--limit_per_class", type=int, default=200)
    args = ap.parse_args()

    set_seed(42)
    out_dir = os.path.join("data", "processed", f"{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    # vocab/alias
    vocab_path = os.path.join(out_dir, "canonical_vocab.json")
    alias_path = os.path.join(out_dir, "alias_map.json")
    if not os.path.exists(vocab_path) or not os.path.exists(alias_path):
        print("[INFO] Bootstrapping canonical vocab + alias map from raw...")
        bootstrap_from_raw(args.data_root, out_dir, top_k=200)
    vocab = load_canonical_vocab(vocab_path)
    alias_map = load_alias_map(alias_path)

    # map canonical IDs -> indices
    cids = sorted({cid for cid in alias_map.values()})
    cid_to_idx = {c: i for i, c in enumerate(cids)}

    # clients
    clients = load_student_phrases_by_client(args.data_root)
    print(f"[INFO] Simulating {len(clients)} clients")

    # anchor set (global evaluation) — sample a subset from alias_map
    phrases = list(alias_map.keys())
    np.random.shuffle(phrases)
    anchor = phrases[: min(2000, len(phrases))]
    X_anchor, y_anchor = build_labeled_dataset(anchor, alias_map, cid_to_idx, limit_per_class=args.limit_per_class)

    # init model
    model = HashingBiEncoder(in_dim=1024, emb_dim=64, num_classes=len(cids))

    history = []
    for r in range(args.rounds):
        deltas = []
        weights = []
        for cid, plist in clients.items():
            Xc, yc = build_labeled_dataset(plist, alias_map, cid_to_idx, limit_per_class=args.limit_per_class)
            state_c, n_c = train_local(model, Xc, yc, epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr)
            if n_c <= 0:
                continue
            deltas.append(state_c)
            weights.append(float(n_c))

        # FedAvg aggregation
        if deltas:
            new_sd = {k: v.clone() for k, v in model.state_dict().items()}
            denom = max(1e-8, float(sum(weights)))
            keys = list(new_sd.keys())
            agg = {k: torch.zeros_like(new_sd[k]) for k in keys}
            for w, st in zip(weights, deltas):
                for k in keys:
                    agg[k] = agg[k] + (w / denom) * st[k]
            model.load_state_dict(agg)

        # Evaluate on anchor
        stats = eval_anchor(model, X_anchor, y_anchor)
        stats.update({"round": r, "num_clients": len(clients)})
        history.append(stats)
        print(f"[RD {r}] anchor_loss={stats['loss']:.4f} anchor_acc={stats['acc']:.4f}")

    enc_path = os.path.join(out_dir, "phrase_encoder.pt")
    save_encoder(enc_path, model)
    with open(os.path.join(out_dir, "fl_train_report.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "params": sum(p.numel() for p in model.parameters())}, f, indent=2)
    print(f"[OK] Saved global encoder to {enc_path}")


if __name__ == "__main__":
    main()

