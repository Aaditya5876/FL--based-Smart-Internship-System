import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .bi_encoder import HashingBiEncoder, set_seed, phrase_to_hashvec, save_encoder
from ..ontology.canonical_vocab import bootstrap_from_raw, load_alias_map, load_canonical_vocab


def build_dataset(alias_map: Dict[str, str], limit_per_class: int = 200) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # alias_map: phrase -> canonical_id
    inv: Dict[str, List[str]] = {}
    for p, cid in alias_map.items():
        inv.setdefault(cid, []).append(p)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    cids = sorted(inv.keys())
    cid_to_idx = {cid: i for i, cid in enumerate(cids)}
    for cid in cids:
        phrases = inv[cid][:limit_per_class]
        for ph in phrases:
            X_list.append(phrase_to_hashvec(ph))
            y_list.append(cid_to_idx[cid])
    X = np.stack(X_list) if X_list else np.zeros((0, 1024), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64) if y_list else np.zeros((0,), dtype=np.int64)
    return X, y, cids


def train_epoch(model: HashingBiEncoder, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn) -> float:
    model.train(); total=0.0; n=0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        total += float(loss.item()) * len(xb)
        n += len(xb)
    return total/max(1,n)


def eval_top1(model: HashingBiEncoder, loader: DataLoader, loss_fn) -> Dict:
    model.eval(); total=0.0; n=0; correct=0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total += float(loss.item()) * len(xb)
            n += len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
    return {"loss": total/max(1,n), "acc": correct/max(1,n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="version tag, e.g., v2_alignment_bootstrap")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data_root", type=str, default="data/raw")
    ap.add_argument("--limit_per_class", type=int, default=200)
    args = ap.parse_args()

    set_seed(42)
    out_dir = os.path.join("data", "processed", f"{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    # Ensure vocabulary exists or bootstrap
    vocab_path = os.path.join(out_dir, "canonical_vocab.json")
    alias_path = os.path.join(out_dir, "alias_map.json")
    if not os.path.exists(vocab_path) or not os.path.exists(alias_path):
        print("[INFO] Bootstrapping canonical vocab + alias map from raw...")
        bootstrap_from_raw(args.data_root, out_dir, top_k=200)

    vocab = load_canonical_vocab(vocab_path)
    alias_map = load_alias_map(alias_path)
    X, y, cids = build_dataset(alias_map, limit_per_class=args.limit_per_class)
    if len(X) == 0:
        raise RuntimeError("No training data built from alias_map. Check bootstrap outputs.")

    # Train/val split
    n = len(X); idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.9 * n)
    tr_idx, va_idx = idx[:split], idx[split:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=args.batch_size, shuffle=False)

    model = HashingBiEncoder(in_dim=1024, emb_dim=64, num_classes=len(cids))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best = {"acc": -1.0, "state": None}
    for ep in range(args.epochs):
        tr_loss = train_epoch(model, tr_loader, opt, loss_fn)
        eval_stats = eval_top1(model, va_loader, loss_fn)
        eval_stats.update({"epoch": ep, "train_loss": tr_loss})
        history.append(eval_stats)
        print(f"[EP {ep}] train_loss={tr_loss:.4f} val_loss={eval_stats['loss']:.4f} val_acc={eval_stats['acc']:.4f}")
        if eval_stats["acc"] > best["acc"]:
            best = {"acc": eval_stats["acc"], "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}

    if best["state"] is not None:
        model.load_state_dict(best["state"])  # type: ignore

    enc_path = os.path.join(out_dir, "phrase_encoder.pt")
    save_encoder(enc_path, model)
    with open(os.path.join(out_dir, "centralized_train_report.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best["acc"], "history": history}, f, indent=2)
    print(f"[OK] Saved encoder to {enc_path}")


if __name__ == "__main__":
    main()

