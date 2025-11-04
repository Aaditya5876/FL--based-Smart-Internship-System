# src/research/RQ1/algorithms/pfl/run.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from src.loaders.federated_dataset import load_federated_clients, summarize_clients

# ------------------- constants -------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

NUMERIC_COLS = [
    "gpa","projects_count","internships_count",
    "impressions","clicks","saves","apply","dwell_time","revisit_count",
]
CATEGORICAL_SINGLE = [
    "major","student_location","title","role_family",
    "salary_band","job_location","company_size","org_type",
]
MULTI_LABEL_COLS = ["courses","student_skills","required_skills","nice_to_have"]
TARGET = "recommended"

# ------------------- multi-hot encoder -------------------
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=","):
        self.sep = sep
        self.vocabs_ = {}
        self.offset_ = {}
        self.total_dim_ = 0

    def get_params(self, deep=True): return {"sep": self.sep}
    def set_params(self, **p):
        for k,v in p.items(): setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        start = 0
        for col in X.columns:
            vals = X[col].fillna("").astype(str)
            toks = set()
            for s in vals:
                toks.update(t.strip() for t in s.split(self.sep) if t.strip())
            vocab = sorted(toks)
            self.vocabs_[col] = vocab
            self.offset_[col] = start
            start += len(vocab)
        self.total_dim_ = start
        return self

    def transform(self, X):
        n = len(X)
        if n == 0 or self.total_dim_ == 0:
            return sparse.csr_matrix((n,0), dtype=np.float32)
        data, ri, ci = [], [], []
        for i, (_, row) in enumerate(X.iterrows()):
            for col in X.columns:
                s = "" if pd.isna(row[col]) else str(row[col])
                if not s: continue
                vocab = self.vocabs_.get(col, [])
                base = self.offset_.get(col, 0)
                for tok in (t.strip() for t in s.split(self.sep) if t.strip()):
                    if tok in vocab:
                        j = vocab.index(tok)
                        data.append(1); ri.append(i); ci.append(base+j)
        return sparse.csr_matrix((data,(ri,ci)), shape=(n,self.total_dim_), dtype=np.float32)

# ------------------- shared preprocessor -------------------
def build_shared_preprocessor(all_clients: Dict[str, Dict[str, pd.DataFrame]]) -> ColumnTransformer:
    feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS
    frames = []
    for cid in sorted(all_clients.keys()):
        df = all_clients[cid]["train"].copy()
        for col in feats:
            if col not in df.columns: df[col] = np.nan
        frames.append(df[feats])
    union_train = pd.concat(frames, ignore_index=True)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    preproc = ColumnTransformer(
        transformers=[
            ("num","passthrough",NUMERIC_COLS),
            ("cat",ohe,CATEGORICAL_SINGLE),
            ("mlb",MultiHotEncoder(),MULTI_LABEL_COLS),
        ],
        sparse_threshold=1.0,
    )
    preproc.fit(union_train)
    return preproc

def csr_to_loader(X_csr, y_np, bs, shuffle):
    X = torch.tensor(X_csr.toarray(), dtype=torch.float32) if sparse.issparse(X_csr) else torch.tensor(X_csr, dtype=torch.float32)
    y = torch.tensor(y_np.reshape(-1,1), dtype=torch.float32)
    return DataLoader(TensorDataset(X,y), batch_size=bs, shuffle=shuffle)

# ------------------- model (FedPer-style split) -------------------
class PFLNet(nn.Module):
    """
    Backbone (shared) + Head (personalized).
    Only backbone weights are aggregated; each client keeps its own head.
    """
    def __init__(self, input_dim: int, hidden=[256,128,64], p_drop=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        self.backbone = nn.Sequential(*layers)      # shared
        self.head = nn.Linear(d, 1)                 # personalized (client-specific)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

    # convenience helpers
    def backbone_state(self):
        return {k: v for k,v in self.state_dict().items() if not k.startswith("head.")}
    def head_state(self):
        return {k: v for k,v in self.state_dict().items() if k.startswith("head.")}
    def load_backbone(self, state_dict):
        own = self.state_dict()
        own.update({k:v for k,v in state_dict.items() if k in own})
        self.load_state_dict(own, strict=False)

# ------------------- client -------------------
class PFLClient:
    def __init__(self, client_id: str, preproc: ColumnTransformer, config: Dict):
        self.client_id = client_id
        self.preproc = preproc
        self.bs = config.get("batch_size", 256)
        self.local_epochs = config.get("local_epochs", 1)
        self.lr = config.get("local_lr", 1e-3)
        self.adapt_epochs = config.get("adapt_epochs", 1)  # optional head-only finetune

        df_tr = ALL_CLIENTS[client_id]["train"].copy()
        df_te = ALL_CLIENTS[client_id]["test"].copy()
        feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS
        for col in feats:
            for d in (df_tr, df_te):
                if col not in d.columns: d[col] = np.nan

        self.X_tr = preproc.transform(df_tr[feats]); self.y_tr = df_tr[TARGET].astype(int).values
        self.X_te = preproc.transform(df_te[feats]); self.y_te = df_te[TARGET].astype(int).values

        self.input_dim = self.X_tr.shape[1]
        self.model = PFLNet(self.input_dim)

    @property
    def n_train(self): return self.X_tr.shape[0]

    def set_backbone(self, bb_state):
        self.model.load_backbone(bb_state)

    def backbone_weights(self):
        return self.model.backbone_state()

    def local_train(self):
        """
        Train full model on local data (head learns personalization naturally).
        """
        self.model.train()
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loader = csr_to_loader(self.X_tr, self.y_tr, self.bs, shuffle=True)
        last = 0.0
        for _ in range(self.local_epochs):
            run = 0.0
            for xb,yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                run += loss.item()
            last = run / max(1, len(loader))
        return {"loss": float(last)}

    def adapt_head(self):
        """
        Optional tiny head-only finetune after global rounds.
        """
        if self.adapt_epochs <= 0: return
        self.model.train()
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.Adam(self.model.head.parameters(), lr=self.lr)
        loader = csr_to_loader(self.X_tr, self.y_tr, self.bs, shuffle=True)
        for _ in range(self.adapt_epochs):
            for xb,yb in loader:
                opt.zero_grad(); loss = loss_fn(self.model(xb), yb)
                loss.backward(); opt.step()
        for p in self.model.backbone.parameters():
            p.requires_grad = True

    def evaluate_test(self):
        self.model.eval()
        with torch.no_grad():
            loader = csr_to_loader(self.X_te, self.y_te, self.bs, shuffle=False)
            probs, ys = [], []
            for xb,yb in loader:
                p = torch.sigmoid(self.model(xb)).cpu().numpy().ravel()
                probs.append(p); ys.append(yb.cpu().numpy().ravel())
        y_prob = np.concatenate(probs) if probs else np.array([])
        y_true = np.concatenate(ys) if ys else np.array([])
        if len(y_true)==0 or y_true.sum()==0:
            return {"client_id": self.client_id, "pr_auc": 0.0, "roc_auc": 0.5, "macro_f1": 0.0}
        pr  = float(average_precision_score(y_true, y_prob))
        roc = float(roc_auc_score(y_true, y_prob))
        y_pred = (y_prob>=0.5).astype(int)
        f1  = float(f1_score(y_true, y_pred, average="macro"))
        return {"client_id": self.client_id, "pr_auc": pr, "roc_auc": roc, "macro_f1": f1}

# ------------------- server -------------------
class PFLServer:
    """
    FedPer-like: aggregate only the backbone; keep each client's head local.
    """
    def __init__(self, clients: List[PFLClient]):
        self.clients = clients
        self.global_backbone = clients[0].model.backbone_state()
        self.history: List[Dict] = []

    def aggregate_backbone(self, deltas_and_sizes: List[Tuple[Dict[str,torch.Tensor], int]]):
        # compute weighted avg backbone state from client models (weights = n_samples)
        total = sum(n for _,n in deltas_and_sizes)
        agg = {k: torch.zeros_like(v) for k,v in self.global_backbone.items()}
        for state, n in deltas_and_sizes:
            w = n / total
            for k in agg.keys():
                agg[k] += w * state[k]
        self.global_backbone = agg

    def round(self, rnd: int, fraction: float):
        print(f"\n=== PFL Round {rnd} ===")
        m = max(1, int(round(len(self.clients)*fraction)))
        idx = np.random.choice(len(self.clients), size=m, replace=False)
        selected = [self.clients[i] for i in idx]

        # broadcast backbone to selected
        for c in selected:
            c.set_backbone(self.global_backbone)

        # local training
        states, logs = [], []
        total_n = 0
        for c in selected:
            log = c.local_train()
            states.append((c.backbone_weights(), c.n_train))
            logs.append({"client_id": c.client_id, "loss": log["loss"], "n": c.n_train})
            total_n += c.n_train

        # aggregate backbone
        self.aggregate_backbone(states)

        # push updated backbone to all clients and evaluate
        evals = []
        for c in self.clients:
            c.set_backbone(self.global_backbone)
            ev = c.evaluate_test()
            evals.append(ev)

        avg_pr = float(np.mean([e["pr_auc"] for e in evals]))
        avg_roc = float(np.mean([e["roc_auc"] for e in evals]))
        avg_f1 = float(np.mean([e["macro_f1"] for e in evals]))
        pr_by = {e["client_id"]: e["pr_auc"] for e in evals}
        gap = max(pr_by.values()) - min(pr_by.values())
        worst_id, worst_pr = min(pr_by.items(), key=lambda kv: kv[1])

        out = {
            "round": rnd,
            "avg_pr_auc": avg_pr,
            "avg_roc_auc": avg_roc,
            "avg_macro_f1": avg_f1,
            "worst_client": {"client_id": worst_id, "pr_auc": worst_pr},
            "gap": gap,
            "train_logs": logs,
            "eval_metrics": evals,
        }
        self.history.append(out)
        print(f"[PFL r{rnd:03d}] PR-AUC(avg)={avg_pr:.4f} | ROC-AUC(avg)={avg_roc:.4f} "
              f"| Macro-F1(avg)={avg_f1:.4f} | Gap={gap:.4f} | Worst={worst_id}:{worst_pr:.4f}")
        return out

# ------------------- runner -------------------
def main(rounds=200, fraction=0.5, adapt_epochs=1, out_dir: Path|None=None):
    global ALL_CLIENTS
    ALL_CLIENTS = load_federated_clients()
    summarize_clients(ALL_CLIENTS)
    preproc = build_shared_preprocessor(ALL_CLIENTS)

    cfg = {"batch_size":256, "local_epochs":1, "local_lr":1e-3, "adapt_epochs":adapt_epochs}
    cids = sorted(ALL_CLIENTS.keys())
    clients = [PFLClient(cid, preproc, cfg) for cid in cids]

    server = PFLServer(clients)
    for r in range(1, rounds+1):
        server.round(r, fraction=fraction)

    # Optional quick personalization pass (head-only) after global training
    if adapt_epochs and adapt_epochs > 0:
        for c in clients:
            c.set_backbone(server.global_backbone)
            c.adapt_head()

    # Final evaluation after adaptation
    evals = []
    for c in clients:
        c.set_backbone(server.global_backbone)
        evals.append(c.evaluate_test())

    avg_pr = float(np.mean([e["pr_auc"] for e in evals]))
    avg_roc = float(np.mean([e["roc_auc"] for e in evals]))
    avg_f1 = float(np.mean([e["macro_f1"] for e in evals]))
    pr_by = {e["client_id"]: e["pr_auc"] for e in evals}
    gap = max(pr_by.values()) - min(pr_by.values())
    worst_id, worst_pr = min(pr_by.items(), key=lambda kv: kv[1])

    final = {
        "round": rounds,
        "avg_pr_auc": avg_pr,
        "avg_roc_auc": avg_roc,
        "avg_macro_f1": avg_f1,
        "worst_client": {"client_id": worst_id, "pr_auc": worst_pr},
        "gap": gap,
        "eval_metrics": evals,
    }

    if out_dir is None:
        out_dir = Path("experiments") / "rq1_alpha_0.1" / f"pfl_p{fraction}" / f"seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("\nFinal summary (after personalization):")
    print(json.dumps({
        "avg_pr_auc": final["avg_pr_auc"],
        "avg_roc_auc": final["avg_roc_auc"],
        "avg_macro_f1": final["avg_macro_f1"],
        "worst_client": final["worst_client"],
        "gap": final["gap"],
    }, indent=2))

if __name__ == "__main__":
    # Example:
    #   python -m src.research.RQ1.algorithms.pfl.run
    # or
    #   $env:PYTHONPATH="$PWD"; python src\research\RQ1\algorithms\pfl\run.py
    main(rounds=200, fraction=0.5, adapt_epochs=1)
