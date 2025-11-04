# src/research/RQ1/algorithms/fedprox/run.py
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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

from src.loaders.federated_dataset import load_federated_clients, summarize_clients

# ------------------- Constants -------------------
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

# ------------------- MultiHotEncoder -------------------
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=","):
        self.sep = sep
        self.vocabs_ = {}
        self.offset_ = {}
        self.total_dim_ = 0

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
            return sparse.csr_matrix((n, 0), dtype=np.float32)
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
                        data.append(1)
                        ri.append(i)
                        ci.append(base + j)
        return sparse.csr_matrix((data,(ri,ci)), shape=(n,self.total_dim_), dtype=np.float32)

# ------------------- Model -------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden=[256,128,64], p_drop=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, 1))  # binary logit
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

# ------------------- Data utils -------------------
def csr_to_loader(X_csr, y_np, bs, shuffle):
    X = torch.tensor(X_csr.toarray(), dtype=torch.float32) if sparse.issparse(X_csr) else torch.tensor(X_csr, dtype=torch.float32)
    y = torch.tensor(y_np.reshape(-1,1), dtype=torch.float32)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

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
    preproc = ColumnTransformer([
        ("num","passthrough",NUMERIC_COLS),
        ("cat",ohe,CATEGORICAL_SINGLE),
        ("mlb",MultiHotEncoder(),MULTI_LABEL_COLS),
    ], sparse_threshold=1.0)
    preproc.fit(union_train)
    return preproc

# ------------------- FedProx Client -------------------
class FedProxClient:
    def __init__(self, client_id, preproc, config):
        self.client_id = client_id
        self.preproc = preproc
        self.mu = config.get("mu", 0.1)
        self.bs = config.get("batch_size", 256)
        self.local_epochs = config.get("local_epochs", 1)
        self.lr = config.get("local_lr", 1e-3)

        df_tr = ALL_CLIENTS[client_id]["train"].copy()
        df_te = ALL_CLIENTS[client_id]["test"].copy()
        feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS
        for col in feats:
            for d in (df_tr, df_te):
                if col not in d.columns: d[col] = np.nan

        self.X_tr = preproc.transform(df_tr[feats])
        self.y_tr = df_tr[TARGET].astype(int).values
        self.X_te = preproc.transform(df_te[feats])
        self.y_te = df_te[TARGET].astype(int).values

        self.input_dim = self.X_tr.shape[1]
        self.model = MLP(self.input_dim)

    @property
    def n_train(self): return self.X_tr.shape[0]

    def set_weights(self, state): self.model.load_state_dict(state, strict=True)
    def get_weights(self): return {k:v.data.clone() for k,v in self.model.state_dict().items()}

    def train_one_round(self, global_state):
        self.model.train()
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loader = csr_to_loader(self.X_tr, self.y_tr, self.bs, shuffle=True)

        global_params = {k: v.clone().detach() for k,v in global_state.items()}
        for _ in range(self.local_epochs):
            total = 0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                prox = 0.0
                for name, param in self.model.named_parameters():
                    prox += torch.norm(param - global_params[name])**2
                loss = loss + (self.mu / 2.0) * prox
                loss.backward()
                opt.step()
                total += loss.item()
        return {"loss": total / max(1,len(loader))}

    def compute_delta(self, global_state):
        local = self.get_weights()
        delta = {}
        for k in global_state.keys():
            delta[k] = local[k] - global_state[k]
        return delta

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

# ------------------- FedProx Server -------------------
class FedProxServer:
    def __init__(self, clients):
        self.clients = clients
        self.global_model = MLP(clients[0].input_dim)
        self.history: List[Dict] = []

    def round(self, rnd: int, fraction: float):
        print(f"\n=== FedProx Round {rnd} ===")
        m = max(1, int(round(len(self.clients)*fraction)))
        idx = np.random.choice(len(self.clients), size=m, replace=False)
        selected = [self.clients[i] for i in idx]
        g_state = self.global_model.state_dict()

        weighted_deltas, logs, total_n = [], [], 0
        for c in selected:
            c.set_weights(g_state)
            train_log = c.train_one_round(g_state)
            delta = c.compute_delta(g_state)
            n = c.n_train
            weighted_deltas.append((delta, n))
            logs.append({"client_id": c.client_id, "loss": train_log["loss"], "n": n})
            total_n += n

        agg = {k: torch.zeros_like(v) for k,v in g_state.items()}
        for delta, n in weighted_deltas:
            w = n / total_n
            for k in agg.keys():
                agg[k] += w * delta[k]

        new_state = {k: g_state[k] + agg[k] for k in g_state.keys()}
        self.global_model.load_state_dict(new_state, strict=True)

        evals = []
        for c in self.clients:
            c.set_weights(self.global_model.state_dict())
            evals.append(c.evaluate_test())

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
        print(f"[FedProx r{rnd:03d}] PR-AUC(avg)={avg_pr:.4f} | ROC-AUC(avg)={avg_roc:.4f} "
              f"| Macro-F1(avg)={avg_f1:.4f} | Gap={gap:.4f} | Worst={worst_id}:{worst_pr:.4f}")
        return out

# ------------------- Runner -------------------
def main(rounds=200, fraction=0.5, mu=0.1, out_dir: Path|None=None):
    global ALL_CLIENTS
    ALL_CLIENTS = load_federated_clients()
    summarize_clients(ALL_CLIENTS)
    preproc = build_shared_preprocessor(ALL_CLIENTS)

    cfg = {"batch_size":256, "local_epochs":1, "local_lr":1e-3, "mu":mu}
    cids = sorted(ALL_CLIENTS.keys())
    clients = [FedProxClient(cid, preproc, cfg) for cid in cids]

    server = FedProxServer(clients)
    for r in range(1, rounds+1):
        server.round(r, fraction=fraction)

    final = server.history[-1]
    if out_dir is None:
        out_dir = Path("experiments") / "rq1_alpha_0.1" / f"fedprox_p{fraction}" / f"seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("\nFinal summary:")
    print(json.dumps({
        "avg_pr_auc": final["avg_pr_auc"],
        "avg_roc_auc": final["avg_roc_auc"],
        "avg_macro_f1": final["avg_macro_f1"],
        "worst_client": final["worst_client"],
        "gap": final["gap"],
    }, indent=2))

if __name__ == "__main__":
    main(rounds=200, fraction=0.5, mu=0.1)
