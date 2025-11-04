# src/research/RQ1/algorithms/fedopt/run.py
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

# ------------------ Repro ------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------ Feature contract ------------------
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

# ------------------ Multi-hot for list columns ------------------
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=","):
        self.sep = sep
        self.vocabs_ : Dict[str, List[str]] = {}
        self.offset_ : Dict[str, int] = {}
        self.total_dim_ = 0

    def get_params(self, deep=True): return {"sep": self.sep}
    def set_params(self, **params):
        for k,v in params.items(): setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        start = 0
        self.vocabs_.clear(); self.offset_.clear()
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
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        n = len(X)
        if n == 0 or self.total_dim_ == 0:
            return sparse.csr_matrix((n,0), dtype=np.float32)
        data, ri, ci = [], [], []
        for i, (_, row) in enumerate(X.iterrows()):
            for col in X.columns:
                s = "" if pd.isna(row[col]) else str(row[col])
                if not s: continue
                vocab = self.vocabs_.get(col, [])
                base  = self.offset_.get(col, 0)
                for tok in (t.strip() for t in s.split(self.sep) if t.strip()):
                    try: j = vocab.index(tok)
                    except ValueError: continue
                    data.append(1); ri.append(i); ci.append(base+j)
        return sparse.csr_matrix((data,(ri,ci)), shape=(n,self.total_dim_), dtype=np.float32)

# ------------------ MLP ------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden=[256,128,64], p_drop=0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d,1)]  # logit
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ------------------ Utils ------------------
def csr_to_loader(X_csr, y_np, bs, shuffle):
    X = torch.tensor(X_csr.toarray(), dtype=torch.float32) if sparse.issparse(X_csr) else torch.tensor(X_csr, dtype=torch.float32)
    y = torch.tensor(y_np.reshape(-1,1), dtype=torch.float32)
    ds = TensorDataset(X,y)
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

# ------------------ Client ------------------
class FedOptClient:
    def __init__(self, client_id: str, preproc: ColumnTransformer, config: Dict):
        self.client_id = client_id
        self.preproc = preproc
        self.bs = config.get("batch_size", 256)
        self.local_epochs = config.get("local_epochs", 1)
        self.lr = config.get("local_lr", 1e-3)

        df_tr = ALL_CLIENTS[client_id]["train"].copy()
        df_va = ALL_CLIENTS[client_id]["val"].copy()
        df_te = ALL_CLIENTS[client_id]["test"].copy()

        feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS
        for col in feats:
            for d in (df_tr, df_va, df_te):
                if col not in d.columns: d[col] = np.nan

        self.X_tr = preproc.transform(df_tr[feats]); self.y_tr = df_tr[TARGET].astype(int).values
        self.X_va = preproc.transform(df_va[feats]); self.y_va = df_va[TARGET].astype(int).values
        self.X_te = preproc.transform(df_te[feats]); self.y_te = df_te[TARGET].astype(int).values

        self.input_dim = self.X_tr.shape[1]
        self.model = MLP(self.input_dim)

    @property
    def n_train(self): return self.X_tr.shape[0]

    def set_weights(self, state): self.model.load_state_dict(state, strict=True)
    def get_weights(self): return {k:v.data.clone() for k,v in self.model.state_dict().items()}

    def train_one_round(self) -> Dict[str,float]:
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        lossfn = nn.BCEWithLogitsLoss()
        loader = csr_to_loader(self.X_tr, self.y_tr, self.bs, shuffle=True)
        last = 0.0
        for _ in range(self.local_epochs):
            run = 0.0
            for xb,yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = lossfn(logits, yb)
                loss.backward()
                opt.step()
                run += loss.item()
            last = run / max(1,len(loader))
        return {"loss": float(last)}

    def compute_delta(self, global_state) -> Dict[str, torch.Tensor]:
        """Return (local_weights - global_weights) after local training."""
        local = self.get_weights()
        delta = {}
        for k in global_state.keys():
            delta[k] = local[k] - global_state[k]
        return delta

    def evaluate_test(self) -> Dict[str, float]:
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

# ------------------ Server: FedAdam ------------------
# ------------------ Server: FedAdam ------------------
class FedOptServer:
    def __init__(self, clients: List[FedOptClient], server_lr: float = 1e-1,
                 beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-8):
        self.clients = clients
        self.global_model = MLP(clients[0].input_dim)

        # optimizer state (same keys as model state_dict)
        self.m = {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}

        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = server_lr
        self.eps = eps

        self.t = 0
        self.history: List[Dict] = []

    def round(self, rnd: int, fraction: float) -> Dict:
        # select clients
        m = max(1, int(round(len(self.clients) * fraction)))
        idx = np.random.choice(len(self.clients), size=m, replace=False)
        selected = [self.clients[i] for i in idx]

        # broadcast
        g_state = self.global_model.state_dict()
        weighted_deltas: List[Tuple[Dict[str, torch.Tensor], int]] = []
        logs = []
        total_n = 0

        for c in selected:
            c.set_weights(g_state)
            train_log = c.train_one_round()
            delta = c.compute_delta(g_state)
            n = c.n_train
            weighted_deltas.append((delta, n))
            logs.append({"client_id": c.client_id, "loss": train_log["loss"], "n": n})
            total_n += n

        # aggregate (weighted)
        agg = {k: torch.zeros_like(v) for k, v in g_state.items()}
        for delta, n in weighted_deltas:
            w = n / total_n
            for k in agg.keys():
                agg[k] += w * delta[k]

        # FedAdam step
        self.t += 1
        new_state = {}
        for k, w in g_state.items():
            g = agg[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            new_state[k] = w - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        self.global_model.load_state_dict(new_state, strict=True)

        # evaluate on ALL clients (test)
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
        print(f"[FedOpt r{rnd:03d}] PR-AUC(avg)={avg_pr:.4f} | ROC-AUC(avg)={avg_roc:.4f} | "
              f"Macro-F1(avg)={avg_f1:.4f} | Gap={gap:.4f} | Worst={worst_id}:{worst_pr:.4f}")
        return out


# ------------------ Orchestration ------------------
def main(rounds=200, fraction=0.5, out_dir: Path | None = None):
    global ALL_CLIENTS
    ALL_CLIENTS = load_federated_clients()
    summarize_clients(ALL_CLIENTS)

    preproc = build_shared_preprocessor(ALL_CLIENTS)

    cids = sorted(ALL_CLIENTS.keys())
    cfg = {"batch_size":256, "local_epochs":1, "local_lr":1e-3}
    clients = [FedOptClient(cid, preproc, cfg) for cid in cids]

    server = FedOptServer(clients, server_lr=1e-1, beta1=0.9, beta2=0.99, eps=1e-8)

    for r in range(1, rounds+1):
        server.round(r, fraction=fraction)

    final = server.history[-1]
    if out_dir is None:
        out_dir = Path("experiments") / "rq1_alpha_0.1" / f"fedopt_fedadam_p{fraction}" / f"seed{SEED}"
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
    # Example: python -m src.research.RQ1.algorithms.fedopt.run
    main(rounds=200, fraction=0.5)
