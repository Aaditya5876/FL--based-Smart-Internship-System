# src/research/RQ1/algorithms/fedavg/run.py
"""
FedAvg (RQ1 baseline) — Smart Internship Engine

- Task: Binary classification on `recommended`
- Data: data/processed/client_*/data.csv (+ splits/train_test_splits.json)
- Features:
    NUMERIC: gpa, projects_count, internships_count, impressions, clicks, saves, apply, dwell_time, revisit_count
    CATEGORICAL (single): major, student_location, title, role_family, salary_band, job_location, company_size, org_type
    MULTI-LABEL (list, comma-separated): courses, student_skills, required_skills, nice_to_have
- Metrics: PR-AUC, ROC-AUC, macro-F1 (test per client), worst-client gap
- Aggregation: sample-size–weighted averaging (true FedAvg)
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

# Local loader
from src.loaders.federated_dataset import load_federated_clients, summarize_clients


# ------------------------------
# Repro
# ------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------
# Feature contract
# ------------------------------
NUMERIC_COLS = [
    "gpa", "projects_count", "internships_count",
    "impressions", "clicks", "saves", "apply", "dwell_time", "revisit_count",
]
CATEGORICAL_SINGLE = [
    "major", "student_location",
    "title", "role_family", "salary_band", "job_location", "company_size",
    "org_type",
]
MULTI_LABEL_COLS = ["courses", "student_skills", "required_skills", "nice_to_have"]
TARGET = "recommended"


# ------------------------------
# MultiHot for list columns (comma-separated)
# ------------------------------
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=","):
        self.sep = sep
        self.vocabs_ = {}
        self.col_offset_ = {}
        self.total_dim_ = 0
    def get_params(self, deep=True): return {"sep": self.sep}
    def set_params(self, **p): 
        for k,v in p.items(): setattr(self, k, v); 
        return self
    def fit(self, X, y=None):
        import pandas as pd
        start = 0
        self.vocabs_.clear(); self.col_offset_.clear()
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        for col in X.columns:
            toks = set()
            s = X[col].fillna("").astype(str)
            for v in s:
                if not v: continue
                toks.update(t.strip() for t in v.split(self.sep) if t.strip())
            vocab = sorted(toks)
            self.vocabs_[col] = vocab
            self.col_offset_[col] = start
            start += len(vocab)
        self.total_dim_ = start
        return self
    def transform(self, X):
        import pandas as pd
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        n = len(X)
        if n == 0 or self.total_dim_ == 0:
            return sparse.csr_matrix((n, 0), dtype=np.float32)
        data, ri, ci = [], [], []
        for i, (_, row) in enumerate(X.iterrows()):
            for col in X.columns:
                s = "" if pd.isna(row[col]) else str(row[col])
                if not s: continue
                vocab = self.vocabs_.get(col, []); base = self.col_offset_.get(col, 0)
                for tok in (t.strip() for t in s.split(self.sep) if t.strip()):
                    try: j = vocab.index(tok)
                    except ValueError: continue
                    data.append(1); ri.append(i); ci.append(base + j)
        return sparse.csr_matrix((data, (ri, ci)), shape=(n, self.total_dim_), dtype=np.float32)

def build_shared_preprocessor(all_clients):
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
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", ohe, CATEGORICAL_SINGLE),
            ("mlb", MultiHotEncoder(), MULTI_LABEL_COLS),
        ],
        sparse_threshold=1.0,
    )
    preproc.fit(union_train)
    return preproc


# ------------------------------
# Simple MLP (logit output)
# ------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] = [256, 128, 64], p_drop: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]  # logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # shape [B,1] (logits)


# ------------------------------
# Utility: sparse->tensor DataLoader
# ------------------------------
def make_loader(X_csr: sparse.csr_matrix, y_np: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X = torch.tensor(X_csr.toarray(), dtype=torch.float32) if sparse.issparse(X_csr) else torch.tensor(X_csr, dtype=torch.float32)
    y = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float32)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ------------------------------
# Federated Client
# ------------------------------
class FederatedClient:
    def __init__(self, client_id: str, preproc: ColumnTransformer):
        self.client_id = client_id
        self.preproc = preproc

        df_train = ALL_CLIENTS[client_id]["train"].copy()
        df_val   = ALL_CLIENTS[client_id]["val"].copy()
        df_test  = ALL_CLIENTS[client_id]["test"].copy()

        feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS
        # fill missing feature columns if any
        for col in feats:
            if col not in df_train.columns:
                for d in (df_train, df_val, df_test):
                    d[col] = np.nan

        self.X_train = preproc.transform(df_train[feats])
        self.X_val   = preproc.transform(df_val[feats])
        self.X_test  = preproc.transform(df_test[feats])

        self.y_train = df_train[TARGET].astype(int).values
        self.y_val   = df_val[TARGET].astype(int).values
        self.y_test  = df_test[TARGET].astype(int).values

        self.input_dim = self.X_train.shape[1]
        self.model = MLP(self.input_dim)

        self.batch_size = 256
        self.epochs = 1
        self.lr = 1e-3

    def set_weights(self, state: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state, strict=True)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.data.clone() for k, v in self.model.state_dict().items()}

    @property
    def num_train(self) -> int:
        return self.X_train.shape[0]

    def train_one_round(self) -> float:
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        loader = make_loader(self.X_train, self.y_train, self.batch_size, shuffle=True)
        total = 0.0
        for _ in range(self.epochs):
            running = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item()
            total = running / max(1, len(loader))
        return total

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            loader = make_loader(self.X_test, self.y_test, self.batch_size, shuffle=False)
            probs, ys = [], []
            for xb, yb in loader:
                logits = self.model(xb)
                p = torch.sigmoid(logits).cpu().numpy().ravel()
                probs.append(p); ys.append(yb.cpu().numpy().ravel())
        y_prob = np.concatenate(probs) if probs else np.array([])
        y_true = np.concatenate(ys) if ys else np.array([])

        if len(y_true) == 0 or y_true.sum() == 0:
            # edge case: no positives; avoid crashes
            pr_auc = 0.0
            roc = 0.5
            macro_f1 = 0.0
        else:
            pr_auc = float(average_precision_score(y_true, y_prob))
            roc    = float(roc_auc_score(y_true, y_prob))
            y_pred = (y_prob >= 0.5).astype(int)
            macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        return {"client_id": self.client_id, "pr_auc": pr_auc, "roc_auc": roc, "macro_f1": macro_f1}


# ------------------------------
# FedAvg Server
# ------------------------------
class FedAvgServer:
    def __init__(self, clients: List[FederatedClient]):
        self.clients = clients
        self.global_model = MLP(clients[0].input_dim)
        self.history: List[Dict] = []

    def _aggregate(self, weighted_states: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        total_n = sum(n for _, n in weighted_states)
        agg: Dict[str, torch.Tensor] = {}
        for k in weighted_states[0][0].keys():
            agg[k] = sum(state[k] * (n / total_n) for state, n in weighted_states)
        return agg

    def round(self, rnd: int, fraction: float = 0.5) -> Dict:
        # sample clients
        m = max(1, int(round(len(self.clients) * fraction)))
        sel_idx = np.random.choice(len(self.clients), size=m, replace=False)
        selected = [self.clients[i] for i in sel_idx]

        # broadcast
        g_state = self.global_model.state_dict()
        for c in selected:
            c.set_weights(g_state)

        # local train
        weighted_states = []
        train_logs = []
        for c in selected:
            loss = c.train_one_round()
            weighted_states.append((c.get_weights(), c.num_train))
            train_logs.append({"client_id": c.client_id, "loss": float(loss), "n": c.num_train})

        # aggregate
        new_state = self._aggregate(weighted_states)
        self.global_model.load_state_dict(new_state, strict=True)

        # evaluate on ALL clients (test)
        evals = []
        for c in self.clients:
            c.set_weights(new_state)
            ev = c.evaluate()
            evals.append(ev)

        avg_pr = float(np.mean([e["pr_auc"] for e in evals]))
        avg_roc = float(np.mean([e["roc_auc"] for e in evals]))
        avg_f1 = float(np.mean([e["macro_f1"] for e in evals]))

        # fairness stats
        pr_by_client = {e["client_id"]: e["pr_auc"] for e in evals}
        gap = max(pr_by_client.values()) - min(pr_by_client.values())
        worst_id, worst_pr = min(pr_by_client.items(), key=lambda kv: kv[1])

        out = {
            "round": rnd,
            "avg_pr_auc": avg_pr,
            "avg_roc_auc": avg_roc,
            "avg_macro_f1": avg_f1,
            "worst_client": {"client_id": worst_id, "pr_auc": worst_pr},
            "gap": gap,
            "train_logs": train_logs,
            "eval_metrics": evals,
        }
        self.history.append(out)
        print(f"[Round {rnd:03d}] PR-AUC(avg)={avg_pr:.4f} | ROC-AUC(avg)={avg_roc:.4f} | Macro-F1(avg)={avg_f1:.4f} | Gap={gap:.4f} | Worst={worst_id}:{worst_pr:.4f}")
        return out


# ------------------------------
# Build shared preprocessor (fit once on union of all TRAIN)
# ------------------------------
def build_shared_preprocessor(all_clients: Dict[str, Dict[str, pd.DataFrame]]) -> ColumnTransformer:
    feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS

    # gather union of train features to fit vocabularies
    frames = []
    for cid in sorted(all_clients.keys()):
        df = all_clients[cid]["train"]
        # ensure missing feature columns exist
        for col in feats:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[feats])
    union_train = pd.concat(frames, ignore_index=True)

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    multi = MultiHotEncoder()

    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", onehot, CATEGORICAL_SINGLE),
            ("mlb", multi, MULTI_LABEL_COLS),
        ],
        sparse_threshold=1.0,  # stay sparse
    )
    preproc.fit(union_train)
    return preproc

def build_shared_preprocessor(all_clients: Dict[str, Dict[str, pd.DataFrame]]) -> ColumnTransformer:
    """Fit a single ColumnTransformer on the UNION of all clients' TRAIN features."""
    feats = NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS

    # Collect union-of-train features
    frames = []
    for cid in sorted(all_clients.keys()):
        df = all_clients[cid]["train"].copy()
        # ensure all expected feature columns exist
        for col in feats:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[feats])

    union_train = pd.concat(frames, ignore_index=True)

    # OneHotEncoder compatibility across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", ohe, CATEGORICAL_SINGLE),
            ("mlb", MultiHotEncoder(), MULTI_LABEL_COLS),
        ],
        sparse_threshold=1.0,  # keep sparse
    )
    preproc.fit(union_train)
    return preproc


# ------------------------------
# Orchestration
# ------------------------------
def main(rounds: int = 200, fraction: float = 0.5, out_dir: Path | None = None):
    global ALL_CLIENTS
    ALL_CLIENTS = load_federated_clients()
    summarize_clients(ALL_CLIENTS)

    preproc = build_shared_preprocessor(ALL_CLIENTS)

    client_ids = sorted(ALL_CLIENTS.keys())
    clients = [FederatedClient(cid, preproc) for cid in client_ids]

    server = FedAvgServer(clients)

    for r in range(1, rounds + 1):
        server.round(r, fraction=fraction)

    # Save final results (last round)
    final = server.history[-1]
    if out_dir is None:
        out_dir = Path("experiments") / "rq1_alpha_0.1" / f"fedavg_p{fraction}" / f"seed{SEED}"
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
    # You can tweak these quickly when calling as a module
    # Example: python -m src.research.RQ1.algorithms.fedavg.run
    main(rounds=200, fraction=0.5)
