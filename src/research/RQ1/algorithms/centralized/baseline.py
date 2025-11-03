# Centralized Baseline for Smart Internship System (RQ1 upper bound)
# Trains on the union of ALL clients' train data; evaluates on the union of ALL clients' test data
# Uses the same splits as FL via src/loaders/federated_dataset.py

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Reuse the loader we added earlier
from src.loaders.federated_dataset import load_federated_clients, summarize_clients

# sklearn bits
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pyplot as plt
from scipy import sparse


# ----------------------------
# Config
# ----------------------------
SEED = 42
np.random.seed(SEED)

# Feature groups (must exist in data.csv)
NUMERIC_COLS = [
    "gpa", "projects_count", "internships_count",
    "impressions", "clicks", "saves", "apply", "dwell_time", "revisit_count"
]
CATEGORICAL_SINGLE = [
    "major", "student_location",
    "title", "role_family", "salary_band", "job_location", "company_size",
    "org_type"
]
MULTI_LABEL_COLS = [
    "courses", "student_skills", "required_skills", "nice_to_have"
]

TARGET_CLASS = "recommended"   # primary for RQ1
TARGET_REG   = "match_score"   # optional


# ----------------------------
# Multi-hot encoder for list-like columns (comma-separated strings)
# ----------------------------
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=","):
        self.sep = sep
        self.vocabs_ = {}         # per-column vocabulary list (order fixed)
        self.col_index_ = {}      # start index per column in the stacked matrix
        self.total_dim_ = 0

    def fit(self, X, y=None):
        # X is a DataFrame with the multi-label columns only
        start = 0
        self.vocabs_.clear()
        self.col_index_.clear()
        for col in X.columns:
            tokens = set()
            series = X[col].fillna("").astype(str)
            for s in series:
                if not s:
                    continue
                toks = [t.strip() for t in s.split(self.sep) if t.strip()]
                tokens.update(toks)
            vocab = sorted(tokens)
            self.vocabs_[col] = vocab
            self.col_index_[col] = start
            start += len(vocab)
        self.total_dim_ = start
        return self

    def transform(self, X):
        # Build a sparse CSR matrix [n_samples, total_dim_]
        n = len(X)
        data, row_ind, col_ind = [], [], []
        for row_idx, (_, row) in enumerate(X.iterrows()):
            for col in X.columns:
                s = "" if pd.isna(row[col]) else str(row[col])
                if not s:
                    continue
                vocab = self.vocabs_.get(col, [])
                base = self.col_index_.get(col, 0)
                for tok in [t.strip() for t in s.split(self.sep) if t.strip()]:
                    try:
                        j = vocab.index(tok)
                    except ValueError:
                        continue  # unseen token → ignore
                    data.append(1)
                    row_ind.append(row_idx)
                    col_ind.append(base + j)
        if self.total_dim_ == 0:
            return sparse.csr_matrix((n, 0), dtype=np.float32)
        mat = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, self.total_dim_), dtype=np.float32)
        return mat


# ----------------------------
# Load ALL clients with fixed splits, then union them
# ----------------------------
clients = load_federated_clients()
summarize_clients(clients)

def union_split(clients_dict, split_name):
    frames = []
    for cid, parts in clients_dict.items():
        df = parts[split_name].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

train_df = union_split(clients, "train")
val_df   = union_split(clients, "val")
test_df  = union_split(clients, "test")

# Optionally merge train+val for final training; keep val separate for hyperparam tuning if needed
train_full_df = pd.concat([train_df, val_df], ignore_index=True)

# ----------------------------
# Build feature matrices and targets
# ----------------------------
for col in NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS + [TARGET_CLASS, TARGET_REG]:
    if col not in train_full_df.columns:
        print(f"[WARN] Column '{col}' missing in data; creating empty if feature, skipping if target.")
        if col in NUMERIC_COLS or col in CATEGORICAL_SINGLE or col in MULTI_LABEL_COLS:
            for df in (train_full_df, test_df):
                df[col] = np.nan

X_train = train_full_df[NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS].copy()
X_test  = test_df[NUMERIC_COLS + CATEGORICAL_SINGLE + MULTI_LABEL_COLS].copy()

y_train_cls = train_full_df[TARGET_CLASS].astype(int).values
y_test_cls  = test_df[TARGET_CLASS].astype(int).values

y_train_reg = train_full_df[TARGET_REG].astype(float).values
y_test_reg  = test_df[TARGET_REG].astype(float).values

# ----------------------------
# ColumnTransformer: numeric passthrough, single-cat OneHot, multi-label MultiHot
# ----------------------------
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

preproc = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_COLS),
        ("cat", onehot, CATEGORICAL_SINGLE),
        ("mlb", MultiHotEncoder(), MULTI_LABEL_COLS),
    ],
    sparse_threshold=1.0  # keep sparse to save memory
)

# ----------------------------
# 1) Classification baseline on 'recommended' (primary)
# ----------------------------
clf = Pipeline([
    ("prep", preproc),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
        class_weight=None  # you can try 'balanced' later if needed
    ))
])

print("\n[Centralized-CLS] Training classifier on 'recommended' ...")
clf.fit(X_train, y_train_cls)

# Predict probabilities for PR-AUC/ROC-AUC
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

pr_auc  = average_precision_score(y_test_cls, y_prob)
roc_auc = roc_auc_score(y_test_cls, y_prob)
f1_mac  = f1_score(y_test_cls, y_pred, average="macro")

print("\n--- Centralized Classification Results ---")
print(f"PR-AUC        : {pr_auc:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")
print(f"Macro-F1 @0.5 : {f1_mac:.4f}")

# ----------------------------
# 2) Regression baseline on 'match_score' (optional)
# ----------------------------
reg = Pipeline([
    ("prep", preproc),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED
    ))
])

print("\n[Centralized-REG] Training regressor on 'match_score' ...")
reg.fit(X_train, y_train_reg)
y_pred_reg = reg.predict(X_test)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2  = r2_score(y_test_reg, y_pred_reg)

print("\n--- Centralized Regression Results (optional) ---")
print(f"MSE : {mse:.6f}")
print(f"R^2 : {r2:.4f}")

# ----------------------------
# Save artifacts
# ----------------------------
out_dir = Path("experiments") / "rq1_alpha_0.1" / "centralized" / f"seed{SEED}"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "results.json", "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "cls": {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "macro_f1": float(f1_mac)},
        "reg": {"mse": float(mse), "r2": float(r2)},
        "counts": {
            "train_rows": int(len(train_full_df)),
            "test_rows": int(len(test_df))
        }
    }, f, indent=2)

# Quick scatter for regression (optional viz)
plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.3)
lo, hi = min(y_test_reg.min(), y_pred_reg.min()), max(y_test_reg.max(), y_pred_reg.max())
plt.plot([lo, hi], [lo, hi], "r--")
plt.xlabel("Actual match_score")
plt.ylabel("Predicted match_score")
plt.title("Centralized baseline: Actual vs Predicted (regression)")
plt.tight_layout()
plt.savefig(out_dir / "centralized_regression_scatter.png", dpi=120)
print(f"Saved results to: {out_dir}\n")
