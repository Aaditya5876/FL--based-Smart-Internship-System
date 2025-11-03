# src/loaders/federated_dataset.py
from __future__ import annotations
import json, os
from typing import Dict
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
DEFAULT_SPLITS_PATH = os.path.join(REPO_ROOT, "splits", "train_test_splits.json")

def _read_client_dataframe(client_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(client_dir, "data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected data.csv not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    expected = {"client_id", "org_type", "match_score", "recommended"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df

def load_federated_clients(
    data_dir: str = DEFAULT_DATA_DIR,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        split_index = json.load(f)

    client_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name in sorted(os.listdir(data_dir)):
        client_dir = os.path.join(data_dir, name)
        if not (os.path.isdir(client_dir) and name.startswith("client_")):
            continue
        df = _read_client_dataframe(client_dir)
        cid = name
        if cid not in split_index:
            raise KeyError(f"Client '{cid}' missing in {splits_path}.")
        idx = split_index[cid]
        n = len(df)
        for split_name in ("train", "val", "test"):
            bad = [i for i in idx[split_name] if not (0 <= i < n)]
            if bad:
                raise IndexError(f"Out-of-range indices for {cid}:{split_name}: {bad[:5]} (len={n})")
        client_splits[cid] = {
            "train": df.iloc[idx["train"]].reset_index(drop=True),
            "val":   df.iloc[idx["val"]  ].reset_index(drop=True),
            "test":  df.iloc[idx["test"] ].reset_index(drop=True),
        }
    if not client_splits:
        raise RuntimeError(f"No client_* folders found in {data_dir}. Did you run the generator?")
    return client_splits

def summarize_clients(client_splits: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    rows = []
    for cid, parts in client_splits.items():
        full = pd.concat([parts["train"], parts["val"], parts["test"]], ignore_index=True)
        pos_rate = full["recommended"].mean() if len(full) else float("nan")
        rows.append({
            "client": cid,
            "org": full.get("org_type", pd.Series(["?"])).iloc[0] if len(full) else "?",
            "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(parts["test"]),
            "pos_rate": round(float(pos_rate), 4) if pd.notna(pos_rate) else None,
        })
    df = pd.DataFrame(rows).sort_values(["org", "client"]).reset_index(drop=True)
    print("\nLoaded federated clients (split sizes & positive rate):")
    print(df.to_string(index=False) if not df.empty else "(no clients)")

if __name__ == "__main__":
    ALL = load_federated_clients()
    summarize_clients(ALL)
    
