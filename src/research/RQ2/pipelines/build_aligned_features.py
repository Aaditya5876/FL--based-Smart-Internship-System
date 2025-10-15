import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ..encoders.bi_encoder import load_encoder, encode_phrases
from ..ontology.canonical_vocab import load_alias_map, load_canonical_vocab
from ..interfaces.normalizer import Normalizer


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _avg_embed(phrases: List[str], enc, emb_dim: int = 64) -> List[float]:
    if not phrases:
        return [0.0] * emb_dim
    E = encode_phrases(enc, phrases)
    v = E.mean(axis=0)
    n = np.linalg.norm(v) + 1e-8
    return (v / n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--data_root", type=str, default="data/raw")
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    out_dir = os.path.join("data", "processed", f"{args.version}")
    _ensure_dir(out_dir)

    # load ontology + encoder
    alias_map = load_alias_map(os.path.join(out_dir, "alias_map.json"))
    vocab = load_canonical_vocab(os.path.join(out_dir, "canonical_vocab.json"))
    # Auto-detect classifier; embeddings only are needed for features
    enc = load_encoder(os.path.join(out_dir, "phrase_encoder.pt"), emb_dim=args.emb_dim)
    norm = Normalizer(alias_map, vocab)

    # load raw
    s_csv = os.path.join(args.data_root, "students.csv")
    j_csv = os.path.join(args.data_root, "jobs.csv")
    if not (os.path.exists(s_csv) and os.path.exists(j_csv)):
        raise FileNotFoundError("students.csv or jobs.csv not found under data_root")
    df_s = pd.read_csv(s_csv)
    df_j = pd.read_csv(j_csv)

    # normalize students
    users_rows = []
    for _, row in df_s.iterrows():
        uid = row.get("user_id")
        skills = str(row.get("skills", ""))
        ns = norm.normalize_skills(skills)
        phrases = [p for p in (skills.split(",") if isinstance(skills, str) else []) if p.strip()]
        emb = _avg_embed(phrases, enc, emb_dim=args.emb_dim)
        users_rows.append({
            "entity_type": "student",
            "entity_id": uid,
            "canonical_ids": ns["ids"],
            "embedding": emb,
        })

    # normalize jobs
    jobs_rows = []
    for _, row in df_j.iterrows():
        jid = row.get("job_id")
        title = str(row.get("title", ""))
        req = str(row.get("skills_required", ""))
        nt = norm.normalize_title(title)
        ns = norm.normalize_skills(req)
        phrases = []
        if isinstance(title, str) and title.strip():
            phrases.append(title)
        phrases.extend([p.strip() for p in req.split(",") if isinstance(req, str) and p.strip()])
        emb = _avg_embed(phrases, enc, emb_dim=args.emb_dim)
        jobs_rows.append({
            "entity_type": "job",
            "entity_id": jid,
            "canonical_ids": list(dict.fromkeys((nt["ids"] + ns["ids"]))),
            "embedding": emb,
        })

    df_users = pd.DataFrame(users_rows)
    df_jobs = pd.DataFrame(jobs_rows)

    # concat with a type column for downstream routing
    df_all = pd.concat([df_users, df_jobs], ignore_index=True)

    # write parquet (fallback to csv if pyarrow unavailable)
    feats_path = os.path.join(out_dir, "features_aligned.parquet")
    wrote_parquet = False
    try:
        df_all.to_parquet(feats_path, index=False)
        wrote_parquet = True
    except Exception:
        feats_path = os.path.join(out_dir, "features_aligned.csv")
        df_all.to_csv(feats_path, index=False)

    # schema + manifest
    schema = {
        "entity_type": "{'student'|'job'}",
        "entity_id": "string",
        "canonical_ids": "list[string]",
        "embedding": f"list[float[{args.emb_dim}]]",
    }
    with open(os.path.join(out_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    manifest = {
        "version": args.version,
        "notes": args.notes,
        "data_root": args.data_root,
        "features_file": os.path.basename(feats_path),
        "format": "parquet" if wrote_parquet else "csv",
        "num_rows": int(len(df_all)),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Wrote aligned features to {feats_path}")


if __name__ == "__main__":
    main()
