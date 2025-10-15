import argparse
import json
import os
from typing import Dict, List


def load_eval(version: str) -> Dict:
    base = os.path.join("data", "processed", version)
    p = os.path.join(base, "intrinsic_eval.json")
    if not os.path.exists(p):
        return {"version": version, "exists": False}
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {
        "version": version,
        "exists": True,
        "f1": d.get("synonym_f1", {}).get("f1", None),
        "precision": d.get("synonym_f1", {}).get("precision", None),
        "recall": d.get("synonym_f1", {}).get("recall", None),
        "auc": d.get("sim_auc", None),
        "purity": d.get("cluster", {}).get("purity", None),
        "nmi": d.get("cluster", {}).get("nmi", None),
        "num_phrases": d.get("num_phrases", None),
        "top_pairs_used": d.get("top_pairs_used", None),
        "gold_pairs": d.get("gold_pairs", None),
        "unique_labels": d.get("unique_labels", None),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows = [load_eval(v) for v in args.versions]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)
    print(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    main()
