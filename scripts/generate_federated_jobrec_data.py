#!/usr/bin/env python3
"""
Federated Job-Recommendation Data Generator (RQ1 & RQ2 Ready)
-----------------------------------------------------------------
This script generates synthetic, *heterogeneous* federated datasets for
job–student matching across 3 org types (Universities, Companies, Platforms).
It supports:
  • Label skew via Dirichlet α (e.g., 0.1 / 0.3 / 1.0)
  • Feature skew via client-specific skill subsets & distributions
  • Quantity skew via different per-client sample sizes
  • Temporal drift via non-overlapping date windows per org type
  • Semantic mismatch via org-type-specific skill aliases with configurable overlap

Outputs
  data/processed/client_*/data.csv
  metadata/schema.yaml
  metadata/skill_universe.csv
  metadata/skill_aliases.csv
  metadata/client_configs.json
  splits/train_test_splits.json

Usage
  - Adjust CONFIG at the bottom if needed.
  - Run: python generate_federated_jobrec_data.py

Notes
  - Keep this script in your repo root (or adjust OUTPUT_BASE).
  - The generated dataset volume defaults to ~20–30k rows across 20 clients.
  - No external dependencies beyond Python stdlib + numpy + pandas.
"""
from __future__ import annotations
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

# ------------------------------
# Utilities
# ------------------------------
RNG = np.random.default_rng(42)
random.seed(42)

OUTPUT_BASE = "data"
METADATA_DIR = os.path.join("metadata")
SPLITS_DIR = os.path.join("splits")
REPORTS_DIR = os.path.join("reports")

os.makedirs(os.path.join(OUTPUT_BASE, "processed"), exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ------------------------------
# Config dataclasses
# ------------------------------
@dataclass
class ClientSpec:
    client_id: str
    org_type: str  # 'U', 'C', or 'P'
    n_rows: int
    label_rate_hint: float  # rough target used to derive threshold
    date_start: str  # YYYY-MM-DD
    date_end: str    # YYYY-MM-DD
    skill_families_focus: List[str]

@dataclass
class GlobalConfig:
    n_universities: int
    n_companies: int
    n_platforms: int
    total_skill_count: int
    skill_families: Dict[str, int]
    overlap_levels: List[int]  # e.g., [80, 50, 20]
    dirichlet_alpha: float
    train_val_test: Tuple[float, float, float]

# ------------------------------
# Skill universe & aliases
# ------------------------------
FAMILIES = {
    "Web": 12,
    "Data": 12,
    "Cloud": 12,
    "ML": 12,
    "DevOps": 10,
    "Security": 10,
    "Mobile": 10,
    "Design": 10,
    # NEW business families
    "Finance": 12,
    "Marketing": 10,
    "Management": 10,
    "HR": 10,
}  # totals to ~120

FAMILY_PREFIX = {
    "Web": "WEB",
    "Data": "DATA",
    "Cloud": "CLOUD",
    "ML": "ML",
    "DevOps": "DEV",
    "Security": "SEC",
    "Mobile": "MOB",
    "Design": "DES",
    # NEW business families
    "Finance": "FIN",
    "Marketing": "MKT",
    "Management": "MGT",
    "HR": "HR",
}

ROLE_FAMILIES = [
    # Tech
    "Software Engineering", "Data Engineering", "Data Science", "ML Engineering",
    "Cloud/Infra", "DevOps/SRE", "Security", "Mobile",
    "Frontend", "Backend", "Full-Stack", "Design/UI-UX",
    # Business (NEW)
    "Business Analyst", "Product Manager", "Marketing Specialist",
    "Digital Marketing", "Finance Analyst", "Investment Analyst",
    "HR Manager", "Talent Acquisition", "Operations Manager",
    "Sales/BD", "Strategy/Consulting"
]

SALARY_BANDS = ["20-30k", "30-50k", "50-80k", "80-120k", "120k+"]
COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
LOCATIONS = ["Remote", "Hybrid", "Onsite-A", "Onsite-B", "Onsite-C"]
MAJORS = ["CS", "IT", "SE", "EE", "Math", "Business", "Design", "Management", "Finance", "Marketing", "HR", "Economics", "Other"]
COURSE_POOL = [
    "DSA", "DBMS", "OS", "Networks", "AI", "ML", "DL", "NLP", "WebDev",
    "Cloud", "Security", "Mobile", "Human-Computer Interaction",
    "Accounting", "Corporate Finance", "Marketing", "Digital Marketing",
    "Business Analytics", "Operations Management", "HRM", "Strategy"
]

# ------------------------------
# Helper functions for skills
# ------------------------------
def build_skill_universe(families: Dict[str, int]) -> List[Tuple[str, str]]:
    skills = []
    for fam, k in families.items():
        for i in range(1, k + 1):
            skills.append((f"{FAMILY_PREFIX[fam]}_{i:02d}", fam))
    return skills

UNIVERSE = build_skill_universe(FAMILIES)
SKILL_NAMES = [s for s, _ in UNIVERSE]


def make_alias(term: str, org_type: str) -> str:
    # Simple aliasing scheme for demonstration; real mapping can be richer.
    # Ensure deterministic but distinct aliases per org-type.
    if org_type == 'U':
        return term  # universities use canonical
    if org_type == 'C':
        return term.replace("_", "-") + "-req"
    if org_type == 'P':
        return term.lower()
    return term


# ------------------------------
# Client construction
# ------------------------------

def date_range(start: str, end: str) -> Tuple[datetime, datetime]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    return s, e


def random_date_between(s: datetime, e: datetime) -> datetime:
    delta = (e - s).days
    return s + timedelta(days=int(RNG.integers(0, max(1, delta + 1))))


def choose_weighted(items: List[str], weights: List[float], k: int) -> List[str]:
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    idx = RNG.choice(len(items), size=k, replace=False, p=probs)
    return [items[i] for i in idx]


# ------------------------------
# Generation logic
# ------------------------------

def assign_client_skill_subset(client: ClientSpec, overlap_pct: int) -> Set[str]:
    """Return a subset of skills biased to the client's focused families.
    overlap_pct is used downstream for alias evaluation; here we just pick a subset size.
    """
    fam_bias = {fam: (2.0 if fam in client.skill_families_focus else 1.0) for fam in FAMILIES}
    # Build weighted pool
    pool = []
    weights = []
    for s, fam in UNIVERSE:
        pool.append(s)
        weights.append(fam_bias[fam])
    # Client subset size ~ 50–80 skills
    subset_size = int(RNG.integers(50, 81))
    subset = set(choose_weighted(pool, weights, subset_size))
    return subset


def sample_student_profile(client: ClientSpec, client_skills: Set[str]) -> dict:
    major = RNG.choice(MAJORS, 1)[0]
    gpa = float(np.clip(RNG.normal(3.0, 0.4), 1.5, 4.0))
    courses = RNG.choice(COURSE_POOL, size=int(RNG.integers(3, 7)), replace=False).tolist()
    projects = int(np.clip(int(RNG.normal(2.0, 1.2)), 0, 8))
    internships = int(np.clip(int(RNG.normal(1.0, 1.0)), 0, 5))
    grad_year = int(RNG.integers(2023, 2027))
    location = RNG.choice(LOCATIONS, 1)[0]
    # Draw student skills from client subset
    skills = RNG.choice(list(client_skills), size=int(RNG.integers(5, 13)), replace=False).tolist()
    return {
        "major": major,
        "gpa": gpa,
        "courses": courses,
        "projects_count": projects,
        "internships_count": internships,
        "grad_year": grad_year,
        "location": location,
        "skills": [make_alias(s, client.org_type) for s in skills],
    }


def sample_job_profile(client: ClientSpec, client_skills: Set[str]) -> dict:
    title = RNG.choice(ROLE_FAMILIES, 1)[0]
    role_family = title
    req = RNG.choice(list(client_skills), size=int(RNG.integers(5, 11)), replace=False).tolist()
    n2h = RNG.choice(list(client_skills), size=int(RNG.integers(2, 6)), replace=False).tolist()
    salary_band = RNG.choice(SALARY_BANDS, 1)[0]
    location = RNG.choice(LOCATIONS, 1)[0]
    company_size = RNG.choice(COMPANY_SIZES, 1)[0]
    return {
        "title": title,
        "role_family": role_family,
        "required_skills": [make_alias(s, client.org_type) for s in req],
        "nice_to_have": [make_alias(s, client.org_type) for s in n2h],
        "salary_band": salary_band,
        "location": location,
        "company_size": company_size,
    }


def sample_platform_interactions(client: ClientSpec) -> dict:
    impressions = int(np.clip(int(RNG.normal(12, 6)), 1, 50))
    clicks = int(np.clip(int(RNG.normal(2, 2)), 0, impressions))
    saves = int(np.clip(int(RNG.normal(1, 1)), 0, min(clicks, 10)))
    apply = int(RNG.integers(0, 2) if clicks > 0 else 0)
    dwell = float(np.clip(np.abs(RNG.normal(30, 15)), 5, 180))
    revisit = int(np.clip(int(RNG.normal(1, 1)), 0, 6))
    return {
        "impressions": impressions,
        "clicks": clicks,
        "saves": saves,
        "apply": apply,
        "dwell_time": dwell,
        "revisit_count": revisit,
    }


def compatibility_score(student: dict, job: dict, inter: dict, org_type: str) -> float:
    # Skill overlap (aliases already applied per org type)
    sset = set(student.get("skills", []))
    rset = set(job.get("required_skills", []))
    n2h = set(job.get("nice_to_have", []))
    common_req = len(sset & rset)
    common_n2h = len(sset & n2h)
    jacc = (len(sset & (rset | n2h)) / max(1, len(sset | (rset | n2h))))

    # Academic fit (U-weighted)
    gpa = student.get("gpa", 3.0)
    acad = (gpa - 2.0) / 2.0  # ~ [0,1]
    acad = float(np.clip(acad, 0, 1))

    # Role/location match
    loc_match = 1.0 if student.get("location") == job.get("location") else 0.5 if job.get("location") in ("Remote", "Hybrid") else 0.0

    # Interaction signals (P-weighted)
    clicks = inter.get("clicks", 0)
    saves = inter.get("saves", 0)
    apply = inter.get("apply", 0)
    dwell = inter.get("dwell_time", 0.0) / 180.0  # normalize
    intent = np.tanh(0.2 * clicks + 0.4 * saves + 1.0 * apply + 0.5 * dwell)

    # Org-type weighting
    if org_type == 'U':
        w_req, w_n2h, w_acad, w_loc, w_intent = 0.35, 0.1, 0.30, 0.15, 0.10
    elif org_type == 'C':
        w_req, w_n2h, w_acad, w_loc, w_intent = 0.45, 0.15, 0.10, 0.20, 0.10
    else:  # 'P'
        w_req, w_n2h, w_acad, w_loc, w_intent = 0.30, 0.15, 0.10, 0.15, 0.30

    # Optional: slightly reweight for business-type roles (MBA-style)
    business_roles = {
    "Business Analyst", "Product Manager", "Marketing Specialist",
    "Digital Marketing", "Finance Analyst", "Investment Analyst",
    "HR Manager", "Talent Acquisition", "Operations Manager",
    "Sales/BD", "Strategy/Consulting"
    }
    if job.get("role_family") in business_roles:
        # tilt toward location & user intent; reduce req-skill dominance a bit
        w_req = max(0.25, w_req - 0.10)
        w_loc = min(0.30, w_loc + 0.05)
        w_intent = min(0.40, w_intent + 0.05)

    raw = (
        w_req * np.tanh(common_req / 5.0) +
        w_n2h * np.tanh(common_n2h / 5.0) +
        0.10 * jacc +
        w_acad * acad +
        w_loc * loc_match +
        w_intent * float(intent)
    )
    
    # Small org-type-specific noise for realism
    noise = {
        'U': RNG.normal(0, 0.03),
        'C': RNG.normal(0, 0.04),
        'P': RNG.normal(0, 0.05),
    }[org_type]

    score = float(np.clip(raw + noise, 0.0, 1.0))
    return score


def derive_threshold_for_label_rate(scores: np.ndarray, target_rate: float) -> float:
    # Choose τ so that roughly target_rate of samples have score >= τ
    if len(scores) == 0:
        return 0.9
    q = np.clip(1.0 - target_rate, 0.0, 1.0)
    return float(np.quantile(scores, q))


# ------------------------------
# Main generation per client
# ------------------------------

def generate_client_dataframe(client: ClientSpec, overlap_pct: int) -> pd.DataFrame:
    # Build client-specific skill subset
    client_skills = assign_client_skill_subset(client, overlap_pct)
    ds, de = date_range(client.date_start, client.date_end)

    rows = []
    # Two-phase pass: first compute scores, then set threshold for desired positive rate
    tmp_scores = []
    tmp_rows = []

    for i in range(client.n_rows):
        student = sample_student_profile(client, client_skills)
        job = sample_job_profile(client, client_skills)
        inter = sample_platform_interactions(client)
        when = random_date_between(ds, de)
        score = compatibility_score(student, job, inter, client.org_type)

        row = {
            "client_id": client.client_id,
            "org_type": client.org_type,
            "posted_date": when.strftime("%Y-%m-%d"),
            # student
            "major": student["major"],
            "gpa": student["gpa"],
            "courses": ",".join(student["courses"]),
            "projects_count": student["projects_count"],
            "internships_count": student["internships_count"],
            "student_location": student["location"],
            "student_skills": ",".join(student["skills"]),
            # job
            "title": job["title"],
            "role_family": job["role_family"],
            "required_skills": ",".join(job["required_skills"]),
            "nice_to_have": ",".join(job["nice_to_have"]),
            "salary_band": job["salary_band"],
            "job_location": job["location"],
            "company_size": job["company_size"],
            # platform interactions
            "impressions": inter["impressions"],
            "clicks": inter["clicks"],
            "saves": inter["saves"],
            "apply": inter["apply"],
            "dwell_time": inter["dwell_time"],
            "revisit_count": inter["revisit_count"],
            # targets
            "match_score": score,
        }
        tmp_rows.append(row)
        tmp_scores.append(score)

    scores_arr = np.array(tmp_scores)
    tau = derive_threshold_for_label_rate(scores_arr, client.label_rate_hint)

    for r, s in zip(tmp_rows, tmp_scores):
        r["recommended"] = int(s >= tau)
        rows.append(r)

    df = pd.DataFrame(rows)
    return df


# ------------------------------
# Splitter
# ------------------------------

def split_train_val_test(df: pd.DataFrame, seed: int, stratify_on: str | None = None,
                         ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = np.arange(n)

    if stratify_on is None:
        rng.shuffle(idx)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        return {
            "train": idx[:n_train].tolist(),
            "val": idx[n_train:n_train+n_val].tolist(),
            "test": idx[n_train+n_val:].tolist(),
        }
    else:
        # simple stratified split for binary column
        g0 = idx[df[stratify_on].values == 0]
        g1 = idx[df[stratify_on].values == 1]
        rng.shuffle(g0)
        rng.shuffle(g1)
        def take(arr, r):
            k = int(len(arr) * r)
            return arr[:k], arr[k:]
        t0, rem0 = take(g0, ratios[0])
        v0, e0 = take(rem0, ratios[1] / (1 - ratios[0]))
        t1, rem1 = take(g1, ratios[0])
        v1, e1 = take(rem1, ratios[1] / (1 - ratios[0]))
        train = np.concatenate([t0, t1])
        val = np.concatenate([v0, v1])
        test = np.concatenate([e0, e1])
        rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
        return {
            "train": train.tolist(),
            "val": val.tolist(),
            "test": test.tolist(),
        }


# ------------------------------
# Top-level orchestration
# ------------------------------

def build_clients(gc: GlobalConfig) -> List[ClientSpec]:
    clients: List[ClientSpec] = []

    # Date windows for drift
    # U: Jan–Mar, C: Apr–Jul, P: Aug–Oct of 2025
    windows = {
        'U': ("2025-01-01", "2025-03-31"),
        'C': ("2025-04-01", "2025-07-31"),
        'P': ("2025-08-01", "2025-10-31"),
    }

    # Family focuses per org-type
    focus_sets = {
    'U': ["Data", "ML", "Web", "Finance", "Marketing", "Management"],
    'C': ["Cloud", "DevOps", "Security", "Management", "Finance", "Marketing", "HR"],
    'P': ["Web", "Mobile", "Design", "Marketing", "Finance"]
    }

    # Quantity skew: choose from ranges
    def draw_n(org_type: str) -> int:
        if org_type == 'U':
            return int(RNG.integers(300, 1201))
        if org_type == 'C':
            return int(RNG.integers(800, 2501))
        return int(RNG.integers(500, 1201))  # P

    # Label rate hints sampled from Dirichlet across all clients
    n_clients = gc.n_universities + gc.n_companies + gc.n_platforms
    # Draw base proportions, then map into a 1%–15% range
    proportions = RNG.dirichlet(alpha=np.ones(n_clients) * gc.dirichlet_alpha)
    rates = 0.01 + proportions * 0.14  # in [0.01, 0.15]

    # Build
    i = 0
    for k in range(gc.n_universities):
        n_rows = draw_n('U')
        start, end = windows['U']
        clients.append(ClientSpec(
            client_id=f"client_U{k+1}", org_type='U', n_rows=n_rows,
            label_rate_hint=float(rates[i]), date_start=start, date_end=end,
            skill_families_focus=focus_sets['U']
        )); i += 1
    for k in range(gc.n_companies):
        n_rows = draw_n('C')
        start, end = windows['C']
        clients.append(ClientSpec(
            client_id=f"client_C{k+1}", org_type='C', n_rows=n_rows,
            label_rate_hint=float(rates[i]), date_start=start, date_end=end,
            skill_families_focus=focus_sets['C']
        )); i += 1
    for k in range(gc.n_platforms):
        n_rows = draw_n('P')
        start, end = windows['P']
        clients.append(ClientSpec(
            client_id=f"client_P{k+1}", org_type='P', n_rows=n_rows,
            label_rate_hint=float(rates[i]), date_start=start, date_end=end,
            skill_families_focus=focus_sets['P']
        )); i += 1

    return clients


def write_metadata(skill_universe: List[Tuple[str, str]], overlap_levels: List[int], clients: List[ClientSpec]):
    # schema.yaml (high-level)
    schema = {
        "common": {
            "client_id": "string",
            "org_type": "{U,C,P}",
            "posted_date": "YYYY-MM-DD",
            "match_score": "float[0,1]",
            "recommended": "{0,1}",
        },
        "student": {
            "major": f"oneof({','.join(MAJORS)})",
            "gpa": "float[1.5,4.0]",
            "courses": "comma-separated list",
            "projects_count": "int[0,8]",
            "internships_count": "int[0,5]",
            "student_location": f"oneof({','.join(LOCATIONS)})",
            "student_skills": "comma-separated list (org-type aliases)",
        },
        "job": {
            "title": f"oneof({','.join(ROLE_FAMILIES)})",
            "role_family": "string",
            "required_skills": "comma-separated list (org-type aliases)",
            "nice_to_have": "comma-separated list (org-type aliases)",
            "salary_band": f"oneof({','.join(SALARY_BANDS)})",
            "job_location": f"oneof({','.join(LOCATIONS)})",
            "company_size": f"oneof({','.join(COMPANY_SIZES)})",
        },
        "platform": {
            "impressions": "int[1,50]",
            "clicks": "int[0,50]",
            "saves": "int[0,10]",
            "apply": "{0,1}",
            "dwell_time": "seconds[5,180]",
            "revisit_count": "int[0,6]",
        }
    }
    with open(os.path.join(METADATA_DIR, "schema.yaml"), "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(schema, f, sort_keys=False)

    # Universe
    uni_df = pd.DataFrame(skill_universe, columns=["skill", "family"])  # canonical names
    uni_df.to_csv(os.path.join(METADATA_DIR, "skill_universe.csv"), index=False)

    # Aliases (org-type specific) at multiple overlap settings (documented, not enforced here)
    alias_rows = []
    for s, fam in skill_universe:
        alias_rows.append({"skill": s, "org_type": "U", "alias": make_alias(s, 'U')})
        alias_rows.append({"skill": s, "org_type": "C", "alias": make_alias(s, 'C')})
        alias_rows.append({"skill": s, "org_type": "P", "alias": make_alias(s, 'P')})
    pd.DataFrame(alias_rows).to_csv(os.path.join(METADATA_DIR, "skill_aliases.csv"), index=False)

    # Client configs
    with open(os.path.join(METADATA_DIR, "client_configs.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in clients], f, indent=2)

    # Document overlap levels
    with open(os.path.join(METADATA_DIR, "overlap_levels.json"), "w", encoding="utf-8") as f:
        json.dump({"overlap_levels": overlap_levels}, f, indent=2)


def generate_all(gc: GlobalConfig, seed: int = 42):
    rng = np.random.default_rng(seed)
    clients = build_clients(gc)
    write_metadata(UNIVERSE, gc.overlap_levels, clients)

    # Assign an overlap level per *org-type pair* experiment; here we just use the first for generation label purposes.
    overlap_for_generation = gc.overlap_levels[0]  # e.g., start with 80; you can re-run with 50/20

    # Generate per-client dataframes and write to disk
    split_index = {}
    for c in clients:
        df = generate_client_dataframe(c, overlap_for_generation)
        client_dir = os.path.join(OUTPUT_BASE, "processed", c.client_id)
        os.makedirs(client_dir, exist_ok=True)
        out_csv = os.path.join(client_dir, "data.csv")
        df.to_csv(out_csv, index=False)

        # Build splits (stratify if recommended is not too sparse)
        strat = "recommended" if df["recommended"].sum() >= 10 else None
        idx = split_train_val_test(df, seed=seed, stratify_on=strat, ratios=gc.train_val_test)
        split_index[c.client_id] = idx

    with open(os.path.join(SPLITS_DIR, "train_test_splits.json"), "w", encoding="utf-8") as f:
        json.dump(split_index, f, indent=2)

    # Quick sanity summary CSV for reports/
    summary_rows = []
    for c in clients:
        client_dir = os.path.join(OUTPUT_BASE, "processed", c.client_id)
        df = pd.read_csv(os.path.join(client_dir, "data.csv"))
        pos_rate = float(df["recommended"].mean()) if len(df) > 0 else 0.0
        summary_rows.append({
            "client_id": c.client_id,
            "org_type": c.org_type,
            "n_rows": len(df),
            "pos_rate": round(pos_rate, 4),
            "date_min": df["posted_date"].min() if len(df) else None,
            "date_max": df["posted_date"].max() if len(df) else None,
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(REPORTS_DIR, "sanity_client_summary.csv"), index=False)


# ------------------------------
# Default CONFIG (edit as needed)
# ------------------------------
CONFIG = GlobalConfig(
    n_universities=8,
    n_companies=8,
    n_platforms=4,
    total_skill_count=sum(FAMILIES.values()),
    skill_families=FAMILIES,
    overlap_levels=[80, 50, 20],
    dirichlet_alpha=0.1,          # severe label skew by default
    train_val_test=(0.70, 0.15, 0.15),
)


if __name__ == "__main__":
    generate_all(CONFIG, seed=42)
    print("\n✅ Data generation complete.")
    print("   - metadata/: schema + skills + aliases + client configs + overlap levels")
    print("   - data/processed/client_*/data.csv")
    print("   - splits/train_test_splits.json")
    print("   - reports/sanity_client_summary.csv\n")
