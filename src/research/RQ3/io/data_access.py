import json
import os
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd

RNG_DEFAULT_SEED = 42


def load_rawframes(data_root: str = "data/raw") -> Dict[str, pd.DataFrame]:
	paths = {
		"students": os.path.join(data_root, "students.csv"),
		"jobs": os.path.join(data_root, "jobs.csv"),
		"companies": os.path.join(data_root, "companies.csv"),
		"interactions": os.path.join(data_root, "interactions.csv"),
	}
	dfs: Dict[str, pd.DataFrame] = {}
	for k, p in paths.items():
		if not os.path.exists(p):
			raise FileNotFoundError(f"Missing required CSV: {p}")
		dfs[f"df_{k}"] = pd.read_csv(p)
	return dfs


def build_client_partitions(df_students: pd.DataFrame, df_interactions: pd.DataFrame, max_clients: int = 6) -> Dict[str, Dict[str, pd.DataFrame]]:
	df_students = df_students.copy()
	if "university" not in df_students.columns:
		raise ValueError("students.csv must include 'university'")
	if "user_id" not in df_students.columns:
		raise ValueError("students.csv must include 'user_id'")
	top = df_students["university"].value_counts().nlargest(max_clients).index.tolist()
	df_students["client_id"] = np.where(df_students["university"].isin(top), df_students["university"], "OTHER")
	dfu = df_students[["user_id", "client_id"]]
	dfi = df_interactions.copy()
	if "user_id" not in dfi.columns:
		raise ValueError("interactions.csv must include 'user_id'")
	dfi = dfi.merge(dfu, on="user_id", how="left")
	dfi["client_id"] = dfi["client_id"].fillna("OTHER")
	client_map: Dict[str, Dict[str, pd.DataFrame]] = {}
	for cid, grp in dfi.groupby("client_id"):
		users = df_students[df_students["client_id"] == cid]
		client_map[cid] = {"users": users.reset_index(drop=True), "interactions": grp.reset_index(drop=True)}
	return client_map


def _one_hot(series: pd.Series, categories: List[Any]) -> pd.DataFrame:
	return pd.get_dummies(series.astype(pd.CategoricalDtype(categories=categories)), dummy_na=False)


def _safe_list(x) -> List[str]:
	if isinstance(x, list):
		return x
	if pd.isna(x):
		return []
	if isinstance(x, str):
		try:
			if x.startswith("[") and x.endswith("]"):
				return [s.strip().strip("'\"") for s in x.strip("[]").split(",") if s.strip() != ""]
			return [t.strip() for t in x.split(",") if t.strip()]
		except Exception:
			return []
	return []


def encode_features(df_students: pd.DataFrame, df_jobs: pd.DataFrame, df_companies: pd.DataFrame, df_interactions: pd.DataFrame, seed: int = RNG_DEFAULT_SEED) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	s = df_students.copy(); j = df_jobs.copy(); c = df_companies.copy(); x = df_interactions.copy()
	if "user_id" not in s.columns: raise ValueError("students must have user_id")
	if "GPA" not in s.columns: s["GPA"] = 3.0 + 0.5 * rng.standard_normal(len(s))
	if "major" not in s.columns: s["major"] = "General"
	if "skills" not in s.columns: s["skills"] = ""
	if "job_id" not in j.columns: raise ValueError("jobs must have job_id")
	if "work_type" not in j.columns: j["work_type"] = "onsite"
	if "required_skills" not in j.columns: j["required_skills"] = ""
	if "salary" not in j.columns: j["salary"] = 50000 + 10000 * rng.standard_normal(len(j))
	if "company_id" not in c.columns: c["company_id"] = np.arange(len(c))
	if "industry" not in c.columns: c["industry"] = "General"
	j = j.merge(c[["company_id", "industry"]], on="company_id", how="left") if "company_id" in j.columns else j.assign(industry="General")
	req_cols = ["user_id", "job_id", "match_score"]
	for col in req_cols:
		if col not in x.columns:
			raise ValueError(f"interactions must have {req_cols}")
	df = x.merge(s[["user_id", "GPA", "major", "skills"]], on="user_id", how="left") \
		  .merge(j[["job_id", "work_type", "required_skills", "salary", "industry"]], on="job_id", how="left")
	majors = s["major"].dropna().value_counts().nlargest(12).index.tolist()
	work_types = j["work_type"].dropna().value_counts().nlargest(4).index.tolist()
	industries = j["industry"].dropna().value_counts().nlargest(12).index.tolist()
	major_oh = _one_hot(df["major"].fillna("General"), majors); major_oh.columns = [f"major_{c}" for c in major_oh.columns]
	wt_oh = _one_hot(df["work_type"].fillna("onsite"), work_types); wt_oh.columns = [f"work_type_{c}" for c in wt_oh.columns]
	ind_oh = _one_hot(df["industry"].fillna("General"), industries); ind_oh.columns = [f"industry_{c}" for c in ind_oh.columns]
	stu_sk = df["skills"].apply(_safe_list); job_sk = df["required_skills"].apply(_safe_list)
	overlap = []; jacc = []
	for a, b in zip(stu_sk, job_sk):
		sa, sb = set(a), set(b)
		inter = len(sa & sb); union = len(sa | sb) if len(sa | sb) > 0 else 1
		overlap.append(inter); jacc.append(inter / union)
	df_feat = pd.DataFrame({
		"GPA": df["GPA"].fillna(df["GPA"].median()),
		"salary": df["salary"].fillna(df["salary"].median()),
		"skills_overlap": np.array(overlap, dtype=float),
		"skills_jaccard": np.array(jacc, dtype=float),
	})
	features_df = pd.concat([df_feat, major_oh, wt_oh, ind_oh], axis=1).fillna(0.0)
	feature_cols = list(features_df.columns)
	df_out = pd.concat([df[["user_id", "job_id"]], df[["match_score"]].rename(columns={"match_score": "y"}), features_df], axis=1)
	df_out["client_id"] = df.get("client_id") if "client_id" in df.columns else "GLOBAL"
	df_out["features_cols_json"] = json.dumps(feature_cols)
	return df_out.reset_index(drop=True)


def make_pair_frame(client_map: Dict[str, Dict[str, pd.DataFrame]], df_jobs: pd.DataFrame, df_companies: pd.DataFrame, seed: int = RNG_DEFAULT_SEED) -> pd.DataFrame:
	frames: List[pd.DataFrame] = []
	for cid, part in client_map.items():
		users = part["users"]
		ints = part["interactions"].copy(); ints["client_id"] = cid
		df_enc = encode_features(users, df_jobs, df_companies, ints, seed=seed)
		frames.append(df_enc)
	full = pd.concat(frames, axis=0).reset_index(drop=True)
	full = full.sort_values(by=["client_id", "user_id", "job_id"]).reset_index(drop=True)
	return full


