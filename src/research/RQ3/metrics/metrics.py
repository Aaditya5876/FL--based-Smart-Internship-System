from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def mse(y_true, y_pred, sample_weight=None):
	err = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
	if sample_weight is None:
		return float(err.mean())
	w = np.asarray(sample_weight)
	return float((w * err).sum() / w.sum())


def mae(y_true, y_pred, sample_weight=None):
	err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
	if sample_weight is None:
		return float(err.mean())
	w = np.asarray(sample_weight)
	return float((w * err).sum() / w.sum())


def r2(y_true, y_pred, sample_weight=None):
	y = np.asarray(y_true)
	yp = np.asarray(y_pred)
	if sample_weight is None:
		ss_res = float(((y - yp) ** 2).sum())
		ss_tot = float(((y - y.mean()) ** 2).sum())
	else:
		w = np.asarray(sample_weight)
		y_bar = float((w * y).sum() / w.sum())
		ss_res = float((w * (y - yp) ** 2).sum())
		ss_tot = float((w * (y - y_bar) ** 2).sum())
	if ss_tot == 0:
		return 0.0
	return float(1.0 - ss_res / ss_tot)


def _rank_per_user(df: pd.DataFrame, K: int):
	for uid, grp in df.groupby("user_id"):
		g = grp.sort_values("y_pred", ascending=False)
		topk = g.head(K)
		yield uid, g, topk


def ndcg_at_k(df: pd.DataFrame, K: int) -> float:
	scores = []
	for _, g, topk in _rank_per_user(df, K):
		dcg = 0.0
		for i, v in enumerate(topk["y"].to_numpy(), start=1):
			dcg += (2 ** float(v) - 1.0) / np.log2(i + 1)
		ideal = g.sort_values("y", ascending=False).head(K)["y"].to_numpy()
		idcg = 0.0
		for i, v in enumerate(ideal, start=1):
			idcg += (2 ** float(v) - 1.0) / np.log2(i + 1)
		scores.append(dcg / idcg if idcg > 0 else 0.0)
	return float(np.mean(scores)) if scores else 0.0


def hitrate_at_k(df: pd.DataFrame, K: int, threshold: float = None) -> float:
	hits = []
	for _, g, topk in _rank_per_user(df, K):
		if threshold is None:
			hit = topk["y"].max() >= g["y"].max()
		else:
			hit = (topk["y"] >= threshold).any()
		hits.append(1.0 if hit else 0.0)
	return float(np.mean(hits)) if hits else 0.0


def map_at_k(df: pd.DataFrame, K: int, threshold: float = None) -> float:
	aps = []
	for _, g, topk in _rank_per_user(df, K):
		thr = g["y"].quantile(0.8) if threshold is None else threshold
		rel = (topk["y"] >= thr).to_numpy().astype(int)
		if rel.sum() == 0:
			aps.append(0.0)
			continue
		precisions = []
		cum_rel = 0
		for i, r in enumerate(rel, start=1):
			if r == 1:
				cum_rel += 1
				precisions.append(cum_rel / i)
		aps.append(np.mean(precisions) if precisions else 0.0)
	return float(np.mean(aps)) if aps else 0.0


def aggregate_client_weighted(per_client: List[Dict]) -> Dict:
	N = sum(x["n"] for x in per_client)
	if N == 0:
		return {"mse": 0.0, "mae": 0.0, "r2": 0.0, "N": 0}
	mse_w = sum(x["mse"] * x["n"] for x in per_client) / N
	mae_w = sum(x["mae"] * x["n"] for x in per_client) / N
	if all(("sse" in x and "sst" in x) for x in per_client):
		sse = sum(x["sse"] for x in per_client)
		sst = sum(x["sst"] for x in per_client)
		r2_w = 1.0 - (sse / sst) if sst > 0 else 0.0
	else:
		r2_w = sum(x["r2"] * x["n"] for x in per_client) / N
	return {"mse": float(mse_w), "mae": float(mae_w), "r2": float(r2_w), "N": int(N)}


def time_to_threshold(curve: Dict[int, float], threshold: float) -> Optional[int]:
	for step in sorted(curve.keys()):
		if curve[step] >= threshold:
			return step
	return None


