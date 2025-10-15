import argparse, json, os, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from ...io.data_access import load_rawframes, build_client_partitions, make_pair_frame
from ...splits.cold_start_splits import split_new_items, split_new_users, split_new_client_LOCO, sample_k_shot
from ...models.base_model import MLPRegressor, make_loader, set_seed, fit_epoch, evaluate
from ...metrics.metrics import mse, mae, r2, ndcg_at_k, hitrate_at_k, map_at_k


def _to_Xy(df: pd.DataFrame):
	feat_cols = json.loads(df["features_cols_json"].iloc[0])
	X = df[feat_cols].to_numpy(dtype=np.float32)
	y = df["y"].to_numpy(dtype=np.float32)
	meta = df[["user_id", "job_id", "client_id"]]
	return X, y, meta, feat_cols



def _eval(model, df, batch):
	if df is None or len(df) == 0:
		return {"mse": 0, "mae": 0, "r2": 0, "ranking": {"ndcg5": 0, "ndcg10": 0, "hit5": 0, "hit10": 0, "map5": 0, "map10": 0}, "n": 0}
	X, y, meta, _ = _to_Xy(df)
	if len(X) == 0:
		return {"mse": 0, "mae": 0, "r2": 0, "ranking": {"ndcg5": 0, "ndcg10": 0, "hit5": 0, "hit10": 0, "map5": 0, "map10": 0}, "n": 0}
	loader = make_loader(X, y, batch, False)
	import torch.nn as nn
	_, y_true, y_pred = evaluate(model, loader, nn.MSELoss())
	df_eval = meta.copy(); df_eval["y"] = y_true; df_eval["y_pred"] = y_pred
	return {
		"mse": mse(y_true, y_pred),
		"mae": mae(y_true, y_pred),
		"r2": r2(y_true, y_pred),
		"ranking": {
			"ndcg5": ndcg_at_k(df_eval, 5),
			"ndcg10": ndcg_at_k(df_eval, 10),
			"hit5": hitrate_at_k(df_eval, 5),
			"hit10": hitrate_at_k(df_eval, 10),
			"map5": map_at_k(df_eval, 5),
			"map10": map_at_k(df_eval, 10),
		},
		"n": int(len(df_eval))
	}


def _client_embedding(df_client: pd.DataFrame, feat_cols):
	X = df_client[feat_cols].to_numpy(np.float32)
	y = df_client["y"].to_numpy(np.float32)
	xmean = X.mean(axis=0) if len(X) > 0 else np.zeros(len(feat_cols), np.float32)
	ymean = float(y.mean()) if len(y) > 0 else 0.0
	return np.concatenate([xmean, np.array([ymean], dtype=np.float32)])


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--scenario", choices=["new_items","new_users","new_client"], required=True)
	ap.add_argument("--held_client", type=str, default=None)
	ap.add_argument("--k_list", nargs="+", type=int, default=[0,5,10,20,50])
	ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,44,45,46])
	ap.add_argument("--local_epochs", type=int, default=1)
	ap.add_argument("--rounds", type=int, default=20)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--batch_size", type=int, default=128)
	ap.add_argument("--kmeans_k", type=int, default=3)
	ap.add_argument("--no_clustering", action="store_true")
	ap.add_argument("--no_perf_weight", action="store_true")
	ap.add_argument("--no_progressive", action="store_true")
	ap.add_argument("--outdir", type=str, default="src/research/RQ3/results/enhanced_pfl")
	args = ap.parse_args()

	os.makedirs(args.outdir, exist_ok=True)
	raw = load_rawframes("data/raw")
	client_map = build_client_partitions(raw["df_students"], raw["df_interactions"])
	pair_df = make_pair_frame(client_map, raw["df_jobs"], raw["df_companies"], seed=min(args.seeds))

	logs = []

	for seed in args.seeds:
		set_seed(seed)
		if args.scenario == "new_items":
			base_train_df, test_df, _ = split_new_items(pair_df, seed, 0.2)
			key_col = "job_id"
		elif args.scenario == "new_users":
			base_train_df, test_df, _ = split_new_users(pair_df, seed, 0.2)
			key_col = "user_id"
		else:
			if not args.held_client:
				raise ValueError("--held_client required for new_client")
			base_train_df, test_df = split_new_client_LOCO(pair_df, args.held_client)
			key_col = "user_id"

		_, _, _, feat_cols = _to_Xy(base_train_df)
		global_model = MLPRegressor(input_dim=len(feat_cols))

		client_ids = sorted(base_train_df["client_id"].unique().tolist())
		embs = np.vstack([_client_embedding(base_train_df[base_train_df["client_id"] == cid], feat_cols) for cid in client_ids])
		if not args.no_clustering and len(client_ids) > 1:
			k = min(args.kmeans_k, len(client_ids))
			clusters = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embs)
		else:
			clusters = np.zeros(len(client_ids), dtype=int)

		for r in range(args.rounds):
			deltas = []
			total_w = 0.0
			for idx, cid in enumerate(client_ids):
				grp = base_train_df[base_train_df["client_id"] == cid]
				Xc = grp[feat_cols].to_numpy(np.float32)
				yc = grp["y"].to_numpy(np.float32)
				if len(Xc) == 0:
					continue
				Xtr, Xval, ytr, yval = train_test_split(Xc, yc, test_size=0.1, random_state=seed)
				local = MLPRegressor(input_dim=len(feat_cols))
				local.load_state_dict({k: v.clone() for k, v in global_model.state_dict().items()})
				import torch.nn as nn
				opt = torch.optim.Adam(local.parameters(), lr=args.lr)
				tr_loader = make_loader(Xtr, ytr, args.batch_size, True)
				val_loader = make_loader(Xval, yval, args.batch_size, False)
				for _ in range(args.local_epochs):
					_ = fit_epoch(local, tr_loader, opt, nn.MSELoss())
				vloss, _, _ = evaluate(local, val_loader, nn.MSELoss())
				perf_w = 1.0 / (vloss + 1e-8)
				w = float(len(Xtr)) * (perf_w if not args.no_perf_weight else 1.0)
				delta = {k: (local.state_dict()[k] - global_model.state_dict()[k]) for k in global_model.state_dict().keys()}
				deltas.append((w, delta, clusters[idx]))
				total_w += w

			new_sd = {k: v.clone() for k, v in global_model.state_dict().items()}
			if deltas:
				cluster_ids = sorted(set(g for _, _, g in deltas))
				cluster_models = []
				for g in cluster_ids:
					group = [(w, d) for (w, d, cg) in deltas if cg == g]
					tw = sum(w for w, _ in group)
					agg_delta = {k: torch.zeros_like(new_sd[k]) for k in new_sd.keys()}
					for w, d in group:
						for k in agg_delta.keys():
							agg_delta[k] = agg_delta[k] + (w / max(1e-8, tw)) * d[k]
					cluster_sd = {k: new_sd[k] + agg_delta[k] for k in new_sd.keys()}
					cluster_models.append(cluster_sd)
				final_sd = {k: sum(cm[k] for cm in cluster_models) / len(cluster_models) for k in new_sd.keys()}
				new_sd = final_sd
			global_model.load_state_dict(new_sd)

		for k in args.k_list:
			adapt_df, eval_df = sample_k_shot(test_df, key_col=key_col, k=k, seed=seed)
			if k == 0 or len(adapt_df) == 0 or args.no_progressive:
				metrics = _eval(global_model, eval_df, args.batch_size)
			else:
				import torch.nn as nn
				stages = [1e-3, 5e-4, 1e-4]
				m2 = MLPRegressor(input_dim=len(feat_cols)); m2.load_state_dict({k: v.clone() for k, v in global_model.state_dict().items()})
				Xa = adapt_df[feat_cols].to_numpy(np.float32); ya = adapt_df["y"].to_numpy(np.float32)
				for lr in stages:
					opt = torch.optim.Adam(m2.parameters(), lr=lr)
					al = make_loader(Xa, ya, min(args.batch_size, max(8, len(Xa))), True)
					_ = fit_epoch(m2, al, opt, nn.MSELoss())
				metrics = _eval(m2, eval_df, args.batch_size)
			logs.append({"seed": seed, "k": k, "scenario": args.scenario, "held_client": args.held_client,
					 "no_clustering": args.no_clustering, "no_perf_weight": args.no_perf_weight,
					 "no_progressive": args.no_progressive, "metrics": metrics})

	out_path = os.path.join(args.outdir, f"enhanced_pfl_{args.scenario}.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(logs, f, indent=2)
	print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
	main()


