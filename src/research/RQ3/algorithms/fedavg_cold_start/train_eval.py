import argparse, json, os, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from ...io.data_access import load_rawframes, build_client_partitions, make_pair_frame
from ...splits.cold_start_splits import split_new_items, split_new_users, split_new_client_LOCO, sample_k_shot
from ...models.base_model import MLPRegressor, make_loader, set_seed, early_stopping_train, fit_epoch, evaluate
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


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--scenario", choices=["new_items","new_users","new_client"], required=True)
	ap.add_argument("--held_client", type=str, default=None)
	ap.add_argument("--k_list", nargs="+", type=int, default=[0,5,10,20,50])
	ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,44,45,46])
	ap.add_argument("--epochs", type=int, default=3)
	ap.add_argument("--local_epochs", type=int, default=1)
	ap.add_argument("--rounds", type=int, default=20)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--batch_size", type=int, default=128)
	ap.add_argument("--outdir", type=str, default="src/research/RQ3/results/fedavg")
	args = ap.parse_args()

	os.makedirs(args.outdir, exist_ok=True)
	raw = load_rawframes("data/raw")
	client_map = build_client_partitions(raw["df_students"], raw["df_interactions"])
	pair_df = make_pair_frame(client_map, raw["df_jobs"], raw["df_companies"], seed=min(args.seeds))

	all_logs = []

	for seed in args.seeds:
		set_seed(seed)
		if args.scenario == "new_items":
			base_train_df, test_df, held = split_new_items(pair_df, seed, 0.2)
			key_col = "job_id"
		elif args.scenario == "new_users":
			base_train_df, test_df, held = split_new_users(pair_df, seed, 0.2)
			key_col = "user_id"
		else:
			if not args.held_client:
				raise ValueError("--held_client required for new_client")
			base_train_df, test_df = split_new_client_LOCO(pair_df, args.held_client)
			key_col = "user_id"

		_, _, _, feat_cols = _to_Xy(base_train_df)
		global_model = MLPRegressor(input_dim=len(feat_cols))

		for r in range(args.rounds):
			deltas = []
			total_n = 0
			for cid, grp in base_train_df.groupby("client_id"):
				Xc = grp[feat_cols].to_numpy(np.float32)
				yc = grp["y"].to_numpy(np.float32)
				if len(Xc) == 0:
					continue
				Xtr, Xval, ytr, yval = train_test_split(Xc, yc, test_size=0.1, random_state=seed)
				local = MLPRegressor(input_dim=len(feat_cols))
				local.load_state_dict({k: v.clone() for k, v in global_model.state_dict().items()})
				tr_loader = make_loader(Xtr, ytr, args.batch_size, True)
				val_loader = make_loader(Xval, yval, args.batch_size, False)
				from ...models.base_model import early_stopping_train
				early_stopping_train(local, tr_loader, val_loader, epochs=args.local_epochs, lr=args.lr, patience=1)
				delta = {k: (local.state_dict()[k] - global_model.state_dict()[k]) for k in global_model.state_dict().keys()}
				deltas.append((len(Xtr), delta))
				total_n += len(Xtr)
			new_sd = {k: v.clone() for k, v in global_model.state_dict().items()}
			for k in new_sd.keys():
				agg = sum(n * d[k] for (n, d) in deltas) / max(1, total_n)
				new_sd[k] = new_sd[k] + agg
			global_model.load_state_dict(new_sd)

		for k in args.k_list:
			adapt_df, eval_df = sample_k_shot(test_df, key_col=key_col, k=k, seed=seed)
			if k == 0 or len(adapt_df) == 0:
				metrics = _eval(global_model, eval_df, args.batch_size)
			else:
				Xa = adapt_df[feat_cols].to_numpy(np.float32)
				ya = adapt_df["y"].to_numpy(np.float32)
				m2 = MLPRegressor(input_dim=len(feat_cols)); m2.load_state_dict({k: v.clone() for k, v in global_model.state_dict().items()})
				opt = torch.optim.Adam(m2.parameters(), lr=min(args.lr, 5e-4))
				import torch.nn as nn
				al = make_loader(Xa, ya, min(args.batch_size, max(8, len(Xa))), True)
				for _ in range(3):
					_ = fit_epoch(m2, al, opt, nn.MSELoss())
				metrics = _eval(m2, eval_df, args.batch_size)
			all_logs.append({"seed": seed, "k": k, "scenario": args.scenario, "held_client": args.held_client, "metrics": metrics})

	out_path = os.path.join(args.outdir, f"fedavg_{args.scenario}.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(all_logs, f, indent=2)
	print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
	main()


