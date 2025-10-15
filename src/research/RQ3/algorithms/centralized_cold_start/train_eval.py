import argparse, json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from ...io.data_access import load_rawframes, build_client_partitions, make_pair_frame
from ...splits.cold_start_splits import split_new_items, split_new_users, split_new_client_LOCO, sample_k_shot
from ...models.base_model import MLPRegressor, make_loader, set_seed, early_stopping_train, evaluate, fit_epoch
from ...metrics.metrics import mse, mae, r2, ndcg_at_k, hitrate_at_k, map_at_k


def _to_Xy(df: pd.DataFrame):
	feat_cols = json.loads(df["features_cols_json"].iloc[0])
	X = df[feat_cols].to_numpy(dtype=np.float32)
	y = df["y"].to_numpy(dtype=np.float32)
	meta = df[["user_id", "job_id", "client_id"]]
	return X, y, meta


def _eval_batch(model, df, batch=256):
	X, y, meta = _to_Xy(df)
	if len(X) == 0:
		return {"mse": 0, "mae": 0, "r2": 0, "ranking": {"ndcg5": 0, "ndcg10": 0, "hit5": 0, "hit10": 0, "map5": 0, "map10": 0}, "n": 0}
	loader = make_loader(X, y, batch, False)
	import torch, torch.nn as nn
	loss_fn = nn.MSELoss()
	_, y_true, y_pred = evaluate(model, loader, loss_fn)
	df_eval = meta.copy()
	df_eval["y"] = y_true
	df_eval["y_pred"] = y_pred
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
	ap.add_argument("--epochs", type=int, default=20)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--batch_size", type=int, default=128)
	ap.add_argument("--outdir", type=str, default="src/research/RQ3/results/centralized")
	args = ap.parse_args()

	os.makedirs(args.outdir, exist_ok=True)
	all_logs = []

	raw = load_rawframes("data/raw")
	client_map = build_client_partitions(raw["df_students"], raw["df_interactions"])
	pair_df = make_pair_frame(client_map, raw["df_jobs"], raw["df_companies"], seed=min(args.seeds))

	for seed in args.seeds:
		set_seed(seed)
		if args.scenario == "new_items":
			train_df, test_df, _ = split_new_items(pair_df, seed, test_frac=0.2)
			key_col = "job_id"
		elif args.scenario == "new_users":
			train_df, test_df, _ = split_new_users(pair_df, seed, test_frac=0.2)
			key_col = "user_id"
		else:
			if not args.held_client:
				raise ValueError("--held_client required for new_client")
			train_df, test_df = split_new_client_LOCO(pair_df, args.held_client)
			key_col = "user_id"

		Xtr, ytr, _ = _to_Xy(train_df)
		Xtr_tr, Xtr_val, ytr_tr, ytr_val = train_test_split(Xtr, ytr, test_size=0.1, random_state=seed)
		model = MLPRegressor(input_dim=Xtr.shape[1])
		tr_loader = make_loader(Xtr_tr, ytr_tr, args.batch_size, True)
		val_loader = make_loader(Xtr_val, ytr_val, args.batch_size, False)
		early_stopping_train(model, tr_loader, val_loader, epochs=args.epochs, lr=args.lr, patience=5)

		for k in args.k_list:
			adapt_df, eval_df = sample_k_shot(test_df, key_col=key_col, k=k, seed=seed)
			if k == 0 or len(adapt_df) == 0:
				metrics = _eval_batch(model, eval_df, batch=args.batch_size)
			else:
				Xa, ya, _ = _to_Xy(adapt_df)
				m2 = MLPRegressor(input_dim=Xtr.shape[1]); m2.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
				import torch, torch.nn as nn
				opt = torch.optim.Adam(m2.parameters(), lr=min(args.lr, 5e-4))
				al = make_loader(Xa, ya, min(args.batch_size, max(8, len(Xa))), True)
				for _ in range(3):
					_ = fit_epoch(m2, al, opt, nn.MSELoss())
				metrics = _eval_batch(m2, eval_df, batch=args.batch_size)
			all_logs.append({"seed": seed, "k": k, "scenario": args.scenario, "held_client": args.held_client, "metrics": metrics})

	out_path = os.path.join(args.outdir, f"centralized_{args.scenario}.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(all_logs, f, indent=2)
	print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
	main()


