from typing import Tuple, Set
import numpy as np
import pandas as pd


def split_new_items(pair_df: pd.DataFrame, seed: int, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, Set]:
	rng = np.random.default_rng(seed)
	jobs = pair_df["job_id"].unique()
	n_test = max(1, int(len(jobs) * test_frac))
	held = set(rng.choice(jobs, size=n_test, replace=False).tolist())
	test_df = pair_df[pair_df["job_id"].isin(held)].copy()
	train_df = pair_df[~pair_df["job_id"].isin(held)].copy()
	return train_df, test_df, held


def split_new_users(pair_df: pd.DataFrame, seed: int, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, Set]:
	rng = np.random.default_rng(seed)
	users = pair_df["user_id"].unique()
	n_test = max(1, int(len(users) * test_frac))
	held = set(rng.choice(users, size=n_test, replace=False).tolist())
	test_df = pair_df[pair_df["user_id"].isin(held)].copy()
	train_df = pair_df[~pair_df["user_id"].isin(held)].copy()
	return train_df, test_df, held


def split_new_client_LOCO(pair_df: pd.DataFrame, held_client_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	test_df = pair_df[pair_df["client_id"] == held_client_id].copy()
	train_df = pair_df[pair_df["client_id"] != held_client_id].copy()
	return train_df, test_df


def sample_k_shot(df: pd.DataFrame, key_col: str, k: int, seed: int):
	if k <= 0:
		return df.iloc[0:0].copy(), df.copy()
	rng = np.random.default_rng(seed)
	shots = []
	remains = []
	for key, grp in df.groupby(key_col):
		idx = grp.index.to_numpy()
		take = min(k, len(idx))
		sel = rng.choice(idx, size=take, replace=False)
		shots.append(df.loc[sel])
		remains.append(df.drop(sel))
	return pd.concat(shots).reset_index(drop=True), pd.concat(remains).reset_index(drop=True)


