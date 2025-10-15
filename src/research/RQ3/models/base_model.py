from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
	import random, os
	random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class MLPRegressor(nn.Module):
	def __init__(self, input_dim: int, hidden=(128, 64, 32), pdrop=0.2):
		super().__init__()
		layers = []
		d = input_dim
		for h in hidden:
			layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(pdrop)]
			d = h
		layers += [nn.Linear(d, 1)]
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x).squeeze(1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
	ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
	return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fit_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn) -> float:
	model.train(); total = 0.0; n = 0
	for xb, yb in loader:
		opt.zero_grad()
		pred = model(xb)
		loss = loss_fn(pred, yb)
		loss.backward()
		opt.step()
		total += float(loss.item()) * len(xb)
		n += len(xb)
	return total / max(1, n)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> Tuple[float, np.ndarray, np.ndarray]:
	model.eval(); total = 0.0; n = 0; preds = []; trues = []
	for xb, yb in loader:
		pred = model(xb)
		loss = loss_fn(pred, yb)
		total += float(loss.item()) * len(xb)
		n += len(xb)
		preds.append(pred.cpu().numpy()); trues.append(yb.cpu().numpy())
	y_pred = np.concatenate(preds) if preds else np.array([])
	y_true = np.concatenate(trues) if trues else np.array([])
	return total / max(1, n), y_true, y_pred


def early_stopping_train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, patience: int = 5, weight_decay: float = 0.0) -> dict:
	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	loss_fn = nn.MSELoss()
	best = {"val_loss": float("inf"), "state": None, "epoch": -1}
	wait = 0
	history = []
	for ep in range(epochs):
		tr_loss = fit_epoch(model, train_loader, opt, loss_fn)
		val_loss, y_true, y_pred = evaluate(model, val_loader, loss_fn)
		history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss})
		if val_loss < best["val_loss"] - 1e-6:
			best = {"val_loss": val_loss, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}, "epoch": ep}
			wait = 0
		else:
			wait += 1
			if wait >= patience:
				break
	if best["state"] is not None:
		model.load_state_dict(best["state"])
	return {"best_val_loss": best["val_loss"], "best_epoch": best["epoch"], "history": history}


