import os
import matplotlib.pyplot as plt
import pandas as pd


def performance_vs_k(df: pd.DataFrame, scenario: str, metric: str, outdir: str):
	os.makedirs(outdir, exist_ok=True)
	d = df[df["scenario"] == scenario]
	pivot = d.pivot_table(index="k", columns="method", values=metric, aggfunc="mean")
	ax = pivot.plot(marker="o")
	ax.set_title(f"{metric} vs k - {scenario}")
	ax.set_xlabel("k"); ax.set_ylabel(metric)
	plt.tight_layout()
	fp = os.path.join(outdir, f"{scenario}_{metric}_vs_k.png")
	plt.savefig(fp); plt.close()


def adaptation_curve(df: pd.DataFrame, method: str, scenario: str, held_entity: str, metric: str, outdir: str):
	os.makedirs(outdir, exist_ok=True)
	d = df[(df["method"] == method) & (df["scenario"] == scenario)]
	ax = d.sort_values("step").plot(x="step", y=metric, marker="o", legend=False, title=f"{method} {scenario} adaptation ({held_entity})")
	plt.tight_layout()
	fp = os.path.join(outdir, f"{method}_{scenario}_adapt_{metric}.png")
	plt.savefig(fp); plt.close()


def bars_zero_shot(df: pd.DataFrame, scenario: str, metric: str, outdir: str):
	os.makedirs(outdir, exist_ok=True)
	d = df[(df["scenario"] == scenario) & (df["k"] == 0)]
	pivot = d.pivot_table(index="method", values=metric, aggfunc="mean")
	ax = pivot.plot(kind="bar")
	ax.set_title(f"Zero-shot {metric} - {scenario}")
	plt.tight_layout()
	fp = os.path.join(outdir, f"{scenario}_zero_shot_{metric}.png")
	plt.savefig(fp); plt.close()


