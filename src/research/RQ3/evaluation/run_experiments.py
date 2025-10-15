import argparse, os, json, datetime as dt, subprocess, sys
from typing import List
import pandas as pd


def run_cmd(cmd: list):
	print("[RUN]", " ".join(cmd))
	r = subprocess.run(cmd, capture_output=True, text=True)
	if r.returncode != 0:
		print(r.stdout)
		print(r.stderr)
		raise RuntimeError("Command failed")
	return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--k_list", nargs="+", type=int, default=[0, 5, 10, 20, 50])
    ap.add_argument("--scenarios", nargs="+", choices=["new_items", "new_users", "new_client"], default=["new_items", "new_users", "new_client"], help="Scenarios to run")
    ap.add_argument("--methods", nargs="+", choices=["centralized", "fedavg", "fedprox", "enhanced_pfl"], default=["centralized", "fedavg", "fedprox", "enhanced_pfl"], help="Methods to run")
    ap.add_argument("--held_clients", nargs="+", default=None, help="Held-out client IDs for new_client scenario; auto-detected if omitted")
    ap.add_argument("--top_clients", type=int, default=3, help="If auto-detecting, take top-N universities by count")
    ap.add_argument("--data_root", type=str, default="data/raw", help="Path to students.csv for held-client auto-detect")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--mu", type=float, default=0.01)
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"src/research/RQ3/results/summary_{ts}"
    os.makedirs(outdir, exist_ok=True)

    def base_args() -> List[str]:
        return ["--k_list", *map(str, args.k_list), "--seeds", *map(str, args.seeds)]

    # Auto-detect held clients if needed
    held_clients = args.held_clients
    if (held_clients is None) and ("new_client" in args.scenarios):
        stu_csv = os.path.join(args.data_root, "students.csv")
        try:
            s = pd.read_csv(stu_csv)
            if "university" in s.columns:
                held_clients = list(s["university"].value_counts().nlargest(max(1, int(args.top_clients))).index)
            else:
                held_clients = []
        except Exception:
            held_clients = []

    for scenario in args.scenarios:
        if scenario == "new_client":
            if not held_clients:
                print("[WARN] No held_clients available. Skipping new_client scenario.")
                continue
            held_iter = held_clients
        else:
            held_iter = [None]

        for held in held_iter:
            if "centralized" in args.methods:
                cmd = [sys.executable, "-m", "src.research.RQ3.algorithms.centralized_cold_start.train_eval", "--scenario", scenario, *base_args()]
                if held:
                    cmd += ["--held_client", held]
                run_cmd(cmd)

            if "fedavg" in args.methods:
                cmd = [sys.executable, "-m", "src.research.RQ3.algorithms.fedavg_cold_start.train_eval", "--scenario", scenario, *base_args(), "--rounds", str(args.rounds), "--local_epochs", str(args.local_epochs)]
                if held:
                    cmd += ["--held_client", held]
                run_cmd(cmd)

            if "fedprox" in args.methods:
                cmd = [sys.executable, "-m", "src.research.RQ3.algorithms.fedprox_cold_start.train_eval", "--scenario", scenario, *base_args(), "--rounds", str(args.rounds), "--local_epochs", str(args.local_epochs), "--mu", str(args.mu)]
                if held:
                    cmd += ["--held_client", held]
                run_cmd(cmd)

            if "enhanced_pfl" in args.methods:
                cmd = [sys.executable, "-m", "src.research.RQ3.algorithms.enhanced_pfl_cold_start.train_eval", "--scenario", scenario, *base_args(), "--rounds", str(args.rounds), "--local_epochs", str(args.local_epochs)]
                if held:
                    cmd += ["--held_client", held]
                run_cmd(cmd)

    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write("Run completed. Aggregate per-method JSON outputs manually.\n")
    print(f"[OK] Finished. Outputs stored; summary stub at {outdir}")


if __name__ == "__main__":
	main()


