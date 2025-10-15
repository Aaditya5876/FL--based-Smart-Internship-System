import json, os, socket, subprocess, datetime as dt


def get_git_hash():
	try:
		return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
	except Exception:
		return "unknown"


def run_metadata():
	return {
		"timestamp": dt.datetime.utcnow().isoformat() + "Z",
		"host": socket.gethostname(),
		"git_hash": get_git_hash(),
		"pid": os.getpid(),
		"python": os.sys.version.split()[0],
	}


def jsonl_logger(path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	f = open(path, "a", encoding="utf-8")

	def log(obj: dict):
		f.write(json.dumps(obj) + "\n"); f.flush()

	return log


