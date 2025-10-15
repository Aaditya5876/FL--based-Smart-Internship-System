from typing import List, Tuple
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn


RNG_SEED = 42


def set_seed(seed: int = RNG_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s = (s or "").lower()
    s = f"<{s}>"  # boundaries
    return [s[i : i + n] for i in range(max(0, len(s) - n + 1))]


def _hash_ngram(ng: str, dim: int) -> int:
    # Stable hash -> index in [0, dim)
    h = int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16)
    return h % dim


def phrase_to_hashvec(phrase: str, dim: int = 1024, n: int = 3) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for ng in _char_ngrams(phrase, n=n):
        vec[_hash_ngram(ng, dim)] += 1.0
    if vec.sum() > 0:
        vec = vec / np.linalg.norm(vec)
    return vec


class HashingBiEncoder(nn.Module):
    def __init__(self, in_dim: int = 1024, emb_dim: int = 64, num_classes: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.emb = nn.Linear(in_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes) if num_classes and num_classes > 0 else None

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        z = self.emb(x)
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode_tensor(x)
        if self.classifier is None:
            return z
        logits = self.classifier(z)
        return logits


def encode_phrases(model: "HashingBiEncoder", phrases: List[str], in_dim: int = 1024) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        mats = [phrase_to_hashvec(p, dim=in_dim) for p in phrases]
        X = torch.from_numpy(np.stack(mats))
        Z = model.encode_tensor(X).cpu().numpy()
    return Z


def save_encoder(path: str, model: HashingBiEncoder):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": model.in_dim}, path)


def load_encoder(path: str, emb_dim: int, num_classes: int | None = None) -> HashingBiEncoder:
    ckpt = torch.load(path, map_location="cpu")
    in_dim = ckpt.get("in_dim", 1024)
    sd = ckpt["state_dict"]
    # Auto-detect classifier presence if num_classes not provided
    if num_classes is None:
        if any(k.startswith("classifier.") for k in sd.keys()):
            # infer out_features from weight shape if available
            w = sd.get("classifier.weight")
            num_classes = int(w.shape[0]) if hasattr(w, "shape") else 0
        else:
            num_classes = 0
    model = HashingBiEncoder(in_dim=in_dim, emb_dim=emb_dim, num_classes=num_classes)
    # Load state dict leniently: allow missing classifier keys when model has none
    missing, unexpected = model.load_state_dict(sd, strict=False)  # type: ignore
    model.eval()
    return model
