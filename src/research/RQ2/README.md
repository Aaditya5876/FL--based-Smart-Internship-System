# RQ2: Privacy-Preserving Semantic Skill Alignment

## Problem Statement
In a federated internship recommender, institutions cannot share raw text (titles, skills, descriptions). We need a privacy-preserving way to:
- Normalize heterogeneous, free‑text titles/skills into a shared canonical vocabulary.
- Learn phrase encoders that map semantically similar phrases to nearby vectors without exposing raw text.
- Produce aligned, versioned features that downstream tasks (e.g., RQ3 cold‑start) can consume.

## Hypotheses
- Centralized training on public/aggregate phrase statistics can seed a robust baseline encoder.
- Federated fine‑tuning on client‑local phrases (no raw text sharing) improves alignment for local vocabularies.
- Differential privacy (DP) or distillation can further reduce leakage while preserving utility.

## Methods (Initial Skeletons)
- Centralized Ontology: Bootstrap a small canonical vocabulary and alias map from frequency statistics in `data/raw`.
- FL Encoders: Simulated FedAvg training across clients partitioned by `students.university`. Only model weights/gradients are shared.
- DP‑FL: Placeholder hooks for clipping/noise (to be enabled later).
- Fed‑Distill: Placeholder for logit sharing on an anchor set (no raw text from clients).

### New (Publication Upgrades)
- Curated alias groups: provide `src/research/RQ2/data/aliases_review.csv` with `canonical_id,phrase` rows; tooling will export `alias_map.json` and `canonical_vocab.json`.
- Contrastive training: centralized (`encoders/train_contrastive.py`) and FL (`federation/train_fl_contrastive.py`) using triplet loss.
- Robust intrinsic eval: avoids NaNs; reports AUC/purity/NMI.

## Privacy Model
- Not shared: raw phrases, raw titles, raw skills, student/job rows.
- Shared: model weights/gradients; optional secure aggregated updates; optional logits on an agreed anchor set.
- Logged: payload sizes (bytes/parameters), training rounds, aggregate metrics only.

## Outputs (Consumed Later by RQ3)
Versioned under `data/processed/v2_alignment_<tag>/`:
- `canonical_vocab.json` and `alias_map.json` — canonical IDs and alias → canonical mapping.
- `phrase_encoder.pt` — lightweight encoder weights.
- `intrinsic_eval.json` — F1/AUC/purity/NMI for alignment quality.
- `features_aligned.parquet` (+ `schema.json`, `manifest.json`) — normalized and embedded user/job features for downstream use.

## Example Commands
- Centralized encoder:
  `python -m src.research.RQ2.encoders.train_centralized --version v2_alignment_bootstrap --epochs 5`
- Centralized contrastive (curated):
  `python -m src.research.RQ2.encoders.train_contrastive --version v2_alignment_cctr --aliases_csv src/research/RQ2/data/aliases_review.csv --epochs 5`
- FL encoder (simulated):
  `python -m src.research.RQ2.federation.train_fl_encoder --version v2_alignment_flboot --rounds 20 --local_epochs 1`
- FL contrastive (curated):
  `python -m src.research.RQ2.federation.train_fl_contrastive --version v2_alignment_flctr --aliases_csv src/research/RQ2/data/aliases_review.csv --rounds 20 --local_epochs 1`
- Build aligned features:
  `python -m src.research.RQ2.pipelines.build_aligned_features --version v2_alignment_flboot --notes "FL encoder baseline"`
- Intrinsic eval:
  `python -m src.research.RQ2.eval.run_intrinsic_eval --version v2_alignment_flboot`

## Defaults & Quality Bar
- Deterministic seeds and small, fast defaults.
- Read from `data/raw`; write to `data/processed/v2_alignment_<tag>/`.
- Clear logs; modular components to swap in stronger models later.
