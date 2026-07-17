"""Configuration for DPA-KT: dataclass + YAML load/merge + ablation presets."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = ROOT / "data_cache"
RUNS_DIR = ROOT / "runs"
DATASETS_DIR = ROOT / "datasets"


@dataclass
class Config:
    # --- dataset ---
    dataset: str = "assist09"
    seq_len: int = 200
    k_max: int = 4            # KC slots per interaction
    k_rel: int = 20           # related-KC budget per question
    min_interactions: int = 3
    min_q_freq: int = 5       # questions rarer than this share one bucket
    max_questions: int = 100_000  # hash-bucket cap for huge question vocabs
    n_difficulty_bins: int = 20
    valid_fold: int = 0

    # --- model dims ---
    d_emb: int = 64
    d_model: int = 128
    d_z: int = 64             # distributional space dim
    d_v: int = 32             # mastery value dim (16 for large-C datasets)
    d_key: int = 32           # KC key dim for reads/gating
    n_heads: int = 4
    dropout: float = 0.2
    temporal_k: int = 20      # temporal pooling window

    # --- ablation switches ---
    use_pattern_temporal: bool = True
    use_pattern_samekc: bool = True
    use_pattern_prereq: bool = True
    use_pattern_neighbor: bool = True
    use_align_mono: bool = True
    use_align_gs: bool = True
    use_distributional: bool = True
    dual_branch: bool = True

    # --- loss weights ---
    w_mono: float = 0.1
    w_gs: float = 0.1
    w_kl: float = 1e-4
    mono_margin: float = 0.05

    # --- training ---
    batch_size: int = 64
    eval_batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 5
    tbptt: int = 50
    mastery_grad_clip: float = 1.0  # per-step grad-through-time clip on M (0 disables)
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"

    # --- bookkeeping (filled from dataset maps at load time) ---
    n_questions: int = 0
    n_kcs: int = 0
    run_name: str = ""

    def merged(self, **overrides) -> "Config":
        cfg = dataclasses.replace(self)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise KeyError(f"Unknown config field: {k}")
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, *paths: str | Path, **overrides) -> "Config":
        """Merge one or more YAML files (later wins), then keyword overrides."""
        merged: dict = {}
        for p in paths:
            with open(p) as f:
                merged.update(yaml.safe_load(f) or {})
        merged.update(overrides)
        return cls().merged(**merged)


# Named ablation presets: name -> config overrides.
ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_temporal": {"use_pattern_temporal": False},
    "no_samekc": {"use_pattern_samekc": False},
    "no_prereq": {"use_pattern_prereq": False},
    "no_neighbor": {"use_pattern_neighbor": False},
    "no_mono": {"use_align_mono": False},
    "no_gs": {"use_align_gs": False},
    "no_distributional": {"use_distributional": False},
    "single_branch": {"dual_branch": False},
}


def load_config(dataset: str, ablation: str = "full", **overrides) -> Config:
    """Load base.yaml + configs/<dataset>.yaml + ablation preset + overrides."""
    paths = []
    base = ROOT / "configs" / "base.yaml"
    ds = ROOT / "configs" / f"{dataset}.yaml"
    if base.exists():
        paths.append(base)
    if ds.exists():
        paths.append(ds)
    cfg = Config.from_yaml(*paths) if paths else Config()
    cfg = cfg.merged(dataset=dataset, **ABLATIONS[ablation], **overrides)
    if not cfg.run_name:
        cfg.run_name = f"{dataset}_{ablation}"
    return cfg
