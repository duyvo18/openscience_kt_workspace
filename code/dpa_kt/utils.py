"""Seeding, timing, and small tensor helpers."""
from __future__ import annotations

import random
import time

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> dict:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    # RNG states must be CPU uint8 tensors; a checkpoint loaded with
    # map_location="cuda" moves them to the GPU, so coerce them back.
    torch.set_rng_state(states["torch"].to("cpu", torch.uint8))
    if torch.cuda.is_available() and "cuda" in states:
        torch.cuda.set_rng_state_all(
            [s.to("cpu", torch.uint8) for s in states["cuda"]]
        )


class Timer:
    """Wall-clock timer usable as a context manager."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
