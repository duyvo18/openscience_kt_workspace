"""Dataset loader registry. Each loader returns the canonical-input frame
(user, question, kcs, correct, ts) plus optional hierarchy edges / KC names."""
from __future__ import annotations

from typing import Callable

LOADERS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        LOADERS[name] = fn
        return fn
    return deco


# Import every loader module so its @register runs. Registry keys:
#   assist09, algebra05, bridge06, xes3g5m, assist12, eedi, junyi
from . import assist09, assist12, eedi, junyi, kddcup, xes3g5m  # noqa: E402,F401
