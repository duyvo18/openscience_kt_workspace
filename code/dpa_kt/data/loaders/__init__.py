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


from .assist09 import *
from .assist12 import *
from .eedi import *
from .junyi import *
from .kddcup import *
from .xes3g5m import *
