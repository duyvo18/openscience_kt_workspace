"""Append-only per-epoch CSV logger."""
from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        if self.path.exists():  # resuming: reuse existing header
            with open(self.path) as f:
                r = csv.reader(f)
                header = next(r, None)
                if header:
                    self._fieldnames = header

    def log(self, row: dict) -> None:
        row = {k: v for k, v in row.items()}
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self._fieldnames).writeheader()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, self._fieldnames, extrasaction="ignore")
            w.writerow(row)
