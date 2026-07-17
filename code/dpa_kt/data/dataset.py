"""Memmap-backed sequence Dataset + DataLoader factory."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .sequences import load_sequences


class KTSequenceDataset(Dataset):
    """Indexes the memmapped .npy sequence arrays without loading them fully.

    split: 'train' (train_valid, fold != valid_fold), 'valid' (== valid_fold),
           'test' (held-out students).
    """

    def __init__(self, dataset: str, split: str, cfg: Config):
        z = load_sequences(dataset)
        fold, sp = np.asarray(z["fold"]), np.asarray(z["split"])
        if split == "train":
            idx = np.where((sp == 0) & (fold != cfg.valid_fold))[0]
        elif split == "valid":
            idx = np.where((sp == 0) & (fold == cfg.valid_fold))[0]
        elif split == "test":
            idx = np.where(sp == 1)[0]
        else:
            raise ValueError(split)
        self._z = z
        self.idx = idx
        self.q_diff_bin = z["q_diff_bin"]
        self.kc_diff_bin = z["kc_diff_bin"]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = self.idx[i]
        z = self._z
        return {
            "q": torch.from_numpy(np.asarray(z["q"][j], dtype=np.int64)),
            "r": torch.from_numpy(np.asarray(z["r"][j], dtype=np.int64)),
            "kc": torch.from_numpy(np.asarray(z["kc"][j], dtype=np.int64)),
            "selectmask": torch.from_numpy(
                np.asarray(z["selectmask"][j], dtype=np.bool_)
            ),
        }


def make_loader(dataset: str, split: str, cfg: Config) -> DataLoader:
    ds = KTSequenceDataset(dataset, split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size if split == "train" else cfg.eval_batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=False,  # Grace-Blackwell unified memory: pinning buys nothing
        drop_last=False,
    )
