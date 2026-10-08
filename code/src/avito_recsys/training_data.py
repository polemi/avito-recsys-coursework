from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    leaf: torch.Tensor
    action: torch.Tensor
    time_bucket: torch.Tensor
    padding_mask: torch.Tensor
    labels: torch.Tensor
    candidate_mask: torch.Tensor
    snapshot_ids: list[str]


class SnapshotDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        return self.frame.iloc[index]


def make_collate(n_leaf: int, max_len: int, n_time_buckets: int):
    def collate(rows) -> Batch:
        batch_size = len(rows)
        leaf = torch.zeros((batch_size, max_len), dtype=torch.long)
        action = torch.zeros((batch_size, max_len), dtype=torch.long)
        time_bucket = torch.zeros((batch_size, max_len), dtype=torch.long)
        padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
        labels = torch.zeros((batch_size, n_leaf + 1), dtype=torch.float32)
        candidate_mask = torch.ones((batch_size, n_leaf + 1), dtype=torch.bool)
        snapshot_ids: list[str] = []
        for row_id, row in enumerate(rows):
            history_leaf = np.asarray(row.leaf[-max_len:], dtype=np.int64)
            history_action = np.asarray(row.action[-max_len:], dtype=np.int64)
            history_ts = np.asarray(row.ts[-max_len:], dtype=np.int64)
            length = len(history_leaf)
            start = max_len - length
            leaf[row_id, start:] = torch.from_numpy(history_leaf)
            action[row_id, start:] = torch.from_numpy(history_action)
            hours_to_snapshot = np.maximum(0.0, (row.snapshot_epoch - history_ts) / 3600.0)
            buckets = np.floor(np.log2(hours_to_snapshot + 1.0)).astype(np.int64) + 1
            buckets = np.clip(buckets, 1, n_time_buckets - 1)
            time_bucket[row_id, start:] = torch.from_numpy(buckets)
            padding_mask[row_id, start:] = False
            labels[row_id, np.asarray(row.label_leaf, dtype=np.int64)] = 1.0
            candidate_mask[row_id, np.asarray(row.masked_leaf, dtype=np.int64)] = False
            candidate_mask[row_id, 0] = False
            snapshot_ids.append(str(row.snapshot_id))
        return Batch(leaf, action, time_bucket, padding_mask, labels, candidate_mask, snapshot_ids)

    return collate

