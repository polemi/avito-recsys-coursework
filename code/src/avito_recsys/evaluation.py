from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from avito_recsys.metrics import binary_ndcg_at_k, recall_at_k


class RankingAccumulator:
    def __init__(self, ks: Iterable[int] = (5, 10, 20)):
        self.ks = tuple(ks)
        self.values: dict[str, list[float]] = defaultdict(list)
        self.recommended: set[int] = set()

    def add(self, ranking: list[int], labels: list[int]) -> None:
        self.recommended.update(ranking[: max(self.ks)])
        for k in self.ks:
            self.values[f"recall@{k}"].append(recall_at_k(ranking, labels, k))
            self.values[f"ndcg@{k}"].append(binary_ndcg_at_k(ranking, labels, k))

    def result(self, n_leaf: int) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            name: float(np.nanmean(values)) for name, values in self.values.items()
        }
        result["coverage"] = len(self.recommended) / n_leaf
        result["n_evaluated"] = len(next(iter(self.values.values()), []))
        return result


def top_k_from_scores(scores: np.ndarray, masked: Iterable[int], k: int) -> list[int]:
    work = scores.copy()
    work[0] = -np.inf
    blocked = np.fromiter((int(x) for x in masked), dtype=np.int64)
    blocked = blocked[(blocked >= 0) & (blocked < len(work))]
    work[blocked] = -np.inf
    valid_k = min(k, int(np.isfinite(work).sum()))
    if valid_k == 0:
        return []
    candidates = np.argpartition(work, -valid_k)[-valid_k:]
    return candidates[np.argsort(work[candidates])[::-1]].astype(int).tolist()

