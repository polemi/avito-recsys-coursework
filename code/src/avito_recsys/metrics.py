from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def mask_ranking(ranking: Iterable[int], masked: Iterable[int]) -> list[int]:
    blocked = set(masked)
    return [item for item in ranking if item not in blocked and item != 0]


def recall_at_k(ranking: Sequence[int], labels: Iterable[int], k: int) -> float:
    positives = set(labels)
    if not positives:
        return float("nan")
    return len(set(ranking[:k]) & positives) / len(positives)


def binary_ndcg_at_k(ranking: Sequence[int], labels: Iterable[int], k: int) -> float:
    positives = set(labels)
    if not positives:
        return float("nan")
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(ranking[:k]) if item in positives)
    ideal_hits = min(k, len(positives))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg

