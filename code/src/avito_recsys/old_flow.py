from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class OldFlowStatistics:
    """Exact statistical core of the legacy weekly category-flow workbook."""

    pair_count: np.ndarray
    source_pair_count: np.ndarray
    target_pair_count: np.ndarray
    total_pairs: float

    def metric_matrix(self, metric: str) -> np.ndarray:
        pair = self.pair_count.astype(np.float64, copy=False)
        source = self.source_pair_count[:, None].astype(np.float64, copy=False)
        target = self.target_pair_count[None, :].astype(np.float64, copy=False)
        total = float(self.total_pairs)

        p_target_given_source = np.divide(
            pair, source, out=np.zeros_like(pair, dtype=np.float64), where=source > 0
        )
        if metric == "p_target_given_source":
            return p_target_given_source

        target_rate = target / total if total > 0 else np.zeros_like(target)
        lift = np.divide(
            p_target_given_source,
            target_rate,
            out=np.zeros_like(pair, dtype=np.float64),
            where=target_rate > 0,
        )
        if metric == "lift":
            return lift

        p_target_without_source = np.divide(
            target - pair,
            total - source,
            out=np.zeros_like(pair, dtype=np.float64),
            where=(total - source) > 0,
        )
        mu = np.divide(
            p_target_given_source,
            p_target_without_source,
            out=np.zeros_like(pair, dtype=np.float64),
            where=p_target_without_source > 0,
        )
        if metric == "mu":
            return mu
        if metric == "potential":
            return (source - pair) * p_target_given_source * 0.08
        raise ValueError(f"Unknown legacy flow metric: {metric}")


def count_strict_ordered_category_pairs(
    sequences: Iterable[tuple[Sequence[int], Sequence[int]]], n_category: int
) -> OldFlowStatistics:
    """Count all A->B pairs with strictly earlier timestamps, not only adjacent pairs."""

    pair_count = np.zeros((n_category + 1, n_category + 1), dtype=np.float64)
    for categories, timestamps in sequences:
        if len(categories) != len(timestamps):
            raise ValueError("categories and timestamps must have equal length")
        if len(categories) < 2:
            continue
        order = np.argsort(np.asarray(timestamps), kind="stable")
        cat = np.asarray(categories, dtype=np.int64)[order]
        ts = np.asarray(timestamps, dtype=np.int64)[order]
        seen = np.zeros(n_category + 1, dtype=np.float64)
        start = 0
        while start < len(cat):
            end = start + 1
            while end < len(cat) and ts[end] == ts[start]:
                end += 1
            group = np.bincount(cat[start:end], minlength=n_category + 1).astype(np.float64)
            targets = np.flatnonzero(group)
            for target in targets:
                pair_count[:, target] += seen * group[target]
            seen += group
            start = end

    pair_count[0, :] = 0
    pair_count[:, 0] = 0
    source_pair_count = pair_count.sum(axis=1)
    target_pair_count = pair_count.sum(axis=0)
    return OldFlowStatistics(
        pair_count=pair_count,
        source_pair_count=source_pair_count,
        target_pair_count=target_pair_count,
        total_pairs=float(pair_count.sum()),
    )


def leaf_scores_from_category_flow(
    source_category: int | None,
    category_metric: np.ndarray,
    leaf_to_category: np.ndarray,
    leaf_popularity: np.ndarray,
) -> np.ndarray:
    """Adapt the legacy category score to leaf ranking with popularity only as tie-breaker."""

    n_leaf = len(leaf_to_category) - 1
    scores = np.zeros(n_leaf + 1, dtype=np.float64)
    popularity = np.log1p(leaf_popularity.astype(np.float64, copy=False))
    if popularity.max(initial=0) > 0:
        popularity = popularity / popularity.max()
    if source_category is None or source_category <= 0 or source_category >= category_metric.shape[0]:
        scores[:] = popularity
    else:
        scores[:] = category_metric[source_category, leaf_to_category]
        scores += popularity * 1e-9
    scores[0] = -np.inf
    return scores
