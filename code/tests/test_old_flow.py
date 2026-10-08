import math

import numpy as np

from avito_recsys.old_flow import (
    OldFlowStatistics,
    count_strict_ordered_category_pairs,
    leaf_scores_from_category_flow,
)


def test_legacy_excel_formulas_keep_a_to_b_direction():
    pair = np.zeros((3, 3), dtype=float)
    pair[1, 2] = 20
    stats = OldFlowStatistics(
        pair_count=pair,
        source_pair_count=np.array([0, 100, 0], dtype=float),
        target_pair_count=np.array([0, 0, 80], dtype=float),
        total_pairs=1000,
    )
    assert math.isclose(stats.metric_matrix("p_target_given_source")[1, 2], 0.2)
    assert math.isclose(stats.metric_matrix("lift")[1, 2], 2.5)
    assert math.isclose(stats.metric_matrix("mu")[1, 2], 3.0)
    assert math.isclose(stats.metric_matrix("potential")[1, 2], 1.28)


def test_pairs_are_non_adjacent_but_require_strictly_earlier_time():
    stats = count_strict_ordered_category_pairs(
        [([1, 2, 3, 2], [10, 10, 20, 30])], n_category=3
    )
    assert stats.pair_count[1, 2] == 1
    assert stats.pair_count[1, 3] == 1
    assert stats.pair_count[2, 3] == 1
    assert stats.pair_count[2, 2] == 1
    assert stats.pair_count[1, 2] != 2  # equal-time category 2 is not after category 1


def test_category_score_is_expanded_to_leaf_with_popularity_tie_break():
    metric = np.zeros((3, 3), dtype=float)
    metric[1, 2] = 4.0
    leaf_to_category = np.array([0, 2, 2, 1])
    popularity = np.array([0, 5, 10, 100])
    scores = leaf_scores_from_category_flow(1, metric, leaf_to_category, popularity)
    assert scores[2] > scores[1] > scores[3]
