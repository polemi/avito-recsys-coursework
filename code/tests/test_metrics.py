import math

from avito_recsys.metrics import binary_ndcg_at_k, mask_ranking, recall_at_k


def test_mask_is_applied_before_top_k():
    assert mask_ranking([4, 2, 1, 3], {4, 1}) == [2, 3]


def test_multilabel_recall_and_ndcg():
    ranking = [2, 8, 5, 7]
    labels = {2, 5}
    assert recall_at_k(ranking, labels, 2) == 0.5
    assert recall_at_k(ranking, labels, 3) == 1.0
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(binary_ndcg_at_k(ranking, labels, 3), expected)


def test_empty_target_is_not_silently_scored_as_zero():
    assert math.isnan(recall_at_k([1, 2], [], 2))
    assert math.isnan(binary_ndcg_at_k([1, 2], [], 2))

