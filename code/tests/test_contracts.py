from datetime import datetime, timedelta

from avito_recsys.contracts import (
    is_history_event,
    is_target_event,
    leaf_key,
    new_target_leaves,
)


def test_window_boundaries_do_not_overlap():
    snapshot = datetime(2026, 6, 8)
    assert is_history_event(snapshot - timedelta(days=90), snapshot)
    assert not is_history_event(snapshot, snapshot)
    assert is_target_event(snapshot, snapshot)
    assert not is_target_event(snapshot + timedelta(days=14), snapshot)


def test_history_contact_masks_candidate_but_history_is_unchanged():
    snapshot = datetime(2026, 6, 8)
    leaf_a = leaf_key("services", "repair", "phones")
    leaf_b = leaf_key("services", "repair", "laptops")
    history = [(snapshot - timedelta(days=2), leaf_a)]
    future = [(snapshot + timedelta(days=1), leaf_a), (snapshot + timedelta(days=2), leaf_b)]
    assert new_target_leaves(history, future, snapshot) == {leaf_b}
    assert history == [(snapshot - timedelta(days=2), leaf_a)]


def test_same_param_text_in_different_paths_is_not_collapsed():
    first = leaf_key("services", "repair", "other")
    second = leaf_key("goods", "electronics", "other")
    assert first != second


def test_multilabel_target_is_preserved():
    snapshot = datetime(2026, 6, 8)
    future = [
        (snapshot + timedelta(hours=1), "v|c|a"),
        (snapshot + timedelta(hours=2), "v|c|b"),
    ]
    assert new_target_leaves([], future, snapshot) == {"v|c|a", "v|c|b"}

