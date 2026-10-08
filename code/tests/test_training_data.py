from types import SimpleNamespace

import torch

from avito_recsys.training_data import make_collate


def test_collate_left_pads_history_and_builds_multilabel_masks():
    row = SimpleNamespace(
        leaf=[2, 3, 4],
        action=[1, 2, 3],
        ts=[100, 200, 300],
        snapshot_epoch=400,
        label_leaf=[5, 6],
        masked_leaf=[2, 7],
        snapshot_id="example",
    )
    batch = make_collate(n_leaf=8, max_len=5, n_time_buckets=16)([row])

    assert batch.leaf.tolist() == [[0, 0, 2, 3, 4]]
    assert batch.action.tolist() == [[0, 0, 1, 2, 3]]
    assert batch.padding_mask.tolist() == [[True, True, False, False, False]]
    assert batch.labels[0, 5] == 1 and batch.labels[0, 6] == 1
    assert batch.labels.sum() == 2
    assert not batch.candidate_mask[0, 0]
    assert not batch.candidate_mask[0, 2]
    assert not batch.candidate_mask[0, 7]
    assert batch.candidate_mask[0, 5]
    assert batch.snapshot_ids == ["example"]
    assert torch.all(batch.time_bucket[0, :2] == 0)
