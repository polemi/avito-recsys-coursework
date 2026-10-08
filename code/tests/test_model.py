import torch

from avito_recsys.model import (
    MultiInterestTransformer,
    attention_diversity_penalty,
    diversity_penalty,
)


def test_multi_interest_forward_is_finite_and_respects_padding():
    torch.manual_seed(7)
    model = MultiInterestTransformer(
        n_leaf=8,
        num_interests=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        dropout=0.0,
        max_len=5,
        n_time_buckets=8,
    )
    leaf = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
    action = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    time_bucket = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    padding_mask = leaf.eq(0)

    scores, interests, attention = model(leaf, action, time_bucket, padding_mask)

    assert scores.shape == (2, 9)
    assert interests.shape == (2, 2, 8)
    assert attention.shape == (2, 2, 5)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(interests).all()
    assert torch.isfinite(attention).all()
    assert torch.equal(attention.masked_select(padding_mask.unsqueeze(1)), torch.zeros(6))
    assert torch.allclose(attention.sum(dim=-1), torch.ones((2, 2)), atol=1e-6)


def test_training_objective_components_have_finite_gradients():
    torch.manual_seed(11)
    model = MultiInterestTransformer(
        n_leaf=5,
        num_interests=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        dropout=0.0,
        max_len=4,
        n_time_buckets=8,
    )
    leaf = torch.tensor([[1, 2, 3, 4]])
    action = torch.tensor([[1, 1, 2, 3]])
    time_bucket = torch.tensor([[1, 2, 3, 4]])
    padding_mask = torch.zeros_like(leaf, dtype=torch.bool)
    scores, interests, attention = model(leaf, action, time_bucket, padding_mask)
    loss = scores.square().mean() + diversity_penalty(interests) + attention_diversity_penalty(attention)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
