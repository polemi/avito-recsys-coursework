from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiInterestTransformer(nn.Module):
    def __init__(
        self,
        n_leaf: int,
        num_interests: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.15,
        max_len: int = 128,
        n_time_buckets: int = 32,
        use_action_embedding: bool = True,
        use_time_embedding: bool = True,
    ):
        super().__init__()
        self.n_leaf = n_leaf
        self.num_interests = num_interests
        self.d_model = d_model
        self.max_len = max_len
        self.use_action_embedding = use_action_embedding
        self.use_time_embedding = use_time_embedding
        self.leaf_embedding = nn.Embedding(n_leaf + 1, d_model, padding_idx=0)
        self.action_embedding = nn.Embedding(5, d_model, padding_idx=0)
        self.time_embedding = nn.Embedding(n_time_buckets, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
        self.interest_queries = nn.Parameter(torch.empty(num_interests, d_model))
        nn.init.orthogonal_(self.interest_queries)
        self.attention_temperature = 0.2

    def encode(
        self,
        leaf: torch.Tensor,
        action: torch.Tensor,
        time_bucket: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(leaf.shape[1], device=leaf.device).unsqueeze(0)
        hidden = self.leaf_embedding(leaf) + self.position_embedding(positions)
        if self.use_action_embedding:
            hidden = hidden + self.action_embedding(action)
        if self.use_time_embedding:
            hidden = hidden + self.time_embedding(time_bucket)
        hidden = self.input_dropout(self.input_norm(hidden))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        attention = (
            torch.einsum("bld,kd->bkl", hidden, self.interest_queries)
            / math.sqrt(self.d_model)
            / self.attention_temperature
        )
        attention = attention.masked_fill(padding_mask.unsqueeze(1), -torch.inf)

        # Each query should both select useful events and compete with the other
        # queries for them. Independent softmaxes over positions let every query
        # attend to exactly the same events, which collapses K interests into one.
        position_weights = torch.softmax(attention, dim=-1)
        competition_logits = attention.masked_fill(padding_mask.unsqueeze(1), -1e4)
        competition_weights = torch.softmax(competition_logits, dim=1)
        competition_weights = competition_weights.masked_fill(padding_mask.unsqueeze(1), 0.0)
        weights = position_weights * competition_weights
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        interests = torch.einsum("bkl,bld->bkd", weights, hidden)
        return interests, weights

    def score(self, interests: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bkd,nd->bkn", interests, self.leaf_embedding.weight)
        return logits.max(dim=1).values / math.sqrt(self.d_model)

    def forward(
        self,
        leaf: torch.Tensor,
        action: torch.Tensor,
        time_bucket: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interests, attention = self.encode(leaf, action, time_bucket, padding_mask)
        return self.score(interests), interests, attention


def diversity_penalty(interests: torch.Tensor) -> torch.Tensor:
    if interests.shape[1] == 1:
        return interests.new_zeros(())
    normalized = F.normalize(interests, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(similarity.shape[1], device=similarity.device).unsqueeze(0)
    return ((similarity - eye) ** 2).mean()


def attention_diversity_penalty(attention: torch.Tensor) -> torch.Tensor:
    if attention.shape[1] == 1:
        return attention.new_zeros(())
    gram = attention @ attention.transpose(1, 2)
    diagonal = torch.diagonal(gram, dim1=1, dim2=2).clamp_min(1e-8)
    scale = torch.sqrt(diagonal.unsqueeze(2) * diagonal.unsqueeze(1))
    normalized = gram / scale
    eye = torch.eye(normalized.shape[1], device=normalized.device).unsqueeze(0)
    return ((normalized - eye) ** 2).mean()


def mean_off_diagonal_cosine(interests: torch.Tensor) -> float:
    if interests.shape[1] == 1:
        return 0.0
    normalized = F.normalize(interests, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    k = similarity.shape[1]
    mask = ~torch.eye(k, dtype=torch.bool, device=similarity.device)
    return float(similarity[:, mask].mean().detach().cpu())
