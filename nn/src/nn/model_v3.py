"""X7 v3 AttentionBody translated from PX0/Lc0's proven transformer design.

The trunk is deliberately conventional: MHA + Smolgen bias + DeepNorm residual
scale + LayerNorm + two-layer FFN.  It is a Knowledge carrier, not a new
attention research branch.  Reference: px0 `src/neural/onnx/converter.cc`
(`MakeEncoderLayer`, `MakeSmolgen`, `MakeAttentionPolicy`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_common import BOARD_SQUARES, TRANSFORMER_TRUNK_KIND, _build_move_pair_index

SMOLGEN_CHANNELS = 32
SMOLGEN_HIDDEN = 256
SMOLGEN_GENERATED = 256


class Smolgen(nn.Module):
    """Lc0/PX0 learned per-position attention-logit bias."""

    def __init__(self, channels: int, *, heads: int) -> None:
        super().__init__()
        self.compress = nn.Linear(channels, SMOLGEN_CHANNELS, bias=False)
        self.hidden = nn.Linear(BOARD_SQUARES * SMOLGEN_CHANNELS, SMOLGEN_HIDDEN)
        self.hidden_norm = nn.LayerNorm(SMOLGEN_HIDDEN)
        self.generate = nn.Linear(SMOLGEN_HIDDEN, SMOLGEN_GENERATED * heads)
        self.generated_norm = nn.LayerNorm(SMOLGEN_GENERATED * heads)
        self.heads = heads

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        encoded = self.compress(x).reshape(batch, BOARD_SQUARES * SMOLGEN_CHANNELS)
        encoded = self.hidden_norm(F.silu(self.hidden(encoded), inplace=True))
        generated = self.generated_norm(F.silu(self.generate(encoded), inplace=True))
        return F.linear(generated.reshape(batch, self.heads, SMOLGEN_GENERATED), weight).reshape(
            batch, self.heads, BOARD_SQUARES, BOARD_SQUARES
        )


class AttentionBodyBlock(nn.Module):
    def __init__(self, channels: int, *, heads: int, ffn_channels: int, alpha: float) -> None:
        super().__init__()
        self.heads, self.head_dim, self.alpha = heads, channels // heads, alpha
        self.q, self.k, self.v = (nn.Linear(channels, channels) for _ in range(3))
        self.out = nn.Linear(channels, channels)
        self.smolgen = Smolgen(channels, heads=heads)
        self.attention_norm = nn.LayerNorm(channels)
        self.ffn_in = nn.Linear(channels, ffn_channels)
        self.ffn_out = nn.Linear(ffn_channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, smolgen_weight: torch.Tensor) -> torch.Tensor:
        batch, squares, channels = x.shape
        q = self.q(x).reshape(batch, squares, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch, squares, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v(x).reshape(batch, squares, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k) * (self.head_dim**-0.5) + self.smolgen(x, smolgen_weight)
        mixed = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).reshape(batch, squares, channels)
        x = self.attention_norm(x + self.alpha * self.out(mixed))
        return self.ffn_norm(x + self.alpha * self.ffn_out(F.silu(self.ffn_in(x), inplace=True)))


class TransformerPolicyHead(nn.Module):
    def __init__(self, channels: int, *, policy_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.q, self.k = nn.Linear(channels, policy_channels), nn.Linear(channels, policy_channels)
        self.scale = policy_channels**-0.5
        self.register_buffer("move_pair_index", _build_move_pair_index(), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm(tokens)
        scores = torch.matmul(self.q(tokens), self.k(tokens).transpose(1, 2)) * self.scale
        return scores.flatten(1).index_select(1, self.move_pair_index)


class TransformerValueAuxHead(nn.Module):
    def __init__(self, channels: int, *, hidden_dim: int, root_value_head: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.project, self.fc = nn.Linear(channels, hidden_dim), nn.Linear(hidden_dim * 2, hidden_dim)
        self.value_out, self.moves_left_out = nn.Linear(hidden_dim, 3), nn.Linear(hidden_dim, 1)
        self.root_value_out = nn.Linear(hidden_dim, 3) if root_value_head else None

    def forward(self, tokens: torch.Tensor, *, include_root_value: bool = True) -> tuple[torch.Tensor, ...]:
        features = F.silu(self.project(self.norm(tokens)), inplace=True)
        features = F.silu(self.fc(torch.cat((features.mean(1), features.amax(1)), dim=1)), inplace=True)
        outputs: tuple[torch.Tensor, ...] = (
            self.value_out(features),
            F.relu(self.moves_left_out(features), inplace=True),
        )
        return (
            outputs
            if self.root_value_out is None or not include_root_value
            else outputs + (self.root_value_out(features),)
        )


class KnowledgeTransformer(nn.Module):
    """v3 fixed-board AttentionBody, exporting the same formal v2 contract."""

    def __init__(
        self,
        in_planes: int = 124,
        width: int = 512,
        num_blocks: int = 12,
        num_moves: int = 2062,
        *,
        heads: int = 16,
        ffn_channels: int = 768,
        value_head: bool = False,
        moves_left_head: bool = False,
        auxiliary_heads: bool = False,
        trunk_kind: str = TRANSFORMER_TRUNK_KIND,
    ) -> None:
        super().__init__()
        if num_moves != 2062 or width < 4 or num_blocks < 1 or heads < 1 or ffn_channels < width or width % heads:
            raise ValueError("invalid v3 AttentionBody dimensions")
        if auxiliary_heads and (not value_head or not moves_left_head):
            raise ValueError("auxiliary_heads 须与 value_head、moves_left_head 一起启用")
        self.in_planes, self.num_moves, self.num_blocks = in_planes, num_moves, num_blocks
        self.width, self.heads, self.ffn_channels = width, heads, ffn_channels
        self.value_head, self.moves_left_head, self.auxiliary_heads, self.trunk_kind = (
            value_head,
            moves_left_head,
            auxiliary_heads,
            trunk_kind,
        )
        self.input_embedding = nn.Linear(in_planes, width)
        self.smolgen_weight = nn.Parameter(torch.empty(BOARD_SQUARES * BOARD_SQUARES, SMOLGEN_GENERATED))
        nn.init.xavier_uniform_(self.smolgen_weight)
        alpha = math.pow(2.0 * num_blocks, -0.25)
        self.blocks = nn.ModuleList(
            AttentionBodyBlock(width, heads=heads, ffn_channels=ffn_channels, alpha=alpha) for _ in range(num_blocks)
        )
        self.policy_head = TransformerPolicyHead(width, policy_channels=width // 2)
        self.soft_policy_head = TransformerPolicyHead(width, policy_channels=width // 2) if auxiliary_heads else None
        if value_head or moves_left_head:
            self.value_aux_head_module = TransformerValueAuxHead(
                width, hidden_dim=width, root_value_head=auxiliary_heads
            )

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_embedding(x.permute(0, 2, 3, 1).reshape(x.shape[0], BOARD_SQUARES, self.in_planes))
        for block in self.blocks:
            tokens = block(tokens, self.smolgen_weight)
        return tokens

    def forward_heads(self, trunk: torch.Tensor, *, include_auxiliary: bool = True) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = [self.policy_head(trunk)]
        if self.value_head or self.moves_left_head:
            value_outputs = self.value_aux_head_module(trunk, include_root_value=include_auxiliary)
            if self.value_head:
                outputs.append(value_outputs[0])
            if self.moves_left_head:
                outputs.append(value_outputs[1])
            if self.auxiliary_heads and include_auxiliary:
                assert self.soft_policy_head is not None
                outputs.extend((self.soft_policy_head(trunk), value_outputs[2]))
        return tuple(outputs)

    def forward_formal_heads(self, trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.value_head or not self.moves_left_head:
            raise RuntimeError("formal ONNX export requires WDL and moves-left heads")
        value, moves = self.value_aux_head_module(trunk, include_root_value=False)
        return self.policy_head(trunk), value, moves

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        outputs = self.forward_heads(self.forward_trunk(x))
        return outputs[0] if len(outputs) == 1 else outputs
