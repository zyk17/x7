"""PX0/Lc0 AttentionBody for the fixed 124x10x9 Xiangqi contract.

Reference: pxzero-training `tf/tfprocess.py:1259-1621` and PX0
`src/neural/onnx/converter.cc:391-1059`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_common import BOARD_SQUARES, TRANSFORMER_TRUNK_KIND, _build_move_pair_index

SMOLGEN_CHANNELS = 16
SMOLGEN_HIDDEN = 128
SMOLGEN_GENERATED = 128


def _attention_position_encoding() -> torch.Tensor:
    """Lc0 attention-policy-map positional encoding, [90, 90]."""
    index = _build_move_pair_index()
    encoding = torch.zeros((BOARD_SQUARES, BOARD_SQUARES), dtype=torch.float32)
    encoding.diagonal().fill_(-1.0)
    encoding.flatten().index_fill_(0, index, 1.0)
    return encoding


def _xavier_init(module: nn.Module, *, gain: float = 1.0) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class Smolgen(nn.Module):
    def __init__(self, channels: int, *, heads: int) -> None:
        super().__init__()
        self.compress = nn.Linear(channels, SMOLGEN_CHANNELS, bias=False)
        self.hidden = nn.Linear(BOARD_SQUARES * SMOLGEN_CHANNELS, SMOLGEN_HIDDEN)
        self.hidden_norm = nn.LayerNorm(SMOLGEN_HIDDEN, eps=1e-3)
        self.generate = nn.Linear(SMOLGEN_HIDDEN, SMOLGEN_GENERATED * heads)
        self.generated_norm = nn.LayerNorm(SMOLGEN_GENERATED * heads, eps=1e-3)
        self.heads = heads

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        encoded = self.compress(x).reshape(batch, BOARD_SQUARES * SMOLGEN_CHANNELS)
        encoded = self.hidden_norm(F.relu(self.hidden(encoded)))
        encoded = self.generated_norm(F.relu(self.generate(encoded)))
        return F.linear(encoded.reshape(batch, self.heads, SMOLGEN_GENERATED), weight).reshape(
            batch, self.heads, BOARD_SQUARES, BOARD_SQUARES
        )


class AttentionBodyBlock(nn.Module):
    def __init__(self, channels: int, *, heads: int, ffn_channels: int, alpha: float) -> None:
        super().__init__()
        self.heads, self.head_dim, self.alpha = heads, channels // heads, alpha
        self.q, self.k, self.v = (nn.Linear(channels, channels) for _ in range(3))
        self.out = nn.Linear(channels, channels)
        self.smolgen = Smolgen(channels, heads=heads)
        self.attention_norm = nn.LayerNorm(channels, eps=1e-6)
        self.ffn_in = nn.Linear(channels, ffn_channels)
        self.ffn_out = nn.Linear(ffn_channels, channels)
        self.ffn_norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: torch.Tensor, smolgen_weight: torch.Tensor) -> torch.Tensor:
        batch, squares, channels = x.shape
        q = self.q(x).reshape(batch, squares, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch, squares, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v(x).reshape(batch, squares, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k) * (self.head_dim**-0.5) + self.smolgen(x, smolgen_weight)
        mixed = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).reshape(batch, squares, channels)
        x = self.attention_norm(x + self.alpha * self.out(mixed))
        return self.ffn_norm(x + self.alpha * self.ffn_out(F.relu(self.ffn_in(x))))


class TransformerPolicyHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embedding = nn.Linear(channels, channels)
        self.q, self.k = nn.Linear(channels, channels), nn.Linear(channels, channels)
        self.scale = channels**-0.5
        self.register_buffer("move_pair_index", _build_move_pair_index(), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = F.relu(self.embedding(tokens))
        scores = torch.matmul(self.q(tokens), self.k(tokens).transpose(1, 2)) * self.scale
        return scores.flatten(1).index_select(1, self.move_pair_index)


class TransformerValueHead(nn.Module):
    def __init__(self, channels: int, *, output_size: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Linear(channels, embedding_size)
        self.hidden = nn.Linear(BOARD_SQUARES * embedding_size, 128)
        self.output = nn.Linear(128, output_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.embedding(tokens)).flatten(1)
        return self.output(F.relu(self.hidden(features)))


class KnowledgeTransformer(nn.Module):
    """PX0/Lc0 AttentionBody; auxiliary heads remain training-only."""

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
        self.register_buffer("position_encoding", _attention_position_encoding(), persistent=False)
        self.input_embedding = nn.Linear(in_planes + BOARD_SQUARES, width)
        self.input_mult_gate = nn.Parameter(torch.ones(BOARD_SQUARES, width))
        self.input_add_gate = nn.Parameter(torch.zeros(BOARD_SQUARES, width))
        self.smolgen_weight = nn.Parameter(torch.empty(BOARD_SQUARES * BOARD_SQUARES, SMOLGEN_GENERATED))
        alpha = math.pow(2.0 * num_blocks, -0.25)
        self.blocks = nn.ModuleList(
            AttentionBodyBlock(width, heads=heads, ffn_channels=ffn_channels, alpha=alpha) for _ in range(num_blocks)
        )
        self.policy_head = TransformerPolicyHead(width)
        self.soft_policy_head = TransformerPolicyHead(width) if auxiliary_heads else None
        if value_head:
            self.value_head_module = TransformerValueHead(width, output_size=3, embedding_size=32)
        if moves_left_head:
            self.moves_left_head_module = TransformerValueHead(width, output_size=1, embedding_size=8)
        self.root_value_head = (
            TransformerValueHead(width, output_size=3, embedding_size=32) if auxiliary_heads else None
        )
        _xavier_init(self)
        beta = math.pow(8.0 * num_blocks, -0.25)
        for block in self.blocks:
            for layer in (block.q, block.k, block.v, block.out, block.ffn_in, block.ffn_out):
                nn.init.xavier_normal_(layer.weight, gain=math.sqrt(beta))
        nn.init.xavier_normal_(self.smolgen_weight)

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.permute(0, 2, 3, 1).reshape(x.shape[0], BOARD_SQUARES, self.in_planes)
        position = self.position_encoding.to(dtype=tokens.dtype).expand(x.shape[0], -1, -1)
        tokens = self.input_embedding(torch.cat((tokens, position), dim=2))
        tokens = F.relu(tokens) * self.input_mult_gate.to(tokens.dtype) + self.input_add_gate.to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, self.smolgen_weight.to(tokens.dtype))
        return tokens

    def forward_heads(self, trunk: torch.Tensor, *, include_auxiliary: bool = True) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = [self.policy_head(trunk)]
        if self.value_head:
            outputs.append(self.value_head_module(trunk))
        if self.moves_left_head:
            outputs.append(F.relu(self.moves_left_head_module(trunk)))
        if self.auxiliary_heads and include_auxiliary:
            assert self.soft_policy_head is not None and self.root_value_head is not None
            outputs.extend((self.soft_policy_head(trunk), self.root_value_head(trunk)))
        return tuple(outputs)

    def forward_formal_heads(self, trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.value_head or not self.moves_left_head:
            raise RuntimeError("formal ONNX export requires WDL and moves-left heads")
        return self.policy_head(trunk), self.value_head_module(trunk), F.relu(self.moves_left_head_module(trunk))

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        outputs = self.forward_heads(self.forward_trunk(x))
        return outputs[0] if len(outputs) == 1 else outputs
