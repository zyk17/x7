"""小型 px0 风格 ResNet trunk + attention policy + WDL value。"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, *, se_ratio: int = 2) -> None:
        super().__init__()
        if channels % int(se_ratio) != 0:
            raise ValueError(f"channels={channels} 需被 se_ratio={se_ratio} 整除")
        hidden = channels // int(se_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        squeezed = F.relu(self.fc1(pooled))
        excited = self.fc2(squeezed)
        gamma, beta = excited.chunk(2, dim=1)
        gamma = torch.sigmoid(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, se_ratio: int = 2) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.se(self.b2(self.c2(y)))
        return F.relu(x + y)


BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SQUARES = BOARD_ROWS * BOARD_COLS


def _load_px0_policy_attention_map() -> torch.Tensor:
    repo_root = Path(__file__).resolve().parents[3]
    moves_path = repo_root / "crates" / "engin" / "src" / "px0_policy_moves.txt"
    indices: list[int] = []
    for raw in moves_path.read_text(encoding="utf-8").splitlines():
        move = raw.strip()
        if len(move) != 4:
            continue
        from_file = ord(move[0]) - ord("a")
        from_rank = ord(move[1]) - ord("0")
        to_file = ord(move[2]) - ord("a")
        to_rank = ord(move[3]) - ord("0")
        from_sq = from_rank * BOARD_COLS + from_file
        to_sq = to_rank * BOARD_COLS + to_file
        indices.append(from_sq * BOARD_SQUARES + to_sq)
    if len(indices) != 2062:
        raise ValueError(f"unexpected px0 policy map size: {len(indices)}")
    return torch.tensor(indices, dtype=torch.long)


class AttentionPolicyHead(nn.Module):
    def __init__(self, trunk_channels: int, num_moves: int, *, embed_dim: int) -> None:
        super().__init__()
        if int(embed_dim) <= 0 or int(embed_dim) % 4 != 0:
            raise ValueError(f"embed_dim={embed_dim} 须为正数且能被 4 整除")
        self.embed = nn.Sequential(
            nn.Conv2d(trunk_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.square_bias = nn.Parameter(torch.zeros(BOARD_SQUARES, embed_dim))
        self.encoder_ln1 = nn.LayerNorm(embed_dim)
        self.encoder_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder_ln2 = nn.LayerNorm(embed_dim)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(float(embed_dim))
        policy_map = _load_px0_policy_attention_map()
        if policy_map.numel() != int(num_moves):
            raise ValueError(f"policy map size={policy_map.numel()} != num_moves={num_moves}")
        self.register_buffer("policy_map", policy_map, persistent=False)

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(trunk).flatten(2).transpose(1, 2)
        tokens = tokens + self.square_bias.unsqueeze(0)
        attn_input = self.encoder_ln1(tokens)
        attn_output, _ = self.encoder_attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        tokens = tokens + self.encoder_ffn(self.encoder_ln2(tokens))
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
        return attn.flatten(1).index_select(1, self.policy_map)


class PolicyResNet(nn.Module):
    """输入 `[B, C, 10, 9]`，输出 policy logits，或 `(logits, value_wdl_logits)`。"""

    def __init__(
        self,
        in_planes: int = 124,
        width: int = 128,
        num_blocks: int = 8,
        num_moves: int = 2062,
        *,
        value_head: bool = False,
        value_head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_planes = int(in_planes)
        self.num_moves = int(num_moves)
        self.value_head = bool(value_head)
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(width, se_ratio=2) for _ in range(num_blocks)])
        head_channels = min(32, width)
        self.policy_head = AttentionPolicyHead(width, num_moves, embed_dim=width)
        if self.value_head:
            self.value_head_module = nn.Sequential(
                nn.Conv2d(width, head_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
            )
            self.fc_value = self._make_value_head(
                head_channels * BOARD_ROWS * BOARD_COLS,
                int(max(0, value_head_hidden_dim)),
            )

    @staticmethod
    def _make_value_head(in_dim: int, hidden_dim: int) -> nn.Module:
        if hidden_dim <= 0:
            return nn.Linear(in_dim, 3)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        logits = self.policy_head(h)
        if not self.value_head:
            return logits
        value_features = self.value_head_module(h).flatten(1)
        return logits, self.fc_value(value_features)


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    bad = ~legal_mask
    logits = logits.masked_fill(bad, float("-inf"))
    return F.log_softmax(logits, dim=1)


def policy_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")

    device = logits.device
    batch_size, vocab_size = logits.shape
    logp = masked_log_softmax(logits, legal_mask)
    safe_logp = torch.where(legal_mask, logp, torch.zeros_like(logp))

    if label_smoothing <= 0.0:
        nll = -safe_logp[torch.arange(batch_size, device=device), targets]
    else:
        eps = float(label_smoothing)
        one_hot = F.one_hot(targets, num_classes=vocab_size).float()
        legal_f = legal_mask.float()
        legal_count = legal_mask.sum(dim=1).clamp(min=1)
        only_one = (legal_count == 1).unsqueeze(1)
        denom = (legal_count - 1).clamp(min=1).float().unsqueeze(1)
        others = (legal_f - one_hot).clamp(min=0) / denom
        target_dist = (1.0 - eps) * one_hot + eps * others
        target_dist = torch.where(only_one, one_hot, target_dist)
        nll = -(target_dist * safe_logp).sum(dim=1)

    if sample_weight is not None:
        if sample_weight.shape != (batch_size,):
            raise ValueError("sample_weight 形状须为 [B]")
        w = sample_weight.to(device=device, dtype=nll.dtype)
        nll = nll * w
        if reduction == "mean":
            return nll.sum() / w.sum().clamp(min=1e-8)
        return nll

    if reduction == "mean":
        return nll.mean()
    return nll


def soft_policy_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    reduction: str = "mean",
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")

    logp = masked_log_softmax(logits, legal_mask)
    safe_logp = torch.where(legal_mask, logp, torch.zeros_like(logp))
    safe_target = torch.where(legal_mask, target_probs, torch.zeros_like(target_probs))
    target_sum = safe_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    safe_target = safe_target / target_sum
    loss = -(safe_target * safe_logp).sum(dim=1)

    if sample_weight is not None:
        if sample_weight.shape != loss.shape:
            raise ValueError("sample_weight 形状须为 [B]")
        loss = loss * sample_weight
        if reduction == "mean":
            return loss.sum() / sample_weight.sum().clamp(min=1e-8)
        return loss

    if reduction == "mean":
        return loss.mean()
    return loss


def mix_wdl_targets(
    winner_wdl: torch.Tensor,
    search_wdl: torch.Tensor,
    *,
    q_ratio: float,
) -> torch.Tensor:
    q_ratio = float(q_ratio)
    if q_ratio <= 0.0:
        return winner_wdl
    if q_ratio >= 1.0:
        return search_wdl
    return q_ratio * search_wdl + (1.0 - q_ratio) * winner_wdl


def wdl_probs_to_q(wdl_probs: torch.Tensor) -> torch.Tensor:
    return wdl_probs[..., 0] - wdl_probs[..., 2]


def wdl_logits_to_q(wdl_logits: torch.Tensor) -> torch.Tensor:
    return wdl_probs_to_q(F.softmax(wdl_logits, dim=-1))


def value_wdl_cross_entropy(
    pred_value: torch.Tensor,
    tgt_wdl: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    if pred_value.ndim != 2 or pred_value.shape[1] != 3:
        raise ValueError("pred_value 形状须为 [B,3]")
    if tgt_wdl.shape != pred_value.shape:
        raise ValueError("tgt_wdl 形状须与 pred_value 一致 [B,3]")
    target_sum = tgt_wdl.sum(dim=1, keepdim=True).clamp(min=1e-8)
    safe_target = tgt_wdl / target_sum
    logp = F.log_softmax(pred_value, dim=1)
    err = -(safe_target * logp).sum(dim=1)
    weight = torch.ones_like(err)
    if sample_weight is not None:
        if sample_weight.shape != err.shape:
            raise ValueError("sample_weight 形状须为 [B]")
        weight = weight * sample_weight
    err = err * weight
    if reduction == "mean":
        return err.sum() / weight.sum().clamp(min=1e-8)
    return err


def value_q_mse(
    pred_value: torch.Tensor,
    tgt_wdl: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pred_q = wdl_logits_to_q(pred_value)
    tgt_q = wdl_probs_to_q(tgt_wdl)
    err = (pred_q - tgt_q) ** 2
    if sample_weight is not None:
        if sample_weight.shape != err.shape:
            raise ValueError("sample_weight 形状须为 [B]")
        err = err * sample_weight
        if reduction == "mean":
            return err.sum() / sample_weight.sum().clamp(min=1e-8)
        return err
    if reduction == "mean":
        return err.mean()
    return err
