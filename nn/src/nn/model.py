"""小型 ResNet backbone + policy/value 头。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyResNet(nn.Module):
    """输入 `[B, 15, 10, 9]`，输出 policy logits，或 `(logits, value_logit)`。"""

    def __init__(
        self,
        in_planes: int = 15,
        width: int = 128,
        num_blocks: int = 8,
        num_moves: int = 4096,
        *,
        value_head: bool = False,
        value_head_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.value_head = bool(value_head)
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_moves)
        if self.value_head:
            self.fc_value = self._make_scalar_head(
                width,
                int(max(0, value_head_hidden_dim)),
            )

    @staticmethod
    def _make_scalar_head(in_dim: int, hidden_dim: int) -> nn.Module:
        if hidden_dim <= 0:
            return nn.Linear(in_dim, 1)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        h = self.pool(h).flatten(1)
        logits = self.fc(h)
        if not self.value_head:
            return logits
        return logits, self.fc_value(h).squeeze(-1)


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
    safe_target = torch.where(legal_mask, target_probs, torch.zeros_like(target_probs))
    target_sum = safe_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    safe_target = safe_target / target_sum
    loss = -(safe_target * logp).sum(dim=1)

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


def value_head_tanh_mse(
    pred_value: torch.Tensor,
    tgt_value: torch.Tensor,
    *,
    target_weight_alpha: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pred = torch.tanh(pred_value)
    err = (pred - tgt_value) ** 2
    weight = torch.ones_like(err)
    if target_weight_alpha != 0.0:
        weight = weight * (1.0 + float(target_weight_alpha) * tgt_value.abs())
    if sample_weight is not None:
        if sample_weight.shape != pred_value.shape:
            raise ValueError("sample_weight 形状须与 pred_value 一致 [B]")
        weight = weight * sample_weight
    err = err * weight
    if reduction == "mean":
        return err.sum() / weight.sum().clamp(min=1e-8)
    return err
