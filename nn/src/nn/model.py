"""小型 ResNet backbone + 全局池化 policy 头（输出固定词表 logits）。"""

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
    """
    Input: [B, in_planes, 10, 9]
    Output: ``aux_heads=False`` 时 ``[B, num_moves]`` logits；
    ``aux_heads=True`` 时返回 ``(logits, attack, danger, tactical)``，后三者为未经 sigmoid 的标量 logit。
    """

    def __init__(
        self,
        in_planes: int = 15,
        width: int = 128,
        num_blocks: int = 8,
        num_moves: int = 4096,
        *,
        aux_heads: bool = False,
    ) -> None:
        super().__init__()
        self.aux_heads = bool(aux_heads)
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_moves)
        if self.aux_heads:
            self.fc_attack = nn.Linear(width, 1)
            self.fc_danger = nn.Linear(width, 1)
            self.fc_tactical = nn.Linear(width, 1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        h = self.pool(h).flatten(1)
        logits = self.fc(h)
        if not self.aux_heads:
            return logits
        a = self.fc_attack(h).squeeze(-1)
        d = self.fc_danger(h).squeeze(-1)
        t = self.fc_tactical(h).squeeze(-1)
        return logits, a, d, t


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """legal_mask: [B, V] bool，True 为合法；非法位置 logits 置为 -inf 再 log_softmax。"""
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
    """
    对合法子集做 softmax 后的交叉熵。
    label_smoothing>0 时在合法着上平滑；sample_weight 为 [B] 时在 reduction 前逐样本加权。
    """
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")

    device = logits.device
    B, V = logits.shape
    logp = masked_log_softmax(logits, legal_mask)
    safe_logp = torch.where(legal_mask, logp, torch.zeros_like(logp))

    if label_smoothing <= 0.0:
        nll = -safe_logp[torch.arange(B, device=device), targets]
    else:
        eps = float(label_smoothing)
        one_hot = F.one_hot(targets, num_classes=V).float()
        K = legal_mask.sum(dim=1).clamp(min=1)
        legal_f = legal_mask.float()
        K1 = (K == 1).unsqueeze(1)
        denom = (K - 1).clamp(min=1).float().unsqueeze(1)
        others = (legal_f - one_hot).clamp(min=0) / denom
        q = (1.0 - eps) * one_hot + eps * others
        q = torch.where(K1, one_hot, q)
        nll = -(q * safe_logp).sum(dim=1)

    if sample_weight is not None:
        if sample_weight.shape != (B,):
            raise ValueError("sample_weight 形状须为 [B]")
        w = sample_weight.to(device=device, dtype=nll.dtype)
        nll = nll * w
        if reduction == "mean":
            return nll.sum() / w.sum().clamp(min=1e-8)
        return nll

    if reduction == "mean":
        return nll.mean()
    return nll


def aux_heads_sigmoid_mse(
    pred_attack: torch.Tensor,
    pred_danger: torch.Tensor,
    pred_tactical: torch.Tensor,
    tgt_attack: torch.Tensor,
    tgt_danger: torch.Tensor,
    tgt_tactical: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """辅助头：sigmoid 后与 [0,1] 伪标签做逐元素 MSE。"""
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pa = torch.sigmoid(pred_attack)
    pd = torch.sigmoid(pred_danger)
    pt = torch.sigmoid(pred_tactical)
    err = (pa - tgt_attack) ** 2 + (pd - tgt_danger) ** 2 + (pt - tgt_tactical) ** 2
    err = err / 3.0
    if sample_weight is not None:
        if sample_weight.shape != pred_attack.shape:
            raise ValueError("sample_weight 形状须与 pred_* 一致 [B]")
        err = err * sample_weight
        if reduction == "mean":
            return err.sum() / sample_weight.sum().clamp(min=1e-8)
        return err
    if reduction == "mean":
        return err.mean()
    return err
