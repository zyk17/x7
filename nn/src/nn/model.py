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
    Output: 视 ``aux_heads`` / ``value_head`` 组合返回 logits 或元组；
    辅助头为未经 sigmoid 的 logit；value 为未经 tanh 的标量 logit（训练对 ``tanh(value)`` 与目标做 MSE）。
    """

    def __init__(
        self,
        in_planes: int = 15,
        width: int = 128,
        num_blocks: int = 8,
        num_moves: int = 4096,
        *,
        aux_heads: bool = False,
        value_head: bool = False,
        aux_head_hidden_dim: int = 0,
        value_head_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.aux_heads = bool(aux_heads)
        self.value_head = bool(value_head)
        self.aux_head_hidden_dim = int(max(0, aux_head_hidden_dim))
        self.value_head_hidden_dim = int(max(0, value_head_hidden_dim))
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_moves)
        if self.aux_heads:
            self.fc_attack = self._make_scalar_head(
                width, self.aux_head_hidden_dim
            )
            self.fc_danger = self._make_scalar_head(
                width, self.aux_head_hidden_dim
            )
            self.fc_tactical = self._make_scalar_head(
                width, self.aux_head_hidden_dim
            )
        if self.value_head:
            self.fc_value = self._make_scalar_head(
                width, self.value_head_hidden_dim
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

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        h = self.blocks(self.stem(x))
        h = self.pool(h).flatten(1)
        logits = self.fc(h)
        if not self.aux_heads and not self.value_head:
            return logits
        out: list[torch.Tensor] = [logits]
        if self.aux_heads:
            out.append(self.fc_attack(h).squeeze(-1))
            out.append(self.fc_danger(h).squeeze(-1))
            out.append(self.fc_tactical(h).squeeze(-1))
        if self.value_head:
            out.append(self.fc_value(h).squeeze(-1))
        return tuple(out) if len(out) > 1 else logits


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
    """辅助头：sigmoid 后与 [0,1] 伪标签做逐元素 MSE（遗留；主训练见 ``aux_heads_sigmoid_bce``）。"""
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


def aux_heads_sigmoid_bce(
    pred_attack: torch.Tensor,
    pred_danger: torch.Tensor,
    pred_tactical: torch.Tensor,
    tgt_attack: torch.Tensor,
    tgt_danger: torch.Tensor,
    tgt_tactical: torch.Tensor,
    *,
    attack_scale: float = 1.0,
    pos_weight_attack: float = 1.0,
    pos_weight_danger: float = 1.0,
    pos_weight_tactical: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """辅助头：``BCEWithLogits``，目标为 [0,1] 事件型伪标签；``attack_scale`` 可压低 attack 项权重。"""
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")

    def _weighted_bce(
        pred: torch.Tensor,
        tgt: torch.Tensor,
        *,
        pos_weight: float,
    ) -> torch.Tensor:
        pw = torch.as_tensor(
            float(max(pos_weight, 1e-6)),
            device=pred.device,
            dtype=pred.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            pred, tgt, reduction="none", pos_weight=pw
        )

    eps = 1e-6
    ta = tgt_attack.clamp(eps, 1.0 - eps)
    td = tgt_danger.clamp(eps, 1.0 - eps)
    tt = tgt_tactical.clamp(eps, 1.0 - eps)
    la = _weighted_bce(pred_attack, ta, pos_weight=pos_weight_attack)
    ld = _weighted_bce(pred_danger, td, pos_weight=pos_weight_danger)
    lt = _weighted_bce(pred_tactical, tt, pos_weight=pos_weight_tactical)
    denom = 2.0 + float(attack_scale)
    err = (float(attack_scale) * la + ld + lt) / denom
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


def value_head_tanh_mse(
    pred_value: torch.Tensor,
    tgt_value: torch.Tensor,
    *,
    target_weight_alpha: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """``tanh(logit)`` 与 [-1,1] 目标（结局监督 value）的 MSE。"""
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pv = torch.tanh(pred_value)
    err = (pv - tgt_value) ** 2
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
