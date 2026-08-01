"""X7 v2 CNN: pre-activation bottleneck ResNet with Global Broadcast."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_common import BATCH_NORM_MOMENTUM, CNN_TRUNK_KIND, POLICY_PLANES, _build_conv_policy_index


def _batch_norm(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, momentum=BATCH_NORM_MOMENTUM)


def _mean_max_pool(features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (F.adaptive_avg_pool2d(features, 1).flatten(1), F.adaptive_max_pool2d(features, 1).flatten(1)), dim=1
    )


class PreActBottleneck(nn.Module):
    def __init__(self, channels: int, *, bottleneck_channels: int) -> None:
        super().__init__()
        self.bn1, self.conv1 = _batch_norm(channels), nn.Conv2d(channels, bottleneck_channels, 1, bias=False)
        self.bn2, self.conv2 = (
            _batch_norm(bottleneck_channels),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False),
        )
        self.bn3, self.conv3 = (
            _batch_norm(bottleneck_channels),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False),
        )
        self.bn4, self.conv4 = _batch_norm(bottleneck_channels), nn.Conv2d(bottleneck_channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.bn1(x), inplace=True))
        y = self.conv2(F.silu(self.bn2(y), inplace=True))
        y = self.conv3(F.silu(self.bn3(y), inplace=True))
        return x + self.conv4(F.silu(self.bn4(y), inplace=True))


class GlobalBroadcast(nn.Module):
    def __init__(self, channels: int, *, gpool_channels: int) -> None:
        super().__init__()
        self.pre_bn = _batch_norm(channels)
        self.gpool_conv = nn.Conv2d(channels, gpool_channels, 3, padding=1, bias=False)
        self.gpool_bn = _batch_norm(gpool_channels)
        self.gpool_to_bias = nn.Linear(gpool_channels * 2, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.silu(self.gpool_bn(self.gpool_conv(F.silu(self.pre_bn(x), inplace=True))), inplace=True)
        return x + self.gpool_to_bias(_mean_max_pool(pooled)).unsqueeze(-1).unsqueeze(-1)


class PolicyHead(nn.Module):
    def __init__(self, channels: int, *, policy_planes: int = POLICY_PLANES) -> None:
        super().__init__()
        self.pre_bn = _batch_norm(channels)
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gpool_conv = nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False)
        self.gpool_bn = _batch_norm(channels // 2)
        self.gpool_to_bias = nn.Linear(channels, channels, bias=False)
        self.mid_bn = _batch_norm(channels)
        self.out_conv = nn.Conv2d(channels, policy_planes, 3, padding=1, bias=True)
        self.register_buffer("policy_index", _build_conv_policy_index(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = F.silu(self.pre_bn(x), inplace=True)
        spatial = self.spatial_conv(trunk)
        pooled = F.silu(self.gpool_bn(self.gpool_conv(trunk)), inplace=True)
        bias = self.gpool_to_bias(_mean_max_pool(pooled)).unsqueeze(-1).unsqueeze(-1)
        return (
            self.out_conv(F.silu(self.mid_bn(spatial + bias), inplace=True))
            .flatten(1)
            .index_select(1, self.policy_index)
        )


class ValueAuxHead(nn.Module):
    def __init__(self, channels: int, *, hidden_dim: int, root_value_head: bool = False) -> None:
        super().__init__()
        self.pre_bn, self.conv = _batch_norm(channels), nn.Conv2d(channels, hidden_dim, 1, bias=False)
        self.conv_bn, self.fc = _batch_norm(hidden_dim), nn.Linear(hidden_dim * 2, hidden_dim)
        self.value_out, self.moves_left_out = nn.Linear(hidden_dim, 3), nn.Linear(hidden_dim, 1)
        self.root_value_out = nn.Linear(hidden_dim, 3) if root_value_head else None

    def forward(self, x: torch.Tensor, *, include_root_value: bool = True) -> tuple[torch.Tensor, ...]:
        features = F.silu(self.conv_bn(self.conv(F.silu(self.pre_bn(x), inplace=True))), inplace=True)
        features = F.silu(self.fc(_mean_max_pool(features)), inplace=True)
        outputs: tuple[torch.Tensor, ...] = (
            self.value_out(features),
            F.relu(self.moves_left_out(features), inplace=True),
        )
        return (
            outputs
            if self.root_value_out is None or not include_root_value
            else outputs + (self.root_value_out(features),)
        )


class KnowledgeResNet(nn.Module):
    """v2 formal contract: `124x10x9 -> 2062 + WDL + moves-left`."""

    def __init__(
        self,
        in_planes: int = 124,
        width: int = 384,
        num_blocks: int = 15,
        num_moves: int = 2062,
        *,
        bottleneck_channels: int | None = None,
        value_head: bool = False,
        moves_left_head: bool = False,
        auxiliary_heads: bool = False,
        trunk_kind: str = CNN_TRUNK_KIND,
    ) -> None:
        super().__init__()
        if width < 4 or width % 2 or num_moves != 2062 or num_blocks < 3:
            raise ValueError("invalid v2 model dimensions")
        if auxiliary_heads and (not value_head or not moves_left_head):
            raise ValueError("auxiliary_heads 须与 value_head、moves_left_head 一起启用")
        self.in_planes, self.num_moves, self.num_blocks = in_planes, num_moves, num_blocks
        self.value_head, self.moves_left_head, self.auxiliary_heads, self.trunk_kind = (
            value_head,
            moves_left_head,
            auxiliary_heads,
            trunk_kind,
        )
        bottleneck_channels = width // 2 if bottleneck_channels is None else bottleneck_channels
        if bottleneck_channels < 1:
            raise ValueError("bottleneck_channels 须为正整数")
        self.bottleneck_channels = bottleneck_channels
        n1, n2 = num_blocks // 3, num_blocks // 3
        n3 = num_blocks - n1 - n2
        self.stem = nn.Conv2d(in_planes, width, 3, padding=1, bias=False)
        self.stage1 = nn.Sequential(
            *(PreActBottleneck(width, bottleneck_channels=bottleneck_channels) for _ in range(n1))
        )
        self.broadcast4 = GlobalBroadcast(width, gpool_channels=width // 2)
        self.stage2 = nn.Sequential(
            *(PreActBottleneck(width, bottleneck_channels=bottleneck_channels) for _ in range(n2))
        )
        self.broadcast8 = GlobalBroadcast(width, gpool_channels=width // 2)
        self.stage3 = nn.Sequential(
            *(PreActBottleneck(width, bottleneck_channels=bottleneck_channels) for _ in range(n3))
        )
        self.trunk_bn = _batch_norm(width)
        self.policy_head = PolicyHead(width)
        self.soft_policy_head = PolicyHead(width) if auxiliary_heads else None
        if value_head or moves_left_head:
            self.value_aux_head_module = ValueAuxHead(width, hidden_dim=max(64, width), root_value_head=auxiliary_heads)

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.broadcast4(self.stage1(self.stem(x)))
        x = self.broadcast8(self.stage2(x))
        return F.silu(self.trunk_bn(self.stage3(x)), inplace=True)

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
