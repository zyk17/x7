"""纯 CNN 的 KataGo 风格 trunk + px0 卷积 policy + WDL value。"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SQUARES = BOARD_ROWS * BOARD_COLS
POLICY_PLANES = 52

FILES = "abcdefghi"
RANKS = "0123456789"
FILE_TO_INDEX = {ch: i for i, ch in enumerate(FILES)}
RANK_TO_INDEX = {ch: i for i, ch in enumerate(RANKS)}


def _index_to_square(file_idx: int, rank_idx: int) -> str:
    return FILES[file_idx] + RANKS[rank_idx]


def _square_to_index(square: str) -> tuple[int, int]:
    return FILE_TO_INDEX[square[0]], RANK_TO_INDEX[square[1]]


def _valid_coord(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < BOARD_COLS and 0 <= rank_idx < BOARD_ROWS


def _slide_move(start: str, direction: tuple[int, int], steps: int) -> str | None:
    file_idx, rank_idx = _square_to_index(start)
    file_idx += direction[0] * steps
    rank_idx += direction[1] * steps
    if not _valid_coord(file_idx, rank_idx):
        return None
    return _index_to_square(file_idx, rank_idx)


def _knight_move(start: str, direction: tuple[int, int]) -> str | None:
    return _slide_move(start, direction, 1)


def _load_px0_policy_moves() -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    moves_path = repo_root / "crates" / "engin" / "src" / "px0_policy_moves.txt"
    moves = [line.strip() for line in moves_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(moves) != 2062:
        raise ValueError(f"unexpected px0 policy size: {len(moves)}")
    return moves


def _build_conv_policy_index() -> torch.Tensor:
    policy_moves = _load_px0_policy_moves()
    move_to_policy_idx = {move: idx for idx, move in enumerate(policy_moves)}

    conv_moves: list[str | None] = []
    rook_dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    knight_dirs = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    bishop_advisor_dirs = ((1, 1), (2, 2), (1, -1), (2, -2), (-1, -1), (-2, -2), (-1, 1), (-2, 2))

    for dx, dy in rook_dirs:
        for steps in range(1, 10):
            for rank in RANKS:
                for file_ in FILES:
                    start = file_ + rank
                    end = _slide_move(start, (dx, dy), steps)
                    conv_moves.append(None if end is None else start + end)

    for dx, dy in knight_dirs:
        for rank in RANKS:
            for file_ in FILES:
                start = file_ + rank
                end = _knight_move(start, (dx, dy))
                conv_moves.append(None if end is None else start + end)

    for dx, dy in bishop_advisor_dirs:
        for rank in RANKS:
            for file_ in FILES:
                start = file_ + rank
                end = _knight_move(start, (dx, dy))
                conv_moves.append(None if end is None else start + end)

    if len(conv_moves) != POLICY_PLANES * BOARD_SQUARES:
        raise ValueError(f"unexpected conv move table size: {len(conv_moves)}")

    policy_to_conv = [-1] * len(policy_moves)
    for flat_idx, move in enumerate(conv_moves):
        if move is None:
            continue
        policy_idx = move_to_policy_idx.get(move)
        if policy_idx is None:
            continue
        policy_to_conv[policy_idx] = flat_idx

    if any(idx < 0 for idx in policy_to_conv):
        missing = [policy_moves[i] for i, idx in enumerate(policy_to_conv) if idx < 0][:8]
        raise ValueError(f"conv policy map missing moves: {missing}")
    return torch.tensor(policy_to_conv, dtype=torch.long)


class PreActBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        return x + y


class GlobalPoolingResidualBlock(nn.Module):
    def __init__(self, channels: int, *, gpool_channels: int) -> None:
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(channels)
        self.regular_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gpool_conv = nn.Conv2d(channels, gpool_channels, kernel_size=1, bias=False)
        self.gpool_bn = nn.BatchNorm2d(gpool_channels)
        self.gpool_to_bias = nn.Linear(gpool_channels, channels)
        self.mid_bn = nn.BatchNorm2d(channels)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = F.relu(self.pre_bn(x), inplace=True)
        spatial = self.regular_conv(trunk)

        pooled = self.gpool_conv(trunk)
        pooled = F.relu(self.gpool_bn(pooled), inplace=True)
        pooled = F.adaptive_avg_pool2d(pooled, output_size=1).flatten(1)
        bias = self.gpool_to_bias(pooled).unsqueeze(-1).unsqueeze(-1)

        fused = spatial + bias
        fused = self.final_conv(F.relu(self.mid_bn(fused), inplace=True))
        return x + fused


class PolicyHead(nn.Module):
    def __init__(self, channels: int, *, policy_planes: int = POLICY_PLANES) -> None:
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(channels)
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gpool_conv = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.gpool_bn = nn.BatchNorm2d(channels // 2)
        self.gpool_to_bias = nn.Linear(channels // 2, channels)
        self.mid_bn = nn.BatchNorm2d(channels)
        self.out_conv = nn.Conv2d(channels, policy_planes, kernel_size=3, padding=1, bias=True)
        self.register_buffer("policy_index", _build_conv_policy_index(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = F.relu(self.pre_bn(x), inplace=True)
        spatial = self.spatial_conv(trunk)

        pooled = self.gpool_conv(trunk)
        pooled = F.relu(self.gpool_bn(pooled), inplace=True)
        pooled = F.adaptive_avg_pool2d(pooled, output_size=1).flatten(1)
        bias = self.gpool_to_bias(pooled).unsqueeze(-1).unsqueeze(-1)

        logits_2d = self.out_conv(F.relu(self.mid_bn(spatial + bias), inplace=True))
        flat = logits_2d.flatten(1)
        return flat.index_select(1, self.policy_index)


class ValueHead(nn.Module):
    def __init__(self, channels: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.conv_bn = nn.BatchNorm2d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = F.relu(self.pre_bn(x), inplace=True)
        features = F.relu(self.conv_bn(self.conv(trunk)), inplace=True)
        avg = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        maxv = F.adaptive_max_pool2d(features, output_size=1).flatten(1)
        fused = torch.cat([avg, maxv], dim=1)
        fused = F.relu(self.fc1(fused), inplace=True)
        return self.fc2(fused)


class MovesLeftHead(nn.Module):
    def __init__(self, channels: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, hidden_dim // 2, kernel_size=1, bias=False)
        self.conv_bn = nn.BatchNorm2d(hidden_dim // 2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = F.relu(self.pre_bn(x), inplace=True)
        features = F.relu(self.conv_bn(self.conv(trunk)), inplace=True)
        avg = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        maxv = F.adaptive_max_pool2d(features, output_size=1).flatten(1)
        fused = torch.cat([avg, maxv], dim=1)
        fused = F.relu(self.fc1(fused), inplace=True)
        return self.fc2(fused)


class PolicyResNet(nn.Module):
    """输入 `[B,124,10,9]`，输出 policy，或 `(policy, value[, moves_left])`。"""

    def __init__(
        self,
        in_planes: int = 124,
        width: int = 128,
        num_blocks: int = 8,
        num_moves: int = 2062,
        *,
        value_head: bool = False,
        moves_left_head: bool = False,
        trunk_kind: str = "katago_cnn_v1",
    ) -> None:
        super().__init__()
        if int(width) <= 0 or int(width) % 2 != 0:
            raise ValueError(f"width={width} 须为正偶数")
        if int(num_moves) != 2062:
            raise ValueError(f"当前只支持 px0 2062 policy，got num_moves={num_moves}")
        self.in_planes = int(in_planes)
        self.num_moves = int(num_moves)
        self.value_head = bool(value_head)
        self.moves_left_head = bool(moves_left_head)
        self.trunk_kind = str(trunk_kind)

        self.stem = nn.Conv2d(in_planes, width, kernel_size=3, padding=1, bias=False)
        gpool_channels = max(16, width // 2)
        blocks: list[nn.Module] = []
        for block_idx in range(int(num_blocks)):
            if (block_idx + 1) % 3 == 0:
                blocks.append(GlobalPoolingResidualBlock(width, gpool_channels=gpool_channels))
            else:
                blocks.append(PreActBlock(width))
        self.blocks = nn.Sequential(*blocks)
        self.trunk_bn = nn.BatchNorm2d(width)

        self.policy_head = PolicyHead(width)
        if self.value_head:
            self.value_head_module = ValueHead(width, hidden_dim=max(64, width))
        if self.moves_left_head:
            self.moves_left_head_module = MovesLeftHead(width, hidden_dim=max(64, width))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trunk = self.blocks(self.stem(x))
        trunk = F.relu(self.trunk_bn(trunk), inplace=True)
        logits = self.policy_head(trunk)

        outputs: list[torch.Tensor] = [logits]
        if self.value_head:
            outputs.append(self.value_head_module(trunk))
        if self.moves_left_head:
            outputs.append(self.moves_left_head_module(trunk))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)  # type: ignore[return-value]


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~legal_mask, float("-inf"))
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
        nll = nll * sample_weight.to(device=device, dtype=nll.dtype)
        if reduction == "mean":
            return nll.sum() / sample_weight.sum().clamp(min=1e-8)
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
    safe_target = safe_target / safe_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
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

    safe_target = tgt_wdl / tgt_wdl.sum(dim=1, keepdim=True).clamp(min=1e-8)
    err = -(safe_target * F.log_softmax(pred_value, dim=1)).sum(dim=1)
    weight = torch.ones_like(err)
    if sample_weight is not None:
        if sample_weight.shape != err.shape:
            raise ValueError("sample_weight 形状须为 [B]")
        weight = weight * sample_weight
    err = err * weight
    if reduction == "mean":
        return err.sum() / weight.sum().clamp(min=1e-8)
    return err


def value_q_mse_from_scalar(
    pred_value: torch.Tensor,
    tgt_q: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pred_q = wdl_logits_to_q(pred_value)
    if tgt_q.ndim == 2 and tgt_q.shape[1] == 1:
        tgt_q = tgt_q.squeeze(1)
    if tgt_q.shape != pred_q.shape:
        raise ValueError("tgt_q 形状须为 [B] 或 [B,1]")
    err = (pred_q - tgt_q) ** 2
    if sample_weight is not None:
        if sample_weight.ndim == 2 and sample_weight.shape[1] == 1:
            sample_weight = sample_weight.squeeze(1)
        if sample_weight.shape != err.shape:
            raise ValueError("sample_weight 形状须为 [B] 或 [B,1]")
        err = err * sample_weight
        if reduction == "mean":
            return err.sum() / sample_weight.sum().clamp(min=1e-8)
        return err
    if reduction == "mean":
        return err.mean()
    return err


def normalize_plies_left(plies_left: torch.Tensor, *, scale: float = 256.0) -> torch.Tensor:
    return torch.log1p(plies_left.clamp_min(0.0)) / math.log1p(float(scale))


def moves_left_loss(
    pred_moves_left: torch.Tensor,
    tgt_plies_left: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    if pred_moves_left.ndim == 2 and pred_moves_left.shape[1] == 1:
        pred_moves_left = pred_moves_left.squeeze(1)
    if tgt_plies_left.ndim == 2 and tgt_plies_left.shape[1] == 1:
        tgt_plies_left = tgt_plies_left.squeeze(1)
    if pred_moves_left.shape != tgt_plies_left.shape:
        raise ValueError("moves_left 形状须匹配")
    target = normalize_plies_left(tgt_plies_left)
    err = F.smooth_l1_loss(pred_moves_left, target, reduction="none")
    if sample_weight is not None:
        if sample_weight.ndim == 2 and sample_weight.shape[1] == 1:
            sample_weight = sample_weight.squeeze(1)
        if sample_weight.shape != err.shape:
            raise ValueError("sample_weight 形状须为 [B] 或 [B,1]")
        err = err * sample_weight
        if reduction == "mean":
            return err.sum() / sample_weight.sum().clamp(min=1e-8)
        return err
    if reduction == "mean":
        return err.mean()
    return err


def visits_to_sample_weight(visits: torch.Tensor) -> torch.Tensor:
    if visits.ndim == 2 and visits.shape[1] == 1:
        visits = visits.squeeze(1)
    visits = visits.clamp_min(0.0)
    scale = math.log1p(256.0)
    weight = torch.log1p(visits) / scale
    return weight.clamp_(min=0.25, max=2.0)


def policy_kld_to_weight(policy_kld: torch.Tensor) -> torch.Tensor:
    if policy_kld.ndim == 2 and policy_kld.shape[1] == 1:
        policy_kld = policy_kld.squeeze(1)
    policy_kld = torch.nan_to_num(policy_kld, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    weight = 1.0 + 0.35 * torch.log1p(policy_kld)
    return weight.clamp_(min=1.0, max=2.0)
