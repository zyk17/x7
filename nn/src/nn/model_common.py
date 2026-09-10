"""Shared Xiangqi NN contract, policy maps, and training losses."""

from __future__ import annotations

from importlib.resources import files

import torch
import torch.nn.functional as F

BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_SQUARES = BOARD_ROWS * BOARD_COLS


def _square_to_index(square: str) -> tuple[int, int]:
    return ord(square[0]) - ord("a"), ord(square[1]) - ord("0")


def _load_move_vocab() -> list[str]:
    moves = [
        line.strip()
        for line in files("nn").joinpath("px0_policy_moves.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(moves) != 2062:
        raise ValueError(f"unexpected move vocab size: {len(moves)}")
    return moves


def _build_move_pair_index() -> torch.Tensor:
    pairs: list[int] = []
    for move in _load_move_vocab():
        start_file, start_rank = _square_to_index(move[:2])
        end_file, end_rank = _square_to_index(move[2:])
        pairs.append((start_rank * BOARD_COLS + start_file) * BOARD_SQUARES + end_rank * BOARD_COLS + end_file)
    return torch.tensor(pairs, dtype=torch.long)


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~legal_mask, float("-inf")), dim=1)


def soften_policy_targets(
    target_probs: torch.Tensor, legal_mask: torch.Tensor, *, temperature: float = 4.0
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature 须为正数")
    target = torch.where(legal_mask, target_probs.clamp_min(0.0), torch.zeros_like(target_probs))
    softened = target.pow(1.0 / float(temperature))
    return softened / softened.sum(dim=1, keepdim=True).clamp(min=1e-8)


def soft_policy_cross_entropy(
    logits: torch.Tensor, target_probs: torch.Tensor, legal_mask: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    safe_logp = torch.where(legal_mask, masked_log_softmax(logits, legal_mask), torch.zeros_like(logits))
    target = torch.where(legal_mask, target_probs, torch.zeros_like(target_probs))
    loss = -(target / target.sum(dim=1, keepdim=True).clamp(min=1e-8) * safe_logp).sum(dim=1)
    return loss.mean() if reduction == "mean" else loss


def wdl_probs_to_q(wdl_probs: torch.Tensor) -> torch.Tensor:
    return wdl_probs[..., 0] - wdl_probs[..., 2]


def wdl_logits_to_q(wdl_logits: torch.Tensor) -> torch.Tensor:
    return wdl_probs_to_q(F.softmax(wdl_logits, dim=-1))


def value_wdl_cross_entropy(
    pred_value: torch.Tensor, tgt_wdl: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    if (
        reduction not in ("mean", "none")
        or pred_value.ndim != 2
        or pred_value.shape[1] != 3
        or tgt_wdl.shape != pred_value.shape
    ):
        raise ValueError("WDL shapes/reduction invalid")
    loss = -(tgt_wdl / tgt_wdl.sum(dim=1, keepdim=True).clamp(min=1e-8) * F.log_softmax(pred_value, dim=1)).sum(dim=1)
    return loss.mean() if reduction == "mean" else loss


def value_q_mse_from_wdl(pred_value: torch.Tensor, tgt_q: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pred_q = wdl_logits_to_q(pred_value)
    target = tgt_q.squeeze(1) if tgt_q.ndim == 2 and tgt_q.shape[1] == 1 else tgt_q
    if target.shape != pred_q.shape:
        raise ValueError("tgt_q 形状须为 [B] 或 [B,1]")
    loss = (pred_q - target).square()
    return loss.mean() if reduction == "mean" else loss


def moves_left_loss(
    pred_moves_left: torch.Tensor, tgt_plies_left: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    if reduction not in ("mean", "none"):
        raise ValueError("reduction 须为 mean 或 none")
    pred = (
        pred_moves_left.squeeze(1) if pred_moves_left.ndim == 2 and pred_moves_left.shape[1] == 1 else pred_moves_left
    )
    target = tgt_plies_left.squeeze(1) if tgt_plies_left.ndim == 2 and tgt_plies_left.shape[1] == 1 else tgt_plies_left
    if pred.shape != target.shape:
        raise ValueError("moves_left 形状须匹配")
    loss = F.huber_loss(pred / 20.0, target / 20.0, delta=0.5, reduction="none")
    return loss.mean() if reduction == "mean" else loss
