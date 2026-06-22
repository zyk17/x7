"""XRSH Dataset 样本 dict 与 DataLoader collate。"""

from __future__ import annotations

from typing import Any

import torch

from nn.board_compact import compact_boards_to_torch_planes

# 单样本 / batch 共用字段名（batch 维在 collate 后 stack）
SAMPLE_BOARD = "board"
SAMPLE_MASK = "mask"
SAMPLE_TARGET = "target"
SAMPLE_WEIGHT = "weight"
SAMPLE_T_VAL = "t_val"
SAMPLE_SEARCH_VISITS = "search_visits"
SAMPLE_VISIT_TARGET = "visit_target"
SAMPLE_PLY = "ply"
SAMPLE_SRC_ID = "src_id"
SAMPLE_BOARD90 = "_board90"
SAMPLE_STM = "_stm"
SAMPLE_LEGAL_IDX = "_legal_idx"
SAMPLE_SEARCH_COUNTS = "_search_counts"
SAMPLE_VOCAB_SIZE = "_vocab_size"

def _stack_index_lists(
    batch: list[dict[str, torch.Tensor]],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = [item[key] for item in batch]
    lengths = torch.tensor([int(part.numel()) for part in parts], dtype=torch.long)
    if lengths.sum().item() == 0:
        return torch.zeros(0, dtype=torch.long), lengths
    return torch.cat(parts, dim=0), lengths


def collate_xrsh_samples(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("空 batch")

    boards90 = torch.stack([item[SAMPLE_BOARD90] for item in batch], dim=0)
    stms = torch.stack([item[SAMPLE_STM] for item in batch], dim=0)
    legal_flat, legal_lengths = _stack_index_lists(batch, SAMPLE_LEGAL_IDX)

    out: dict[str, torch.Tensor] = {
        SAMPLE_BOARD: compact_boards_to_torch_planes(boards90, stms),
        SAMPLE_TARGET: torch.stack([item[SAMPLE_TARGET] for item in batch], dim=0),
        SAMPLE_WEIGHT: torch.stack([item[SAMPLE_WEIGHT] for item in batch], dim=0),
    }

    width = int(batch[0][SAMPLE_VOCAB_SIZE].item())
    mask = torch.zeros((len(batch), width), dtype=torch.bool)
    if legal_flat.numel():
        batch_index = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.long),
            legal_lengths,
        )
        mask[batch_index, legal_flat.long()] = True
    out[SAMPLE_MASK] = mask

    if SAMPLE_T_VAL in batch[0]:
        out[SAMPLE_T_VAL] = torch.stack([item[SAMPLE_T_VAL] for item in batch], dim=0)
        out[SAMPLE_SEARCH_VISITS] = torch.stack(
            [item[SAMPLE_SEARCH_VISITS] for item in batch],
            dim=0,
        )

    if SAMPLE_SEARCH_COUNTS in batch[0]:
        visit_target = torch.zeros((len(batch), width), dtype=torch.float32)
        counts_flat, count_lengths = _stack_index_lists(batch, SAMPLE_SEARCH_COUNTS)
        if not torch.equal(legal_lengths, count_lengths):
            raise ValueError("legal_idx 与 search_counts 批量长度不一致")
        if counts_flat.numel():
            batch_index = torch.repeat_interleave(
                torch.arange(len(batch), dtype=torch.long),
                legal_lengths,
            )
            visit_target[batch_index, legal_flat.long()] = counts_flat.to(torch.float32)
            totals = visit_target.sum(dim=1, keepdim=True).clamp_min_(1.0)
            visit_target /= totals
        out[SAMPLE_VISIT_TARGET] = visit_target

    if SAMPLE_PLY in batch[0]:
        out[SAMPLE_PLY] = torch.stack([item[SAMPLE_PLY] for item in batch], dim=0)
        out[SAMPLE_SRC_ID] = torch.stack([item[SAMPLE_SRC_ID] for item in batch], dim=0)

    return out


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    pin = device.type == "cuda"
    return {k: v.to(device, non_blocking=pin) for k, v in batch.items()}


def batch_field_names(
    *,
    with_value_labels: bool,
    with_search_labels: bool,
    with_row_meta: bool,
) -> tuple[str, ...]:
    names = [SAMPLE_BOARD, SAMPLE_MASK, SAMPLE_TARGET, SAMPLE_WEIGHT]
    if with_value_labels:
        names.extend([SAMPLE_T_VAL, SAMPLE_SEARCH_VISITS])
    if with_search_labels:
        names.append(SAMPLE_VISIT_TARGET)
    if with_row_meta:
        names.extend([SAMPLE_PLY, SAMPLE_SRC_ID])
    return tuple(names)
