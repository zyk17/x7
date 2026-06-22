"""单 epoch 训练/验证步。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn.dataset_batch import (
    SAMPLE_BOARD,
    SAMPLE_MASK,
    SAMPLE_PLY,
    SAMPLE_SEARCH_VISITS,
    SAMPLE_SRC_ID,
    SAMPLE_T_VAL,
    SAMPLE_TARGET,
    SAMPLE_VISIT_TARGET,
    SAMPLE_WEIGHT,
    move_batch_to_device,
)
from nn.metrics import ValMetricsState, format_val_metrics_report
from nn.model import soft_policy_cross_entropy, value_head_tanh_mse

from train_common import LABEL_SMOOTHING
from train_loss import compute_policy_value_loss


def _model_forward(model, boards: torch.Tensor, value_head: bool):
    out = model(boards)
    if value_head:
        return out[0], out[1]
    return out, None


def run_train_epoch(
    model,
    train_loader: DataLoader,
    opt,
    *,
    device: torch.device,
    value_head: bool,
    search_policy_head: bool,
    amp_enabled: bool,
    args,
    epoch: int,
    end_epoch: int,
) -> float:
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((), device=device, dtype=torch.float32)

    for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{end_epoch} train"):
        bt = move_batch_to_device(batch, device)
        boards = bt[SAMPLE_BOARD]
        masks = bt[SAMPLE_MASK]
        targets = bt[SAMPLE_TARGET]
        weights = bt[SAMPLE_WEIGHT]
        t_val = bt.get(SAMPLE_T_VAL) if value_head else None
        search_visits = bt.get(SAMPLE_SEARCH_VISITS) if value_head else None
        visit_target = bt.get(SAMPLE_VISIT_TARGET) if search_policy_head else None

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp_enabled
            else nullcontext()
        )
        with autocast_ctx:
            logits, pred_value = _model_forward(model, boards, value_head)
            loss = compute_policy_value_loss(
                logits,
                pred_value,
                targets=targets,
                masks=masks,
                weights=weights,
                value_head=value_head,
                search_policy_head=search_policy_head,
                t_val=t_val,
                search_visits=search_visits,
                visit_target=visit_target,
                label_smoothing=LABEL_SMOOTHING,
                value_loss_weight=float(args.value_loss_weight),
                value_target_weight_alpha=float(args.value_target_weight_alpha),
                value_min_visits=int(args.value_min_visits),
                search_policy_weight=float(args.search_policy_weight),
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        batch_weight = weights.sum()
        total_loss += loss.detach().to(dtype=torch.float32) * batch_weight.to(dtype=torch.float32)
        weight_sum += batch_weight.to(dtype=torch.float32)

    return float((total_loss / weight_sum.clamp(min=1e-8)).item())


def run_val_epoch(
    model,
    val_loader: DataLoader,
    *,
    device: torch.device,
    value_head: bool,
    search_policy_head: bool,
    amp_enabled: bool,
    args,
    val_ds,
) -> dict[str, Any]:
    model.eval()
    vloss = 0.0
    vweight_sum = 0.0
    vcount = 0
    correct = 0
    value_mse = 0.0
    value_eval_count = 0
    search_ce = 0.0
    val_metrics = ValMetricsState()

    with torch.no_grad():
        for batch in val_loader:
            bv = move_batch_to_device(batch, device)
            boards = bv[SAMPLE_BOARD]
            masks = bv[SAMPLE_MASK]
            targets = bv[SAMPLE_TARGET]
            weights = bv[SAMPLE_WEIGHT]
            plies = bv[SAMPLE_PLY]
            src_ids = bv[SAMPLE_SRC_ID]
            t_val = bv.get(SAMPLE_T_VAL) if value_head else None
            search_visits = bv.get(SAMPLE_SEARCH_VISITS) if value_head else None
            visit_target = bv.get(SAMPLE_VISIT_TARGET) if search_policy_head else None

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits, pred_value = _model_forward(model, boards, value_head)
                loss = compute_policy_value_loss(
                    logits,
                    pred_value,
                    targets=targets,
                    masks=masks,
                    weights=weights,
                    value_head=value_head,
                    search_policy_head=search_policy_head,
                    t_val=t_val,
                    search_visits=search_visits,
                    visit_target=visit_target,
                    label_smoothing=0.0,
                    value_loss_weight=float(args.value_loss_weight),
                    value_target_weight_alpha=float(args.value_target_weight_alpha),
                    value_min_visits=int(args.value_min_visits),
                    search_policy_weight=float(args.search_policy_weight),
                )
                if value_head and pred_value is not None and t_val is not None and search_visits is not None:
                    value_mask = search_visits >= int(args.value_min_visits)
                    if value_mask.any():
                        n_val = int(value_mask.sum().item())
                        masked_pred = pred_value[value_mask]
                        masked_tgt = t_val[value_mask]
                        value_mse += (
                            value_head_tanh_mse(
                                masked_pred,
                                masked_tgt,
                                reduction="mean",
                            ).item()
                            * n_val
                        )
                        value_eval_count += n_val
                if search_policy_head and visit_target is not None:
                    search_ce += (
                        soft_policy_cross_entropy(
                            logits,
                            visit_target,
                            masks,
                            reduction="mean",
                        ).item()
                        * boards.size(0)
                    )

            batch_w = float(weights.sum().item())
            vloss += loss.item() * batch_w
            vweight_sum += batch_w
            vcount += boards.size(0)
            pred = logits.masked_fill(~masks, float("-inf")).argmax(dim=1)
            correct += (pred == targets).sum().item()
            val_metrics.update_batch(logits, targets, masks, plies, src_ids)

    val_mean = vloss / max(vweight_sum, 1e-8)
    return {
        "val_mean": val_mean,
        "acc": correct / max(1, vcount),
        "vcount": vcount,
        "value_mse": value_mse,
        "value_eval_count": value_eval_count,
        "search_ce": search_ce,
        "val_metrics": val_metrics,
        "metrics_report": format_val_metrics_report(val_metrics, pgn_source_vocab=val_ds.pgn_source_vocab),
    }


def format_val_log(result: dict[str, Any], *, value_head: bool, search_policy_head: bool) -> str:
    tail = ""
    vcount = int(result["vcount"])
    if value_head and int(result["value_eval_count"]) > 0:
        tail += f" | val_value_mse {result['value_mse'] / result['value_eval_count']:.4f} (n={result['value_eval_count']})"
    elif value_head and vcount > 0:
        tail += " | val_value_mse n/a (无 search 标注样本)"
    if search_policy_head and vcount > 0:
        tail += f" | val_search_ce {result['search_ce'] / vcount:.4f}"
    return (
        f"val loss {result['val_mean']:.4f} acc {result['acc']:.4f}{tail}\n"
        f"{result['metrics_report']}"
    )
