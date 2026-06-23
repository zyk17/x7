"""训练/验证共用损失。"""

from __future__ import annotations

import torch

from nn.model import policy_cross_entropy, soft_policy_cross_entropy, value_wdl_cross_entropy


def compute_policy_value_loss(
    logits: torch.Tensor,
    pred_value: torch.Tensor | None,
    *,
    targets: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    value_head: bool,
    search_policy_head: bool,
    t_val: torch.Tensor | None = None,
    search_visits: torch.Tensor | None = None,
    visit_target: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    value_loss_weight: float = 0.5,
    value_min_visits: int = 1,
    search_policy_weight: float = 0.0,
) -> torch.Tensor:
    loss = policy_cross_entropy(
        logits,
        targets,
        masks,
        label_smoothing=label_smoothing,
        sample_weight=weights,
    )
    if value_head and pred_value is not None and t_val is not None and search_visits is not None:
        value_mask = search_visits >= int(value_min_visits)
        value_weights = weights * value_mask.to(weights.dtype)
        if value_weights.sum() > 0:
            loss = loss + float(value_loss_weight) * value_wdl_cross_entropy(
                pred_value,
                t_val,
                sample_weight=value_weights,
            )
    if search_policy_head and visit_target is not None:
        loss = loss + float(search_policy_weight) * soft_policy_cross_entropy(
            logits,
            visit_target,
            masks,
            sample_weight=weights,
        )
    return loss
