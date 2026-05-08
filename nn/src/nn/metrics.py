"""验证集 policy 指标：合法 softmax 下的熵、top-k、人类着法 NLL（surprise）。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    bad = ~legal_mask
    x = logits.masked_fill(bad, float("-inf"))
    return F.log_softmax(x, dim=1)


def per_sample_entropy_bits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """合法子集上归一化分布的 Shannon 熵，单位 nat；打印时可 /ln(2) 为 bits。"""
    logp = masked_log_softmax(logits, legal_mask)
    p = logp.exp()
    pl = p * logp
    pl = torch.where(legal_mask, pl, torch.zeros_like(pl))
    return -pl.sum(dim=1)


def per_sample_nll(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """-log p(human_move)，与无平滑 CE 一致。"""
    logp = masked_log_softmax(logits, legal_mask)
    return -logp[torch.arange(logits.size(0), device=logits.device), targets]


def metric_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """单次 log_softmax + 单次 topk，产出验证聚合所需逐样本指标。"""
    logp = masked_log_softmax(logits, legal_mask)
    nll = -logp[torch.arange(logits.size(0), device=logits.device), targets]

    p = logp.exp()
    ent = -(torch.where(legal_mask, p * logp, torch.zeros_like(logp))).sum(dim=1)

    maxk = min(5, logits.shape[1])
    top_idx = logits.masked_fill(~legal_mask, float("-inf")).topk(maxk, dim=1).indices
    tgt = targets.long().unsqueeze(1)
    top1 = (top_idx[:, 0] == targets).float()
    top3 = (top_idx[:, : min(3, maxk)] == tgt).any(dim=1).float()
    top5 = (top_idx == tgt).any(dim=1).float()
    return nll, ent, top1, top3, top5


def per_sample_topk_hit(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """逐样本：人类着法是否落在「按 logits 排序的合法着」前 min(k, n_legal) 名。"""
    maxk = min(k, logits.shape[1])
    top_idx = logits.masked_fill(~legal_mask, float("-inf")).topk(maxk, dim=1).indices
    return (top_idx == targets.long().unsqueeze(1)).any(dim=1).float()


PLY_BIN_LABELS = ("ply0-19", "ply20-39", "ply40-59", "ply60+")


def ply_bin_index(ply: int) -> int:
    if ply < 20:
        return 0
    if ply < 40:
        return 1
    if ply < 60:
        return 2
    return 3


@dataclass
class ValMetricTotals:
    n: int = 0
    sum_nll: float = 0.0
    sum_nll_sq: float = 0.0
    sum_entropy: float = 0.0
    sum_top1: float = 0.0
    sum_top3: float = 0.0
    sum_top5: float = 0.0

    def update(
        self,
        nll: torch.Tensor,
        entropy: torch.Tensor,
        top1: torch.Tensor,
        top3: torch.Tensor,
        top5: torch.Tensor,
    ) -> None:
        n = int(nll.numel())
        self.n += n
        self.sum_nll += float(nll.sum().item())
        self.sum_nll_sq += float((nll * nll).sum().item())
        self.sum_entropy += float(entropy.sum().item())
        self.sum_top1 += float(top1.sum().item())
        self.sum_top3 += float(top3.sum().item())
        self.sum_top5 += float(top5.sum().item())

    def mean_std_nll(self) -> tuple[float, float]:
        if self.n < 1:
            return 0.0, 0.0
        m = self.sum_nll / self.n
        v = max(0.0, self.sum_nll_sq / self.n - m * m)
        return m, math.sqrt(v)

    def update_from_sums(
        self,
        *,
        n: int,
        sum_nll: float,
        sum_nll_sq: float,
        sum_entropy: float,
        sum_top1: float,
        sum_top3: float,
        sum_top5: float,
    ) -> None:
        self.n += n
        self.sum_nll += sum_nll
        self.sum_nll_sq += sum_nll_sq
        self.sum_entropy += sum_entropy
        self.sum_top1 += sum_top1
        self.sum_top3 += sum_top3
        self.sum_top5 += sum_top5


@dataclass
class ValMetricsState:
    overall: ValMetricTotals = field(default_factory=ValMetricTotals)
    by_ply_bin: dict[int, ValMetricTotals] = field(
        default_factory=lambda: defaultdict(ValMetricTotals)
    )
    by_source_id: dict[int, ValMetricTotals] = field(
        default_factory=lambda: defaultdict(ValMetricTotals)
    )

    def update_batch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        legal_mask: torch.Tensor,
        plies: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> None:
        nll, ent, top1, top3, top5 = metric_tensors(logits, targets, legal_mask)

        self.overall.update(nll, ent, top1, top3, top5)

        plies_cpu = plies.detach().cpu()
        src_cpu = source_ids.detach().cpu()
        nll_cpu = nll.detach().cpu()
        ent_cpu = ent.detach().cpu()
        t1_cpu = top1.detach().cpu()
        t3_cpu = top3.detach().cpu()
        t5_cpu = top5.detach().cpu()

        ply_bins = torch.bucketize(
            plies_cpu,
            boundaries=torch.tensor([20, 40, 60], dtype=plies_cpu.dtype),
        )
        for pb in range(len(PLY_BIN_LABELS)):
            mask = ply_bins == pb
            if not mask.any():
                continue
            self.by_ply_bin[pb].update(
                nll_cpu[mask], ent_cpu[mask], t1_cpu[mask], t3_cpu[mask], t5_cpu[mask]
            )

        for sid in src_cpu.unique(sorted=True).tolist():
            mask = src_cpu == sid
            self.by_source_id[int(sid)].update(
                nll_cpu[mask], ent_cpu[mask], t1_cpu[mask], t3_cpu[mask], t5_cpu[mask]
            )


def format_val_metrics_report(
    state: ValMetricsState,
    *,
    pgn_source_vocab: list[str],
) -> str:
    """多行文本，供 train 脚本 print。"""
    lines: list[str] = []
    o = state.overall
    if o.n < 1:
        return "val metrics: (no samples)"
    mn, sd = o.mean_std_nll()
    ent_m = o.sum_entropy / o.n
    lines.append(
        f"val human_NLL mean={mn:.4f} std={sd:.4f} | "
        f"entropy(nat) mean={ent_m:.4f} | "
        f"top1={o.sum_top1/o.n:.4f} top3={o.sum_top3/o.n:.4f} top5={o.sum_top5/o.n:.4f}"
    )
    lines.append("val by ply bin:")
    for bi, label in enumerate(PLY_BIN_LABELS):
        t = state.by_ply_bin[bi]
        if t.n < 1:
            continue
        mn_b, sd_b = t.mean_std_nll()
        lines.append(
            f"  {label} n={t.n} NLL={mn_b:.4f}+-{sd_b:.4f} "
            f"H={t.sum_entropy/t.n:.4f} top1={t.sum_top1/t.n:.4f} top3={t.sum_top3/t.n:.4f}"
        )
    lines.append("val by pgn_source:")
    for sid in sorted(state.by_source_id.keys()):
        t = state.by_source_id[sid]
        if t.n < 1:
            continue
        name = pgn_source_vocab[sid] if sid < len(pgn_source_vocab) else f"id={sid}"
        mn_b, sd_b = t.mean_std_nll()
        lines.append(
            f"  {name} n={t.n} NLL={mn_b:.4f}+-{sd_b:.4f} "
            f"H={t.sum_entropy/t.n:.4f} top1={t.sum_top1/t.n:.4f} top3={t.sum_top3/t.n:.4f}"
        )
    return "\n".join(lines)
