"""网络基础设施入口。"""

from nn.board_compact import (
    compact_board_to_planes,
    compact_board_to_torch_planes,
    fen_to_compact_board,
)
from nn.dataset_px0 import Px0ChunkDataset, Px0DatasetConfig
from nn.dataset_xrsh import PolicyXrshDataset
from nn.fen_tensor import fen_to_planes
from nn.model import (
    PolicyResNet,
    mix_wdl_targets,
    policy_cross_entropy,
    soft_policy_cross_entropy,
    value_q_mse,
    value_wdl_cross_entropy,
    wdl_logits_to_q,
    wdl_probs_to_q,
)

__all__ = [
    "PolicyResNet",
    "Px0ChunkDataset",
    "Px0DatasetConfig",
    "PolicyXrshDataset",
    "compact_board_to_planes",
    "compact_board_to_torch_planes",
    "fen_to_compact_board",
    "fen_to_planes",
    "mix_wdl_targets",
    "policy_cross_entropy",
    "soft_policy_cross_entropy",
    "value_q_mse",
    "value_wdl_cross_entropy",
    "wdl_logits_to_q",
    "wdl_probs_to_q",
]
