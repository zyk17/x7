"""网络基础设施入口。"""

from nn.dataset_px0 import Px0ChunkDataset, Px0DatasetConfig
from nn.model import (
    CNN_TRUNK_KIND,
    TRANSFORMER_TRUNK_KIND,
    KnowledgeResNet,
    KnowledgeTransformer,
    build_model,
    moves_left_loss,
    soften_policy_targets,
    soft_policy_cross_entropy,
    value_q_mse_from_wdl,
    value_wdl_cross_entropy,
    wdl_logits_to_q,
    wdl_probs_to_q,
)

__all__ = [
    "KnowledgeResNet",
    "KnowledgeTransformer",
    "build_model",
    "CNN_TRUNK_KIND",
    "TRANSFORMER_TRUNK_KIND",
    "Px0ChunkDataset",
    "Px0DatasetConfig",
    "moves_left_loss",
    "soften_policy_targets",
    "soft_policy_cross_entropy",
    "value_q_mse_from_wdl",
    "value_wdl_cross_entropy",
    "wdl_logits_to_q",
    "wdl_probs_to_q",
]
