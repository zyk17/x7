"""人类策略网络：棋盘编码、ResNet policy、训练与 ONNX 导出。"""

from nn.board_compact import (
    compact_board_to_planes,
    compact_board_to_torch_planes,
    fen_to_compact_board,
)
from nn.dataset import PolicyJsonlDataset, PolicyJsonlMmapDataset
from nn.fen_tensor import fen_to_planes
from nn.jsonl_index import (
    SAMPLER_ORDER,
    SAMPLER_SEG_PTR,
    build_jsonl_index,
    index_dir_is_complete,
    index_sampler_is_complete,
)
from nn.model import PolicyResNet, policy_cross_entropy

__all__ = [
    "SAMPLER_ORDER",
    "SAMPLER_SEG_PTR",
    "PolicyJsonlDataset",
    "PolicyJsonlMmapDataset",
    "PolicyResNet",
    "build_jsonl_index",
    "compact_board_to_planes",
    "compact_board_to_torch_planes",
    "fen_to_compact_board",
    "fen_to_planes",
    "index_dir_is_complete",
    "index_sampler_is_complete",
    "policy_cross_entropy",
]
