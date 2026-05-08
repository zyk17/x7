"""人类策略网络：棋盘编码、ResNet policy、训练与 ONNX 导出。"""

from nn.board_compact import (
    compact_board_to_planes,
    compact_board_to_torch_planes,
    fen_to_compact_board,
)
from nn.dataset_xrsh import PolicyXrshDataset
from nn.fen_tensor import fen_to_planes
from nn.model import PolicyResNet, policy_cross_entropy

__all__ = [
    "PolicyResNet",
    "PolicyXrshDataset",
    "compact_board_to_planes",
    "compact_board_to_torch_planes",
    "fen_to_compact_board",
    "fen_to_planes",
    "policy_cross_entropy",
]
