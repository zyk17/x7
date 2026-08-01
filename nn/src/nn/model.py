"""Public NN factory and compatibility imports.

Architecture implementations live in `model_v2` and `model_v3`; shared model
contract and losses live in `model_common`.
"""

from __future__ import annotations

from .model_common import *  # noqa: F403
from .model_common import CNN_TRUNK_KIND, TRANSFORMER_TRUNK_KIND, _load_move_vocab  # noqa: F401
from .model_v2 import GlobalBroadcast, KnowledgeResNet, PreActBottleneck, ValueAuxHead  # noqa: F401
from .model_v3 import KnowledgeTransformer


def build_model(
    *,
    trunk_kind: str,
    in_planes: int,
    width: int,
    blocks: int,
    num_moves: int,
    bottleneck_channels: int | None = None,
    heads: int | None = None,
    ffn_channels: int | None = None,
    value_head: bool = False,
    moves_left_head: bool = False,
    auxiliary_heads: bool = False,
) -> KnowledgeResNet | KnowledgeTransformer:
    if trunk_kind == CNN_TRUNK_KIND:
        return KnowledgeResNet(
            in_planes=in_planes,
            width=width,
            num_blocks=blocks,
            num_moves=num_moves,
            bottleneck_channels=bottleneck_channels,
            value_head=value_head,
            moves_left_head=moves_left_head,
            auxiliary_heads=auxiliary_heads,
            trunk_kind=trunk_kind,
        )
    if trunk_kind == TRANSFORMER_TRUNK_KIND:
        return KnowledgeTransformer(
            in_planes=in_planes,
            width=width,
            num_blocks=blocks,
            num_moves=num_moves,
            heads=16 if heads is None else heads,
            ffn_channels=width * 3 // 2 if ffn_channels is None else ffn_channels,
            value_head=value_head,
            moves_left_head=moves_left_head,
            auxiliary_heads=auxiliary_heads,
            trunk_kind=trunk_kind,
        )
    raise ValueError(f"未知 trunk_kind: {trunk_kind}")
