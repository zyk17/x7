"""校验仓库根目录 `data/policy.onnx` 与 `export_onnx.py` 约定一致。

`data/` 通常 gitignore，本地导出后放入即可运行本测试；缺失则跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, shape_inference

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ONNX = REPO_ROOT / "data" / "policy.onnx"

_ALLOWED_SUFFIXES = (
    (),
    ("attack", "danger", "tactical"),
    ("attack", "danger", "tactical", "value"),
)


def _tensor_elem_type_and_shape(vi) -> tuple[int, list[int | str | None]]:
    tt = vi.type.tensor_type
    elem = int(tt.elem_type)
    dims: list[int | str | None] = []
    for d in tt.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return elem, dims


@pytest.mark.skipif(
    not POLICY_ONNX.is_file(),
    reason=f"optional artifact missing: {POLICY_ONNX} (gitignored; export via scripts/export/export_onnx.py)",
)
def test_policy_onnx_contract_matches_export_script() -> None:
    raw = onnx.load(str(POLICY_ONNX))
    onnx.checker.check_model(raw, full_check=True)
    model = shape_inference.infer_shapes(raw)

    init_names = {x.name for x in model.graph.initializer}
    net_inputs = [i for i in model.graph.input if i.name not in init_names]
    assert len(net_inputs) == 1, f"expected single network input, got {[x.name for x in net_inputs]}"
    assert net_inputs[0].name == "board"
    elem, dims = _tensor_elem_type_and_shape(net_inputs[0])
    assert elem == TensorProto.FLOAT
    assert dims == [1, 15, 10, 9], f"board shape expected [1,15,10,9], got {dims}"

    out_list = list(model.graph.output)
    names = [o.name for o in out_list]
    if not names or names[0] != "logits":
        raise AssertionError(f"first output must be logits, got {names}")
    suffix = tuple(names[1:])
    assert suffix in _ALLOWED_SUFFIXES, (
        f"output names must be logits + one of {_ALLOWED_SUFFIXES}, got {names}"
    )

    logits_info = out_list[0]
    elem, dims = _tensor_elem_type_and_shape(logits_info)
    assert elem == TensorProto.FLOAT
    assert len(dims) == 2 and dims[0] == 1 and dims[1] not in (None, ""), (
        f"logits expected float32[1,V] with static V, got dims={dims}"
    )
    assert isinstance(dims[1], int), f"logits second dim must be static vocab size, got {dims!r}"
    assert dims[1] >= 2

    for o in out_list[1:]:
        elem, dims = _tensor_elem_type_and_shape(o)
        assert elem == TensorProto.FLOAT
        flat = 1
        for d in dims:
            if isinstance(d, int):
                flat *= d
        assert flat == 1, (
            f"{o.name} expected one scalar element total, shape={dims}"
        )


@pytest.mark.skipif(
    not POLICY_ONNX.is_file(),
    reason=f"optional artifact missing: {POLICY_ONNX}",
)
def test_policy_onnx_optional_runtime_inference_smoke() -> None:
    """若安装了 onnxruntime，则用全零输入跑通 Session（与引擎侧用法一致）。"""
    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(POLICY_ONNX), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    inputs = {in_name: np.zeros((1, 15, 10, 9), dtype=np.float32)}
    out = sess.run(None, inputs)
    n_out = len(sess.get_outputs())
    assert len(out) == n_out
    assert out[0].shape[0] == 1 and out[0].ndim == 2
    for i in range(1, len(out)):
        assert out[i].size == 1
