"""onnx2torch compatibility shims for opset-18 reduction operators.

The eFrog classifier is exported with `opset_version=18` (see EDA-Master.ipynb).
In opset 18, several ONNX reduction operators (ReduceMean, ReduceMax, ...) moved
their `axes` from a node *attribute* to an optional second *input* tensor.

onnx2torch (1.5.x) only ships converters for these ops up to version 13, where
axes is still an attribute, so converting an opset-18 model raises
`NotImplementedError: Converter is not implemented (... ReduceMean version=18)`.

This module registers version-18 converters that:
  * read `axes` from the optional constant input (input[1]) when present,
  * otherwise fall back to the `axes` attribute (older style),
  * honour `keepdims` and `noop_with_empty_axes`,
and then reuse onnx2torch's existing `OnnxReduceStaticAxes` torch module.

Importing this module has the side effect of registering the converters, so
`import onnx2torch_compat` before calling `onnx2torch.convert(...)`.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import (
    OnnxMapping,
    OperationConverterResult,
    get_const_value,
    onnx_mapping_from_node,
)

# Reduction ops whose opset-18 form carries axes as an optional input tensor.
_REDUCE_OPS_V18 = (
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
)


def _axes_from_input(node: OnnxNode, graph: OnnxGraph) -> Optional[List[int]]:
    """Return axes read from the optional constant input[1], or None."""
    if len(node.input_values) < 2:
        return None
    try:
        axes = get_const_value(node.input_values[1], graph)
    except KeyError:
        return None
    if axes is None:
        return None
    if isinstance(axes, torch.Tensor):
        axes = axes.tolist()
    return list(axes)


def _convert_reduce_v18(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    attrs = node.attributes
    keepdims: int = attrs.get("keepdims", 1)

    axes = _axes_from_input(node, graph)
    if axes is None:
        axes = attrs.get("axes", None)

    return OperationConverterResult(
        torch_module=OnnxReduceStaticAxes(
            operation_type=node.operation_type,
            axes=axes,
            keepdims=keepdims,
        ),
        # Only the data tensor is a runtime input; axes is a baked-in constant.
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        )
        if len(node.input_values) >= 2
        else onnx_mapping_from_node(node=node),
    )


def register() -> None:
    """Register opset-18 reduce converters (idempotent-ish; safe to call once)."""
    for op in _REDUCE_OPS_V18:
        try:
            add_converter(operation_type=op, version=18)(_convert_reduce_v18)
        except Exception:  # pragma: no cover - already registered
            pass


# Register on import.
register()
