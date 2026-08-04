"""Manifest-driven benchmark for fused per-channel FP8 quantization."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import (
    QuantPerChannelCastFusedExpandOp,
    QuantPerChannelCastFusedOp,
    QuantPerChannelCastFusedRescaleExpandOp,
    QuantPerChannelCastFusedRescaleOp,
)
from tileops.testing.per_channel_cast_fused import per_channel_cast_fused_reference
from workloads.per_channel_cast_fused import PerChannelCastFusedWorkload

_PLAIN_OP = "QuantPerChannelCastFusedOp"
_EXPAND_OP = "QuantPerChannelCastFusedExpandOp"
_RESCALE_OP = "QuantPerChannelCastFusedRescaleOp"
_RESCALE_EXPAND_OP = "QuantPerChannelCastFusedRescaleExpandOp"

_PLAIN_WORKLOADS = load_workloads(_PLAIN_OP)
_EXPAND_WORKLOADS = load_workloads(_EXPAND_OP)
_RESCALE_WORKLOADS = load_workloads(_RESCALE_OP)
_RESCALE_EXPAND_WORKLOADS = load_workloads(_RESCALE_EXPAND_OP)

_OP_CLASSES = {
    _PLAIN_OP: QuantPerChannelCastFusedOp,
    _EXPAND_OP: QuantPerChannelCastFusedExpandOp,
    _RESCALE_OP: QuantPerChannelCastFusedRescaleOp,
    _RESCALE_EXPAND_OP: QuantPerChannelCastFusedRescaleExpandOp,
}

# Keep one literal ManifestBenchmark call per manifest entry so the L4 AST
# validator can prove that every variant consumes its own manifest roofline.
_BENCHMARK_FACTORIES = {
    _PLAIN_OP: lambda op, workload: ManifestBenchmark(_PLAIN_OP, op, workload),
    _EXPAND_OP: lambda op, workload: ManifestBenchmark(_EXPAND_OP, op, workload),
    _RESCALE_OP: lambda op, workload: ManifestBenchmark(_RESCALE_OP, op, workload),
    _RESCALE_EXPAND_OP: lambda op, workload: ManifestBenchmark(_RESCALE_EXPAND_OP, op, workload),
}


def _manifest_params() -> list:
    params = []
    groups = (
        (_PLAIN_OP, _PLAIN_WORKLOADS, False, False),
        (_EXPAND_OP, _EXPAND_WORKLOADS, False, True),
        (_RESCALE_OP, _RESCALE_WORKLOADS, True, False),
        (_RESCALE_EXPAND_OP, _RESCALE_EXPAND_WORKLOADS, True, True),
    )
    for op_name, workloads, with_rescale, with_expand in groups:
        for workload_index, workload in enumerate(workloads):
            label = workload.get("label", "unlabeled")
            num_tokens_out = workload["pos_to_token_shape"][0] if with_expand else None
            tier = pytest.mark.smoke if workload_index == 0 else pytest.mark.full
            for dtype_name in workload["dtypes"]:
                params.append(
                    pytest.param(
                        op_name,
                        tuple(workload["x_shape"]),
                        getattr(torch, dtype_name),
                        num_tokens_out,
                        bool(workload.get("round_sf", False)),
                        with_rescale,
                        with_expand,
                        id=f"{op_name}-{label}-{dtype_name}",
                        marks=tier,
                    )
                )
    return params


def _make_op(op_name: str, round_sf: bool):
    try:
        op_cls = _OP_CLASSES[op_name]
    except KeyError as exc:
        raise ValueError(f"unknown op_name {op_name!r}") from exc
    return op_cls(round_sf=round_sf)


def _make_reference(
    *,
    with_rescale: bool,
    with_expand: bool,
    round_sf: bool,
) -> Callable:
    if with_rescale and with_expand:
        return lambda x, x_sf_invs, pos_to_token: per_channel_cast_fused_reference(
            x,
            x_sf_invs=x_sf_invs,
            pos_to_token=pos_to_token,
            round_sf=round_sf,
        )
    if with_rescale:
        return lambda x, x_sf_invs: per_channel_cast_fused_reference(
            x, x_sf_invs=x_sf_invs, round_sf=round_sf
        )
    if with_expand:
        return lambda x, pos_to_token: per_channel_cast_fused_reference(
            x, pos_to_token=pos_to_token, round_sf=round_sf
        )
    return lambda x: per_channel_cast_fused_reference(x, round_sf=round_sf)


def _make_manifest_benchmark(op_name: str, op, workload):
    try:
        factory = _BENCHMARK_FACTORIES[op_name]
    except KeyError as exc:
        raise ValueError(f"unknown op_name {op_name!r}") from exc
    return factory(op, workload)


@pytest.mark.parametrize(
    "op_name, x_shape, dtype, num_tokens_out, round_sf, with_rescale, with_expand",
    _manifest_params(),
)
def test_per_channel_cast_fused_bench(
    op_name: str,
    x_shape: tuple[int, int],
    dtype: torch.dtype,
    num_tokens_out: int | None,
    round_sf: bool,
    with_rescale: bool,
    with_expand: bool,
) -> None:
    workload = PerChannelCastFusedWorkload(
        x_shape,
        dtype,
        with_rescale=with_rescale,
        with_expand=with_expand,
        num_tokens_out=num_tokens_out,
        round_sf=round_sf,
    )
    inputs = workload.gen_inputs()
    op = _make_op(op_name, round_sf)
    reference = _make_reference(
        with_rescale=with_rescale,
        with_expand=with_expand,
        round_sf=round_sf,
    )
    benchmark = _make_manifest_benchmark(op_name, op, workload)

    # Compile before profiling so JIT time is excluded from both measurements.
    op(*inputs)
    reference(*inputs)
    torch.cuda.synchronize()

    result = benchmark.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_ref = benchmark.profile(reference, *inputs)
    BenchmarkReport.record(op, locals(), result_ref, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
