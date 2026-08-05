"""Manifest-driven benchmark for fused per-channel FP8 quantization."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.per_channel_cast_fused_baselines import (
    PerChannelCastFusedTileLangBaseline,
)
from tileops.kernels.quant.per_channel_cast_fused import PerChannelCastFusedKernel
from tileops.manifest import load_workloads
from tileops.ops import (
    QuantPerChannelCastFusedExpandOp,
    QuantPerChannelCastFusedOp,
    QuantPerChannelCastFusedRescaleExpandOp,
    QuantPerChannelCastFusedRescaleOp,
)
from workloads.per_channel_cast_fused import PerChannelCastFusedWorkload

_PLAIN_OP = "QuantPerChannelCastFusedOp"
_EXPAND_OP = "QuantPerChannelCastFusedExpandOp"
_RESCALE_OP = "QuantPerChannelCastFusedRescaleOp"
_RESCALE_EXPAND_OP = "QuantPerChannelCastFusedRescaleExpandOp"

_PLAIN_WORKLOADS = load_workloads(_PLAIN_OP)
_EXPAND_WORKLOADS = load_workloads(_EXPAND_OP)
_RESCALE_WORKLOADS = load_workloads(_RESCALE_OP)
_RESCALE_EXPAND_WORKLOADS = load_workloads(_RESCALE_EXPAND_OP)

assert tuple(
    len(workloads)
    for workloads in (
        _PLAIN_WORKLOADS,
        _EXPAND_WORKLOADS,
        _RESCALE_WORKLOADS,
        _RESCALE_EXPAND_WORKLOADS,
    )
) == (5, 5, 5, 5)

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


_FP8_MAX = 448.0
_NUM_PER_TOKENS = 128
_NUM_PER_CHANNELS = 128


def _torch_eager_reference(
    x: torch.Tensor,
    x_sf_invs: torch.Tensor | None = None,
    pos_to_token: torch.Tensor | None = None,
    round_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent benchmark-local PyTorch baseline."""
    logical = x.to(torch.float32)
    if x_sf_invs is not None:
        logical = logical * x_sf_invs.repeat_interleave(_NUM_PER_CHANNELS, dim=1)
    if pos_to_token is not None:
        valid = pos_to_token >= 0
        logical = logical[pos_to_token.clamp(min=0).long()]
        logical = torch.where(valid[:, None], logical, torch.zeros_like(logical))

    num_tokens_out, hidden = logical.shape
    sf_rows = (num_tokens_out + _NUM_PER_TOKENS - 1) // _NUM_PER_TOKENS
    padded_rows = sf_rows * _NUM_PER_TOKENS - num_tokens_out
    if padded_rows:
        logical_for_reduce = torch.cat(
            (
                logical,
                torch.zeros((padded_rows, hidden), dtype=torch.float32, device=x.device),
            ),
            dim=0,
        )
    else:
        logical_for_reduce = logical

    amax = (
        logical_for_reduce.reshape(sf_rows, _NUM_PER_TOKENS, hidden)
        .abs()
        .amax(dim=1)
        .clamp(min=1e-4)
    )
    fp8_max = torch.full_like(amax, _FP8_MAX)
    out_sf = amax / fp8_max
    if round_sf:
        rounded_bits = (out_sf.view(torch.int32) + 0x007FFFFF) & 0x7F800000
        out_sf = rounded_bits.view(torch.float32)
        quant_sf = out_sf.reciprocal()
    else:
        quant_sf = fp8_max / amax
    quant_sf_per_token = quant_sf.repeat_interleave(_NUM_PER_TOKENS, dim=0)[
        :num_tokens_out
    ]
    out = torch.clamp(logical * quant_sf_per_token, -_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return out.contiguous(), out_sf.contiguous()


def _assert_quant_equal(actual, expected):
    """Check quantized output and scale against a trusted reference."""
    out, out_sf = actual
    out_ref, out_sf_ref = expected

    assert out.shape == out_ref.shape
    assert out.dtype == torch.float8_e4m3fn
    assert out_sf.shape == out_sf_ref.shape
    assert out_sf.dtype == torch.float32
    torch.testing.assert_close(out.float(), out_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(out_sf, out_sf_ref, atol=1e-7, rtol=1e-6)


def _make_reference(
    *,
    with_rescale: bool,
    with_expand: bool,
    round_sf: bool,
) -> Callable:
    if with_rescale and with_expand:
        return lambda x, x_sf_invs, pos_to_token: _torch_eager_reference(
            x,
            x_sf_invs=x_sf_invs,
            pos_to_token=pos_to_token,
            round_sf=round_sf,
        )
    if with_rescale:
        return lambda x, x_sf_invs: _torch_eager_reference(
            x, x_sf_invs=x_sf_invs, round_sf=round_sf
        )
    if with_expand:
        return lambda x, pos_to_token: _torch_eager_reference(
            x, pos_to_token=pos_to_token, round_sf=round_sf
        )
    return lambda x: _torch_eager_reference(x, round_sf=round_sf)


def _make_tilelang_baseline(
    *,
    hidden: int,
    in_dtype: torch.dtype,
    with_rescale: bool,
    with_expand: bool,
    round_sf: bool,
) -> Callable:
    baseline = PerChannelCastFusedTileLangBaseline(
        hidden=hidden,
        in_dtype=in_dtype,
        with_rescale=with_rescale,
        with_expand=with_expand,
        round_sf=round_sf,
    )
    if with_rescale and with_expand:
        return lambda x, x_sf_invs, pos_to_token: baseline(
            x,
            x_sf_invs=x_sf_invs,
            pos_to_token=pos_to_token,
        )
    if with_rescale:
        return lambda x, x_sf_invs: baseline(x, x_sf_invs=x_sf_invs)
    if with_expand:
        return lambda x, pos_to_token: baseline(x, pos_to_token=pos_to_token)
    return lambda x: baseline(x)


def _make_shared_staging_baseline(
    *,
    x_shape: tuple[int, int],
    num_tokens_out: int | None,
    in_dtype: torch.dtype,
    with_expand: bool,
    round_sf: bool,
) -> Callable:
    """Build the pre-optimization plain path for an in-process A/B."""
    kernel = PerChannelCastFusedKernel(
        num_tokens=x_shape[0],
        num_tokens_out=num_tokens_out if with_expand else x_shape[0],
        hidden=x_shape[1],
        in_dtype=in_dtype,
        with_rescale=False,
        with_expand=with_expand,
        round_sf=round_sf,
        device=torch.device("cuda"),
        config={"register_staging": False},
    )
    if with_expand:
        return lambda x, pos_to_token: kernel(x, pos_to_token=pos_to_token)
    return lambda x: kernel(x)


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
    compiled_reference = torch.compile(reference, fullgraph=True)
    tilelang_baseline = _make_tilelang_baseline(
        hidden=x_shape[1],
        in_dtype=dtype,
        with_rescale=with_rescale,
        with_expand=with_expand,
        round_sf=round_sf,
    )
    shared_staging_baseline = None
    if not with_rescale:
        shared_staging_baseline = _make_shared_staging_baseline(
            x_shape=x_shape,
            num_tokens_out=num_tokens_out,
            in_dtype=dtype,
            with_expand=with_expand,
            round_sf=round_sf,
        )
    benchmark = _make_manifest_benchmark(op_name, op, workload)

    # Establish correctness before timing. Compile TileLang and torch.compile
    # paths here so JIT/graph compilation time is excluded from steady-state timing.
    expected = reference(*inputs)
    _assert_quant_equal(op(*inputs), expected)
    _assert_quant_equal(compiled_reference(*inputs), expected)
    _assert_quant_equal(tilelang_baseline(*inputs), expected)
    if shared_staging_baseline is not None:
        _assert_quant_equal(shared_staging_baseline(*inputs), expected)
    torch.cuda.synchronize()

    result = benchmark.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if shared_staging_baseline is not None:
        result_shared = benchmark.profile(shared_staging_baseline, *inputs)
        result_shared["speedup_vs_tileops"] = (
            result_shared["latency_ms"] / result["latency_ms"]
        )
        BenchmarkReport.record(
            op_name,
            locals(),
            result_shared,
            tag="tileops-shared-staging",
        )

    result_tilelang = benchmark.profile(tilelang_baseline, *inputs)
    result_tilelang["speedup_vs_tileops"] = (
        result_tilelang["latency_ms"] / result["latency_ms"]
    )
    BenchmarkReport.record(op_name, locals(), result_tilelang, tag="tilekernels-tilelang")

    result_compiled = benchmark.profile(compiled_reference, *inputs)
    result_compiled["speedup_vs_tileops"] = (
        result_compiled["latency_ms"] / result["latency_ms"]
    )
    BenchmarkReport.record(
        op_name, locals(), result_compiled, tag="torch-compile"
    )

    result_ref = benchmark.profile(reference, *inputs)
    result_ref["speedup_vs_tileops"] = (
        result_ref["latency_ms"] / result["latency_ms"]
    )
    BenchmarkReport.record(op_name, locals(), result_ref, tag="torch-eager")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
