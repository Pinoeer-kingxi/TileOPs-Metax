"""Augenstern 32-case performance matrix for per-channel FP8 cast.

The existing benchmark file remains the stable four-way A/B suite.  This
file preserves the broader workload matrix from
``exp/quant-per-channel-register-resident@387119e`` and compares only the
production TileOps implementation with the pinned eager-PyTorch reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

_Variant = Literal["plain", "expand", "rescale", "rescale_expand"]


@dataclass(frozen=True)
class _PerformanceCase:
    variant: _Variant
    label: str
    x_shape: tuple[int, int]
    dtype: torch.dtype
    round_sf: bool
    num_tokens_out: int | None = None


_OP_NAMES = {
    "plain": "QuantPerChannelCastFusedOp",
    "expand": "QuantPerChannelCastFusedExpandOp",
    "rescale": "QuantPerChannelCastFusedRescaleOp",
    "rescale_expand": "QuantPerChannelCastFusedRescaleExpandOp",
}
_SOURCE_SUITE = "augenstern-performance"


def _load_source_cases(variant: _Variant) -> tuple[_PerformanceCase, ...]:
    """Adapt tagged source workloads to the current Quant* Op contracts."""
    with_rescale = variant in ("rescale", "rescale_expand")
    with_expand = variant in ("expand", "rescale_expand")
    cases = []
    for workload in load_workloads(_OP_NAMES[variant]):
        if workload.get("__suite") != _SOURCE_SUITE:
            continue

        x_shape = tuple(workload["x_shape"])
        num_tokens_out = workload["pos_to_token_shape"][0] if with_expand else None
        assert workload["fmt"] == "e4m3"
        assert workload["num_per_tokens"] == 128
        if with_rescale:
            assert workload["num_per_channels"] == 128
            assert tuple(workload["x_sf_invs_shape"]) == (
                x_shape[0],
                x_shape[1] // 128,
            )

        for dtype_name in workload["dtypes"]:
            cases.append(
                _PerformanceCase(
                    variant=variant,
                    label=workload["label"],
                    x_shape=x_shape,
                    dtype=getattr(torch, dtype_name),
                    round_sf=bool(workload["round_sf"]),
                    num_tokens_out=num_tokens_out,
                )
            )
    return tuple(cases)


_CASE_GROUPS = tuple(
    _load_source_cases(variant) for variant in ("plain", "expand", "rescale", "rescale_expand")
)
_ALL_CASES = tuple(case for group in _CASE_GROUPS for case in group)
_CASE_IDS = tuple(f"{case.label}-{str(case.dtype).removeprefix('torch.')}" for case in _ALL_CASES)

# Fail collection if the manifest matrix is accidentally truncated or duplicated.
assert all(len(group) == 8 for group in _CASE_GROUPS)
assert len(_ALL_CASES) == 32
assert len(set(_CASE_IDS)) == len(_CASE_IDS)


class _AugensternWorkload(PerChannelCastFusedWorkload):
    """Use the source branch's coupled FP8-rescale and random gather inputs."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        num_tokens, hidden = self.x_shape
        if self.with_rescale:
            real = torch.randn(self.x_shape, dtype=torch.float32, device="cuda")
            x_sf_invs = (
                torch.rand((num_tokens, hidden // 128), dtype=torch.float32, device="cuda") * 0.02
                + 1e-4
            )
            expanded_sf = x_sf_invs.repeat_interleave(128, dim=1)
            x = torch.clamp(real / expanded_sf, -448.0, 448.0).to(torch.float8_e4m3fn)
            inputs: tuple[torch.Tensor, ...] = (x, x_sf_invs)
        else:
            x = torch.randn(self.x_shape, dtype=self.dtype, device="cuda")
            inputs = (x,)

        if self.with_expand:
            assert self.num_tokens_out is not None
            pos_to_token = torch.randint(
                0,
                num_tokens,
                (self.num_tokens_out,),
                dtype=torch.int32,
                device="cuda",
            )
            pos_to_token[::17] = -1
            inputs = (*inputs, pos_to_token)
        return inputs


def _make_op(case: _PerformanceCase):
    if case.variant == "plain":
        return QuantPerChannelCastFusedOp(round_sf=case.round_sf)
    if case.variant == "expand":
        return QuantPerChannelCastFusedExpandOp(round_sf=case.round_sf)
    if case.variant == "rescale":
        return QuantPerChannelCastFusedRescaleOp(round_sf=case.round_sf)
    return QuantPerChannelCastFusedRescaleExpandOp(round_sf=case.round_sf)


def _make_reference(case: _PerformanceCase):
    if case.variant == "plain":
        return lambda x: per_channel_cast_fused_reference(x, round_sf=case.round_sf)
    if case.variant == "expand":
        return lambda x, pos: per_channel_cast_fused_reference(
            x, pos_to_token=pos, round_sf=case.round_sf
        )
    if case.variant == "rescale":
        return lambda x, x_sf: per_channel_cast_fused_reference(
            x, x_sf_invs=x_sf, round_sf=case.round_sf
        )
    return lambda x, x_sf, pos: per_channel_cast_fused_reference(
        x,
        x_sf_invs=x_sf,
        pos_to_token=pos,
        round_sf=case.round_sf,
    )


def _assert_quant_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-7, rtol=1e-6)


def _case_params() -> list:
    params = []
    for group in _CASE_GROUPS:
        for index, case in enumerate(group):
            tier = pytest.mark.smoke if index == 0 else pytest.mark.full
            params.append(pytest.param(case, id=_CASE_IDS[len(params)], marks=tier))
    return params


@pytest.mark.parametrize("case", _case_params())
def test_per_channel_cast_fused_augenstern_bench(case: _PerformanceCase) -> None:
    with_rescale = case.variant in ("rescale", "rescale_expand")
    with_expand = case.variant in ("expand", "rescale_expand")
    workload = _AugensternWorkload(
        case.x_shape,
        case.dtype,
        with_rescale=with_rescale,
        with_expand=with_expand,
        num_tokens_out=case.num_tokens_out,
        round_sf=case.round_sf,
    )
    inputs = workload.gen_inputs()
    op = _make_op(case)
    reference = _make_reference(case)
    benchmark = ManifestBenchmark(_OP_NAMES[case.variant], op, workload)

    # Compile the production kernel and reject numerical regressions before
    # timing.  Reference OOMs and mismatches are intentionally not suppressed.
    expected = reference(*inputs)
    actual = op(*inputs)
    _assert_quant_equal(actual, expected)
    del actual, expected
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    params = {
        "variant": case.variant,
        "label": case.label,
        "x_shape": case.x_shape,
        "dtype": case.dtype,
        "num_tokens_out": case.num_tokens_out or case.x_shape[0],
        "round_sf": case.round_sf,
    }
    result = benchmark.profile(op, *inputs)
    BenchmarkReport.record(op, params, result, tag="tileops")

    result_reference = benchmark.profile(reference, *inputs)
    BenchmarkReport.record(
        _OP_NAMES[case.variant],
        params,
        result_reference,
        tag="torch-eager",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
