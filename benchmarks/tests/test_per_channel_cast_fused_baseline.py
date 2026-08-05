"""Credibility tests for the pinned TileKernels performance baseline."""

from __future__ import annotations

import pytest
import torch

from benchmarks.ops.per_channel_cast_fused_baselines import (
    tile_k_for,
    UPSTREAM_COMMIT,
    UPSTREAM_KERNEL_SHA256,
    UPSTREAM_TORCH_REFERENCE_SHA256,
    PerChannelCastFusedTileLangBaseline,
)
from benchmarks.ops.bench_per_channel_cast_fused import _torch_eager_reference


def _make_fp8(shape: tuple[int, int]) -> torch.Tensor:
    values = torch.linspace(-4.0, 4.0, shape[0] * shape[1], device="cuda")
    return values.reshape(shape).to(torch.float8_e4m3fn)


def _assert_quant_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    out, out_sf = actual
    out_ref, out_sf_ref = expected
    assert out.shape == out_ref.shape
    assert out.dtype == torch.float8_e4m3fn
    assert out_sf.shape == out_sf_ref.shape
    assert out_sf.dtype == torch.float32
    torch.testing.assert_close(out.float(), out_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(out_sf, out_sf_ref, atol=1e-7, rtol=1e-6)


@pytest.mark.smoke
def test_per_channel_cast_fused_baseline_provenance_is_pinned() -> None:
    assert UPSTREAM_COMMIT == "0266ab740980de7dc03a828b8259cd73d100c2eb"
    assert UPSTREAM_KERNEL_SHA256 == (
        "64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc"
    )
    assert UPSTREAM_TORCH_REFERENCE_SHA256 == (
        "6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f"
    )
    assert tile_k_for(torch.bfloat16, with_rescale=False) == 128
    assert tile_k_for(torch.float32, with_rescale=False) == 64
    assert tile_k_for(torch.float8_e4m3fn, with_rescale=True) == 256


@pytest.mark.parametrize(
    "case, round_sf",
    [
        pytest.param("plain-bf16", False, marks=pytest.mark.smoke),
        pytest.param("plain-fp32", True, marks=pytest.mark.full),
        pytest.param("expand-bf16", False, marks=pytest.mark.full),
        pytest.param("rescale", False, marks=pytest.mark.full),
        pytest.param("rescale-expand", True, marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_tilelang_baseline_matches_torch_eager(
    case: str,
    round_sf: bool,
) -> None:
    with_rescale = case.startswith("rescale")
    with_expand = "expand" in case

    if case == "plain-bf16":
        x = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    elif case == "plain-fp32":
        x = torch.linspace(-896.0, 896.0, 128 * 128, device="cuda").reshape(128, 128)
    elif case == "expand-bf16":
        x = torch.randn((64, 128), dtype=torch.bfloat16, device="cuda")
    elif case == "rescale":
        x = _make_fp8((128, 256))
    else:
        x = _make_fp8((64, 256))

    x_sf_invs = None
    if with_rescale:
        x_sf_invs = torch.linspace(
            0.01,
            1.01,
            x.shape[0] * (x.shape[1] // 128),
            device="cuda",
        ).reshape(x.shape[0], x.shape[1] // 128)

    pos_to_token = None
    if with_expand:
        pos_to_token = torch.arange(127, -1, -1, dtype=torch.int32, device="cuda")
        pos_to_token %= x.shape[0]
        pos_to_token[::11] = -1
        pos_to_token[1::17] = 3

    baseline = PerChannelCastFusedTileLangBaseline(
        hidden=x.shape[1],
        in_dtype=x.dtype,
        with_rescale=with_rescale,
        with_expand=with_expand,
        round_sf=round_sf,
    )
    actual = baseline(
        x,
        x_sf_invs=x_sf_invs,
        pos_to_token=pos_to_token,
    )
    expected = _torch_eager_reference(
        x,
        x_sf_invs=x_sf_invs,
        pos_to_token=pos_to_token,
        round_sf=round_sf,
    )
    _assert_quant_equal(actual, expected)


@pytest.mark.smoke
def test_per_channel_cast_fused_tilelang_baseline_rejects_invalid_expand_alignment() -> None:
    x = torch.randn((64, 128), dtype=torch.bfloat16, device="cuda")
    pos_to_token = torch.arange(15, dtype=torch.int32, device="cuda") % x.shape[0]
    baseline = PerChannelCastFusedTileLangBaseline(
        hidden=x.shape[1],
        in_dtype=x.dtype,
        with_rescale=False,
        with_expand=True,
        round_sf=False,
    )
    with pytest.raises(ValueError, match="num_tokens_out divisible by 128"):
        baseline(x, pos_to_token=pos_to_token)
