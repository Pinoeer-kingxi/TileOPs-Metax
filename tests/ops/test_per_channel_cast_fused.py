"""Correctness and contract tests for fused per-channel FP8 quantization."""

from __future__ import annotations

import pytest
import torch

from tileops.ops import (
    QuantPerChannelCastFusedExpandOp,
    QuantPerChannelCastFusedOp,
    QuantPerChannelCastFusedRescaleExpandOp,
    QuantPerChannelCastFusedRescaleOp,
)
from tileops.testing.per_channel_cast_fused import per_channel_cast_fused_reference


def _make_fp8(shape: tuple[int, int]) -> torch.Tensor:
    return (
        torch.randn(shape, device="cuda", dtype=torch.float32).clamp(-4, 4).to(torch.float8_e4m3fn)
    )


def _make_positions(num_tokens: int, num_tokens_out: int) -> torch.Tensor:
    if num_tokens == 0:
        return torch.full((num_tokens_out,), -1, device="cuda", dtype=torch.int32)
    positions = torch.arange(num_tokens_out, device="cuda", dtype=torch.int32)
    positions %= num_tokens
    positions[::11] = -1
    return positions


def _assert_quant_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    assert isinstance(actual, tuple)
    out, out_sf = actual
    out_ref, out_sf_ref = expected
    assert out.shape == out_ref.shape
    assert out.dtype == torch.float8_e4m3fn
    assert out.is_contiguous()
    assert out_sf.shape == out_sf_ref.shape
    assert out_sf.dtype == torch.float32
    assert out_sf.is_contiguous()
    torch.testing.assert_close(out.float(), out_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(out_sf, out_sf_ref, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "shape, dtype, round_sf",
    [
        pytest.param((128, 128), torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param((128, 128), torch.float32, False, marks=pytest.mark.smoke),
        pytest.param((256, 384), torch.bfloat16, True, marks=pytest.mark.full),
        pytest.param((256, 384), torch.float32, True, marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_plain(
    shape: tuple[int, int], dtype: torch.dtype, round_sf: bool
) -> None:
    x = torch.randn(shape, device="cuda", dtype=dtype)
    op = QuantPerChannelCastFusedOp(round_sf=round_sf)
    _assert_quant_equal(op(x), per_channel_cast_fused_reference(x, round_sf=round_sf))


@pytest.mark.smoke
def test_per_channel_cast_fused_zero_input_uses_minimum_scale() -> None:
    x = torch.zeros((128, 128), device="cuda", dtype=torch.bfloat16)
    actual = QuantPerChannelCastFusedOp()(x)
    expected = per_channel_cast_fused_reference(x)
    _assert_quant_equal(actual, expected)
    torch.testing.assert_close(
        actual[1],
        torch.full_like(actual[1], 1e-4 / 448.0),
        atol=0,
        rtol=1e-6,
    )


@pytest.mark.smoke
def test_per_channel_cast_fused_signed_extremes() -> None:
    values = torch.tensor(
        [-448.0, -127.5, -1e-4, 0.0, 1e-4, 127.5, 448.0, 896.0],
        device="cuda",
        dtype=torch.float32,
    )
    x = values.repeat(128 * 128 // values.numel()).reshape(128, 128)
    _assert_quant_equal(QuantPerChannelCastFusedOp()(x), per_channel_cast_fused_reference(x))


@pytest.mark.parametrize(
    "num_tokens_out, position_case",
    [
        pytest.param(16, "mixed", marks=pytest.mark.smoke),
        pytest.param(128, "repeated", marks=pytest.mark.full),
        pytest.param(144, "all_padding", marks=pytest.mark.full),
        pytest.param(256, "reverse", marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_expand_output_sizes(
    num_tokens_out: int, position_case: str
) -> None:
    x = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    if position_case == "repeated":
        pos_to_token = torch.full((num_tokens_out,), 7, device="cuda", dtype=torch.int32)
        pos_to_token[::9] = -1
    elif position_case == "all_padding":
        pos_to_token = torch.full((num_tokens_out,), -1, device="cuda", dtype=torch.int32)
    elif position_case == "reverse":
        pos_to_token = torch.arange(num_tokens_out - 1, -1, -1, device="cuda", dtype=torch.int32)
        pos_to_token %= x.shape[0]
        pos_to_token[::13] = -1
    else:
        pos_to_token = _make_positions(x.shape[0], num_tokens_out)

    op = QuantPerChannelCastFusedExpandOp()
    _assert_quant_equal(
        op(x, pos_to_token),
        per_channel_cast_fused_reference(x, pos_to_token=pos_to_token),
    )


@pytest.mark.parametrize(
    "shape, round_sf",
    [
        pytest.param((128, 256), False, marks=pytest.mark.smoke),
        pytest.param((256, 512), True, marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_rescale(shape: tuple[int, int], round_sf: bool) -> None:
    x = _make_fp8(shape)
    x_sf_invs = torch.rand((shape[0], shape[1] // 128), device="cuda", dtype=torch.float32) + 0.01
    op = QuantPerChannelCastFusedRescaleOp(round_sf=round_sf)
    _assert_quant_equal(
        op(x, x_sf_invs),
        per_channel_cast_fused_reference(x, x_sf_invs=x_sf_invs, round_sf=round_sf),
    )


@pytest.mark.parametrize(
    "shape, num_tokens_out, round_sf",
    [
        pytest.param((64, 256), 144, True, marks=pytest.mark.smoke),
        pytest.param((256, 512), 256, False, marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_rescale_expand(
    shape: tuple[int, int], num_tokens_out: int, round_sf: bool
) -> None:
    x = _make_fp8(shape)
    x_sf_invs = torch.rand((shape[0], shape[1] // 128), device="cuda", dtype=torch.float32) + 0.01
    pos_to_token = _make_positions(shape[0], num_tokens_out)
    pos_to_token[1::19] = 3
    op = QuantPerChannelCastFusedRescaleExpandOp(round_sf=round_sf)
    _assert_quant_equal(
        op(x, x_sf_invs, pos_to_token),
        per_channel_cast_fused_reference(
            x,
            x_sf_invs=x_sf_invs,
            pos_to_token=pos_to_token,
            round_sf=round_sf,
        ),
    )


@pytest.mark.smoke
def test_per_channel_cast_fused_empty_input() -> None:
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    out, out_sf = QuantPerChannelCastFusedOp()(x)
    assert out.shape == (0, 128)
    assert out.dtype == torch.float8_e4m3fn
    assert out_sf.shape == (0, 128)
    assert out_sf.dtype == torch.float32


@pytest.mark.smoke
def test_per_channel_cast_fused_expand_empty_source_uses_padding() -> None:
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    pos_to_token = torch.full((16,), -1, device="cuda", dtype=torch.int32)
    actual = QuantPerChannelCastFusedExpandOp()(x, pos_to_token)
    expected = per_channel_cast_fused_reference(x, pos_to_token=pos_to_token)
    _assert_quant_equal(actual, expected)


@pytest.mark.parametrize(
    "ctor, match",
    [
        (lambda: QuantPerChannelCastFusedOp(fmt="e5m2"), "fmt must be"),
        (
            lambda: QuantPerChannelCastFusedOp(num_per_tokens=64),
            "num_per_tokens must be 128",
        ),
        (
            lambda: QuantPerChannelCastFusedRescaleOp(num_per_channels=64),
            "num_per_channels must be 128",
        ),
    ],
)
@pytest.mark.smoke
def test_per_channel_cast_fused_rejects_invalid_params(ctor, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ctor()


@pytest.mark.parametrize(
    "case, match",
    [
        ("dtype", "x.dtype must be"),
        ("rank", "x must have shape"),
        ("hidden", "hidden must be positive and divisible by 128"),
        ("tokens", "num_tokens must be divisible by 128"),
        ("noncontiguous", "x must be contiguous"),
    ],
)
@pytest.mark.smoke
def test_per_channel_cast_fused_rejects_invalid_plain_input(case: str, match: str) -> None:
    if case == "dtype":
        x = torch.ones((128, 128), device="cuda", dtype=torch.float16)
    elif case == "rank":
        x = torch.ones((128, 128, 1), device="cuda", dtype=torch.bfloat16)
    elif case == "hidden":
        x = torch.ones((128, 192), device="cuda", dtype=torch.bfloat16)
    elif case == "tokens":
        x = torch.ones((16, 128), device="cuda", dtype=torch.bfloat16)
    else:
        x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16).T

    with pytest.raises(ValueError, match=match):
        QuantPerChannelCastFusedOp()(x)


@pytest.mark.smoke
def test_per_channel_cast_fused_expand_rejects_invalid_positions() -> None:
    x = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16)
    pos_to_token = torch.zeros((16,), device="cuda", dtype=torch.int32)
    pos_to_token[3] = 32
    with pytest.raises(ValueError, match="must be less than num_tokens"):
        QuantPerChannelCastFusedExpandOp()(x, pos_to_token)


@pytest.mark.parametrize(
    "sf_shape, sf_dtype, match",
    [
        ((127, 2), torch.float32, r"shape\[0\] must equal"),
        ((128, 1), torch.float32, r"shape\[1\] \* 128 must equal hidden"),
        ((128, 2), torch.float16, "x_sf_invs.dtype must be float32"),
    ],
)
@pytest.mark.smoke
def test_per_channel_cast_fused_rescale_rejects_invalid_scale_tensor(
    sf_shape: tuple[int, int], sf_dtype: torch.dtype, match: str
) -> None:
    x = _make_fp8((128, 256))
    x_sf_invs = torch.ones(sf_shape, device="cuda", dtype=sf_dtype)
    with pytest.raises(ValueError, match=match):
        QuantPerChannelCastFusedRescaleOp()(x, x_sf_invs)
