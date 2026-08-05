"""Correctness and contract tests for fused per-channel FP8 quantization."""

from __future__ import annotations

import pytest
import torch

from tileops.kernels.quant.per_channel_cast_fused import (
    _C500_SHARED_MEMORY_LIMIT_BYTES,
    _rescale_threads_per_token,
    _shared_memory_bytes,
)
from tileops.ops import (
    QuantPerChannelCastFusedExpandOp,
    QuantPerChannelCastFusedOp,
    QuantPerChannelCastFusedRescaleExpandOp,
    QuantPerChannelCastFusedRescaleOp,
)


_FP8_MAX = 448.0
_NUM_PER_TOKENS = 128
_NUM_PER_CHANNELS = 128


def per_channel_cast_fused_reference(
    x: torch.Tensor,
    x_sf_invs: torch.Tensor | None = None,
    pos_to_token: torch.Tensor | None = None,
    round_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch correctness oracle for all four variants."""
    logical = x.to(torch.float32)
    if x_sf_invs is not None:
        logical = logical * x_sf_invs.repeat_interleave(_NUM_PER_CHANNELS, dim=1)
    if pos_to_token is not None:
        if x.shape[0] == 0:
            logical = torch.zeros(
                (pos_to_token.shape[0], x.shape[1]),
                dtype=torch.float32,
                device=x.device,
            )
        else:
            valid = pos_to_token >= 0
            logical = logical[pos_to_token.clamp(min=0).long()]
            logical = torch.where(valid[:, None], logical, torch.zeros_like(logical))

    num_tokens_out, hidden = logical.shape
    sf_rows = (num_tokens_out + _NUM_PER_TOKENS - 1) // _NUM_PER_TOKENS
    if num_tokens_out == 0:
        return (
            torch.empty((0, hidden), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((0, hidden), dtype=torch.float32, device=x.device),
        )
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


@pytest.mark.smoke
def test_per_channel_cast_fused_c500_shared_memory_budget() -> None:
    assert _shared_memory_bytes(64, torch.float32, register_staging=True) == 1_024
    assert _shared_memory_bytes(64, torch.bfloat16, register_staging=True) == 1_024
    assert _shared_memory_bytes(64, torch.float32) == 33_792
    assert _shared_memory_bytes(128, torch.float32) == 67_584
    assert _shared_memory_bytes(128, torch.bfloat16) == 34_816
    assert _shared_memory_bytes(64, torch.float8_e4m3fn) == 9_216
    assert _shared_memory_bytes(
        64, torch.float8_e4m3fn, threads_per_token=16
    ) == 12_288
    assert _shared_memory_bytes(256, torch.float8_e4m3fn) == 36_864
    assert _shared_memory_bytes(128, torch.float32) > _C500_SHARED_MEMORY_LIMIT_BYTES
    assert _shared_memory_bytes(64, torch.float32) <= _C500_SHARED_MEMORY_LIMIT_BYTES


@pytest.mark.smoke
@pytest.mark.parametrize(
    "num_tokens_out, with_expand, expected",
    [
        (256, False, 8),
        (257, False, 16),
        (2048, False, 16),
        (2048, True, 32),
        (4096, False, 32),
    ],
)
def test_per_channel_cast_fused_rescale_thread_mapping(
    num_tokens_out: int, with_expand: bool, expected: int
) -> None:
    assert _rescale_threads_per_token(num_tokens_out, with_expand) == expected


@pytest.mark.parametrize(
    "shape, dtype, round_sf",
    [
        pytest.param((128, 128), torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param((128, 128), torch.float32, False, marks=pytest.mark.smoke),
        pytest.param((256, 384), torch.bfloat16, True, marks=pytest.mark.full),
        pytest.param((256, 384), torch.float32, True, marks=pytest.mark.full),
        pytest.param((1024, 7168), torch.bfloat16, True, marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_plain(
    shape: tuple[int, int], dtype: torch.dtype, round_sf: bool
) -> None:
    x = torch.randn(shape, device="cuda", dtype=dtype)
    op = QuantPerChannelCastFusedOp(round_sf=round_sf)
    _assert_quant_equal(op(x), per_channel_cast_fused_reference(x, round_sf=round_sf))
    assert op.kernel.tile_k == 64
    assert op.kernel.register_staging is True
    assert op.kernel.shared_memory_bytes <= _C500_SHARED_MEMORY_LIMIT_BYTES


@pytest.mark.parametrize(
    "magnitude",
    [
        pytest.param(0.0, marks=pytest.mark.smoke, id="zero"),
        pytest.param(1e-8, marks=pytest.mark.full, id="below-amax-floor"),
    ],
)
def test_per_channel_cast_fused_minimum_scale(magnitude: float) -> None:
    x = torch.full((128, 128), magnitude, device="cuda", dtype=torch.bfloat16)
    x[:, 1::2].neg_()
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
    "x_shape, dtype, num_tokens_out, position_case, round_sf",
    [
        pytest.param((64, 128), torch.bfloat16, 16, "mixed", False, marks=pytest.mark.smoke),
        pytest.param((153, 128), torch.float32, 160, "mixed", True, marks=pytest.mark.smoke),
        pytest.param((64, 128), torch.bfloat16, 128, "repeated", False, marks=pytest.mark.full),
        pytest.param((64, 128), torch.bfloat16, 144, "all_padding", False, marks=pytest.mark.full),
        pytest.param((64, 128), torch.bfloat16, 256, "reverse", False, marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_expand_output_sizes(
    x_shape: tuple[int, int],
    dtype: torch.dtype,
    num_tokens_out: int,
    position_case: str,
    round_sf: bool,
) -> None:
    x = torch.randn(x_shape, device="cuda", dtype=dtype)
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

    op = QuantPerChannelCastFusedExpandOp(round_sf=round_sf)
    _assert_quant_equal(
        op(x, pos_to_token),
        per_channel_cast_fused_reference(
            x,
            pos_to_token=pos_to_token,
            round_sf=round_sf,
        ),
    )
    assert op.kernel.tile_k == 64
    assert op.kernel.register_staging is True


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
    assert op.kernel.tile_k == 64
    assert op.kernel.register_staging is False


@pytest.mark.parametrize(
    "shape, num_tokens_out, round_sf",
    [
        pytest.param((64, 256), 144, True, marks=pytest.mark.smoke),
        pytest.param((256, 512), 256, False, marks=pytest.mark.full),
        pytest.param((513, 7168), 1024, False, marks=pytest.mark.full),
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
    assert op.kernel.tile_k == 64
    assert op.kernel.register_staging is False


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
        (
            lambda: QuantPerChannelCastFusedOp(round_sf=1),
            "round_sf must be bool",
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

@pytest.mark.parametrize(
    "case, match",
    [
        pytest.param("alignment", "divisible by 16", marks=pytest.mark.smoke),
        pytest.param("rank", "pos_to_token must be 1D", marks=pytest.mark.full),
        pytest.param("dtype", "pos_to_token.dtype must be int32", marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_expand_rejects_invalid_metadata(
    case: str,
    match: str,
) -> None:
    x = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16)
    if case == "alignment":
        pos_to_token = torch.zeros((15,), device="cuda", dtype=torch.int32)
    elif case == "rank":
        pos_to_token = torch.zeros((4, 4), device="cuda", dtype=torch.int32)
    else:
        pos_to_token = torch.zeros((16,), device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match=match):
        QuantPerChannelCastFusedExpandOp()(x, pos_to_token)


@pytest.mark.parametrize(
    "case, match",
    [
        pytest.param("x-dtype", "x.dtype must be float8_e4m3fn", marks=pytest.mark.smoke),
        pytest.param("hidden", "hidden must be positive and divisible by 256", marks=pytest.mark.full),
        pytest.param("scale-device", "x_sf_invs must be a CUDA tensor", marks=pytest.mark.full),
    ],
)
def test_per_channel_cast_fused_rescale_rejects_invalid_input_contract(
    case: str,
    match: str,
) -> None:
    if case == "x-dtype":
        x = torch.ones((128, 256), device="cuda", dtype=torch.bfloat16)
        x_sf_invs = torch.ones((128, 2), device="cuda", dtype=torch.float32)
    elif case == "hidden":
        x = _make_fp8((128, 128))
        x_sf_invs = torch.ones((128, 1), device="cuda", dtype=torch.float32)
    else:
        x = _make_fp8((128, 256))
        x_sf_invs = torch.ones((128, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match=match):
        QuantPerChannelCastFusedRescaleOp()(x, x_sf_invs)
