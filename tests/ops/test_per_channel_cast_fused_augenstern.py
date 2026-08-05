"""The 70-case correctness matrix from augenstern's register-resident branch.

This is a one-for-one port of the suite at
``exp/quant-per-channel-register-resident@387119e154c6cfa4d09b25823635814380af91f9``.
It uses this repository's pinned eager-PyTorch reference and never converts a
reference OOM or a mismatch into a successful test.
"""

from __future__ import annotations

import pytest
import torch

from tests.ops.test_per_channel_cast_fused import _assert_quant_equal
from tileops.ops import (
    QuantPerChannelCastFusedExpandOp,
    QuantPerChannelCastFusedOp,
    QuantPerChannelCastFusedRescaleExpandOp,
    QuantPerChannelCastFusedRescaleOp,
)
from tileops.testing.per_channel_cast_fused import per_channel_cast_fused_reference

AUGENSTERN_UPSTREAM_HEAD = "387119e154c6cfa4d09b25823635814380af91f9"
AUGENSTERN_TEST_BLOB = "11238539f3f64e2b5585a7a906067eab9d7e1e7c"
AUGENSTERN_TEST_COMMITS = (
    "ff86cdff073f124130ed87f37c1f6a9078eca24c",  # Feature and numeric boundaries.
    "ea5d33e7aa7b031e25c97c84693613e2ec5c95b7",  # Token-scale matrix.
    "c77cc75caa4b8a281a814b66b06102564ef33a01",  # Token-by-hidden matrix.
)


def _case(*values: object, id: str, smoke: bool = False):
    """Build a full-tier case, or an explicitly selected smoke case."""
    tier = pytest.mark.smoke if smoke else pytest.mark.full
    return pytest.param(*values, marks=tier, id=id)


# (variant, num_tokens, hidden, num_tokens_out, dtype, round_sf)
_VARIANT_CASES = (
    _case("plain", 128, 128, None, torch.bfloat16, False, id="plain-bf16", smoke=True),
    _case("plain", 128, 128, None, torch.float32, False, id="plain-fp32", smoke=True),
    _case(
        "expand",
        137,
        128,
        144,
        torch.bfloat16,
        False,
        id="expand-bf16-tail",
        smoke=True,
    ),
    _case(
        "expand",
        153,
        128,
        160,
        torch.float32,
        True,
        id="expand-fp32-tail-rounded",
        smoke=True,
    ),
    _case(
        "rescale",
        128,
        256,
        None,
        torch.float8_e4m3fn,
        False,
        id="rescale-fp8",
        smoke=True,
    ),
    _case(
        "rescale_expand",
        137,
        256,
        144,
        torch.float8_e4m3fn,
        False,
        id="rescale-expand-fp8-tail",
        smoke=True,
    ),
    _case("plain", 128, 128, None, torch.bfloat16, True, id="plain-bf16-rounded"),
    _case("plain", 256, 512, None, torch.bfloat16, False, id="plain-bf16-multitile"),
    _case(
        "plain",
        256,
        384,
        None,
        torch.float32,
        True,
        id="plain-fp32-multitile-rounded",
    ),
    _case("expand", 137, 128, 144, torch.bfloat16, True, id="expand-bf16-tail-rounded"),
    _case("expand", 256, 256, 256, torch.bfloat16, False, id="expand-bf16-aligned"),
    _case(
        "expand",
        257,
        512,
        272,
        torch.bfloat16,
        False,
        id="expand-bf16-multitile-tail",
    ),
    _case("rescale", 128, 256, None, torch.float8_e4m3fn, True, id="rescale-fp8-rounded"),
    _case(
        "rescale",
        256,
        512,
        None,
        torch.float8_e4m3fn,
        False,
        id="rescale-fp8-multitile",
    ),
    _case(
        "rescale_expand",
        137,
        256,
        144,
        torch.float8_e4m3fn,
        True,
        id="rescale-expand-fp8-tail-rounded",
    ),
    _case(
        "rescale_expand",
        256,
        512,
        256,
        torch.float8_e4m3fn,
        False,
        id="rescale-expand-fp8-aligned",
    ),
    _case(
        "rescale_expand",
        257,
        512,
        272,
        torch.float8_e4m3fn,
        True,
        id="rescale-expand-fp8-multitile-tail-rounded",
    ),
)

# (variant, num_tokens, num_tokens_out, dtype, round_sf), hidden=512
_TOKEN_SCALE_CASES = (
    _case("plain", 128, None, torch.bfloat16, False, id="plain-t128-bf16", smoke=True),
    _case(
        "expand",
        17,
        32,
        torch.float32,
        True,
        id="expand-t17-to32-fp32-rounded",
        smoke=True,
    ),
    _case(
        "rescale",
        128,
        None,
        torch.float8_e4m3fn,
        False,
        id="rescale-t128-fp8",
        smoke=True,
    ),
    _case("plain", 256, None, torch.bfloat16, False, id="plain-t256-bf16"),
    _case("plain", 512, None, torch.bfloat16, True, id="plain-t512-bf16-rounded"),
    _case("plain", 1024, None, torch.bfloat16, False, id="plain-t1024-bf16"),
    _case("plain", 256, None, torch.float32, True, id="plain-t256-fp32-rounded"),
    _case("plain", 512, None, torch.float32, False, id="plain-t512-fp32"),
    _case("plain", 1024, None, torch.float32, True, id="plain-t1024-fp32-rounded"),
    _case("expand", 17, 32, torch.bfloat16, False, id="expand-t17-to32-bf16"),
    _case("expand", 137, 144, torch.bfloat16, True, id="expand-t137-to144-bf16-rounded"),
    _case("expand", 513, 1024, torch.bfloat16, False, id="expand-t513-to1024-bf16"),
    _case(
        "expand",
        1001,
        2048,
        torch.bfloat16,
        True,
        id="expand-t1001-to2048-bf16-rounded",
    ),
    _case("rescale", 256, None, torch.float8_e4m3fn, True, id="rescale-t256-fp8-rounded"),
    _case("rescale", 512, None, torch.float8_e4m3fn, False, id="rescale-t512-fp8"),
    _case("rescale", 1024, None, torch.float8_e4m3fn, True, id="rescale-t1024-fp8-rounded"),
    _case(
        "rescale_expand",
        17,
        32,
        torch.float8_e4m3fn,
        False,
        id="rescale-expand-t17-to32-fp8",
    ),
    _case(
        "rescale_expand",
        137,
        144,
        torch.float8_e4m3fn,
        True,
        id="rescale-expand-t137-to144-fp8-rounded",
    ),
    _case(
        "rescale_expand",
        513,
        1024,
        torch.float8_e4m3fn,
        False,
        id="rescale-expand-t513-to1024-fp8",
    ),
    _case(
        "rescale_expand",
        1001,
        2048,
        torch.float8_e4m3fn,
        True,
        id="rescale-expand-t1001-to2048-fp8-rounded",
    ),
)

# (num_tokens, hidden, dtype, round_sf)
_PLAIN_CROSS_SCALE_CASES = (
    _case(128, 3072, torch.bfloat16, False, id="t128-h3072-bf16", smoke=True),
    _case(128, 3072, torch.float32, True, id="t128-h3072-fp32-rounded", smoke=True),
    _case(512, 128, torch.bfloat16, True, id="t512-h128-bf16-rounded"),
    _case(512, 7168, torch.bfloat16, False, id="t512-h7168-bf16"),
    _case(1024, 3072, torch.bfloat16, True, id="t1024-h3072-bf16-rounded"),
    _case(1024, 7168, torch.bfloat16, False, id="t1024-h7168-bf16"),
    _case(512, 7168, torch.float32, False, id="t512-h7168-fp32"),
    _case(1024, 3072, torch.float32, True, id="t1024-h3072-fp32-rounded"),
)

# (num_tokens, hidden, num_tokens_out, dtype, round_sf)
_EXPAND_CROSS_SCALE_CASES = (
    _case(137, 3072, 144, torch.bfloat16, False, id="t137-to144-h3072-bf16", smoke=True),
    _case(
        137,
        3072,
        144,
        torch.float32,
        True,
        id="t137-to144-h3072-fp32-rounded",
        smoke=True,
    ),
    _case(513, 128, 1024, torch.bfloat16, True, id="t513-to1024-h128-bf16-rounded"),
    _case(513, 7168, 1024, torch.bfloat16, False, id="t513-to1024-h7168-bf16"),
    _case(
        1001,
        3072,
        2048,
        torch.bfloat16,
        True,
        id="t1001-to2048-h3072-bf16-rounded",
    ),
    _case(1001, 7168, 2048, torch.bfloat16, False, id="t1001-to2048-h7168-bf16"),
    _case(513, 7168, 1024, torch.float32, False, id="t513-to1024-h7168-fp32"),
    _case(
        1001,
        3072,
        2048,
        torch.float32,
        True,
        id="t1001-to2048-h3072-fp32-rounded",
    ),
)

# (num_tokens, hidden, round_sf)
_RESCALE_CROSS_SCALE_CASES = (
    _case(128, 3072, False, id="t128-h3072", smoke=True),
    _case(512, 256, True, id="t512-h256-rounded"),
    _case(512, 7168, False, id="t512-h7168"),
    _case(1024, 3072, True, id="t1024-h3072-rounded"),
    _case(1024, 7168, False, id="t1024-h7168"),
)

# (num_tokens, hidden, num_tokens_out, round_sf)
_RESCALE_EXPAND_CROSS_SCALE_CASES = (
    _case(137, 3072, 144, False, id="t137-to144-h3072", smoke=True),
    _case(513, 256, 1024, True, id="t513-to1024-h256-rounded"),
    _case(513, 7168, 1024, False, id="t513-to1024-h7168"),
    _case(1001, 3072, 2048, True, id="t1001-to2048-h3072-rounded"),
    _case(1001, 7168, 2048, False, id="t1001-to2048-h7168"),
)

_EXPAND_INDEX_PATTERNS = (
    _case("sequential", id="sequential", smoke=True),
    _case("repeated", id="repeated-token"),
    _case("all_invalid", id="all-invalid"),
)
_NUMERIC_BOUNDARIES = (
    _case("zeros", id="zeros", smoke=True),
    _case("tiny", id="below-amax-clamp"),
    _case("large", id="large-alternating-sign"),
)

_EXPECTED_GROUP_COUNTS = {
    "variants": 17,
    "token_scale": 20,
    "plain_cross_scale": 8,
    "expand_cross_scale": 8,
    "rescale_cross_scale": 5,
    "rescale_expand_cross_scale": 5,
    "expand_index_patterns": 3,
    "numeric_boundaries": 3,
    "parameter_validation": 1,
}
_CASE_GROUPS = {
    "variants": _VARIANT_CASES,
    "token_scale": _TOKEN_SCALE_CASES,
    "plain_cross_scale": _PLAIN_CROSS_SCALE_CASES,
    "expand_cross_scale": _EXPAND_CROSS_SCALE_CASES,
    "rescale_cross_scale": _RESCALE_CROSS_SCALE_CASES,
    "rescale_expand_cross_scale": _RESCALE_EXPAND_CROSS_SCALE_CASES,
    "expand_index_patterns": _EXPAND_INDEX_PATTERNS,
    "numeric_boundaries": _NUMERIC_BOUNDARIES,
}
_ACTUAL_GROUP_COUNTS = {name: len(cases) for name, cases in _CASE_GROUPS.items()}
_ACTUAL_GROUP_COUNTS["parameter_validation"] = 1
assert _ACTUAL_GROUP_COUNTS == _EXPECTED_GROUP_COUNTS
assert sum(_ACTUAL_GROUP_COUNTS.values()) == 70
assert all(len({case.id for case in cases}) == len(cases) for cases in _CASE_GROUPS.values())


def _make_op(variant: str, round_sf: bool):
    if variant == "plain":
        return QuantPerChannelCastFusedOp(round_sf=round_sf)
    if variant == "expand":
        return QuantPerChannelCastFusedExpandOp(round_sf=round_sf)
    if variant == "rescale":
        return QuantPerChannelCastFusedRescaleOp(round_sf=round_sf)
    if variant == "rescale_expand":
        return QuantPerChannelCastFusedRescaleExpandOp(round_sf=round_sf)
    raise AssertionError(f"unknown variant: {variant}")


def _check_case(
    variant: str,
    num_tokens: int,
    hidden: int,
    num_tokens_out: int | None,
    dtype: torch.dtype,
    round_sf: bool,
) -> None:
    torch.manual_seed(0)
    with_rescale = variant in ("rescale", "rescale_expand")
    with_expand = variant in ("expand", "rescale_expand")

    if with_rescale:
        real = torch.randn((num_tokens, hidden), dtype=torch.float32, device="cuda")
        x_sf_invs = (
            torch.rand(
                (num_tokens, hidden // 128),
                dtype=torch.float32,
                device="cuda",
            )
            * 0.02
            + 1e-4
        )
        expanded_sf = x_sf_invs.repeat_interleave(128, dim=1)
        x = torch.clamp(real / expanded_sf, -448.0, 448.0).to(torch.float8_e4m3fn)
    else:
        x = torch.randn((num_tokens, hidden), dtype=dtype, device="cuda")
        x_sf_invs = None

    if with_expand:
        assert num_tokens_out is not None
        pos_to_token = torch.randint(
            0,
            num_tokens,
            (num_tokens_out,),
            dtype=torch.int32,
            device="cuda",
        )
        pos_to_token[::17] = -1
    else:
        assert num_tokens_out is None
        pos_to_token = None

    op = _make_op(variant, round_sf)
    if with_rescale and with_expand:
        assert x_sf_invs is not None and pos_to_token is not None
        actual = op(x, x_sf_invs, pos_to_token)
    elif with_rescale:
        assert x_sf_invs is not None
        actual = op(x, x_sf_invs)
    elif with_expand:
        assert pos_to_token is not None
        actual = op(x, pos_to_token)
    else:
        actual = op(x)

    expected = per_channel_cast_fused_reference(
        x,
        x_sf_invs=x_sf_invs,
        pos_to_token=pos_to_token,
        round_sf=round_sf,
    )
    _assert_quant_equal(actual, expected)


@pytest.mark.parametrize(
    "variant, num_tokens, hidden, num_tokens_out, dtype, round_sf",
    _VARIANT_CASES,
)
def test_augenstern_per_channel_cast_fused_variants(
    variant: str,
    num_tokens: int,
    hidden: int,
    num_tokens_out: int | None,
    dtype: torch.dtype,
    round_sf: bool,
) -> None:
    _check_case(variant, num_tokens, hidden, num_tokens_out, dtype, round_sf)


@pytest.mark.parametrize(
    "variant, num_tokens, num_tokens_out, dtype, round_sf",
    _TOKEN_SCALE_CASES,
)
def test_augenstern_per_channel_cast_fused_token_scale(
    variant: str,
    num_tokens: int,
    num_tokens_out: int | None,
    dtype: torch.dtype,
    round_sf: bool,
) -> None:
    _check_case(variant, num_tokens, 512, num_tokens_out, dtype, round_sf)


@pytest.mark.parametrize("num_tokens, hidden, dtype, round_sf", _PLAIN_CROSS_SCALE_CASES)
def test_augenstern_per_channel_cast_fused_plain_cross_scale(
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    round_sf: bool,
) -> None:
    _check_case("plain", num_tokens, hidden, None, dtype, round_sf)


@pytest.mark.parametrize(
    "num_tokens, hidden, num_tokens_out, dtype, round_sf",
    _EXPAND_CROSS_SCALE_CASES,
)
def test_augenstern_per_channel_cast_fused_expand_cross_scale(
    num_tokens: int,
    hidden: int,
    num_tokens_out: int,
    dtype: torch.dtype,
    round_sf: bool,
) -> None:
    _check_case("expand", num_tokens, hidden, num_tokens_out, dtype, round_sf)


@pytest.mark.parametrize("num_tokens, hidden, round_sf", _RESCALE_CROSS_SCALE_CASES)
def test_augenstern_per_channel_cast_fused_rescale_cross_scale(
    num_tokens: int,
    hidden: int,
    round_sf: bool,
) -> None:
    _check_case("rescale", num_tokens, hidden, None, torch.float8_e4m3fn, round_sf)


@pytest.mark.parametrize(
    "num_tokens, hidden, num_tokens_out, round_sf",
    _RESCALE_EXPAND_CROSS_SCALE_CASES,
)
def test_augenstern_per_channel_cast_fused_rescale_expand_cross_scale(
    num_tokens: int,
    hidden: int,
    num_tokens_out: int,
    round_sf: bool,
) -> None:
    _check_case(
        "rescale_expand",
        num_tokens,
        hidden,
        num_tokens_out,
        torch.float8_e4m3fn,
        round_sf,
    )


@pytest.mark.parametrize("pattern", _EXPAND_INDEX_PATTERNS)
def test_augenstern_per_channel_cast_fused_expand_index_patterns(pattern: str) -> None:
    torch.manual_seed(0)
    x = torch.randn((137, 128), dtype=torch.bfloat16, device="cuda")
    if pattern == "sequential":
        pos_to_token = torch.arange(144, dtype=torch.int32, device="cuda") % 137
    elif pattern == "repeated":
        pos_to_token = torch.full((144,), 17, dtype=torch.int32, device="cuda")
    else:
        pos_to_token = torch.full((144,), -1, dtype=torch.int32, device="cuda")

    actual = QuantPerChannelCastFusedExpandOp()(x, pos_to_token)
    expected = per_channel_cast_fused_reference(x, pos_to_token=pos_to_token)
    _assert_quant_equal(actual, expected)


@pytest.mark.parametrize("pattern", _NUMERIC_BOUNDARIES)
def test_augenstern_per_channel_cast_fused_numeric_boundaries(pattern: str) -> None:
    shape = (128, 128)
    if pattern == "zeros":
        x = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    elif pattern == "tiny":
        x = torch.full(shape, 1e-8, dtype=torch.bfloat16, device="cuda")
        x[:, 1::2].neg_()
    else:
        x = torch.full(shape, 1e4, dtype=torch.bfloat16, device="cuda")
        x[:, 1::2].neg_()

    actual = QuantPerChannelCastFusedOp()(x)
    expected = per_channel_cast_fused_reference(x)
    _assert_quant_equal(actual, expected)


@pytest.mark.smoke
def test_augenstern_per_channel_cast_fused_parameter_validation() -> None:
    with pytest.raises(ValueError, match="fmt"):
        QuantPerChannelCastFusedOp("e5m6", 128)
    with pytest.raises(ValueError, match="num_per_tokens"):
        QuantPerChannelCastFusedOp("e4m3", 64)
    with pytest.raises(ValueError, match="num_per_channels"):
        QuantPerChannelCastFusedRescaleOp("e4m3", 128, 64)
