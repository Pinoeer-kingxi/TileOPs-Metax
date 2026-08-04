"""Pinned eager-PyTorch reference for fused per-channel FP8 quantization.

This acceptance baseline intentionally stays as ordinary eager PyTorch
primitives in correctness and performance tests.
"""

from __future__ import annotations

from typing import Optional

import torch

_FP8_MAX = 448.0
_NUM_PER_TOKENS = 128
_NUM_PER_CHANNELS = 128

UPSTREAM_COMMIT = "0266ab740980de7dc03a828b8259cd73d100c2eb"
UPSTREAM_TORCH_REFERENCE_PATH = "tile_kernels/torch/per_channel_cast_fused.py"
UPSTREAM_TORCH_REFERENCE_SHA256 = "6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f"


def per_channel_cast_fused_reference(
    x: torch.Tensor,
    x_sf_invs: Optional[torch.Tensor] = None,
    pos_to_token: Optional[torch.Tensor] = None,
    round_sf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the four manifest variants with PyTorch primitives."""
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

    quant_sf_per_token = quant_sf.repeat_interleave(_NUM_PER_TOKENS, dim=0)[:num_tokens_out]
    out = torch.clamp(logical * quant_sf_per_token, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return out.contiguous(), out_sf.contiguous()
