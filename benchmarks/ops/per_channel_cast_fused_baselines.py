# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""Pinned TileKernels baseline for the per-channel fused-cast benchmark.

The TileLang baseline follows MetaX-MACA/TileKernels-Metax ``dev`` at the
commit recorded below. It preserves the upstream schedule:

* plain/expand BF16: ``tile_k=128``;
* rescale/rescale-expand FP8: ``tile_k=256``.

The sole scheduling compatibility change is plain/expand FP32
``tile_k=128 -> 64``. The unmodified layout needs 67,584 bytes of explicit
shared memory and cannot fit within the C500's 65,536-byte workgroup limit.

This module is intentionally independent from the TileOPs kernel so later
optimizations cannot silently move the performance baseline.
"""

from __future__ import annotations

from typing import Optional

import tilelang
import tilelang.language as T
import torch

__all__ = [
    "tile_k_for",
    "UPSTREAM_COMMIT",
    "UPSTREAM_KERNEL_PATH",
    "UPSTREAM_KERNEL_SHA256",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_TORCH_REFERENCE_PATH",
    "UPSTREAM_TORCH_REFERENCE_SHA256",
    "PerChannelCastFusedTileLangBaseline",
]

UPSTREAM_COMMIT = "0266ab740980de7dc03a828b8259cd73d100c2eb"
UPSTREAM_TORCH_REFERENCE_PATH = "tile_kernels/torch/per_channel_cast_fused.py"
UPSTREAM_TORCH_REFERENCE_SHA256 = "6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f"

UPSTREAM_REPOSITORY = "https://github.com/MetaX-MACA/TileKernels-Metax"
UPSTREAM_KERNEL_PATH = "tile_kernels/quant/per_channel_cast_fused_kernel.py"
UPSTREAM_KERNEL_SHA256 = "64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc"

TILE_M = 128
NUM_THREADS = 256
NUM_THREADS_PER_TOKEN = 64
NUM_PER_CHANNELS = 128
FP8_MAX = 448.0
MIN_AMAX = 1e-4


def tile_k_for(in_dtype: torch.dtype | str, *, with_rescale: bool) -> int:
    """Return upstream ``tile_k``, except for the C500-infeasible FP32 path."""
    dtype_name = str(in_dtype).removeprefix("torch.")
    if with_rescale:
        return 256
    if dtype_name == "float32":
        return 64
    return 128


def _source_token(with_expand: bool, index: int, token: int, positions):
    if with_expand:
        return positions[index]
    return token


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _get_tilelang_baseline_kernel(
    hidden: int,
    in_dtype: str,
    with_rescale: bool,
    with_expand: bool,
    round_sf: bool,
):
    """Build the pinned upstream kernel with only the required FP32 tile fix."""
    num_tokens = T.dynamic("baseline_num_tokens")
    num_tokens_out = T.dynamic("baseline_num_tokens_out")
    tile_k = tile_k_for(in_dtype, with_rescale=with_rescale)
    vec_k = tile_k // NUM_THREADS_PER_TOKEN
    vec_m = TILE_M * NUM_THREADS_PER_TOKEN // NUM_THREADS

    @T.macro
    def _scale_and_inverse(amax: T.float32):
        clamped_amax = T.max(amax, MIN_AMAX)
        sf = T.alloc_var(T.float32)
        sf_inv = T.alloc_var(T.float32)
        sf = clamped_amax / FP8_MAX
        sf_inv = FP8_MAX / clamped_amax
        if round_sf:
            bits = T.reinterpret(sf, T.uint32)
            exp_sf = ((bits - 1) >> 23) + 1 - 127
            sf = T.reinterpret((127 + exp_sf) << 23, T.float32)
            sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
        return sf, sf_inv

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), in_dtype),
        out: T.Tensor((num_tokens_out, hidden), T.float8_e4m3fn),
        out_sf: T.Tensor((T.ceildiv(num_tokens_out, TILE_M), hidden), T.float32),
        x_sf_invs: T.Tensor(
            (num_tokens, T.ceildiv(hidden, NUM_PER_CHANNELS)),
            T.float32,
        ),
        pos_to_token: T.Tensor((num_tokens_out,), T.int32),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens_out, TILE_M),
            T.ceildiv(hidden, tile_k),
            threads=NUM_THREADS,
        ) as (pid_token, pid_hidden):
            x_shared = T.alloc_shared((TILE_M, tile_k), in_dtype)
            pos_to_token_local = T.alloc_local((vec_m,), T.int32)
            sf_invs_local = T.alloc_local((vec_m,), T.float32)
            amax_local = T.alloc_local((vec_k,), T.float32)
            amax_shared = T.alloc_shared((vec_k, NUM_THREADS), T.float32)
            in_local = T.alloc_local((vec_k,), in_dtype)
            out_local = T.alloc_local((vec_k,), T.float8_e4m3fn)

            tid = T.get_thread_binding(0)
            m_id = tid // NUM_THREADS_PER_TOKEN
            k_id = tid % NUM_THREADS_PER_TOKEN
            m_offset = pid_token * TILE_M + m_id * vec_m
            k_offset = pid_hidden * tile_k + k_id * vec_k
            logical_value = T.alloc_var(T.float32)

            if with_expand:
                position = T.alloc_var(T.int32)
                if k_id < vec_m:
                    position = pos_to_token[k_id + m_offset]
                for i in T.serial(vec_m):
                    pos_to_token_local[i] = T.shfl_sync(position, i)

            if with_rescale:
                for i in T.serial(vec_m):
                    source_token = _source_token(
                        with_expand,
                        i,
                        i + m_offset,
                        pos_to_token_local,
                    )
                    T.assume(source_token < num_tokens)
                    sf_invs_local[i] = T.Select(
                        with_expand and source_token < 0,
                        0.0,
                        x_sf_invs[
                            source_token,
                            (pid_hidden * tile_k + k_id * vec_k) // NUM_PER_CHANNELS,
                        ],
                    )

            T.clear(amax_local)
            for i in T.serial(vec_m):
                source_token = _source_token(
                    with_expand,
                    i,
                    i + m_offset,
                    pos_to_token_local,
                )
                T.assume(source_token < num_tokens)
                if not with_expand or source_token >= 0:
                    for j in T.vectorized(vec_k):
                        in_local[j] = x[source_token, k_offset + j]
                        x_shared[m_id * vec_m + i, k_id * vec_k + j] = in_local[j]
                    for j in T.vectorized(vec_k):
                        logical_value = T.cast(in_local[j], T.float32)
                        if with_rescale:
                            logical_value = logical_value * sf_invs_local[i]
                        amax_local[j] = T.max(amax_local[j], T.abs(logical_value))
                else:
                    for j in T.vectorized(vec_k):
                        x_shared[m_id * vec_m + i, k_id * vec_k + j] = 0.0

            for i in T.unroll(vec_k):
                amax_shared[i, tid] = amax_local[i]

            sf = T.alloc_var(T.float32)
            sf_inv = T.alloc_var(T.float32)
            sf = 0.0
            sf_inv = 0.0
            col_id = tid % NUM_THREADS_PER_TOKEN * vec_k + tid // NUM_THREADS_PER_TOKEN
            if tid < tile_k:
                for i in T.serial(
                    col_id // vec_k,
                    NUM_THREADS,
                    NUM_THREADS_PER_TOKEN,
                ):
                    sf = T.max(sf, amax_shared[col_id % vec_k, i])
                sf, sf_inv = _scale_and_inverse(sf)
                out_sf[pid_token, pid_hidden * tile_k + col_id] = sf
                amax_shared[0, tid] = sf_inv

            for i in T.serial(vec_k):
                amax_local[i] = amax_shared[0, k_id + i * NUM_THREADS_PER_TOKEN]

            for i in T.serial(vec_m):
                for j in T.vectorized(vec_k):
                    in_local[j] = x_shared[m_id * vec_m + i, k_id * vec_k + j]
                for j in T.vectorized(vec_k):
                    logical_value = T.cast(in_local[j], T.float32)
                    if with_rescale:
                        logical_value = logical_value * sf_invs_local[i]
                    out_local[j] = logical_value * amax_local[j]
                for j in T.vectorized(vec_k):
                    out[i + m_offset, j + k_offset] = out_local[j]

    return main


class PerChannelCastFusedTileLangBaseline:
    """Callable pinned TileKernels baseline used only by tests and benchmarks."""

    def __init__(
        self,
        *,
        hidden: int,
        in_dtype: torch.dtype,
        with_rescale: bool,
        with_expand: bool,
        round_sf: bool,
    ) -> None:
        self.with_rescale = with_rescale
        self.with_expand = with_expand
        self.tile_k = tile_k_for(in_dtype, with_rescale=with_rescale)
        if hidden <= 0 or hidden % self.tile_k != 0:
            raise ValueError(
                f"hidden must be positive and divisible by {self.tile_k}, got {hidden}"
            )
        self.kernel = _get_tilelang_baseline_kernel(
            hidden,
            str(in_dtype).removeprefix("torch."),
            with_rescale,
            with_expand,
            round_sf,
        )

    def __call__(
        self,
        x: torch.Tensor,
        x_sf_invs: Optional[torch.Tensor] = None,
        pos_to_token: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.with_rescale and x_sf_invs is None:
            raise ValueError("x_sf_invs is required for the rescale baseline")
        if self.with_expand and pos_to_token is None:
            raise ValueError("pos_to_token is required for the expand baseline")

        num_tokens_out = pos_to_token.shape[0] if self.with_expand else x.shape[0]
        # The pinned upstream schedule has no masked TILE_M tail path.  Keep
        # performance comparisons on complete 128-token tiles even though the
        # production TileOP supports the broader 16-token Expand contract.
        required_alignment = TILE_M
        if num_tokens_out % required_alignment != 0:
            raise ValueError(
                "the pinned upstream-style TileLang baseline requires "
                f"num_tokens_out divisible by {required_alignment}, got {num_tokens_out}"
            )

        out = torch.empty(
            (num_tokens_out, x.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        out_sf = torch.empty(
            ((num_tokens_out + TILE_M - 1) // TILE_M, x.shape[1]),
            dtype=torch.float32,
            device=x.device,
        )
        if num_tokens_out > 0:
            self.kernel(x, out, out_sf, x_sf_invs, pos_to_token)
        return out, out_sf
