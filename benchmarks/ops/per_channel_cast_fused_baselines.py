# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""Pinned baselines for the per-channel fused-cast benchmark.

The TileLang baseline follows MetaX-MACA/TileKernels-Metax ``dev`` at the
commit recorded below. Its only scheduling change is ``tile_k=64``: the
upstream FP32 ``tile_k=128`` layout needs about 66 KiB of explicit shared
memory and cannot fit within the C500's 64 KiB workgroup limit.

This module is intentionally independent from the TileOPs kernel so later
optimizations cannot silently move the performance baseline.
"""

from __future__ import annotations

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.testing.per_channel_cast_fused import (
    UPSTREAM_COMMIT,
    UPSTREAM_TORCH_REFERENCE_PATH,
    UPSTREAM_TORCH_REFERENCE_SHA256,
)

__all__ = [
    "TILE_K",
    "UPSTREAM_COMMIT",
    "UPSTREAM_KERNEL_PATH",
    "UPSTREAM_KERNEL_SHA256",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_TORCH_REFERENCE_PATH",
    "UPSTREAM_TORCH_REFERENCE_SHA256",
    "PerChannelCastFusedTileLangBaseline",
]

UPSTREAM_REPOSITORY = "https://github.com/MetaX-MACA/TileKernels-Metax"
UPSTREAM_KERNEL_PATH = "tile_kernels/quant/per_channel_cast_fused_kernel.py"
UPSTREAM_KERNEL_SHA256 = "64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc"

TILE_M = 128
TILE_K = 64
NUM_THREADS = 256
NUM_THREADS_PER_TOKEN = 64
NUM_PER_CHANNELS = 128
FP8_MAX = 448.0
MIN_AMAX = 1e-4


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
    """Build the pinned upstream-style kernel with the C500-safe tile."""
    num_tokens = T.dynamic("baseline_num_tokens")
    num_tokens_out = T.dynamic("baseline_num_tokens_out")
    vec_k = TILE_K // NUM_THREADS_PER_TOKEN
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
            T.ceildiv(hidden, TILE_K),
            threads=NUM_THREADS,
        ) as (pid_token, pid_hidden):
            x_shared = T.alloc_shared((TILE_M, TILE_K), in_dtype)
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
            k_offset = pid_hidden * TILE_K + k_id * vec_k
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
                            (pid_hidden * TILE_K + k_id * vec_k) // NUM_PER_CHANNELS,
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
            if tid < TILE_K:
                for i in T.serial(
                    col_id // vec_k,
                    NUM_THREADS,
                    NUM_THREADS_PER_TOKEN,
                ):
                    sf = T.max(sf, amax_shared[col_id % vec_k, i])
                sf, sf_inv = _scale_and_inverse(sf)
                out_sf[pid_token, pid_hidden * TILE_K + col_id] = sf
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
        if hidden <= 0 or hidden % TILE_K != 0:
            raise ValueError(f"hidden must be positive and divisible by {TILE_K}, got {hidden}")
        self.with_rescale = with_rescale
        self.with_expand = with_expand
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
        if num_tokens_out % TILE_M != 0:
            raise ValueError(
                "the pinned upstream-style TileLang baseline requires "
                f"num_tokens_out divisible by {TILE_M}, got {num_tokens_out}"
            )

        out = torch.empty(
            (num_tokens_out, x.shape[1]),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        out_sf = torch.empty(
            (num_tokens_out // TILE_M, x.shape[1]),
            dtype=torch.float32,
            device=x.device,
        )
        if num_tokens_out > 0:
            self.kernel(x, out, out_sf, x_sf_invs, pos_to_token)
        return out, out_sf
