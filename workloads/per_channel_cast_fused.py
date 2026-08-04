"""Input generation shared by fused per-channel cast tests and benchmarks."""

from __future__ import annotations

import torch

from workloads.workload_base import WorkloadBase


class PerChannelCastFusedWorkload(WorkloadBase):
    def __init__(
        self,
        x_shape: tuple[int, int],
        dtype: torch.dtype,
        *,
        with_rescale: bool = False,
        with_expand: bool = False,
        num_tokens_out: int | None = None,
        round_sf: bool = False,
    ) -> None:
        self.x_shape = x_shape
        self.shape = x_shape
        self.dtype = dtype
        self.with_rescale = with_rescale
        self.with_expand = with_expand
        self.num_tokens_out = num_tokens_out
        self.round_sf = round_sf

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        num_tokens, hidden = self.x_shape
        if self.with_rescale:
            x = (
                torch.randn(self.x_shape, dtype=torch.float32, device="cuda")
                .clamp(-4.0, 4.0)
                .to(torch.float8_e4m3fn)
            )
            x_sf_invs = (
                torch.rand(
                    (num_tokens, hidden // 128),
                    dtype=torch.float32,
                    device="cuda",
                )
                + 0.01
            )
        else:
            x = torch.randn(self.x_shape, dtype=self.dtype, device="cuda")
            x_sf_invs = None

        pos_to_token = None
        if self.with_expand:
            if self.num_tokens_out is None:
                raise ValueError("num_tokens_out is required when with_expand=True")
            if num_tokens == 0:
                pos_to_token = torch.full(
                    (self.num_tokens_out,), -1, dtype=torch.int32, device="cuda"
                )
            else:
                pos_to_token = torch.arange(self.num_tokens_out, dtype=torch.int32, device="cuda")
                pos_to_token %= num_tokens
                pos_to_token[::17] = -1

        inputs: list[torch.Tensor] = [x]
        if x_sf_invs is not None:
            inputs.append(x_sf_invs)
        if pos_to_token is not None:
            inputs.append(pos_to_token)
        return tuple(inputs)
