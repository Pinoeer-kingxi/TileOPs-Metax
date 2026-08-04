"""Ops for fused per-channel FP8 quantization."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ...kernels.kernel_base import Kernel
from ...kernels.quant import PerChannelCastFusedKernel
from ..op_base import Op

__all__ = [
    "QuantPerChannelCastFusedExpandOp",
    "QuantPerChannelCastFusedOp",
    "QuantPerChannelCastFusedRescaleExpandOp",
    "QuantPerChannelCastFusedRescaleOp",
]

_NUM_PER_TOKENS = 128
_NUM_PER_CHANNELS = 128
_PLAIN_DTYPES = (torch.bfloat16, torch.float32)


class _QuantPerChannelCastFusedBase(Op):
    _with_rescale = False
    _with_expand = False

    def __init__(
        self,
        fmt: str = "e4m3",
        num_per_tokens: int = 128,
        num_per_channels: Optional[int] = None,
        round_sf: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if fmt != "e4m3":
            raise ValueError(f"fmt must be 'e4m3', got {fmt!r}")
        if num_per_tokens != _NUM_PER_TOKENS:
            raise ValueError(f"num_per_tokens must be {_NUM_PER_TOKENS}, got {num_per_tokens}")
        if self._with_rescale and num_per_channels != _NUM_PER_CHANNELS:
            raise ValueError(
                f"num_per_channels must be {_NUM_PER_CHANNELS}, got {num_per_channels}"
            )
        if not isinstance(round_sf, bool):
            raise TypeError(f"round_sf must be bool, got {type(round_sf).__name__}")

        self.fmt = fmt
        self.num_per_tokens = num_per_tokens
        self.num_per_channels = num_per_channels
        self.round_sf = round_sf
        self.tune = tune
        self.num_tokens: Optional[int] = None
        self.num_tokens_out: Optional[int] = None
        self.hidden: Optional[int] = None
        self.sf_rows: Optional[int] = None
        self.sf_cols: Optional[int] = None
        self.elem_bytes: Optional[int] = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._validated_position_maps: set[tuple[int, int, int, int]] = set()
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"per_channel_cast_fused_kernel": PerChannelCastFusedKernel}

    def _get_kernel(
        self,
        x: torch.Tensor,
        num_tokens_out: int,
    ) -> Kernel:
        key = (
            x.shape[0],
            num_tokens_out,
            x.shape[1],
            x.dtype,
            self._with_rescale,
            self._with_expand,
            self.round_sf,
            x.device.index,
            self.tune,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["per_channel_cast_fused_kernel"](
                num_tokens=x.shape[0],
                num_tokens_out=num_tokens_out,
                hidden=x.shape[1],
                in_dtype=x.dtype,
                with_rescale=self._with_rescale,
                with_expand=self._with_expand,
                round_sf=self.round_sf,
                device=x.device,
                tune=self.tune,
            )
        return self._kernel_cache[key]

    @staticmethod
    def _validate_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _validate_common_shapes(
        self,
        x: torch.Tensor,
        x_sf_invs: Optional[torch.Tensor],
        pos_to_token: Optional[torch.Tensor],
    ) -> int:
        if x.ndim != 2:
            raise ValueError(f"x must have shape [num_tokens, hidden], got {tuple(x.shape)}")
        self._validate_cuda_contiguous("x", x)
        num_tokens, hidden = x.shape
        hidden_alignment = 256 if self._with_rescale else 128
        if hidden <= 0 or hidden % hidden_alignment != 0:
            raise ValueError(
                f"hidden must be positive and divisible by {hidden_alignment}, got {hidden}"
            )

        num_tokens_out = num_tokens
        if self._with_expand:
            if pos_to_token is None:
                raise ValueError("pos_to_token is required for the expand variant")
            if pos_to_token.ndim != 1:
                raise ValueError(f"pos_to_token must be 1D, got shape {tuple(pos_to_token.shape)}")
            self._validate_cuda_contiguous("pos_to_token", pos_to_token)
            if pos_to_token.device != x.device:
                raise ValueError("pos_to_token must be on the same device as x")
            num_tokens_out = pos_to_token.shape[0]
            if num_tokens_out % 16 != 0:
                raise ValueError(f"num_tokens_out must be divisible by 16, got {num_tokens_out}")
            version = int(getattr(pos_to_token, "_version", 0))
            validation_key = (
                pos_to_token.data_ptr(),
                version,
                pos_to_token.numel(),
                num_tokens,
            )
            if validation_key not in self._validated_position_maps:
                if num_tokens_out > 0 and bool(torch.any(pos_to_token >= num_tokens).item()):
                    max_pos = int(pos_to_token.max().item())
                    raise ValueError(
                        "each non-negative pos_to_token entry must be less than "
                        f"num_tokens={num_tokens}, got maximum {max_pos}"
                    )
                if len(self._validated_position_maps) >= 16:
                    self._validated_position_maps.clear()
                self._validated_position_maps.add(validation_key)
        elif num_tokens % self.num_per_tokens != 0:
            raise ValueError(
                f"num_tokens must be divisible by {self.num_per_tokens}, got {num_tokens}"
            )

        if self._with_rescale:
            if x_sf_invs is None:
                raise ValueError("x_sf_invs is required for the rescale variant")
            if x_sf_invs.ndim != 2:
                raise ValueError(f"x_sf_invs must be 2D, got shape {tuple(x_sf_invs.shape)}")
            self._validate_cuda_contiguous("x_sf_invs", x_sf_invs)
            if x_sf_invs.device != x.device:
                raise ValueError("x_sf_invs must be on the same device as x")
            if x_sf_invs.shape[0] != num_tokens:
                raise ValueError(
                    "x_sf_invs.shape[0] must equal x.shape[0], got "
                    f"{x_sf_invs.shape[0]} and {num_tokens}"
                )
            if x_sf_invs.shape[1] * self.num_per_channels != hidden:
                raise ValueError(
                    "x_sf_invs.shape[1] * 128 must equal hidden, got "
                    f"{x_sf_invs.shape[1]} * 128 != {hidden}"
                )

        return num_tokens_out

    def _run(
        self,
        x: torch.Tensor,
        x_sf_invs: Optional[torch.Tensor] = None,
        pos_to_token: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens_out = self._validate_common_shapes(x, x_sf_invs, pos_to_token)
        self.num_tokens = x.shape[0]
        self.num_tokens_out = num_tokens_out
        self.hidden = x.shape[1]
        self.sf_rows = (num_tokens_out + self.num_per_tokens - 1) // self.num_per_tokens
        self.sf_cols = None if x_sf_invs is None else x_sf_invs.shape[1]
        self.elem_bytes = x.element_size()

        if num_tokens_out == 0:
            return (
                torch.empty((0, self.hidden), dtype=torch.float8_e4m3fn, device=x.device),
                torch.empty((0, self.hidden), dtype=torch.float32, device=x.device),
            )

        self.kernel = self._get_kernel(x, num_tokens_out)
        return self.kernel(x, x_sf_invs, pos_to_token)

    def _bound_roofline_dims(self) -> tuple[int, int, int]:
        if self.num_tokens_out is None or self.hidden is None or self.sf_rows is None:
            raise RuntimeError("eval_roofline() requires a prior forward() call")
        return self.num_tokens_out, self.hidden, self.sf_rows


class QuantPerChannelCastFusedOp(_QuantPerChannelCastFusedBase):
    """Quantize BF16/FP32 input with 128-token per-channel FP8 scales."""

    def __init__(
        self,
        fmt: str = "e4m3",
        num_per_tokens: int = 128,
        round_sf: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
            fmt=fmt,
            num_per_tokens=num_per_tokens,
            round_sf=round_sf,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_dtypes(x)
        return self._run(x)

    def _infer_output_shapes(self, x_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        num_tokens, hidden = x_shape
        sf_rows = (num_tokens + self.num_per_tokens - 1) // self.num_per_tokens
        return {"out": (num_tokens, hidden), "out_sf": (sf_rows, hidden)}

    def _validate_dtypes(self, x: torch.Tensor) -> None:
        if x.dtype not in _PLAIN_DTYPES:
            raise ValueError(f"x.dtype must be bfloat16 or float32, got {x.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        num_tokens, hidden, sf_rows = self._bound_roofline_dims()
        assert self.elem_bytes is not None
        flops = 3 * num_tokens * hidden + 2 * sf_rows * hidden
        bytes_ = num_tokens * hidden * self.elem_bytes + num_tokens * hidden + sf_rows * hidden * 4
        return int(flops), int(bytes_)


class QuantPerChannelCastFusedExpandOp(_QuantPerChannelCastFusedBase):
    """Gather tokens, then quantize with 128-token per-channel FP8 scales."""

    _with_expand = True

    def __init__(
        self,
        fmt: str = "e4m3",
        num_per_tokens: int = 128,
        round_sf: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
            fmt=fmt,
            num_per_tokens=num_per_tokens,
            round_sf=round_sf,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_to_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_dtypes(x, pos_to_token)
        return self._run(x, pos_to_token=pos_to_token)

    def _infer_output_shapes(
        self,
        x_shape: tuple[int, ...],
        pos_to_token_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        hidden = x_shape[1]
        num_tokens_out = pos_to_token_shape[0]
        sf_rows = (num_tokens_out + self.num_per_tokens - 1) // self.num_per_tokens
        return {"out": (num_tokens_out, hidden), "out_sf": (sf_rows, hidden)}

    def _validate_dtypes(self, x: torch.Tensor, pos_to_token: torch.Tensor) -> None:
        if x.dtype not in _PLAIN_DTYPES:
            raise ValueError(f"x.dtype must be bfloat16 or float32, got {x.dtype}")
        if pos_to_token.dtype != torch.int32:
            raise ValueError(f"pos_to_token.dtype must be int32, got {pos_to_token.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        num_tokens_out, hidden, sf_rows = self._bound_roofline_dims()
        assert self.elem_bytes is not None
        flops = 3 * num_tokens_out * hidden + 2 * sf_rows * hidden
        bytes_ = (
            num_tokens_out * hidden * self.elem_bytes
            + num_tokens_out * 4
            + num_tokens_out * hidden
            + sf_rows * hidden * 4
        )
        return int(flops), int(bytes_)


class QuantPerChannelCastFusedRescaleOp(_QuantPerChannelCastFusedBase):
    """Rescale FP8 input, then requantize with per-channel FP8 scales."""

    _with_rescale = True

    def __init__(
        self,
        fmt: str = "e4m3",
        num_per_tokens: int = 128,
        num_per_channels: int = 128,
        round_sf: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
            fmt=fmt,
            num_per_tokens=num_per_tokens,
            num_per_channels=num_per_channels,
            round_sf=round_sf,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_sf_invs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_dtypes(x, x_sf_invs)
        return self._run(x, x_sf_invs=x_sf_invs)

    def _infer_output_shapes(
        self,
        x_shape: tuple[int, ...],
        x_sf_invs_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        del x_sf_invs_shape
        num_tokens, hidden = x_shape
        sf_rows = (num_tokens + self.num_per_tokens - 1) // self.num_per_tokens
        return {"out": (num_tokens, hidden), "out_sf": (sf_rows, hidden)}

    def _validate_dtypes(self, x: torch.Tensor, x_sf_invs: torch.Tensor) -> None:
        if x.dtype != torch.float8_e4m3fn:
            raise ValueError(f"x.dtype must be float8_e4m3fn, got {x.dtype}")
        if x_sf_invs.dtype != torch.float32:
            raise ValueError(f"x_sf_invs.dtype must be float32, got {x_sf_invs.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        num_tokens, hidden, sf_rows = self._bound_roofline_dims()
        assert self.sf_cols is not None
        flops = 5 * num_tokens * hidden + 2 * sf_rows * hidden
        bytes_ = 2 * num_tokens * hidden + num_tokens * self.sf_cols * 4 + sf_rows * hidden * 4
        return int(flops), int(bytes_)


class QuantPerChannelCastFusedRescaleExpandOp(_QuantPerChannelCastFusedBase):
    """Gather and rescale FP8 input, then requantize per channel."""

    _with_rescale = True
    _with_expand = True

    def __init__(
        self,
        fmt: str = "e4m3",
        num_per_tokens: int = 128,
        num_per_channels: int = 128,
        round_sf: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
            fmt=fmt,
            num_per_tokens=num_per_tokens,
            num_per_channels=num_per_channels,
            round_sf=round_sf,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_sf_invs: torch.Tensor,
        pos_to_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_dtypes(x, x_sf_invs, pos_to_token)
        return self._run(x, x_sf_invs=x_sf_invs, pos_to_token=pos_to_token)

    def _infer_output_shapes(
        self,
        x_shape: tuple[int, ...],
        x_sf_invs_shape: tuple[int, ...],
        pos_to_token_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        del x_sf_invs_shape
        hidden = x_shape[1]
        num_tokens_out = pos_to_token_shape[0]
        sf_rows = (num_tokens_out + self.num_per_tokens - 1) // self.num_per_tokens
        return {"out": (num_tokens_out, hidden), "out_sf": (sf_rows, hidden)}

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        x_sf_invs: torch.Tensor,
        pos_to_token: torch.Tensor,
    ) -> None:
        if x.dtype != torch.float8_e4m3fn:
            raise ValueError(f"x.dtype must be float8_e4m3fn, got {x.dtype}")
        if x_sf_invs.dtype != torch.float32:
            raise ValueError(f"x_sf_invs.dtype must be float32, got {x_sf_invs.dtype}")
        if pos_to_token.dtype != torch.int32:
            raise ValueError(f"pos_to_token.dtype must be int32, got {pos_to_token.dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        num_tokens_out, hidden, sf_rows = self._bound_roofline_dims()
        assert self.sf_cols is not None
        flops = 5 * num_tokens_out * hidden + 2 * sf_rows * hidden
        bytes_ = (
            2 * num_tokens_out * hidden
            + num_tokens_out * self.sf_cols * 4
            + num_tokens_out * 4
            + sf_rows * hidden * 4
        )
        return int(flops), int(bytes_)
