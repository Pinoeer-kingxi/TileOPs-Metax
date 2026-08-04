"""Stable mcProfiler driver for per-channel fused-cast staging experiments."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import torch

from tileops.kernels.quant.per_channel_cast_fused import PerChannelCastFusedKernel

_CASES = {
    "plain-medium": {
        "x_shape": (1024, 4096),
        "dtype": torch.bfloat16,
        "with_rescale": False,
        "with_expand": False,
        "num_tokens_out": 1024,
        "round_sf": True,
    },
    "expand-medium": {
        "x_shape": (512, 4096),
        "dtype": torch.bfloat16,
        "with_rescale": False,
        "with_expand": True,
        "num_tokens_out": 1024,
        "round_sf": False,
    },
    "rescale-control": {
        "x_shape": (1024, 4096),
        "dtype": torch.float8_e4m3fn,
        "with_rescale": True,
        "with_expand": False,
        "num_tokens_out": 1024,
        "round_sf": True,
    },
}


def _make_inputs(case: dict) -> tuple[torch.Tensor, ...]:
    num_tokens, hidden = case["x_shape"]
    x = torch.ones(case["x_shape"], dtype=case["dtype"], device="cuda")
    inputs: list[torch.Tensor] = [x]
    if case["with_rescale"]:
        inputs.append(torch.ones((num_tokens, hidden // 128), dtype=torch.float32, device="cuda"))
    if case["with_expand"]:
        positions = torch.arange(case["num_tokens_out"], dtype=torch.int32)
        positions %= num_tokens
        positions[::17] = -1
        inputs.append(positions.to("cuda"))
    return tuple(inputs)


def _call_kernel(
    kernel: PerChannelCastFusedKernel,
    inputs: tuple[torch.Tensor, ...],
    *,
    with_rescale: bool,
    with_expand: bool,
) -> None:
    if with_rescale and with_expand:
        kernel(inputs[0], x_sf_invs=inputs[1], pos_to_token=inputs[2])
    elif with_rescale:
        kernel(inputs[0], x_sf_invs=inputs[1])
    elif with_expand:
        kernel(inputs[0], pos_to_token=inputs[1])
    else:
        kernel(inputs[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(_CASES), required=True)
    parser.add_argument("--staging", choices=("production", "shared"), default="production")
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    case = _CASES[args.case]
    register_staging = not case["with_rescale"] and args.staging == "production"
    kernel = PerChannelCastFusedKernel(
        num_tokens=case["x_shape"][0],
        num_tokens_out=case["num_tokens_out"],
        hidden=case["x_shape"][1],
        in_dtype=case["dtype"],
        with_rescale=case["with_rescale"],
        with_expand=case["with_expand"],
        round_sf=case["round_sf"],
        device=torch.device("cuda"),
        config={"register_staging": register_staging},
    )
    inputs = _make_inputs(case)
    for _ in range(args.warmup):
        _call_kernel(
            kernel,
            inputs,
            with_rescale=case["with_rescale"],
            with_expand=case["with_expand"],
        )
    torch.cuda.synchronize()

    expected_workgroups = ((case["num_tokens_out"] + 127) // 128) * (
        (case["x_shape"][1] + 63) // 64
    )
    repo_root = Path(__file__).parents[2]
    source = repo_root / "tileops/kernels/quant/per_channel_cast_fused.py"
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
        ).strip()
    )
    print(
        f"commit={commit} dirty={dirty} torch={torch.__version__} "
        f"device={torch.cuda.get_device_name(0)!r} case={args.case} staging={args.staging} "
        f"register_staging={register_staging} expected_workgroups={expected_workgroups} "
        f"shared_bytes={kernel.shared_memory_bytes} "
        f"kernel_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}"
    )
    _call_kernel(
        kernel,
        inputs,
        with_rescale=case["with_rescale"],
        with_expand=case["with_expand"],
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
