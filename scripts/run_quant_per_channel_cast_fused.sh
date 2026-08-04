#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
tilelang_root=${TILELANG_ROOT:-/opt/tilelang-metax-v0.1.10}
export PYTHONPATH="${tilelang_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    echo "usage: $0 {smoke|correctness|baselines|benchmark|gates|profile|profile-driver} [args...]" >&2
}

command_name=${1:-}
if [[ -z "${command_name}" ]]; then
    usage
    exit 2
fi
shift

cd "${repo_root}"
case "${command_name}" in
    smoke)
        python -m pytest -q -m smoke tests/ops/test_per_channel_cast_fused.py "$@"
        ;;
    correctness)
        python -m pytest -q tests/ops/test_per_channel_cast_fused.py "$@"
        ;;
    baselines)
        python -m pytest -q tests/ops/test_per_channel_cast_fused_baselines.py "$@"
        ;;
    benchmark)
        python -m pytest -vvs benchmarks/ops/bench_per_channel_cast_fused.py "$@"
        ;;
    gates)
        git diff --check
        python scripts/validate_manifest.py
        python scripts/validate_manifest.py \
            --check-op QuantPerChannelCastFusedOp \
            --strict
        python -m pytest -q benchmarks/tests
        python -m pytest -q tests/test_ops_manifest.py
        python -m ruff check \
            tileops/kernels/quant/per_channel_cast_fused.py \
            tileops/ops/quant/per_channel_cast_fused.py \
            tileops/testing/per_channel_cast_fused.py \
            tests/ops/test_per_channel_cast_fused.py \
            tests/ops/test_per_channel_cast_fused_baselines.py \
            benchmarks/ops/bench_per_channel_cast_fused.py \
            benchmarks/ops/per_channel_cast_fused_baselines.py \
            benchmarks/ops/profile_per_channel_cast_fused.py
        ;;
    profile-driver)
        python benchmarks/ops/profile_per_channel_cast_fused.py "$@"
        ;;
    profile)
        profile_case=${1:-}
        staging=${2:-production}
        case "${profile_case}" in
            plain-medium|expand-medium|rescale-control) ;;
            *) usage; exit 2 ;;
        esac
        case "${staging}" in
            production|shared) ;;
            *) usage; exit 2 ;;
        esac
        profiler_bin=${MCPROFILER_BIN:-/opt/mcProfiler-ubuntu18.04/mcProfiler}
        "${profiler_bin}" perf_exec \
            --cmdline "${repo_root}/scripts/run_quant_per_channel_cast_fused.sh profile-driver --case ${profile_case} --staging ${staging}" \
            --cwd "${repo_root}" \
            --kernelname main_kernel \
            --casename "quant_${profile_case//-/_}_${staging}" \
            --counts 1 \
            --single-pass
        ;;
    *)
        usage
        exit 2
        ;;
esac
