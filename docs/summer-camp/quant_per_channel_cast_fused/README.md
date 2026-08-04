# quant_per_channel_cast_fused 迁移与 MetaX C500 实测报告

本文记录 `quant_per_channel_cast_fused` 从 TileKernels-Metax 迁移到
TileOPs-Metax 的完整过程，包括接口对应关系、代码目录、TileLang Kernel
计算过程、访存模型、正确性测试、Benchmark、mcProfiler 和 Roofline 实测。

迁移完成后的外部实现对比、`tile_k=64` 两轮 A/B、失败实验和分阶段开发
计划见 [深度优化分析与开发计划](OPTIMIZATION_ANALYSIS.md)。

## 1. 迁移信息

| 项目 | 内容 |
|---|---|
| 算子 | `quant_per_channel_cast_fused` |
| TileOPs 分支 | `feat/quant-per-channel-cast-fused` |
| TileOPs 迁移基线 | `70c85c45bd476ee53134afcd36d90a19fbf409c5` |
| 原迁移实现提交 | `967b65b7775f9523b884bacf410bb66032c505f0` |
| `tile_k=64` 与固定基线提交 | `253fe14e7c33e5c9851e6120d46caf78cbe3d13b` |
| 上游仓库 | `https://github.com/MetaX-MACA/TileKernels-Metax` |
| 上游提交 | `0266ab740980de7dc03a828b8259cd73d100c2eb` |
| 上游 TileLang 源码 | `tile_kernels/quant/per_channel_cast_fused_kernel.py` |
| 上游 TileLang SHA256 | `64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc` |
| 上游 PyTorch reference | `tile_kernels/torch/per_channel_cast_fused.py` |
| 上游 PyTorch SHA256 | `6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f` |
| 仓内固定 TileLang baseline blob | `4a1c4722fd6dcd3f9fbc295ed3cb2fcbdb76d652` |
| 仓内 eager PyTorch baseline blob | `adc604bcd8c10c06f01bc6f84f9bb79f582aa286` |
| Manifest PR | https://gitlink.org.cn/ccf-ai-infra/TileOPs-Metax/pulls/29 |
| 实测日期 | 2026-08-04 |
| 实测设备 | MetaX C500，25% sGPU 切片 |

迁移结果包含四个公开算子：

1. `QuantPerChannelCastFusedOp`
2. `QuantPerChannelCastFusedExpandOp`
3. `QuantPerChannelCastFusedRescaleOp`
4. `QuantPerChannelCastFusedRescaleExpandOp`

四个变体共用一个 TileLang Kernel，通过编译期的 `with_rescale` 和
`with_expand` 参数生成专用代码。

## 2. 代码目录与职责

```text
TileOPs-Metax/
├── tileops/manifest/quantization.yaml
├── tileops/ops/__init__.py
├── tileops/ops/quant/
│   ├── __init__.py
│   └── per_channel_cast_fused.py
├── tileops/kernels/quant/
│   ├── __init__.py
│   └── per_channel_cast_fused.py
├── tileops/testing/per_channel_cast_fused.py
├── tests/ops/test_per_channel_cast_fused.py
├── tests/ops/test_per_channel_cast_fused_baselines.py
├── workloads/per_channel_cast_fused.py
├── benchmarks/ops/per_channel_cast_fused_baselines.py
└── benchmarks/ops/bench_per_channel_cast_fused.py
```

各文件职责如下：

| 路径 | 作用 |
|---|---|
| `tileops/manifest/quantization.yaml` | 定义四个算子的输入输出、参数、shape 规则、workload 和 Roofline 公式 |
| `tileops/ops/quant/per_channel_cast_fused.py` | 对外 Op 接口；负责参数、dtype、shape、CUDA、连续性和索引检查，以及 Kernel 缓存和调度 |
| `tileops/kernels/quant/per_channel_cast_fused.py` | 真正执行设备计算的 TileLang Kernel |
| `tileops/testing/per_channel_cast_fused.py` | 独立 PyTorch 参考实现，不调用被测 Kernel |
| `tests/ops/test_per_channel_cast_fused.py` | 正确性、边界和异常测试 |
| `tests/ops/test_per_channel_cast_fused_baselines.py` | 固定 TileLang 基线与 eager PyTorch 的独立一致性测试 |
| `workloads/per_channel_cast_fused.py` | Benchmark 输入生成 |
| `benchmarks/ops/per_channel_cast_fused_baselines.py` | 从官方 `dev` 源码固定的 benchmark-only TileLang 基线，C500 唯一改动为 `tile_k=64` |
| `benchmarks/ops/bench_per_channel_cast_fused.py` | Manifest 驱动的 TileOPs、固定 TileLang 与 eager PyTorch 三方 Benchmark |
| `tileops/ops/__init__.py` | 导出四个公开 Op |

调用链为：

```text
用户调用 QuantPerChannelCastFused*Op
        │
        ▼
Op.forward()
  ├── dtype/shape/device/layout 校验
  ├── QuantTensor 拆分参数校验
  └── 按输入 shape 缓存 Kernel
        │
        ▼
PerChannelCastFusedKernel.forward()
        │
        ▼
@tilelang.jit + T.prim_func + T.Kernel
        │
        ▼
返回 (out_fp8, out_sf)
```

Op 层不使用 PyTorch 代替设备计算；量化、归约、gather 和 rescale 均在
TileLang Kernel 中完成。

## 3. 接口与上游类型对应

上游函数签名中的 `x` 类型为：

```python
x: Union[torch.Tensor, QuantTensor]
```

其中 `QuantTensor` 表示：

```python
(data, sf_invs)
```

TileOPs 根据是否需要 rescale，将上游的一个 `QuantTensor` 拆成两个显式
Tensor 输入。这样更符合 Manifest 对每个输入分别描述 dtype 和 shape 的约定。

| 上游调用形态 | TileOPs Op | TileOPs 输入 |
|---|---|---|
| 普通 Tensor | `QuantPerChannelCastFusedOp` | `x` |
| 普通 Tensor + gather | `QuantPerChannelCastFusedExpandOp` | `x, pos_to_token` |
| `QuantTensor(data, sf_invs)` | `QuantPerChannelCastFusedRescaleOp` | `x=data, x_sf_invs=sf_invs` |
| `QuantTensor` + gather | `QuantPerChannelCastFusedRescaleExpandOp` | `x=data, x_sf_invs=sf_invs, pos_to_token` |

### 3.1 输入输出 dtype

| 变体 | `x` dtype | 其他输入 | 输出 |
|---|---|---|---|
| plain | `bfloat16 \| float32` | 无 | `out: float8_e4m3fn`，`out_sf: float32` |
| expand | `bfloat16 \| float32` | `pos_to_token: int32` | `out: float8_e4m3fn`，`out_sf: float32` |
| rescale | `float8_e4m3fn` | `x_sf_invs: float32` | `out: float8_e4m3fn`，`out_sf: float32` |
| rescale-expand | `float8_e4m3fn` | `x_sf_invs: float32`，`pos_to_token: int32` | `out: float8_e4m3fn`，`out_sf: float32` |

### 3.2 主要 shape 约束

- `num_per_tokens == 128`。
- rescale 变体要求 `num_per_channels == 128`。
- plain/rescale 的 `num_tokens` 必须能被 128 整除。
- expand 输出 token 数必须能被 16 整除，最后一个 128-token scale block
  可以不足 128 个有效 token。
- plain/expand 的 `hidden` 必须是 128 的正整数倍。
- rescale 变体的 `hidden` 必须是 256 的正整数倍。
- `x_sf_invs.shape == (num_tokens, hidden / 128)`。
- `pos_to_token >= 0` 的元素必须小于源 `num_tokens`；负数表示 padding。
- 所有设备输入必须位于同一 CUDA/MACA 设备并且连续。

## 4. Kernel 计算过程

设：

- 输出 token 数为 `N`；
- hidden 大小为 `H`；
- 每个 scale block 包含 `128` 个 token；
- FP8 e4m3 最大有限值为 `448`；
- 最小 amax 为 `1e-4`。

### 4.1 Tile 与线程映射

Kernel 固定使用：

- `tile_m = 128`；
- `threads = 256`；
- MetaX wavefront 大小为 64；
- 每个 token 方向线程组使用 64 个线程。

所有路径在 MetaX C500 上统一固定：

| 路径 | `tile_k` | 原因 |
|---|---:|---|
| BF16 plain/expand | 64 | 17,408 B shared/CTA；增加 hidden 方向并行度 |
| FP32 plain/expand | 64 | 33,792 B shared/CTA；低于 C500 65,536 B 上限 |
| FP8 rescale/rescale-expand | 64 | 9,216 B shared/CTA；显著降低 CTA 资源占用 |

显式 shared-memory 预算为：

```text
shared_bytes = 128 × tile_k × sizeof(input)
             + (tile_k / 64) × 256 × sizeof(float32)
```

需要准确区分：不是所有 `tile_k=128` 都会溢出。BF16 的 128 tile 为
34,816 B，可以容纳；但 FP32 的 128 tile 为 67,584 B，超过 C500 每个
workgroup 的 65,536 B 上限。统一使用 64 同时解决 FP32 编译/launch 风险，
并在已完成的 C500 A/B 中显著提高 BF16 和 FP8 路径的并行度。

### 4.2 可选 gather/expand

当 `with_expand=True` 时：

1. 读取 `pos_to_token[out_token]`；
2. 使用 `T.shfl_sync` 在 64-thread 线程组中广播索引；
3. 非负索引指向源 token；
4. 负索引视为 padding，逻辑输入为零；
5. 最后一个非完整 128-token tile 使用边界判断，不访问越界位置。

该边界保护修复了 expand 尾块可能越界的问题。

### 4.3 可选 FP8 rescale

rescale 输入对应上游 QuantTensor：

```text
logical[t, h] = float32(x[t, h])
              × x_sf_invs[t, h // 128]
```

其中：

- `x` 是 `float8_e4m3fn` 数据；
- `x_sf_invs` 是每个 token、每 128 个 hidden channel 一组的
  FP32 反量化 scale。

若同时启用 expand，则先通过 `pos_to_token` 找到源 token，再读取其
`x` 和 `x_sf_invs`。

### 4.4 第一遍：加载和计算 amax

每个工作组处理一个 `128 × tile_k` tile：

1. 从 HBM 读取 `x`；
2. 把原始输入写入 `x_shared`；
3. 转为 FP32；
4. rescale 路径乘以 `x_sf_invs`；
5. 每个线程在寄存器中累计对应 channel 的
   `max(abs(logical_value))`；
6. 把局部 amax 写入 `amax_shared`。

输入值只从 HBM 读取一次。后续量化阶段再次使用输入时，从 shared memory
读取，不会再次读取 HBM 中的 `x`。

### 4.5 归约和输出 scale

对每个输出 channel，在 128 个 token 上归约：

```text
amax = max(max(abs(logical[:, h])), 1e-4)
```

不启用 `round_sf` 时：

```text
out_sf = amax / 448
quant_sf = 448 / amax
```

启用 `round_sf` 时：

```text
out_sf = ceil_power_of_two(amax / 448)
quant_sf = 1 / out_sf
```

Kernel 使用 FP32 位操作生成向上取整的 2 的幂，避免额外的高层算子。
每个 128-token block、每个 hidden channel 写一个 FP32 `out_sf`。

### 4.6 第二遍：量化并写回

1. 从 `x_shared` 读取原始输入；
2. 转为 FP32；
3. rescale 路径再次乘 `x_sf_invs`；
4. 乘 `quant_sf`；
5. 转为 `float8_e4m3fn`；
6. 写入 `out`。

rescale 的乘法执行两次，分别服务于 amax 计算和最终量化；第二遍输入读取
来自 shared memory。选择存储原始 FP8 而不是完整 FP32 logical tile，
可以避免超过 C500 的 64 KiB shared-memory 限制。

## 5. 访存次数与 Roofline 字节模型

### 5.1 每个有效输出元素

| 访问 | plain/expand | rescale/rescale-expand |
|---|---:|---:|
| HBM 读取 `x` | 1 次 | 1 次 |
| shared 写入输入 | 1 次 | 1 次 |
| shared 读取输入 | 1 次 | 1 次 |
| HBM 写入 FP8 `out` | 1 次 | 1 次 |
| FP32 `x_sf_invs` | 无 | 每 token、每 128 channel 组逻辑读取一次 |
| `pos_to_token` | expand 时每输出 token 逻辑读取一次 | expand 时每输出 token 逻辑读取一次 |
| FP32 `out_sf` | 每 128-token block、每 channel 写 1 次 | 同左 |

这里的 shared 读写不计入 HBM Roofline bytes。特别需要注意：Kernel 的第二遍
输入读取来自 `x_shared`，不能把 `x` 重复计为第二次 HBM 读取。

### 5.2 Manifest 公式

记：

- `N`：输出 token 数；
- `H`：hidden；
- `S = ceil(N / 128)`；
- `C = H / 128`；
- `e`：plain 输入元素字节数，BF16 为 2、FP32 为 4。

| 变体 | FLOPs | HBM 逻辑字节数 |
|---|---|---|
| plain | `3NH + 2SH` | `NHe + NH + 4SH` |
| expand | `3NH + 2SH` | `NHe + 4N + NH + 4SH` |
| rescale | `5NH + 2SH` | `2NH + 4NC + 4SH` |
| rescale-expand | `5NH + 2SH` | `2NH + 4NC + 4N + 4SH` |

这些公式与 `tileops/manifest/quantization.yaml` 和各 Op 的
`eval_roofline()` 一致。

## 6. 正确性测试

### 6.1 独立参考实现

参考实现位于：

```text
tileops/testing/per_channel_cast_fused.py
```

参考实现固定到上游提交 `0266ab7` 的语义，只使用 eager PyTorch 原语，
不经过其他编译器或调用被测 TileLang Kernel。它独立完成：

- QuantTensor 反量化；
- gather/padding；
- 128-token per-channel amax；
- `round_sf`；
- FP8 量化。

比较标准：

- FP8 输出转为 FP32 后要求逐元素完全一致：
  `atol=0, rtol=0`；
- FP32 scale：
  `atol=1e-7, rtol=1e-6`；
- 同时检查输出 shape、dtype 和 contiguous。

### 6.2 测试覆盖

主算子共 29 项测试，覆盖：

- BF16 和 FP32 plain 输入；
- FP8 e4m3 QuantTensor/rescale 输入；
- 四个公开变体；
- `round_sf=False/True`；
- 全零输入与最小 scale；
- 正负极值；
- 重复、逆序和包含 padding 的 gather；
- 全 padding gather；
- 输出 token 数 16、128、144、256；
- 多个 token block；
- 多个 hidden tile；
- 空输入；
- 空源输入加 padding；
- 非法构造参数；
- 非法 dtype、rank、hidden、token 数；
- 非连续输入；
- 非法 `pos_to_token`；
- 非法 `x_sf_invs` shape 和 dtype。

另外有 7 项固定基线测试，覆盖：

- 官方上游提交、TileLang 文件 SHA256 和 PyTorch 文件 SHA256 门禁；
- 固定 `tile_k == 64`；
- BF16、FP32、FP8 输入；
- plain、expand、rescale、rescale-expand；
- 正负极值、随机输入、逆序、重复和 padding position；
- `round_sf=False/True`；
- 固定 TileLang 基线与 eager PyTorch 的 FP8 逐元素一致性；
- 上游式 TileLang 性能基线拒绝非完整 128-token scale block，避免把其
  无尾块保护的适用范围误当作 TileOps 公共语义。

### 6.3 实测命令与结果

环境：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH
```

执行：

```bash
python scripts/validate_manifest.py
python scripts/validate_manifest.py \
  --check-op QuantPerChannelCastFusedOp --strict
python -m pytest -q tests/ops/test_per_channel_cast_fused.py
python -m pytest -q tests/ops/test_per_channel_cast_fused_baselines.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
```

结果：

| 检查 | 结果 |
|---|---|
| 全量 Manifest | 通过；仓库既有 advisory warning 不阻塞 |
| 本算子 strict Manifest | 通过；10 条 synthetic shape precondition warning |
| 算子正确性测试 | `29 passed in 26.07s` |
| 固定基线正确性测试 | `7 passed in 41.66s` |
| Benchmark 基础测试 | `17 passed in 24.17s` |
| Ops Manifest 测试 | `7 passed in 23.09s` |
| Ruff check | `All checks passed!` |
| Ruff format | 通过 |
| `py_compile` | 通过 |
| `git diff --check` | 通过 |
| `pre-commit` | 环境未安装，未执行 |

本次只安装了独立检查工具 `ruff==0.14.13`：

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps ruff==0.14.13
```

下载源为已配置的清华/阿里云国内镜像。没有安装或覆盖 TileOps、TileLang、
PyTorch，也没有使用会解析其依赖的安装命令。`pre-commit` 当前未安装，故以
Ruff、`py_compile`、Manifest 和 pytest 的分项结果作为检查证据。

## 7. Benchmark 实测

### 7.1 方法

Benchmark 位于：

```text
benchmarks/ops/bench_per_channel_cast_fused.py
```

运行命令：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH
python -m pytest -q -s benchmarks/ops/bench_per_channel_cast_fused.py
```

三条被测路径使用完全相同的 Manifest workload 和计时协议：

1. `tileops`：提交 `253fe14` 的公共 Op 和安全 Kernel。
2. `tilelang-baseline`：固定自官方 TileKernels-Metax `dev@0266ab7` 的
   TileLang 算法；唯一调度适配是 C500 全路径 `tile_k=64`，代码独立保存在
   `benchmarks/ops/per_channel_cast_fused_baselines.py`，不会随生产 Kernel
   后续优化而漂移。
3. `torch-eager`：`tileops/testing/per_channel_cast_fused.py` 中的独立
   eager PyTorch 原语组合。

每个 case 在计时前先让 TileOps 和固定 TileLang 基线分别与 eager PyTorch
比较；FP8 输出必须逐元素完全一致，scale 使用 `atol=1e-7, rtol=1e-6`。

稳定态计时协议：

- JIT 编译在计时前完成；
- 10 次 warmup；
- 每个 trial 50 次测量；
- 共 3 个 trial；
- 每次计时前刷新 L2；
- 输入总量允许时使用 3 组 clone 轮换地址；
- 每次测量前后同步；
- 优先使用 CUPTI kernel-only 时间；
- 结果取三个 trial 平均值的中位数；
- 三条路径使用完全相同的输入和计时框架。

原始 Benchmark 报告生成于 `/data/profile_run.log`，未提交大型原始日志。
其 SHA256 为：

```text
08e2ed60869bfa2c2919e872ecf8edbddc5c80b3df78cbd78b1b8351390ec1f4
```

### 7.2 结果

| 变体与 workload | TileOps ms | 固定 TileLang ms | eager PyTorch ms | TileOps/TileLang | TileOps/PyTorch |
|---|---:|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.0325 | 0.0243 | 0.0769 | 0.748× | 2.366× |
| plain 1024×4096 BF16，rounded | 0.0705 | 0.0517 | 0.2619 | 0.733× | 3.715× |
| plain 4096×8192 FP32 | 1.2296 | 0.8020 | 1.5036 | 0.652× | 1.223× |
| expand 512×4096 → 1024 BF16 | 0.0756 | 0.0778 | 0.3926 | 1.029× | 5.193× |
| expand 1024×4096 → 2048 FP32，rounded | 0.3310 | 0.3359 | 0.6582 | 1.015× | 1.989× |
| rescale 128×1024 FP8 | 0.0549 | 0.0525 | 0.0800 | 0.956× | 1.457× |
| rescale 1024×4096 FP8，rounded | 0.0853 | 0.0844 | 0.3731 | 0.989× | 4.374× |
| rescale-expand 512×4096 → 1024 FP8 | 0.1259 | 0.1281 | 0.4482 | 1.017× | 3.560× |
| rescale-expand 1024×4096 → 2048 FP8，rounded | 0.1949 | 0.1974 | 0.7917 | 1.013× | 4.062× |

结论：

- 9 个 workload 全部快于 eager PyTorch，范围为 1.223×～5.193×。
- Expand 与 Rescale-Expand 的 TileOps 安全实现比固定 TileLang 基线快
  1.3%～2.9%；Rescale 两组相差 1.1%～4.6%，基本处于同一量级。
- Plain 三组相对固定 TileLang 基线的吞吐低 25.2%～34.8%，对应延迟高
  33.7%～53.3%（加速口径为 0.652×～0.748×）。固定基线只运行完整
  128-token block，没有 TileOps 公共 Kernel 的尾块边界分支；动态 token
  设计也不同。两项必须分别 A/B，不能直接删除安全检查。
- 这组“三方基线”用于判断当前绝对实现质量；`tile_k=64` 相对迁移初版
  128/256 tile 的收益见优化分析文档中的 22 组独立 A/B。

## 8. mcProfiler 实测

### 8.1 环境

| 项目 | 实测值 |
|---|---|
| GPU | MetaX C500 |
| sGPU Compute | 25% |
| sGPU Vram Quota | 16000 MiB |
| 整卡显存 | 65536 MiB |
| Driver | 3.8.30 |
| MACA | 3.7.1.5 |
| mx-smi | 2.3.1 |
| Python | 3.12.11 |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | 0.1.10，backend=maca |
| mcProfiler | 3.8.1.4 |
| C500 shared memory/workgroup | 64 KiB |
| wavefront | 64 |

### 8.2 可复现 profiling 入口

可使用下面的最小驱动脚本分别运行两个 workload：

```python
# /tmp/profile_per_channel_cast_fused.py
import argparse

import torch

from tileops.ops import (
    QuantPerChannelCastFusedOp,
    QuantPerChannelCastFusedRescaleExpandOp,
)
from workloads.per_channel_cast_fused import PerChannelCastFusedWorkload

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("plain-medium", "rescale-expand"), required=True)
args = parser.parse_args()

torch.manual_seed(1235)
torch.cuda.manual_seed_all(1235)

if args.case == "plain-medium":
    workload = PerChannelCastFusedWorkload(
        (1024, 4096),
        torch.bfloat16,
        round_sf=True,
    )
    op = QuantPerChannelCastFusedOp(round_sf=True)
else:
    workload = PerChannelCastFusedWorkload(
        (512, 4096),
        torch.float8_e4m3fn,
        with_rescale=True,
        with_expand=True,
        num_tokens_out=1024,
    )
    op = QuantPerChannelCastFusedRescaleExpandOp()

inputs = workload.gen_inputs()
for _ in range(30):
    op(*inputs)
torch.cuda.synchronize()

op(*inputs)
torch.cuda.synchronize()
```

命令：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH

mcProfiler perf_exec \
  --cmdline "python /tmp/profile_per_channel_cast_fused.py --case plain-medium" \
  --cwd /data/TileOPs-Metax \
  --kernelname main_kernel \
  --casename quant_plain_medium \
  --counts 1 \
  --single-pass

mcProfiler perf_exec \
  --cmdline "python /tmp/profile_per_channel_cast_fused.py --case rescale-expand" \
  --cwd /data/TileOPs-Metax \
  --kernelname main_kernel \
  --casename quant_rescale_expand \
  --counts 1 \
  --single-pass
```

### 8.3 原迁移实现有效报告

下面两份报告记录提交 `967b65b` 的初始 tile 配置，作为 `tile_k=64` 优化
前的固定 A/B 对照：

| workload | 报告 |
|---|---|
| plain 1024×4096 BF16 rounded | `/opt/mcProfiler-ubuntu18.04/output20260804080605/1_main_kernel.txt` |
| rescale-expand 512×4096 → 1024 FP8 | `/opt/mcProfiler-ubuntu18.04/output20260804080829/1_main_kernel.txt` |

报告 SHA256：

```text
33acb500f76d6d42051d99f52a22cb8f37f9b5380e2a25a26ed7ee337e59731f  plain
4b99c58344153267b4271ab0666570db6d4fe58d6946f655aa035d701362d6f6  rescale-expand
```

### 8.4 原迁移实现关键指标

| 指标 | plain medium | rescale-expand |
|---|---:|---:|
| Workgroups | 256 | 128 |
| Waves | 1024 | 512 |
| Kernel cycles | 108061 | 174296 |
| Kernel duration | 96.054 μs | 154.930 μs |
| AP busy | 13.92% | 17.80% |
| Compute instructions busy | 24.70% | 21.83% |
| MTE duty | 24.70% | 21.73% |
| STE duty | 14.34% | 20.22% |
| VL1 hit | 96.80% | 96.08% |
| L2 hit | 12.41% | 46.08% |
| Shared-memory efficiency | 100% | 100% |
| HBM read | 8,404,992 B | 2,213,888 B |
| HBM write | 4,317,184 B | 4,325,376 B |
| 实测 HBM 总流量 | 12,722,176 B | 6,539,264 B |
| mcProfiler 实测带宽 | 132.448 GB/s | 42.208 GB/s |

plain workload 的 Manifest 逻辑字节数为 12,713,984 B，与 mcProfiler
实测 12,722,176 B 只差 8,192 B，差异约 0.064%，说明访存模型与实际
HBM 流量基本一致。

rescale-expand 的写流量：

```text
FP8 out   = 1024 × 4096 × 1 = 4,194,304 B
FP32 sf   =    8 × 4096 × 4 =   131,072 B
total     =                       4,325,376 B
```

与 mcProfiler 的 HBM write 完全一致。

该 workload 只包含 512 个源 token，却产生 1024 个输出 token，同一源数据
会被 gather 两次，并且 `pos_to_token[::17]` 为 padding。Manifest 按每个
逻辑输出读取计数，而实际重复读取部分被 L2/VL1 缓存吸收，因此实测 HBM
read 低于逻辑公式；46.08% 的 L2 hit 也支持这一解释。

### 8.5 小 workload 限制

对 plain 128×1024 BF16 也运行了 single-pass profiler：

```text
/opt/mcProfiler-ubuntu18.04/output20260804081122/1_main_kernel.txt
```

该 workload 只有 8 个 workgroup、32 个 wave，未充分覆盖设备。
mcProfiler 报告：

```text
RoofLine: cannot draw from data: 'processed_data'
```

因此该次运行只用于说明“小 workload 无法满足 single-pass Roofline
采样要求”，不作为成功 Roofline 报告引用。

### 8.6 `tile_k=64` A/B 实测

提交 `253fe14` 的 Kernel 与隔离 worktree 中已验证的全路径 `tile_k=64`
候选计算代码一致。正式 A/B 使用相同输入、相同 profiler 参数，并通过
Workgroups 数检查排除输入初始化 Kernel 污染。

有效报告：

```text
33acb500f76d6d42051d99f52a22cb8f37f9b5380e2a25a26ed7ee337e59731f
  /opt/mcProfiler-ubuntu18.04/output20260804080605/1_main_kernel.txt
  # plain 1024x4096 BF16, tile128

ba2c9873af6a2be0786e9adf0c312d1c7fd031637ace3c5ba9ecc6013d1513d3
  /opt/mcProfiler-ubuntu18.04/output20260804132836/report.txt
  # plain 1024x4096 BF16, tile64

598bf721f99b7ce5fc26038c230d4b501d8dd9f41b6af1f48100fa425bfa97c0
  /opt/mcProfiler-ubuntu18.04/output20260804133542/report.txt
  # rescale 1024x4096 FP8, tile256

45280caf7e0dea476083267ef336f73ebd9985076df7f5b8b87e67994f42566e
  /opt/mcProfiler-ubuntu18.04/output20260804133241/report.txt
  # rescale 1024x4096 FP8, tile64
```

| 指标 | Plain tile128 | Plain tile64 | Rescale tile256 | Rescale tile64 |
|---|---:|---:|---:|---:|
| Workgroups | 256 | 512 | 128 | 512 |
| Waves | 1024 | 2048 | 512 | 2048 |
| 显式 shared/CTA | 34,816 B | 17,408 B | 36,864 B | 9,216 B |
| Average wave cycles | 13,174 | 7,618 | 37,912 | 18,551 |
| Compute busy | 24.70% | 46.04% | 17.70% | 50.70% |
| MTE duty | 24.70% | 45.94% | 17.66% | 50.62% |
| STE duty | 14.34% | 11.03% | 18.42% | 45.43% |
| VL1 hit | 96.80% | 98.42% | 96.13% | 98.43% |
| L2 hit | 12.41% | 37.56% | 30.90% | 73.06% |
| HBM total | 12,722,176 B | 12,765,344 B | 8,710,784 B | 8,753,600 B |
| Shared efficiency | 100% | 67.78% | 100% | 41.19% |

`tile_k=64` 的语义访存没有减少，实测 HBM 流量仅增加 0.34%～0.49%。
收益主要来自更多 workgroup、更低 shared/CTA、更短 wave 生命周期，以及更高
的 Compute/MTE 利用率。shared efficiency 的下降是真实现象，但并行度收益
显著更大，因此不应退回 128/256 tile。

## 9. Roofline 实测分析

### 9.1 峰值与 sGPU 折算

mcProfiler 给出的整卡 Roofline 参数：

| 参数 | 整卡 |
|---|---:|
| HBM 峰值带宽 | 1843.2 GB/s |
| Ridge point | 260 FLOP/B |
| 推导计算峰值 | 479.232 TFLOP/s |

当前容器使用 25% Compute、16000 MiB 的 sGPU 切片。按配额线性折算：

| 参数 | 25% sGPU |
|---|---:|
| HBM 峰值带宽 | 460.8 GB/s |
| 计算峰值 | 119.808 TFLOP/s |
| Ridge point | 260 FLOP/B |

Roofline 效率必须使用切片峰值计算，不能把切片实测性能直接除以整卡峰值。

### 9.2 算术强度

各 workload 的 Manifest 算术强度范围为：

| 类型 | 典型 AI |
|---|---:|
| BF16 plain/expand | 约 0.995 FLOP/B |
| FP32 plain/expand | 约 0.599 FLOP/B |
| FP8 rescale/rescale-expand | 约 2.431 FLOP/B |

全部远低于 260 FLOP/B 的 ridge point，因此四个变体均属于
memory-bound，而不是 compute-bound。

### 9.3 plain medium（原迁移 `tile_k=128`）

输入：

```text
N = 1024
H = 4096
S = 8
e = 2 B
```

计算：

```text
FLOPs = 3 × 1024 × 4096 + 2 × 8 × 4096
      = 12,648,448

Bytes = 1024 × 4096 × 2
      + 1024 × 4096
      + 8 × 4096 × 4
      = 12,713,984 B

AI = 12,648,448 / 12,713,984
   = 0.994845 FLOP/B
```

使用 25% sGPU 带宽峰值：

```text
memory roof       = 0.994845 × 460.8 GB/s
                  = 0.458425 TFLOP/s
ideal memory time = 12,713,984 / 460.8 GB/s
                  = 27.591 μs
measured time     = 96.054 μs
Roofline efficiency
                  = 27.591 / 96.054
                  = 28.72%
```

Manifest 逻辑字节对应的 achieved bandwidth 为 132.363 GB/s；
mcProfiler 实际流量对应 132.448 GB/s，两种口径几乎一致。

### 9.4 rescale-expand（原迁移 `tile_k=256`）

输入：

```text
source tokens = 512
N output      = 1024
H             = 4096
C             = 32
S             = 8
```

计算：

```text
FLOPs = 5 × 1024 × 4096 + 2 × 8 × 4096
      = 21,037,056

Bytes = 2 × 1024 × 4096
      + 4 × 1024 × 32
      + 4 × 1024
      + 4 × 8 × 4096
      = 8,654,848 B

AI = 21,037,056 / 8,654,848
   = 2.430667 FLOP/B
```

使用 25% sGPU 带宽峰值：

```text
memory roof       = 2.430667 × 460.8 GB/s
                  = 1.120051 TFLOP/s
ideal memory time = 8,654,848 / 460.8 GB/s
                  = 18.782 μs
measured time     = 154.930 μs
Roofline efficiency
                  = 18.782 / 154.930
                  = 12.12%
```

按 Manifest 逻辑流量计算的 achieved bandwidth 为 55.863 GB/s。
按 mcProfiler 实际 HBM 流量计算为 42.208 GB/s，约为切片峰值的 9.16%。
两者差异来自重复 gather 数据的缓存命中；验收 Roofline 与 Manifest
保持同一逻辑字节口径，因此采用 12.12%。

### 9.5 当前 `tile_k=64` Roofline

`tile_k` 只改变调度和资源占用，不改变 Manifest 的 FLOPs、语义 bytes 或
算术强度。按 25% sGPU 峰值 `460.8 GB/s` 重新计算：

| workload | AI | 原配置效率 | `tile_k=64` 效率 | `tile_k=64` 语义带宽 |
|---|---:|---:|---:|---:|
| plain 1024×4096 BF16 | 0.995 FLOP/B | 19.74% | 39.10% | 180.15 GB/s |
| rescale 1024×4096 FP8 | 2.432 FLOP/B | 6.08% | 21.92% | 101.00 GB/s |
| plain 4096×7168 BF16 | 0.995 FLOP/B | 23.15% | 46.59% | 214.66 GB/s |
| rescale 4096×3072 FP8 | 2.432 FLOP/B | 9.21% | 25.13% | 115.80 GB/s |

最小 rescale 128×1024 的效率也由约 0.38% 提升到约 1.07%，但仍受启动
延迟和设备覆盖不足限制，不适合单独代表稳定吞吐。

### 9.6 性能结论

- 所有 workload 的 AI 都显著低于 ridge point，主瓶颈是数据搬运、访存
  延迟和小规模固定开销。
- `tile_k=64` 没有改变算法 AI，提升来自并行度、occupancy 和实际搬运效率。
- Plain/Rescale 的 shared-memory efficiency 分别降到 67.78%/41.19%，
  表明子字宽访问存在 bank conflict；但 Compute/MTE 利用率和总性能显著
  提升，当前不应为了 shared efficiency 退回大 tile。
- plain medium 的实际 HBM 流量仍与公式高度一致。
- gather/rescale 的 L2 hit 较高，缓存有效复用了重复源 token。
- 当前三方 Benchmark 中 Plain 仍落后于无尾块保护的固定 TileLang 基线，
  下一步应分别验证完整块快速路径和动态 token Kernel，而不是删除公共路径
  的尾块安全检查。
- 小型 rescale workload 受启动和固定调度影响明显，不应仅用小 shape
  判断稳定吞吐。

## 10. 本次迁移中的适配与修复

1. 将上游 QuantTensor tuple 拆成 Manifest 可描述的 `x` 和
   `x_sf_invs` 两个输入。
2. 提供四个清晰的公开 Op，同时共用一个 TileLang Kernel。
3. 使用编译期布尔参数生成 plain、expand、rescale 和
   rescale-expand 专用路径。
4. 为 C500 的 64 KiB shared memory 和调度效率将四条路径统一固定为
   `tile_k=64`，并加入显式 shared-memory 字节门禁。
5. 保持完整 128-token scale block，避免改变上游量化语义。
6. 为 expand 尾块增加输出和源 token 边界保护。
7. 对负 `pos_to_token` 统一产生零 padding。
8. 加入 Kernel shape/dtype/device 缓存和输入合法性检查。
9. 使用独立 PyTorch 参考实现建立精度证据。
10. 固定官方 `dev@0266ab7` 的 benchmark-only TileLang 基线和 eager
    PyTorch 基线，并记录源文件 SHA256。
11. 增加 Manifest 驱动三方 Benchmark、访存公式和 Roofline 实测。

## 11. 已知限制与后续优化方向

- 当前 Kernel 按 C500 固定 `tile_k=64`，`tune=True` 尚无本算子的候选
  配置空间；在没有新实测证据前不重新引入 128/256 tile。
- scale shuffle 广播已经独立 A/B，虽然正确，但稳定回退 9.74%～21.27%，
  因此明确不采用。
- plain 非 gather 路径是连续访存，可评估独立 `T.copy` 快速路径；任何
  改动都需要重新执行正确性、Benchmark 和 mcProfiler A/B。
- 固定 TileLang 基线采用上游动态 token 维度，并且只运行完整 128-token
  block；当前 Plain 差距需要把“动态 token”和“完整块无边界分支”拆成两个
  独立实验。
- 首次检查新的 `pos_to_token` 会执行范围校验并同步 Host；当前用有限
  cache 避免重复校验，生产模式可进一步评估可选校验策略。
- 小 workload 不适合 mcProfiler `--single-pass` Roofline，应该选择能
  充分覆盖设备的中等 workload。

## 12. 一键复核命令

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH

python scripts/validate_manifest.py
python scripts/validate_manifest.py \
  --check-op QuantPerChannelCastFusedOp --strict

python -m pytest -q tests/ops/test_per_channel_cast_fused.py
python -m pytest -q tests/ops/test_per_channel_cast_fused_baselines.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py

python -m pytest -q -s benchmarks/ops/bench_per_channel_cast_fused.py

ruff check benchmarks/ops/bench_per_channel_cast_fused.py \
  benchmarks/ops/per_channel_cast_fused_baselines.py \
  tests/ops/test_per_channel_cast_fused.py \
  tests/ops/test_per_channel_cast_fused_baselines.py \
  tileops/kernels/quant/per_channel_cast_fused.py \
  tileops/testing/per_channel_cast_fused.py
git diff --check
```

## 13. 完成状态

| 项目 | 状态 |
|---|---|
| Manifest | 完成并通过校验 |
| Op 接口 | 完成 |
| TileLang Kernel | 完成 |
| QuantTensor/FP8 输入 | 完成 |
| plain/expand/rescale/rescale-expand | 完成 |
| C500 全路径 `tile_k=64` | 完成；含 shared-memory 门禁 |
| 正确性、边界、异常测试 | 29 项通过 |
| 固定基线测试 | 7 项通过 |
| eager PyTorch 基线 | 完成并固定上游 SHA |
| 官方式 TileLang 基线 | 完成；`dev@0266ab7` + C500 `tile_k=64` |
| Benchmark | 9 组三方 workload 实测完成 |
| mcProfiler | Plain/Rescale 优化前后 4 份 A/B 报告完成 |
| Roofline | 公式、切片折算、原配置与 tile64 实测效率完成 |
| MetaX C500 验证 | 完成 |
