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
| Plain/Expand thread-local staging 提交 | `6ec6bde` |
| 测试与 profile 入口提交 | `3a0f2c0` |
| augenstern 70 项矩阵提交 | `266fabb` |
| 精确 profiler metrics runner 提交 | `ec3f87d` |
| Rescale shared 预算门禁提交 | `c9655a2` |
| mcProfiler 实测代码基线 | `ec3f87d6b357f9d97ab80cc49ee783ed4b4db742` |
| 生产 Kernel SHA256 | `3b7df342a2dba2db0988210dc5aa608793cc708cc08bb35e748a1010e854d30c` |
| 上游仓库 | `https://github.com/MetaX-MACA/TileKernels-Metax` |
| 上游提交 | `0266ab740980de7dc03a828b8259cd73d100c2eb` |
| 上游 TileLang 源码 | `tile_kernels/quant/per_channel_cast_fused_kernel.py` |
| 上游 TileLang SHA256 | `64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc` |
| 上游 PyTorch reference | `tile_kernels/torch/per_channel_cast_fused.py` |
| 上游 PyTorch SHA256 | `6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f` |
| 仓内固定 TileLang baseline blob | `4a1c4722fd6dcd3f9fbc295ed3cb2fcbdb76d652` |
| 仓内 eager PyTorch baseline blob | `adc604bcd8c10c06f01bc6f84f9bb79f582aa286` |
| ACoolFIsh 最新可访问审查版本 | `497e7237546c5df4ec8056b0e423151bffbfad0c` |
| augenstern register-resident 审查版本 | `387119e154c6cfa4d09b25823635814380af91f9` |
| augenstern 测试最后提交 / blob | `c77cc75caa4b8a281a814b66b06102564ef33a01` / `11238539f3f64e2b5585a7a906067eab9d7e1e7c` |
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
├── tests/ops/test_per_channel_cast_fused_augenstern.py
├── tests/ops/test_per_channel_cast_fused_baselines.py
├── workloads/per_channel_cast_fused.py
├── benchmarks/ops/per_channel_cast_fused_baselines.py
├── benchmarks/ops/bench_per_channel_cast_fused.py
├── benchmarks/ops/profile_per_channel_cast_fused.py
└── scripts/run_quant_per_channel_cast_fused.sh
```

各文件职责如下：

| 路径 | 作用 |
|---|---|
| `tileops/manifest/quantization.yaml` | 定义四个算子的输入输出、参数、shape 规则、workload 和 Roofline 公式 |
| `tileops/ops/quant/per_channel_cast_fused.py` | 对外 Op 接口；负责参数、dtype、shape、CUDA、连续性和索引检查，以及 Kernel 缓存和调度 |
| `tileops/kernels/quant/per_channel_cast_fused.py` | 真正执行设备计算的 TileLang Kernel |
| `tileops/testing/per_channel_cast_fused.py` | 独立 PyTorch 参考实现，不调用被测 Kernel |
| `tests/ops/test_per_channel_cast_fused.py` | 本地 33 项正确性、边界、资源和失败契约测试 |
| `tests/ops/test_per_channel_cast_fused_augenstern.py` | augenstern `387119e` 的 70 项功能/正确性矩阵；保留原始 shape、dtype、round 和用例 ID，适配本地 Op 与独立 reference |
| `tests/ops/test_per_channel_cast_fused_baselines.py` | 固定 TileLang 基线与 eager PyTorch 的独立一致性测试 |
| `workloads/per_channel_cast_fused.py` | Benchmark 输入生成 |
| `benchmarks/ops/per_channel_cast_fused_baselines.py` | 固定官方 `dev` 算法结构和语义的 benchmark-only 上游式 TileLang 基线；显式适配当前 Benchmark 接口，并在 C500 上使用 `tile_k=64` |
| `benchmarks/ops/bench_per_channel_cast_fused.py` | Manifest 驱动的生产 Kernel、shared-staging 对照、固定 TileLang 与 eager PyTorch 四方 Benchmark |
| `benchmarks/ops/profile_per_channel_cast_fused.py` | mcProfiler 稳定驱动；固定输入构造、workgroup 期望值、commit、dirty 状态和 Kernel SHA256 |
| `scripts/run_quant_per_channel_cast_fused.sh` | smoke、完整正确性、固定基线、Benchmark、质量门禁和 mcProfiler 的统一入口 |
| `docs/summer-camp/quant_per_channel_cast_fused/artifacts/` | 三轮 Benchmark、最终回归和五份目标 Kernel profiler 原始文本 |
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

| 路径 | `tile_k` | 输入 staging | 生产配置显式 shared/CTA |
|---|---:|---|---:|
| BF16 plain/expand | 64 | `T.alloc_local` thread-local | 1,024 B |
| FP32 plain/expand | 64 | `T.alloc_local` thread-local | 1,024 B |
| FP8 rescale/rescale-expand | 64 | `T.alloc_shared` | 9,216 B |

Plain 和 Expand 默认 `register_staging=True`。每线程的 32 个输入值在第一次
amax 遍历与第二次量化遍历之间保存在 `T.alloc_local` 数组，不再写入并读回
输入 shared tile；四条路径仍保留 1,024 B 的 `amax_shared[1, 256]` 归约
scratch。`T.alloc_local` 表示线程私有存储，不能仅凭源码断言全部落入物理
寄存器；是否发生 private-memory spill 以 mcProfiler 指标为准。

Rescale 和 Rescale-Expand 默认 `register_staging=False`。它们还需要让每线程
32 个 FP32 输入 scale 跨越归约保持存活；把输入也改为 thread-local 会增加
local/private 资源压力；augenstern `387119e` 的 register-resident 消融记录了
Rescale 回退。本仓最终三轮 A/B 没有把 Rescale-local 当作候选，因此生产路径
保守地保留 shared staging。其显式 shared 预算为：

```text
128 × 64 × sizeof(FP8) + 256 × sizeof(FP32)
= 8,192 + 1,024
= 9,216 B
```

作为单变量 A/B 的 shared-staging Plain/Expand 对照分别占用 17,408 B
（BF16）和 33,792 B（FP32）。

原迁移/上游调度的历史预算用于解释 `tile_k=64` 的选择，不是当前生产占用：

| 历史路径 | tile | 显式 shared/CTA |
|---|---:|---:|
| BF16 plain/expand | 128 | 34,816 B |
| FP32 plain/expand | 128 | 67,584 B |
| FP8 rescale/rescale-expand | 256 | 36,864 B |

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

每个工作组处理一个 `128 × 64` tile：

1. 发起一次全局 `x` load；该 load 可能命中 VL1/L2，也可能到达 HBM；
2. Plain/Expand 把原始输入写入每线程 `x_staging[32, 1]`；
3. Rescale/Rescale-Expand 把原始输入写入 `x_shared[128, 64]`；
4. 转为 FP32；
5. rescale 路径乘以 `x_sf_invs`；
6. 每个线程在 local 标量中累计对应 channel 的
   `max(abs(logical_value))`；
7. 把局部 amax 写入 `amax_shared`，完成跨线程归约。

源码对每个有效输出元素只发起一次全局 `x` load。后续量化阶段再次使用输入
时，从 shared memory 或 thread-local staging 读取，不会再次发起 `x` 的全局
load；实际到达 HBM 的 transaction 数取决于 VL1/L2 命中。

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

1. Plain/Expand 从 `x_staging` 读取原始输入，Rescale 两条路径从
   `x_shared` 读取；
2. 转为 FP32；
3. rescale 路径再次乘 `x_sf_invs`；
4. 乘 `quant_sf`；
5. 转为 `float8_e4m3fn`；
6. 写入 `out`。

rescale 的乘法执行两次，分别服务于 amax 计算和最终量化；第二遍输入读取
来自 shared memory。选择存储原始 FP8 而不是完整 FP32 logical tile，
可以避免超过 C500 的 64 KiB shared-memory 限制。Plain/Expand 则通过
thread-local staging 删除一次输入 shared 写和一次输入 shared 读。

## 5. 访存次数与 Roofline 字节模型

### 5.1 每个有效输出元素

| 访问 | plain/expand | rescale/rescale-expand |
|---|---:|---:|
| 全局/逻辑读取 `x` | 1 次 | 1 次 |
| thread-local staging 写/读 | 各 1 次 | 无 |
| shared 输入 staging 写/读 | 无 | 各 1 次 |
| 全局/逻辑写入 FP8 `out` | 1 次 | 1 次 |
| FP32 `x_sf_invs` | 无 | 每 token、每 128 channel 组逻辑读取一次 |
| `pos_to_token` | expand 时每输出 token 逻辑读取一次 | expand 时每输出 token 逻辑读取一次 |
| 全局/逻辑写入 FP32 `out_sf` | 每 128-token block、每 channel 写 1 次 | 同左 |

`amax_shared` 的归约与 reciprocal scale 广播访问在四条路径中都存在。这里的
thread-local/shared 读写不计入 HBM Roofline bytes。特别需要注意：Kernel
第二遍输入读取来自片上 staging，不能把 `x` 重复计为第二次 HBM 读取。

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

正确性分为两个互补套件：

- 本地 33 项测试，重点覆盖公开接口、边界、资源配置和失败契约；
- augenstern 兼容矩阵 70 项，与
  `exp/quant-per-channel-register-resident@387119e` 一一对应。远端 HEAD
  只修改文档，测试代码最后更新为 `c77cc75`。

70 项兼容矩阵的分组固定为：

| 分组 | 数量 | 覆盖 |
|---|---:|---|
| 基础变体 | 17 | 四个 Op、BF16/FP32/FP8、round/non-round、对齐/尾块 |
| token sweep，`hidden=512` | 20 | `17/137/513/1001` 的 Expand 到 `32/144/1024/2048`，以及 `128/256/512/1024` Plain/Rescale |
| token × hidden | 26 | `hidden=128/256/3072/7168` 与中大 token 组合 |
| Gather pattern | 3 | sequential、repeated、all-invalid |
| 数值边界 | 3 | zero、`1e-8`、交替正负 `1e4` |
| 构造参数 | 1 | 非法 `fmt`、token/channel group |
| **合计** | **70** | 源码内有分组计数和总数断言 |

矩阵沿用远端 `torch.manual_seed(0)` 和 Rescale 的
`real / x_sf_invs -> FP8` 输入构造。与远端不同的是，本地不引入
`FixtureBase/TestBase`，不允许 reference OOM 静默返回，也不使用非 MetaX
的宽松 match-ratio/cosine 分支。所有用例统一使用本地独立 eager
PyTorch reference 和相同的严格输出门禁。

本地 33 项套件额外覆盖：

- BF16 和 FP32 plain 输入；
- FP8 e4m3 QuantTensor/rescale 输入；
- 四个公开变体；
- `round_sf=False/True`；
- 全零输入与最小 scale；
- 正负极值；
- 重复、逆序和包含 padding 的 gather；
- 全 padding gather；
- 输出 token 数 16、128、144、256；
- FP32 Expand 非完整 scale block：`(153, 128) -> 160`，并启用
  `round_sf=True`；
- 绝对值低于 `1e-4` floor 的交替正负非零输入；
- 大型 Plain：`(1024, 7168)`；
- 大型 Rescale-Expand：`(513, 7168) -> 1024`；
- 多个 token block；
- 多个 hidden tile；
- 空输入；
- 空源输入加 padding；
- 非法构造参数；
- 非法 dtype、rank、hidden、token 数；
- 非连续输入；
- 非法 `pos_to_token`；
- 非法 `x_sf_invs` shape 和 dtype。
- Plain/Expand 必须启用 thread-local staging，Rescale 两条路径必须保留
  shared staging；
- 精确门禁 Plain/Expand 为 1,024 B、Rescale 为 9,216 B，并保留
  BF16/FP32 shared-staging 对照和历史大 tile 预算断言。

另外有 7 项固定基线测试，覆盖：

- 上游提交与两个源文件 SHA256 的 provenance 元数据常量门禁；测试不联网重新
  下载或哈希上游仓库；
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
python -m pytest -q tests/ops/test_per_channel_cast_fused_augenstern.py
python -m pytest -q \
  tests/ops/test_per_channel_cast_fused.py \
  tests/ops/test_per_channel_cast_fused_augenstern.py
python -m pytest -q tests/ops/test_per_channel_cast_fused_baselines.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
```

结果：

| 检查 | 结果 |
|---|---|
| 全量 Manifest | 通过；仓库既有 advisory warning 不阻塞 |
| 本算子 strict Manifest | 通过；10 条 synthetic shape precondition warning |
| 本地契约测试 | `33 passed in 26.60s` |
| augenstern 70 项兼容矩阵 | `70 passed in 183.35s` |
| 统一 `correctness` 入口 | `103 passed in 32.08s`（编译缓存已热） |
| 固定基线正确性测试 | `7 passed in 23.51s` |
| Benchmark 基础测试 | `17 passed in 24.36s` |
| Ops Manifest 测试 | `7 passed in 23.44s` |
| 最终四方 Benchmark 回归 | `9 passed in 32.64s` |
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

Benchmark 与固定基线位于：

```text
benchmarks/ops/bench_per_channel_cast_fused.py
benchmarks/ops/per_channel_cast_fused_baselines.py
```

运行命令：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH
./scripts/run_quant_per_channel_cast_fused.sh benchmark
```

四条被测路径使用完全相同的 Manifest workload、输入和计时框架：

1. `tileops`：当前生产 Op；Plain/Expand 使用 thread-local staging，Rescale
   两条路径使用 shared staging。
2. `tileops-shared-staging`：仅为 Plain/Expand 构造的同 Kernel 单变量对照；
   唯一变化是编译期 `register_staging=False`。
3. `tilelang-baseline`：固定自官方 TileKernels-Metax `dev@0266ab7` 的
   上游式 TileLang 算法基线。它保留核心 tile、线程映射、shared staging、
   两遍归约/量化和动态 token 结构，并显式展开配置与 scale helper 以接入当前
   Benchmark；C500 调度选择为全路径 `tile_k=64`。它不是上游源文件的逐字副本，
   因此文档不把差异夸大为“只有一行 tile 修改”。该基线独立保存在
   `benchmarks/ops/per_channel_cast_fused_baselines.py`，不会随生产 Kernel
   后续优化而漂移。
4. `torch-eager`：`tileops/testing/per_channel_cast_fused.py` 中的独立
   eager PyTorch 原语组合。

每个 case 在计时前先让所有可用 TileLang 路径分别与 eager PyTorch 比较；
FP8 输出必须逐元素完全一致，scale 使用 `atol=1e-7, rtol=1e-6`。

稳定态计时协议：

- JIT 编译在计时前完成；
- 10 次 warmup；
- 每个 trial 50 次测量；
- 共 3 个 trial；
- 每次计时前刷新 L2；
- 输入总量允许时使用 3 组 clone 轮换地址；
- 每次测量前后同步；
- 优先使用 CUPTI kernel-only 时间；
- 每份报告取三个 trial mean 的中位数；
- 完整 9-workload 报告独立运行 3 次；
- 最终交付表再取三份独立报告 latency 的中位数。

三份原始报告已原样提交，GitHub 上可按 SHA256 复核：

| 运行 | 仓内原始报告 | SHA256 |
|---|---|---|
| run 1 | [benchmark_run1.txt](artifacts/benchmark_run1.txt) | `c8657fc39c8270dc9dfd7cd6609832426924253612dd7fb7af96000a45a92d8d` |
| run 2 | [benchmark_run2.txt](artifacts/benchmark_run2.txt) | `802777a578097eb8dad1b30c0db35042549e71046ceed40dced19c7176490dfd` |
| run 3 | [benchmark_run3.txt](artifacts/benchmark_run3.txt) | `08fe02d82be3be13b973bc9375bc45d93b52faae30c30e91bae2ff9671c6db8b` |

文档完成后的交付回归再次得到 `9 passed in 32.64s`。该轮只用于确认最终
代码仍能完整运行，不替换上表的三轮中位统计；原始报告为
[benchmark_final_regression.txt](artifacts/benchmark_final_regression.txt)，
SHA256 为 `9566cdc0f60a86153d296efe5975649eb8bba40f505803434bd857e696fff3dc`。

### 7.2 结果

单位均为 ms。`N/A` 表示 Rescale 生产路径本身就是 shared staging，不构造
没有单变量意义的重复对照。

| 变体与 workload | production | shared staging | 固定 TileLang | eager PyTorch |
|---|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.0314 | 0.0325 | 0.0243 | 0.0748 |
| plain 1024×4096 BF16，rounded | 0.0507 | 0.0705 | 0.0520 | 0.2618 |
| plain 4096×8192 FP32 | 0.2852 | 1.2286 | 0.7889 | 1.5036 |
| expand 512×4096 → 1024 BF16 | 0.0513 | 0.0754 | 0.0778 | 0.3931 |
| expand 1024×4096 → 2048 FP32，rounded | 0.0862 | 0.3322 | 0.3374 | 0.6570 |
| rescale 128×1024 FP8 | 0.0546 | N/A | 0.0526 | 0.0801 |
| rescale 1024×4096 FP8，rounded | 0.0851 | N/A | 0.0845 | 0.3731 |
| rescale-expand 512×4096 → 1024 FP8 | 0.1259 | N/A | 0.1282 | 0.4485 |
| rescale-expand 1024×4096 → 2048 FP8，rounded | 0.1972 | N/A | 0.1972 | 0.7908 |

### 7.3 thread-local staging 单变量 A/B

| workload | shared ms | production ms | 加速 | 延迟下降 |
|---|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.0325 | 0.0314 | 1.035× | 3.38% |
| plain 1024×4096 BF16 rounded | 0.0705 | 0.0507 | 1.391× | 28.09% |
| plain 4096×8192 FP32 | 1.2286 | 0.2852 | 4.308× | 76.78% |
| expand 512×4096 → 1024 BF16 | 0.0754 | 0.0513 | 1.470× | 31.96% |
| expand 1024×4096 → 2048 FP32 rounded | 0.3322 | 0.0862 | 3.854× | 74.05% |

结论：

- 生产配置在 9 个 workload 上全部快于 eager PyTorch，范围为
  1.467×～7.663×。
- thread-local staging 对五组 Plain/Expand 均无回退；中大型 BF16 延迟下降
  约 28%～32%，FP32 延迟下降约 74%～77%。
- 最小 Plain 只有 16 个 workgroup，固定启动成本占主导；生产配置仍比固定 TileLang
  慢约 29%，不能再概括为所有 Plain workload 都落后 25%～35%。中大型
  Plain 已持平或显著超过固定 TileLang。
- Rescale 两条路径保留 shared staging，与固定 TileLang 基本持平；augenstern
  分支已记录 local/register 消融回退，而本仓尚无可复核的 Rescale-local
  单变量报告，因此不把该候选带入生产路径。
- 固定 TileLang 仍是独立的上游算法基线；shared-staging 则是用于归因
  thread-local staging 收益的单变量对照，两者用途不同。

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

### 8.2 采样入口与数据隔离

稳定驱动位于：

```text
benchmarks/ops/profile_per_channel_cast_fused.py
```

驱动固定 workload、输入构造和目标调用，并在执行前打印 commit、dirty
状态、Kernel SHA256、期望 workgroups、staging 策略和显式 shared bytes：

| case | workload | staging | 期望 workgroups | shared/CTA |
|---|---|---|---:|---:|
| `plain-medium` | Plain 1024×4096 BF16，rounded | production | 512 | 1,024 B |
| `plain-medium` | Plain 1024×4096 BF16，rounded | shared | 512 | 17,408 B |
| `expand-medium` | Expand 512×4096 → 1024 BF16 | production | 512 | 1,024 B |
| `expand-medium` | Expand 512×4096 → 1024 BF16 | shared | 512 | 17,408 B |
| `rescale-control` | Rescale 1024×4096 FP8，rounded | production | 512 | 9,216 B |

固定 runner 命令为：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH

./scripts/run_quant_per_channel_cast_fused.sh profile plain-medium production
./scripts/run_quant_per_channel_cast_fused.sh profile plain-medium shared
./scripts/run_quant_per_channel_cast_fused.sh profile expand-medium production
./scripts/run_quant_per_channel_cast_fused.sh profile expand-medium shared
./scripts/run_quant_per_channel_cast_fused.sh profile rescale-control production
```

runner 使用明确的 metrics 列表并让 mcProfiler 分多个 event batch 精确采样。
生成的 `mcProfiler.json` 过滤条件为：

```text
kernelnames  = ["main_kernel"]
include      = true
is_perkernel = true
counts       = 2
```

这里 `counts=2` 包含 aggregate 和 1 次目标 Kernel 记录。每次结果都以
512 workgroups、2,048 waves 作为目标 Kernel 身份门禁。

只引用每个输出目录中的 `1_main_kernel.txt`。顶层 `report.txt` 聚合了
驱动中的 `torch.ones`、`torch.arange`、Host-to-device 初始化等 Kernel；
例如 Expand production 的 `report.txt` 为 768 workgroups、6,144 waves，
而 `1_main_kernel.txt` 才是目标 Kernel 的 512 workgroups、2,048 waves。
因此不能用 `report.txt` 的 HBM、workgroup 或 RoofLine 数据评价本算子。

### 8.3 最终报告与 SHA256

| case | staging | 仓内目标 Kernel 报告 | mcProfiler 输出目录 |
|---|---|---|---|
| Plain medium | shared | [mcprofiler_plain_shared.txt](artifacts/mcprofiler_plain_shared.txt) | `output20260804171933` |
| Plain medium | production | [mcprofiler_plain_production.txt](artifacts/mcprofiler_plain_production.txt) | `output20260804170503` |
| Expand medium | shared | [mcprofiler_expand_shared.txt](artifacts/mcprofiler_expand_shared.txt) | `output20260804170841` |
| Expand medium | production | [mcprofiler_expand_production.txt](artifacts/mcprofiler_expand_production.txt) | `output20260804171217` |
| Rescale control | production | [mcprofiler_rescale_production.txt](artifacts/mcprofiler_rescale_production.txt) | `output20260804171558` |

```text
e6703fe9f2f17e82771ea1337d4a5d7bc72f710292442d390a549c7057f1b60f
  artifacts/mcprofiler_plain_shared.txt
577a203009afdbc9c234ff4b309584a72ec9a174ce40c23a5e2ad2141e0f2a3e
  artifacts/mcprofiler_plain_production.txt
871b4086dccf93fd41db24271b2a535362aee6639b4b07092ffdeaa44ee9e0be
  artifacts/mcprofiler_expand_shared.txt
e35974d3011c18da534bf90451940211c06faeb7c021e67443d1fbffaf7cd198
  artifacts/mcprofiler_expand_production.txt
adc21ab9605e765aad0ec44baf10fad3cde94c1999699188fe8e373dd758b3ee
  artifacts/mcprofiler_rescale_production.txt
```

### 8.4 multi-batch、per-kernel 指标

下表的 `AP busy / during` 是 mcProfiler RoofLine
`processed_data.during` 的原始 cycle 计数，不是百分比。S 表示 shared
staging，P 表示 production。

| 指标 | Plain S | Plain P | Expand S | Expand P | Rescale P |
|---|---:|---:|---:|---:|---:|
| Workgroups | 512 | 512 | 512 | 512 | 512 |
| Waves | 2,048 | 2,048 | 2,048 | 2,048 | 2,048 |
| 显式 shared/CTA | 17,408 B | 1,024 B | 17,408 B | 1,024 B | 9,216 B |
| Average wave cycles | 9,548.75 | 11,242.30 | 10,321.08 | 11,427.86 | 17,190.72 |
| AP busy / during | 77,231 | 55,324 | 84,562 | 57,347 | 79,659 |
| Shared load instructions | 68,272 | 4,016 | 68,272 | 4,016 | 68,408 |
| Shared store instructions | 66,766 | 2,510 | 66,766 | 2,510 | 66,899 |
| Private read instructions | 0 | 0 | 0 | 0 | 0 |
| Private write instructions | 0 | 0 | 0 | 0 | 0 |
| VL1 hit | 98.42% | 98.42% | 98.41% | 98.41% | 98.43% |
| L2 hit | 38.70% | 25.76% | 62.69% | 50.38% | 74.09% |
| HBM read | 8,438,752 B | 8,429,216 B | 4,248,032 B | 4,241,824 B | 4,428,896 B |
| HBM write | 4,327,232 B | 4,325,696 B | 4,328,256 B | 4,325,952 B | 4,325,760 B |
| HBM total | 12,765,984 B | 12,754,912 B | 8,576,288 B | 8,567,776 B | 8,754,656 B |
| Shared access efficiency | 67.76% | 100% | 75.52% | 100% | 41.22% |
| hardware `case_I` | 159.729 | 148.183 | 237.690 | 219.647 | 324.795 |
| hardware `case_bandwith` (GB/s) | 185.958 | 259.368 | 114.098 | 168.078 | 123.639 |

### 8.5 mcProfiler 结论

- Plain production 相对 shared 将 shared load 从 68,272 降到 4,016，
  shared store 从 66,766 降到 2,510；AP busy/during 下降 28.37%，硬件
  `case_bandwith` 提高 1.395×。
- Expand production 的 shared 指令下降幅度相同；AP busy/during 下降
  32.18%，硬件 `case_bandwith` 提高 1.473×。
- 两组 A/B 的 HBM 总流量变化都小于 0.1%，说明收益不是减少语义 HBM
  访问，而是删除输入 tile 的 shared 往返并降低每 CTA 资源占用。
- 五份报告的 private read/write 都为 0，当前选定 workload 没有观察到
  thread-local staging spill 到 profiler 可见的 private memory。
- production 的 Average wave cycles 略高，但总 AP busy/during 明显降低；
  单 wave 生命周期不能替代完整 Kernel 延迟或吞吐指标。
- Rescale 仍有约 6.8 万次 shared load/store，且 shared efficiency 为
  41.22%，与生产路径有意保留 shared staging 一致。是否改用 local 必须另做
  资源压力和 spill A/B，不能从 Plain/Expand 结果外推。
- mcProfiler 的 `case_I` 和 `case_bandwith` 来自硬件指令与 memory
  transaction 口径，只用于同一 profiler 配置下的 A/B，不与下一节的
  Manifest FLOPs、语义 bytes 或算术强度混用。

## 9. Roofline 实测分析

### 9.1 口径与峰值参考

Roofline 使用第 5 节 Manifest 公式和第 7 节三次独立 Benchmark 的中位延迟：

```text
Manifest AI             = Manifest FLOPs / Manifest semantic bytes
achieved TFLOP/s        = Manifest FLOPs / benchmark latency
semantic bandwidth      = Manifest semantic bytes / benchmark latency
full-card reference     = semantic bandwidth / 1843.2 GB/s
25% linear reference    = semantic bandwidth / 460.8 GB/s
```

mcProfiler 报告的整卡参数为 1,843.2 GB/s HBM 峰值和 260 FLOP/B ridge
point。当前环境分配 25% Compute，但没有经过校准的 sGPU HBM 带宽上限。
`460.8 GB/s = 1843.2 × 25%` 只是未校准的线性参考线，不是实际切片峰值。

尤其要注意，表中的 bandwidth 由 Manifest 语义字节数计算。Expand 的重复
gather 可被 cache 复用，因此语义字节可能显著大于物理 HBM transaction。
所以两列百分比都只是量纲一致的归一化参考，不是物理 HBM 利用率、效率或
下界。超过 100% 只说明不能把“Manifest 语义流量”和“Compute 配额线性折算的
HBM 峰值”组合成效率指标；它本身不能校准或证明切片的物理带宽策略。

### 9.2 三轮 Benchmark 语义 Roofline

| production workload | Manifest AI (FLOP/B) | achieved TFLOP/s | semantic BW (GB/s) | semantic BW / 1843.2 | semantic BW / 460.8 |
|---|---:|---:|---:|---:|---:|
| Plain 128×1024 BF16 | 0.994845 | 0.0126 | 12.65 | 0.69% | 2.75% |
| Plain 1024×4096 BF16，rounded | 0.994845 | 0.2495 | 250.77 | 13.61% | 54.42% |
| Plain 4096×8192 FP32 | 0.599379 | 0.3548 | 591.94 | 32.11% | 128.46% |
| Expand 512×4096 → 1024 BF16 | 0.994525 | 0.2466 | 247.92 | 13.45% | 53.80% |
| Expand 1024×4096 → 2048 FP32，rounded | 0.599263 | 0.2935 | 489.71 | 26.57% | 106.27% |
| Rescale 128×1024 FP8 | 2.431818 | 0.0120 | 4.95 | 0.27% | 1.07% |
| Rescale 1024×4096 FP8，rounded | 2.431818 | 0.2472 | 101.65 | 5.52% | 22.06% |
| Rescale-Expand 512×4096 → 1024 FP8 | 2.430667 | 0.1671 | 68.74 | 3.73% | 14.92% |
| Rescale-Expand 1024×4096 → 2048 FP8，rounded | 2.430667 | 0.2134 | 87.78 | 4.76% | 19.05% |

### 9.3 解释与结论

- 九组 Manifest AI 为 0.599263～2.431818 FLOP/B，说明算法语义的计算/搬运
  比很低，优化应优先关注数据路径；小 workload 还受启动延迟和设备覆盖不足
  影响。Manifest AI 与 mcProfiler 的硬件 `MAX_I` 计数定义不同，不能仅凭
  `0.599～2.432 < 260` 把九组 workload 严格分类为硬件 memory-bound。
- Plain/Expand 的 thread-local staging 不改变 Manifest FLOPs、语义 bytes
  或 AI；收益体现在同等语义工作量下更高的 achieved TFLOP/s 和 semantic
  bandwidth，并由第 8 节的 shared 指令下降提供硬件侧佐证。
- 25% 线性参考列仅展示归一化结果：两组 FP32 workload 达到 106.27% 和
  128.46%，不能解释为超过物理峰值、物理利用率超过 100%，或切片策略证据。
- 表中 bandwidth 是 Benchmark 延迟下的 Manifest 语义带宽，不是
  mcProfiler 的 `case_bandwith`。同理，Manifest AI 不能与硬件
  `case_I=148.183～324.795` 比较；两套计数回答不同问题。
- Plain medium 的 Manifest 逻辑流量为 12,713,984 B，production profiler
  的物理 HBM 流量为 12,754,912 B，两者接近。连续 Plain 的语义带宽与物理
  流量具有较好的对应关系。
- Expand medium 的 Manifest 逻辑流量为 12,718,080 B，而 production
  profiler 的物理 HBM 流量为 8,567,776 B。Expand 会重复 gather 源 token，
  Manifest 按每个逻辑输出计入读取，硬件则通过 VL1/L2 复用，因此两者差异
  是预期的，不能据此修改语义 bytes 公式。
- 对五份已采样报告，mcProfiler 自己的硬件口径给出 Plain/Expand
  `case_I=148.183～237.690 < 260`，位于其 memory-side；Rescale control
  `case_I=324.795 > 260`，位于其 compute-side。这个结论只适用于对应的五份
  Kernel 报告，不能外推到未 profile 的全部九组 workload。
- 当前最强的中大型 Plain/Expand 已达到 247.92～591.94 GB/s 语义带宽；
  Rescale 路径因额外 scale 读取、二次乘法和 shared staging，仍是后续
  独立优化重点。

## 10. 本次迁移中的适配与修复

1. 将上游 QuantTensor tuple 拆成 Manifest 可描述的 `x` 和
   `x_sf_invs` 两个输入。
2. 提供四个清晰的公开 Op，同时共用一个 TileLang Kernel。
3. 使用编译期布尔参数生成 plain、expand、rescale 和
   rescale-expand 专用路径。
4. 为 C500 的 64 KiB shared memory 和调度效率将四条路径统一固定为
   `tile_k=64`，并加入显式 shared-memory 字节门禁。
5. 按变体采用混合 staging：Plain/Expand 使用 thread-local staging，
   Rescale/Rescale-Expand 因 scale 生命周期和 local/private 压力保留 shared。
6. 保持完整 128-token scale block，避免改变上游量化语义。
7. 为 expand 尾块增加输出和源 token 边界保护。
8. 对负 `pos_to_token` 统一产生零 padding；对非负越界索引明确报错。
9. 加入 Kernel shape/dtype/device 缓存和输入合法性检查。
10. 使用独立 PyTorch 参考实现建立精度证据。
11. 固定官方 `dev@0266ab7` 的 benchmark-only TileLang 基线和 eager
    PyTorch 基线，并记录源文件 SHA256。
12. 增加 Manifest 驱动四方 Benchmark，使用 shared-staging 单变量对照归因
    thread-local staging 收益。
13. 增加稳定 profile driver 与一键 runner，打印 commit、dirty、Kernel
    SHA256、期望 workgroups 和显式 shared bytes，并指定精确 profiler metrics。
14. 审查 ACoolFIsh `497e723` 与 augenstern `387119e`；仅吸收经过当前
    Kernel 独立 A/B 的优化，不整体替换现有接口、校验和测试结构。
15. 将 augenstern `ff86cdf` / `ea5d33e` / `c77cc75` 的完整 70 项
    功能/正确性矩阵迁移为独立兼容套件，并保留本地更严格的比较与失败语义。

### 10.1 真正吸收的 Kernel 优化

从原始迁移提交 `967b65b` 到当前生产 Kernel，只有两项改变设备
执行的性能优化。测试矩阵、固定基线、Benchmark、mcProfiler 和文档
提高可信度和可复现性，但不应计为 Kernel 加速。

| 项目 | 原始迁移 `967b65b` | 当前代码 | 来源与结论 |
|---|---|---|---|
| Hidden tile | Rescale=256，BF16 Plain/Expand=128，FP32=64 | 四路径固定 `tile_k=64` | 与 augenstern C500 配置一致，也印证 ACoolFIsh 的 tile64 并行度结论；本地 22 组隔离 A/B 中 21 组降低 19.64%～72.94%，原本已是 tile64 的 FP32 用例持平 |
| 输入 staging | 四路径都写入并重读 shared tile | Plain/Expand 使用 thread-local；Rescale 两路保留 shared | 吸收 augenstern `fa6bafe` 消融与 `87b083c` 最终策略；五组单变量 A/B 延迟降低 3.38%～76.78% |
| 显式 shared/CTA | BF16 tile128 34,816 B；FP32 tile64 33,792 B；FP8 Rescale tile256 36,864 B | Plain/Expand 1,024 B；Rescale 9,216 B | Plain/Expand 删除输入 tile 的一次 shared store 和一次 shared load；加入 C500 64 KiB 预算门禁 |
| mcProfiler | shared-staging 对照 | Plain/Expand local staging | shared load `68,272 -> 4,016` (-94.12%)；store `66,766 -> 2,510` (-96.24%)；HBM 变化小于 0.1%；未观察到 private spill |
| 计算语义 | 128-token 两遍 amax/量化 | 不变 | FLOPs、语义 HBM bytes、四个 Op 接口和 Roofline 公式均未改变 |

`tile_k=64` 和 local staging 的 A/B 使用了不同 workload 矩阵，因此不能把
两阶段百分比相乘，也不报告一个没有直接实测的“总加速”。

### 10.2 审查但未吸收的候选

- augenstern 的 Rescale register staging 回退约 24%～33%，本地保留 shared。
- 单 lane scale shuffle 在本地独立 A/B 回退 9.74%～21.27%，不采用。
- ACoolFIsh 的 Rescale `threads_per_token=16/vec_k=4` 同时混入多项调度改动，
  尚不能单独归因，未吸收。
- ACoolFIsh 的 `shared_rows=120`、Op-side CUDA `F.pad`、动态 tile 分派、
  custom-op/fake/meta 和 `torch.compile` 都未采用。

## 11. 已知限制与后续优化方向

- 当前 Kernel 按 C500 固定 `tile_k=64`，`tune=True` 尚无本算子的候选
  配置空间；在没有新实测证据前不重新引入 128/256 tile。
- 当前生产 Kernel cache key 仍包含精确 token 维度；动态 token Kernel 是
  后续独立实验，不能与 staging、边界分支或接口变更混在同一次 A/B 中。
- `T.alloc_local` 只保证线程私有语义，可能映射到物理寄存器或 private
  memory；性能结论必须同时检查 private read/write 与 spill 证据。
- Rescale/Rescale-Expand 的 register staging 已明确禁用；其输入 scale 与
  输入值同时跨归约存活会增加 local/private 压力，现阶段 shared 更稳定。
- 旧的标量 lane-0 scale shuffle 广播已经独立 A/B，虽然正确，但稳定回退
  9.74%～21.27%，因此不采用。ACoolFIsh 最新 vec4 映射与 subgroup shuffle
  并不等价于该失败实验，需要在当前 Kernel 上做隔离 A/B 后才能吸收。
- plain 非 gather 路径是连续访存，可评估独立 `T.copy` 快速路径；任何
  改动都需要重新执行正确性、Benchmark 和 mcProfiler A/B。
- 必须保留正 `pos_to_token` 越界拒绝、负索引 padding 语义和 Kernel 尾块
  guard；不能用 Op-side CUDA `F.pad` 或静默 OOB 处理换取不可归因的性能变化。
- 首次检查新的 `pos_to_token` 会执行范围校验并同步 Host；当前用有限
  cache 避免重复校验。缓存身份与 allocator 地址复用仍需单独审查，不能
  假设相同地址永久代表相同 position 内容。
- 小 workload 不适合 mcProfiler `--single-pass` Roofline，应该选择能
  充分覆盖设备的中等 workload。
- 本项目不采用 ACoolFIsh 分支中的 `torch.compile`、custom-op、fake/meta
  注册；公共执行边界保持 eager Python Op + TileLang JIT，不声明
  `torch.compile` 支持。

## 12. 一键复核命令

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax:$PYTHONPATH

./scripts/run_quant_per_channel_cast_fused.sh smoke
./scripts/run_quant_per_channel_cast_fused.sh matrix
./scripts/run_quant_per_channel_cast_fused.sh correctness
./scripts/run_quant_per_channel_cast_fused.sh baselines
./scripts/run_quant_per_channel_cast_fused.sh benchmark
./scripts/run_quant_per_channel_cast_fused.sh gates

./scripts/run_quant_per_channel_cast_fused.sh profile plain-medium production
./scripts/run_quant_per_channel_cast_fused.sh profile plain-medium shared
./scripts/run_quant_per_channel_cast_fused.sh profile expand-medium production
./scripts/run_quant_per_channel_cast_fused.sh profile expand-medium shared
./scripts/run_quant_per_channel_cast_fused.sh profile rescale-control production
```

`smoke` 快速检查本地套件和兼容矩阵的 smoke 用例；`matrix` 单独执行
augenstern 70 项兼容矩阵；`correctness` 执行本地 33 项与矩阵 70 项，
共 103 个 pytest 节点；`baselines`
验证固定 provenance 常量和数值门禁；`benchmark` 运行四方 9-workload 报告；`gates`
统一运行 diff、Manifest、Benchmark 基础测试、Ops Manifest 和 Ruff 检查。
`profile` 由脚本调用 mcProfiler，并用稳定 driver 隔离目标 Kernel。

## 13. 完成状态

| 项目 | 状态 |
|---|---|
| Manifest | 完成并通过校验 |
| Op 接口 | 完成 |
| TileLang Kernel | 完成 |
| QuantTensor/FP8 输入 | 完成 |
| plain/expand/rescale/rescale-expand | 完成 |
| C500 全路径 `tile_k=64` | 完成；含 shared-memory 门禁 |
| 混合 thread-local/shared staging | 完成；Plain/Expand local，Rescale shared |
| 正确性、边界、异常测试 | 本地 33 项 + augenstern 兼容矩阵 70 项，均已通过 |
| 固定基线测试 | 7 项通过 |
| eager PyTorch 基线 | 完成并固定上游 SHA |
| 官方式 TileLang 基线 | 完成；`dev@0266ab7` + C500 `tile_k=64` |
| Benchmark | 9 组四方 workload、3 次完整独立报告实测完成 |
| profile driver/runner | 完成；支持 production/shared 单变量采样 |
| mcProfiler | Plain/Expand 同协议 A/B 与 Rescale 控制组共 5 份精确报告完成 |
| Roofline | 9 组语义 Roofline、整卡保守参考与未校准 25% 线性参考完成 |
| `torch.compile` 集成 | 按设计不采用；保持 eager Python Op + TileLang JIT |
| MetaX C500 验证 | 完成 |
