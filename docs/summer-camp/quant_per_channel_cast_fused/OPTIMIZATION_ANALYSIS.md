# quant_per_channel_cast_fused 深度优化分析与开发计划

> 2026-08-05 的测试入口合并、torch.compile性能基线和原始TileKernels调度回退以
> [本轮重构记录](REFACTOR_LOG_2026-08-05.md)为准。本文既有性能表属于历史全路径
> tile64适配baseline，不能与回退后的新基线混用。
> 当前提交前版本已进一步收敛为40项production UT和20项Manifest Benchmark；
> 本文103项/32项矩阵内容仅表示历史消融与覆盖探索。

本文记录 `quant_per_channel_cast_fused` 当前 TileOPs 实现与两个外部实现的
深度对比、MetaX C500 A/B Benchmark、正确性验证、mcProfiler、Roofline、
失败实验以及后续分阶段开发计划。

本文先用独立 worktree 筛选调度参数，再在生产 Kernel 上做单变量 A/B。当前
分支已经完成两轮有实测收益的优化：四条路径统一使用 `tile_k=64`；Plain 和
Expand 再将输入从 shared staging 改为每线程 local staging，Rescale 和
Rescale-Expand 继续使用 shared staging。公开 Op 接口和计算语义没有改变。

## 1. 分析范围与版本

| 项目 | 内容 |
|---|---|
| 主仓库 | `/data/TileOPs-Metax` |
| 主分支 | `main` |
| 分析基线提交 | `2b833dc` |
| 原迁移实现提交 | `967b65b` |
| `tile_k=64`/固定基线提交 | `253fe14e7c33e5c9851e6120d46caf78cbe3d13b` |
| register staging 提交 | `6ec6bde` |
| 测试与精确 A/B 入口提交 | `3a0f2c0` |
| augenstern 70 项矩阵提交 | `266fabb` |
| augenstern 32 项性能矩阵适配提交 | `d85cb35` |
| 当前实测提交 | `ec3f87d6b357f9d97ab80cc49ee783ed4b4db742` |
| Rescale shared 预算门禁提交 | `c9655a2` |
| 生产 Kernel SHA256 | `3b7df342a2dba2db0988210dc5aa608793cc708cc08bb35e748a1010e854d30c` |
| 官方上游提交 | `0266ab740980de7dc03a828b8259cd73d100c2eb` |
| 官方 TileLang 基线 | `tile_kernels/quant/per_channel_cast_fused_kernel.py` |
| 官方 TileLang SHA256 | `64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc` |
| 官方 PyTorch reference | `tile_kernels/torch/per_channel_cast_fused.py` |
| 官方 PyTorch SHA256 | `6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f` |
| 仓内固定 TileLang baseline blob | `4a1c4722fd6dcd3f9fbc295ed3cb2fcbdb76d652` |
| 仓内 eager PyTorch baseline blob | `adc604bcd8c10c06f01bc6f84f9bb79f582aa286` |
| ACoolFIsh 审查版本 | `497e7237546c5df4ec8056b0e423151bffbfad0c` |
| augenstern 审查分支 | `exp/quant-per-channel-register-resident` |
| augenstern 审查版本 | `387119e154c6cfa4d09b25823635814380af91f9` |
| augenstern 测试最后提交 / blob | `c77cc75caa4b8a281a814b66b06102564ef33a01` / `11238539f3f64e2b5585a7a906067eab9d7e1e7c` |
| augenstern 性能矩阵提交 / Benchmark blob | `2d65be6` / `189ac950959893619b77fa4fbe783f8b98cd33f1` |
| 实测设备 | MetaX C500，25% sGPU，16000 MiB Vram Quota |
| 原 A/B 与 mcProfiler 实测日期 | 2026-08-04 |
| 32 项性能矩阵实测日期 | 2026-08-05 |

外部实现来源：

- ACoolFIsh：<https://gitlink.org.cn/ACoolFIsh/TileOPs-Metax/tree/feat%2Fquant-per-channel-cast-fused>
- augenstern：<https://gitlink.org.cn/augenstern/TileOPs-Metax.git>

历史 `tile_k` 与 scale-broadcast 隔离实验 worktree：

```text
/data/TileOPs-Metax-ab-small64
/data/TileOPs-Metax-ab-all64
/data/TileOPs-Metax-ab-all64-broadcast
```

以上实验 worktree 均从分析基线创建。register staging 则直接以同一个生产
Kernel 的编译期 `register_staging=True/False` 做对照，避免把动态 shape、
边界判断或接口差异混入优化收益。Op 未修改。

## 2. 最终结论

### 2.1 总体结论

不应整体替换当前实现。当前实现仍然是最合适的工程基础，原因包括：

1. 与 Manifest PR #29 的四个公开接口一致。
2. Op、Kernel、reference、测试和 Benchmark 分层清楚。
3. 契约覆盖正数越界索引、负数 padding、空输入和非整块尾部；Op 有直接校验，
   但第 12 节记录的 position-cache identity 风险仍需独立修复。
4. 本地 33 项正确性/契约测试与 augenstern 完整 70 项兼容矩阵互补；
   前者覆盖外部分支缺失的失败语义，后者提供完整 shape 广度；另有四变体
   各 8 项的来源性能矩阵，并已在 C500 完整运行。
5. FP8 输出使用逐元素完全一致的正确性门禁。
6. 另有 7 项固定基线测试，以及完整 Benchmark、mcProfiler 和 Roofline
   证据。

当前已落地的最优工程组合是：

```text
当前接口、校验、测试和文档
    +
MetaX C500 全路径 tile_k=64
    +
Plain/Expand thread-local staging
    +
Rescale/Rescale-Expand shared staging
    +
固定官方 TileLang 基线和 eager PyTorch 基线
```

动态 token Kernel 仍是后续独立实验，不与本轮提交混合。

### 2.2 各实现最值得保留的部分

| 实现 | 最有价值的部分 | 不应直接吸收的部分 |
|---|---|---|
| 当前实现 | 接口契约、常规失败校验、空输入、Kernel 内尾块保护、全路径 `tile_k=64`、混合 local/shared staging、固定基线 | position-cache identity 风险；token shape 静态 specialization；完整块路径仍有边界分支成本 |
| ACoolFIsh | 动态 token Kernel、Plain FP32 `tile_k=64`、小 Rescale FP8 vec4 线程映射 | Expand Op-side CUDA `F.pad`、正 OOB 静默当 padding、整体 Op 重写；FP32 Expand 仍用 `shared_rows=120` |
| augenstern | Plain/Expand register staging、Rescale 保留 shared、扩展 shape 测试 | 接口名和目录不兼容、缺少正 OOB 检查、拒绝空输入；CUDA register 策略未经实测 |

### 2.3 哪个实现更好

- 工程质量和可维护性：当前实现更好。
- C500 稳定态 Kernel 调度：augenstern 的 Plain/Expand register staging 已
  选择性吸收；其分支记录的 Rescale register 消融有回退，因此当前保留 shared。
- 小 Rescale：ACoolFIsh 最新 FP8 vec4 映射有其分支内性能证据，但线程映射、
  tile 策略和接口同时不同，只列为当前 Kernel 上的独立候选，尚未吸收。
- 动态 shape：ACoolFIsh 和官方 TileKernels 的动态 token 设计更完整，后续
  可独立验证。
- `torch.compile`/custom-op/fake/meta 调用边界超出本次算子迁移范围，明确不采用。
- 最终方案：保持当前工程基础，只吸收有独立 C500 实测证据的优化。

### 2.4 从原始迁移到当前代码

对 `967b65b..HEAD` 的生产 Op/Kernel diff 进行审计后，只有两项真正
改变设备执行的 Kernel 优化。`tileops/ops/quant/per_channel_cast_fused.py`
在这个区间没有 diff，四个公开 Op、静态 Kernel cache、shape/dtype
校验、空输入和正 OOB 失败语义均是原始迁移已有能力，不是后续
性能优化。

| 项目 | 原始迁移 `967b65b` | 当前代码 | 影响 |
|---|---|---|---|
| `tile_k` | Rescale=256，BF16 Plain/Expand=128，FP32=64 | 四路径固定 64 | BF16 隔离 A/B 增加约 2 倍 hidden workgroup，Rescale 增加约 4 倍；22 组中 21 组延迟降低 19.64%～72.94%，原来已是 tile64 的 FP32 用例持平 |
| input staging | 四路径均使用 shared tile | Plain/Expand thread-local；Rescale 两路 shared | 吸收 augenstern `fa6bafe` 消融和 `87b083c` 变体选择；五组延迟降低 3.38%～76.78% |
| shared/CTA | BF16 tile128 34,816 B；FP32 tile64 33,792 B；FP8 tile256 36,864 B | Plain/Expand 1,024 B；Rescale 9,216 B | Plain/Expand 删除一次 shared store/load 往返；增加 C500 64 KiB 预算门禁 |
| 语义 HBM 与 FLOPs | 两遍 amax/量化 | 不变 | 优化改变片上资源和并行度，不改变 Roofline 字节数与计算量 |

这两阶段的 A/B workload 矩阵不同，不能将百分比相乘成一个没有
直接实测的“最终总加速”。

## 3. 当前语义和代码边界

相关代码：

- Kernel：[`tileops/kernels/quant/per_channel_cast_fused.py`](../../../tileops/kernels/quant/per_channel_cast_fused.py)
- Op：[`tileops/ops/quant/per_channel_cast_fused.py`](../../../tileops/ops/quant/per_channel_cast_fused.py)
- 测试与独立PyTorch oracle：[`tests/ops/test_per_channel_cast_fused.py`](../../../tests/ops/test_per_channel_cast_fused.py)，合并后103项
- 固定基线测试：[`tests/ops/test_per_channel_cast_fused_baselines.py`](../../../tests/ops/test_per_channel_cast_fused_baselines.py)
- 固定 TileLang 基线：[`benchmarks/ops/per_channel_cast_fused_baselines.py`](../../../benchmarks/ops/per_channel_cast_fused_baselines.py)
- Benchmark：[`benchmarks/ops/bench_per_channel_cast_fused.py`](../../../benchmarks/ops/bench_per_channel_cast_fused.py)，统一承载9项稳定消融与32项规模矩阵
- mcProfiler 驱动：[`benchmarks/ops/profile_per_channel_cast_fused.py`](../../../benchmarks/ops/profile_per_channel_cast_fused.py)
- 一键复核脚本：[`scripts/run_quant_per_channel_cast_fused.sh`](../../../scripts/run_quant_per_channel_cast_fused.sh)
- 原始实测文本：[`docs/summer-camp/quant_per_channel_cast_fused/artifacts/`](artifacts/)
- Manifest：[`tileops/manifest/quantization.yaml`](../../../tileops/manifest/quantization.yaml)

上游 `per_channel_cast_fused()` 的 `x` 有两种形式：

1. 普通 `torch.Tensor`：对应 TileOPs plain/expand Op 的 `x`。
2. `QuantTensor`，即 `(data, sf_invs)`：在 TileOPs 中拆成 Rescale Op 的
   `x` 和 `x_sf_invs` 两个输入。

`pos_to_token` 是否存在决定是否使用 Expand 变体。因此当前四个 Op 与上游
一个可选参数函数的对应关系为：

| 上游输入形态 | TileOPs Op |
|---|---|
| 普通 tensor，无 position map | `QuantPerChannelCastFusedOp` |
| 普通 tensor，有 position map | `QuantPerChannelCastFusedExpandOp` |
| QuantTensor，无 position map | `QuantPerChannelCastFusedRescaleOp` |
| QuantTensor，有 position map | `QuantPerChannelCastFusedRescaleExpandOp` |

优化不得改变这一接口对应关系。

## 4. Kernel 计算过程

每个 workgroup 固定处理一个 `128 × 64` 的 token-hidden tile，使用 256 个
线程。`vec_m=32`、`vec_k=1`，即每个线程处理同一 hidden channel 上的
32 个 token。

1. Expand 路径加载 `pos_to_token` 并通过 shuffle 广播；负数表示 padding，
   非负索引已由 Op 保证小于源 token 数。
2. Rescale 路径把对应的 FP32 `x_sf_invs` 保存在每线程 local 数组。
3. 第一遍发起一次全局 `x` load；它可能命中 VL1/L2 或到达 HBM。Plain/Expand 写入每线程
   `x_staging[32, 1]`；Rescale/Rescale-Expand 写入
   `x_shared[128, 64]`。
4. 输入转为 FP32；Rescale 路径乘 `x_sf_invs`；每线程累计 32 个 token 的
   `max(abs(value))`，再通过 `amax_shared` 做跨线程归约。
5. 将最大值截断到至少 `1e-4`，计算：

   ```text
   out_sf     = amax / 448
   out_sf_inv = 448 / amax
   ```

6. 如果 `round_sf=true`，将 scale 向上取整为 2 的幂。
7. 第二遍从 `x_staging` 或 `x_shared` 重新读取输入，乘以
   `out_sf_inv` 并转换为 FP8 E4M3；Rescale 路径再次乘 `x_sf_invs`。
8. 写出 FP8 `out` 和 FP32 `out_sf`。

Rescale 路径的逻辑输入是：

```text
logical_x[t, h] = x[t, h] * x_sf_invs[t, h // 128]
```

### 4.1 语义访存次数

| 访问 | Plain/Expand | Rescale/Rescale-Expand |
|---|---:|---:|
| 全局/逻辑读取 `x` | 每有效输出元素 1 次 | 每有效输出元素 1 次 |
| 每线程 local staging | 写 1 次、读 1 次 | 无 |
| shared 输入 staging | 无 | 写 1 次、读 1 次 |
| 全局/逻辑写 `out` | 每有效输出元素 1 次 | 每有效输出元素 1 次 |
| 全局/逻辑写 `out_sf` | 每 128-token block、每 channel 1 次 | 同左 |
| FP32 `x_sf_invs` | 无 | 每 token、每 128 channel 组逻辑读取 1 次 |
| `pos_to_token` | Expand 时每输出 token 逻辑读取 1 次 | Expand 时同左 |

`amax_shared` 在四条路径中都存在，用于跨线程归约和广播 reciprocal scale；
register staging 只删除输入 tile 的 shared 写/读，不删除归约 scratch。

实现中 position map 和输入 scale 会被不同 hidden tile 重复发起 load 指令，
但大部分重复读取由 VL1/L2 吸收。Manifest Roofline 使用语义最小字节数，
不使用实现的 `tile_k` 计算字节数。优化前后 HBM 语义流量和算术强度不变；
变化的是片上 local/shared 流量和资源占用。

## 5. `tile_k=64` 选择与固定基线

原迁移配置：

```python
tile_k = 256 if with_rescale else (64 if in_dtype == "float32" else 128)
```

当前 C500 配置：

```python
tile_k = 64
```

当前 Kernel 的支持范围为架构门禁 `80` 下的 MetaX C500 路径，不据此扩大
未经验证的 CUDA 支持声明。固定 `tile_k=64` 已在隔离 worktree 完成两轮
A/B，并在提交 `253fe14` 中落地。

固定性能基线来自官方：

```text
https://github.com/MetaX-MACA/TileKernels-Metax
commit: 0266ab740980de7dc03a828b8259cd73d100c2eb
path:   tile_kernels/quant/per_channel_cast_fused_kernel.py
```

benchmark-only 上游式基线保持官方核心算法、动态 token 维度、shared
staging、两遍归约/量化和完整块执行结构；它为当前 Benchmark 显式展开了
配置、scale helper 和调用接口，并把官方 plain/rescale 的 128/256 tile 在
C500 上统一设为 64。它不是官方源文件的逐字副本，因此结论只把它作为固定的
上游式算法基线，不声称生成代码除 tile 外完全等价。该文件与生产 Kernel
独立，后续优化不会静默移动基线。

### 5.1 shared memory 与线程 local 占用

这里只统计源码中的显式分配。`amax_shared[1, 256]` 固定为 1,024 B。

| 路径 | shared-staging 对照 | 当前生产配置 | 输入 staging 变化 |
|---|---:|---:|---:|
| BF16 plain/expand，tile64 | 17,408 B | 1,024 B | 每 CTA -16,384 B |
| FP32 plain/expand，tile64 | 33,792 B | 1,024 B | 每 CTA -32,768 B |
| FP8 rescale/rescale-expand，tile64 | 9,216 B | 9,216 B | 不变 |

Plain 每线程新增 32 个输入 local 值；Expand 同时还保留 32 个 position。
Rescale 若照搬 register staging，还会让 32 个 FP32 scale 与输入跨越归约同时
存活。augenstern `387119e` 的独立消融记录了该路径回退；本仓最终三轮 A/B
没有纳入 Rescale-local 候选，因此生产配置保守地保留 shared。
源码中的 `T.alloc_local` 表示 thread-local staging；是否全部物理驻留寄存器，
还必须结合 private-memory spill 和 profiler 计数判断，不能仅由名称推断。

FP32 shared staging 的 `tile_k=128` 需要 67,584 B，明确超过 C500 的
65,536 B 上限；BF16 `tile_k=128` 并不会溢出，改为 64 的原因主要是并行度
和性能。当前 Plain/Expand 在 tile64 上再改为 local staging 后，显式 shared
只剩 1 KiB 归约 scratch。

## 6. Benchmark 方法

正式筛选协议与仓库 `bench_kernel` 默认协议一致：

- 真实 MetaX C500，25% sGPU。
- 10 次预热。
- 每个 trial 50 次测量。
- 3 个 trial，报告 trial mean 的中位数。
- CUPTI Kernel 时间，不包含编译时间和 Host launch 时间。
- 每次测量前清空 L2。
- 输入使用 3 份 clone 轮换。
- 当前实现、shared-staging 对照、固定 TileLang 和 eager PyTorch 使用相同输入。
- Plain/Expand 的 shared 对照由同一个生产 Kernel 构造，唯一变化是编译期
  `register_staging=False`。
- 完整 9 workload 独立运行三轮。
- 另有独立的 augenstern 来源矩阵：当前 `Quant*` Op、当前 manifest
  `__suite: augenstern-performance` workload、当前 eager reference；不与
  四方 9-workload 长期报告混合。

三轮的候选和 shared-staging 结果高度一致；下表报告三次完整运行各自结果的
中位数，而每次运行内部仍按三个 trial mean 的中位数统计。

## 7. Benchmark 实测结果

### 7.1 历史 tile64 筛选

提交 `253fe14` 之前的 22 组隔离 A/B 比较原迁移的 128/64/256 tile 与
全路径 tile64：21 组延迟下降 19.64%～72.94%，唯一的 FP32 用例因原迁移
已经使用 tile64 而持平，没有发现回退。这组数据只说明 tile 选择，不包含
register staging 收益。

### 7.2 register staging 精确 A/B

以下为三次完整 9-workload 运行的中位数。`shared` 与 `production` 使用同一个
静态 shape 生产 Kernel、相同输入和计时框架，唯一差异是编译期
`register_staging=False/True`。Rescale 两条路径生产配置本来就是 shared，
因此不重复列伪 A/B。

| workload | shared ms | production ms | 加速 | 延迟下降 |
|---|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.0325 | 0.0314 | 1.035× | 3.38% |
| plain 1024×4096 BF16 rounded | 0.0705 | 0.0507 | 1.391× | 28.09% |
| plain 4096×8192 FP32 | 1.2286 | 0.2852 | 4.308× | 76.78% |
| expand 512×4096→1024 BF16 | 0.0754 | 0.0513 | 1.470× | 31.96% |
| expand 1024×4096→2048 FP32 rounded | 0.3322 | 0.0862 | 3.854× | 74.05% |

五组均无回退。最小 BF16 workload 主要受启动和固定开销限制，收益仅 3.38%；
中大型 BF16 收益约 28%～32%，FP32 因删除 32 KiB/CTA 的 shared staging，
收益达到约 74%～77%。这比直接引用远端百分比更可靠，因为对照没有混入
动态 shape、接口或边界判断差异。

### 7.3 当前实现、固定 TileLang 与 eager PyTorch

| workload | production ms | 固定 TileLang ms | eager PyTorch ms | 相对 TileLang | 相对 PyTorch |
|---|---:|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.0314 | 0.0243 | 0.0748 | 0.774× | 2.382× |
| plain 1024×4096 BF16 rounded | 0.0507 | 0.0520 | 0.2618 | 1.026× | 5.164× |
| plain 4096×8192 FP32 | 0.2852 | 0.7889 | 1.5036 | 2.766× | 5.272× |
| expand 512×4096→1024 BF16 | 0.0513 | 0.0778 | 0.3931 | 1.517× | 7.663× |
| expand 1024×4096→2048 FP32 rounded | 0.0862 | 0.3374 | 0.6570 | 3.914× | 7.622× |
| rescale 128×1024 FP8 | 0.0546 | 0.0526 | 0.0801 | 0.963× | 1.467× |
| rescale 1024×4096 FP8 rounded | 0.0851 | 0.0845 | 0.3731 | 0.993× | 4.384× |
| rescale-expand 512×4096→1024 FP8 | 0.1259 | 0.1282 | 0.4485 | 1.018× | 3.562× |
| rescale-expand 1024×4096→2048 FP8 rounded | 0.1972 | 0.1972 | 0.7908 | 1.000× | 4.010× |

这里“相对”使用 `baseline_latency / production_latency`，大于 1 表示生产
Kernel 更快。除最小 Plain 和两组 Rescale 外，生产 Kernel 已经达到或超过
固定 TileLang；全部 9 组均快于 eager PyTorch。最小 Plain 只有 16 个 workgroup，
固定启动成本主导；Rescale 保留 shared 后与固定 TileLang 基本持平。

三轮原始报告已提交，GitHub 上可按 SHA256 复核：

| 运行 | 仓内原始报告 | SHA256 |
|---|---|---|
| run 1 | [benchmark_run1.txt](artifacts/benchmark_run1.txt) | `c8657fc39c8270dc9dfd7cd6609832426924253612dd7fb7af96000a45a92d8d` |
| run 2 | [benchmark_run2.txt](artifacts/benchmark_run2.txt) | `802777a578097eb8dad1b30c0db35042549e71046ceed40dced19c7176490dfd` |
| run 3 | [benchmark_run3.txt](artifacts/benchmark_run3.txt) | `08fe02d82be3be13b973bc9375bc45d93b52faae30c30e91bae2ff9671c6db8b` |

### 7.4 augenstern 来源矩阵的当前 Op 适配

远端性能提交 `2d65be6` 使用旧的 `PerChannelCastFused*Op` 和
`tileops/kernels/per_channel_cast_fused_maca.py`；当前仓库的真实接口是
`QuantPerChannelCastFused*Op`，Kernel 在
`tileops/kernels/quant/per_channel_cast_fused.py`。适配工作因此分为三层：

1. 将 32 个 shape、dtype、`round_sf`、label 和 Expand 输出 token 原样写入
   当前 `quantization.yaml`，标记为 `__suite: augenstern-performance`。
2. 新增独立 Benchmark 文件，从当前 manifest 读取矩阵，并把 Rescale 的
   `x_sf` 映射成当前 `x_sf_invs`；Roofline 通过当前 Op 的
   `ManifestBenchmark(...).profile()` 触发 `eval_roofline()`。
3. 输入生成保持来源的随机 gather 和 FP8 构造，但在计时前调用当前仓库的
   独立 reference 做 FP8 逐元素、scale 容差和异常传播门禁。

旧的 9 项四方入口过滤该 suite，因而仍能做长期 production/shared/fixed
TileLang/eager A/B；新入口只比较当前 production/eager，避免把外部旧算子、
不同 Kernel 路径或不同 staging 策略混入加速归因。

真实 C500 命令和结果：

```text
benchmark-matrix --collect-only -q: 32 tests collected
benchmark-matrix -m smoke: 4 passed, 28 deselected
benchmark-matrix: 32 passed in 117.11s
stable four-way benchmark: 9 passed in 32.94s
```

原始报告为
[`benchmark_augenstern_matrix.txt`](artifacts/benchmark_augenstern_matrix.txt)，
SHA256 为
`0ec62a2288e7bfa890b2b022447962c0d3437d69810636c35b52b15430ee7405`。
production 相对 eager 的 case 算术平均加速比为 Plain `3.859x`、Expand
`6.723x`、Rescale `3.406x`、RescaleExpand `4.199x`，总体 `4.547x`；32
项均无回退。最大单项为 Expand `4097x7168->8192` BF16 rounded，`8.945x`；
最小单项为 Rescale `128x256` FP8，`1.399x`。逐项延迟表和输入构造见主
README 第 7.4 节，原始 121 行报告保留所有 Roofline 字段和 Kernel config。

该矩阵的 production 语义 Roofline 最高为 `0.3745 TFLOP/s` 和
`0.5932 TB/s`。它证明当前实现覆盖了远端中大型 shape，但不等价于 32 份
mcProfiler 物理采样；硬件 shared/private/HBM counter 仍以第 9 节五份
multi-batch per-kernel 报告为准。

## 8. 正确性验证

### 8.1 完整测试套件

本地契约测试结果为：

```text
33 passed
```

augenstern `exp/quant-per-channel-register-resident@387119e` 的完整兼容
矩阵结果为：

```text
70 passed in 183.35s
```

同一进程的统一 `correctness` 入口合并执行两个套件，结果为：

```text
103 passed in 32.08s
```

该统一运行复用了前一次矩阵运行生成的 TileLang 编译缓存，因此耗时
只用于复现当时环境，不与冷缓存的 183.35 秒直接比较。

固定 TileLang/PyTorch 基线自身的 5 种计算路径和 2 项契约测试结果：

```text
7 passed
```

测试覆盖：

- BF16 和 FP32 plain 输入。
- FP8 QuantTensor/Rescale 输入。
- 四个算子变体。
- `round_sf=false/true`。
- 16、128、144、160、256 token 输出，以及 128-token 尾块。
- 重复、逆序、混合和全 padding position map。
- 空输入和空 source。
- 全零、`1e-8` amax-floor 输入和正负极值。
- Plain/Expand 必须启用、Rescale/Rescale-Expand 必须关闭 register staging。
- register staging 的 BF16/FP32 shared 预算均为 1,024 B。
- 非连续输入。
- 非法 dtype、rank、hidden、token 数、scale shape。
- 正的越界 position。
- 输出 shape、dtype 和 contiguous。
- FP8 逐元素完全一致。

### 8.2 augenstern 70 项兼容矩阵

远端测试最后三个功能提交为 `ff86cdf`、`ea5d33e` 和
`c77cc75`。当前分支已将其完整迁移为独立测试文件：

| 分组 | 数量 |
|---|---:|
| 基础四变体 | 17 |
| 固定 `hidden=512` token sweep | 20 |
| Plain token × hidden | 8 |
| Expand token × hidden | 8 |
| Rescale token × hidden | 5 |
| Rescale-Expand token × hidden | 5 |
| Gather pattern | 3 |
| 数值边界 | 3 |
| 构造参数校验 | 1 |
| **合计** | **70** |

所有 shape、dtype、`round_sf`、测试 ID 和 smoke/full 层级与远端一一
对应；源码内用分组计数与总数断言防止后续静默丢失用例。本地
保留 `torch.manual_seed(0)` 与 Rescale 的耦合输入生成，但使用仓内
独立 eager PyTorch reference、FP8 逐元素精确比较和 scale 严格容差。
远端的 reference OOM 静默返回和非 MACA 宽松相似度路径没有吸收。

本地 33 项套件继续保留正 OOB 失败、空输入、空 source、非连续
输入、非法 dtype/shape、输出 contiguous、shared-memory 预算和 staging
策略门禁。两个套件合计 103 个 pytest 节点，职责互补。

## 9. mcProfiler 分析

### 9.1 最终采样协议和无效数据排除

最终采样使用固定入口：

```text
benchmarks/ops/profile_per_channel_cast_fused.py
scripts/run_quant_per_channel_cast_fused.sh profile
```

五次采样命令对应 Plain shared/production、Expand shared/production 和
Rescale production control：

```bash
./scripts/run_quant_per_channel_cast_fused.sh profile plain-medium shared
./scripts/run_quant_per_channel_cast_fused.sh profile plain-medium production
./scripts/run_quant_per_channel_cast_fused.sh profile expand-medium shared
./scripts/run_quant_per_channel_cast_fused.sh profile expand-medium production
./scripts/run_quant_per_channel_cast_fused.sh profile rescale-control production
```

runner 固定 shape、输入构造、Kernel 名和 selected metrics，并记录 commit、
dirty 状态、预期 workgroup 数、显式 shared 字节和 Kernel SHA256。mcProfiler
使用 `--per-kernel`，不使用 `--single-pass`；选定事件由四个 event batch
采集。五份有效目标报告的 Workgroups/Waves 均为 `512/2,048`，与
`ceil(1024/128) × ceil(4096/64) = 512` 和每组 4 waves 完全一致。

只引用输出目录中的 `1_main_kernel.txt`。顶层 `report.txt` 会聚合 runner 的
`torch.ones`、`torch.arange` 和 Host-to-device 初始化 Kernel。例如 Plain
production 的顶层报告为 1,024 workgroups、10,240 waves，目标报告才是
512 workgroups、2,048 waves，因此顶层报告的 HBM、workgroup 和 RoofLine
数据不能用于评价目标 Kernel。

早期 `--single-pass` 报告还出现过缺少 `processed_data` 或超过物理范围的
推算比例。这些结果已经排除，不用于任何结论。下面所有指标均来自同一套
multi-batch、per-kernel、selected-metrics 协议。

### 9.2 最终原始指标

S 表示 shared-staging 对照，P 表示 production。`RoofLine during` 是报告
`processed_data.during` 的原始 cycle 计数；`case_bandwith` 保留 mcProfiler
报告中的字段拼写。

| 指标 | Plain S | Plain P | Expand S | Expand P | Rescale P |
|---|---:|---:|---:|---:|---:|
| Workgroups | 512 | 512 | 512 | 512 | 512 |
| Waves | 2,048 | 2,048 | 2,048 | 2,048 | 2,048 |
| 显式 shared/CTA | 17,408 B | 1,024 B | 17,408 B | 1,024 B | 9,216 B |
| Average wave life cycles | 9,548.75 | 11,242.30 | 10,321.08 | 11,427.86 | 17,190.72 |
| RoofLine `during` | 77,231 | 55,324 | 84,562 | 57,347 | 79,659 |
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
| Hardware `case_I` | 159.729 | 148.183 | 237.690 | 219.647 | 324.795 |
| Hardware `case_bandwith` (GB/s) | 185.958 | 259.368 | 114.098 | 168.078 | 123.639 |

### 9.3 同协议 A/B 结论

Plain 和 Expand 的每组 A/B 使用完全相同的静态 shape、输入、grid、metrics
和采样协议，唯一编译期差异是 `register_staging=False/True`。因此可以归因：

1. Plain 的 `during` 从 77,231 降到 55,324，下降 28.37%；Benchmark
   延迟下降 28.09%。Expand 的 `during` 从 84,562 降到 57,347，下降
   32.18%；Benchmark 延迟下降 31.96%。两套独立计时证据一致。
2. 两组 shared load 都从 68,272 降到 4,016，下降 94.12%；shared store
   都从 66,766 降到 2,510，下降 96.24%。剩余 shared 指令来自四条路径
   共有的 `amax_shared` 归约和广播。
3. Plain HBM 总流量从 12,765,984 B 变为 12,754,912 B，变化 -0.0867%；
   Expand 从 8,576,288 B 变为 8,567,776 B，变化 -0.0993%。收益不是减少
   HBM 语义访问，而是删除输入 tile 的 shared 往返并降低每 CTA 资源占用。
4. 五份报告的 private read/write 都是 0，选定 workload 没有观察到
   thread-local staging spill 到 profiler 可见的 private memory。
5. production 的 Average wave life cycles 略高，L2 hit 也比 shared 对照低，
   但总 `during` 明显下降。因此不能用单 wave 生命周期或 cache hit 单独解释
   Kernel 性能。
6. Rescale production 仍有 68,408/66,899 次 shared load/store，且 shared
   efficiency 为 41.22%，与该路径有意保留 shared staging 一致。Plain/
   Expand 的结果不能直接外推到 Rescale。

Hardware `case_I` 和 `case_bandwith` 基于 profiler 的硬件指令与 memory
transaction 口径，只能用于同一协议内 A/B。它们不等于下一节的 Manifest
FLOPs、语义 bytes、算术强度或语义带宽。

### 9.4 最终报告和 SHA256

| case | 仓内目标 Kernel 报告 | SHA256 |
|---|---|---|
| Plain shared | [mcprofiler_plain_shared.txt](artifacts/mcprofiler_plain_shared.txt) | `e6703fe9f2f17e82771ea1337d4a5d7bc72f710292442d390a549c7057f1b60f` |
| Plain production | [mcprofiler_plain_production.txt](artifacts/mcprofiler_plain_production.txt) | `577a203009afdbc9c234ff4b309584a72ec9a174ce40c23a5e2ad2141e0f2a3e` |
| Expand shared | [mcprofiler_expand_shared.txt](artifacts/mcprofiler_expand_shared.txt) | `871b4086dccf93fd41db24271b2a535362aee6639b4b07092ffdeaa44ee9e0be` |
| Expand production | [mcprofiler_expand_production.txt](artifacts/mcprofiler_expand_production.txt) | `e35974d3011c18da534bf90451940211c06faeb7c021e67443d1fbffaf7cd198` |
| Rescale production | [mcprofiler_rescale_production.txt](artifacts/mcprofiler_rescale_production.txt) | `adc21ab9605e765aad0ec44baf10fad3cde94c1999699188fe8e373dd758b3ee` |

## 10. Roofline 分析

### 10.1 口径和峰值边界

本节只使用 Manifest 公式和第 7 节三次独立完整 Benchmark 的中位延迟：

```text
Manifest AI        = Manifest FLOPs / Manifest semantic bytes
achieved TFLOP/s   = Manifest FLOPs / benchmark latency
semantic bandwidth = Manifest semantic bytes / benchmark latency
```

mcProfiler 给出的整卡参数为：

| 参数 | 整卡值 |
|---|---:|
| HBM 峰值带宽 | 1,843.2 GB/s |
| Ridge point | 260 FLOP/B |
| 推导计算峰值 | 479.232 TFLOP/s |

其中计算峰值由 `1,843.2 GB/s × 260 FLOP/B` 推导。当前环境分配 25%
Compute，但没有经过校准的 sGPU HBM 带宽上限。`460.8 GB/s = 1,843.2 ×
25%` 和 `119.808 TFLOP/s = 479.232 × 25%` 只能作为线性缩放参考，不能称为
切片实测峰值，也不能作为“Roofline 效率”的验收分母。

两组 FP32 workload 的 Manifest 语义带宽分别达到 591.94 GB/s 和
489.71 GB/s，即 460.8 GB/s 线性参考的 128.46% 和 106.27%。语义带宽会把
可由 cache 复用的数据按算子逻辑重复计数，因此这既不是物理 HBM 带宽超过
峰值，也不能单独证明切片的物理带宽策略；它只证明不能把 Manifest 语义流量
除以未校准线性参考后称作物理效率。

### 10.2 Manifest 算术强度

设 `N` 为输出 token 数、`H` 为 hidden、`S=ceil(N/128)`、`C=H/128`，
`e` 为 Plain/Expand 输入元素字节数：

| 变体 | FLOPs | HBM 语义字节数 |
|---|---|---|
| Plain | `3NH + 2SH` | `NHe + NH + 4SH` |
| Expand | `3NH + 2SH` | `NHe + 4N + NH + 4SH` |
| Rescale | `5NH + 2SH` | `2NH + 4NC + 4SH` |
| Rescale-Expand | `5NH + 2SH` | `2NH + 4NC + 4N + 4SH` |

对应的算法语义 AI 为 FP32 Plain/Expand 约 0.599、BF16 Plain/Expand 约
0.995、FP8 Rescale/Rescale-Expand 约 2.431 FLOP/B，说明算子语义的计算/搬运
比很低。它与 mcProfiler 根据硬件指令和 transaction 计算的 `case_I/MAX_I`
不是同一口径，不能直接用 `0.599～2.431 < 260` 对九个 workload 做严格硬件
Roofline 分类。小 workload 还明显受 launch、固定调度和设备覆盖不足影响。

`tile_k` 和 local/shared staging 都是实现参数，不改变这些语义 FLOPs、bytes
或 AI。

### 10.3 三轮 Benchmark 语义 Roofline

`1843.2` 和 `460.8` 两列都只是用物理峰值数值归一化 Manifest 语义带宽的
参考比值。由于分子不是物理 HBM transaction，它们不是物理带宽利用率、效率
或下界；`460.8` 列也不能用来推断实际切片策略。

| production workload | Manifest AI | achieved TFLOP/s | semantic BW | semantic BW / 1843.2 | semantic BW / 460.8 |
|---|---:|---:|---:|---:|---:|
| Plain 128×1024 BF16 | 0.994845 | 0.0126 | 12.65 GB/s | 0.69% | 2.75% |
| Plain 1024×4096 BF16，rounded | 0.994845 | 0.2495 | 250.77 GB/s | 13.61% | 54.42% |
| Plain 4096×8192 FP32 | 0.599379 | 0.3548 | 591.94 GB/s | 32.11% | 128.46% |
| Expand 512×4096→1024 BF16 | 0.994525 | 0.2466 | 247.92 GB/s | 13.45% | 53.80% |
| Expand 1024×4096→2048 FP32，rounded | 0.599263 | 0.2935 | 489.71 GB/s | 26.57% | 106.27% |
| Rescale 128×1024 FP8 | 2.431818 | 0.0120 | 4.95 GB/s | 0.27% | 1.07% |
| Rescale 1024×4096 FP8，rounded | 2.431818 | 0.2472 | 101.65 GB/s | 5.52% | 22.06% |
| Rescale-Expand 512×4096→1024 FP8 | 2.430667 | 0.1671 | 68.74 GB/s | 3.73% | 14.92% |
| Rescale-Expand 1024×4096→2048 FP8，rounded | 2.430667 | 0.2134 | 87.78 GB/s | 4.76% | 19.05% |

### 10.4 语义字节与物理 HBM 的边界

Manifest Roofline 按逻辑输出计数，mcProfiler 则统计到物理 HBM transaction：

- Plain medium 的语义流量是 12,713,984 B，production profiler 的物理 HBM
  流量是 12,754,912 B，两者接近，连续读取下语义模型与物理流量对应良好。
- Expand medium 的语义流量是 12,718,080 B，production profiler 的物理 HBM
  流量只有 8,567,776 B。该 workload 从 512 个源 token gather 到 1,024 个
  输出 token，重复源读取被 VL1/L2 复用，因此物理 HBM 小于语义 bytes 是
  预期结果，不能据此减少 Manifest 公式中的逻辑读取。

同理，第 9 节 mcProfiler 的硬件 `case_I=148.183～324.795` 和
`case_bandwith=114.098～259.368` 不能与本节的 Manifest AI 或 semantic
bandwidth 混用。前者回答硬件执行了多少指令和 transaction，后者回答完成
算子语义所需的工作量和字节数。

按 mcProfiler 自己的硬件口径，四份 Plain/Expand 报告的
`case_I=148.183～237.690 < MAX_I=260`，位于 profiler 的 memory-side；
Rescale control 的 `case_I=324.795 > 260`，位于 profiler 的 compute-side。
这是五个指定 Kernel 报告的严格硬件分类，不能外推到其余未 profile workload。

最终结论是：Plain/Expand thread-local staging 不改变算法 AI 或 HBM 语义
流量；它通过删除片上 shared 往返提高同等语义工作量下的吞吐。Rescale 仍受
shared staging、scale 读取和额外乘法影响，是下一项独立性能实验的重点。

### 10.5 32 项矩阵的语义 Roofline

32 项矩阵复用同一组公式，production 最高 achieved throughput 为
`0.3745 TFLOP/s`，最高 semantic bandwidth 为 `0.5932 TB/s`。后者来自
Expand FP32 `4001x3072->8192`，相当于整卡 1,843.2 GB/s 参考的 32.18%，
但达到未校准 25% 线性参考的 128.73%。这是 Expand 重复 gather 的 Manifest
逻辑流量和 cache 复用共同造成的口径现象，不能解读成物理 HBM 超峰值。

Plain/Expand 的 `0.599～0.995 FLOP/B` 与 Rescale 两路约 `2.431 FLOP/B`
都远低于 mcProfiler 整卡 ridge point 260 FLOP/B，但严格的 hardware-side
分类仍只适用于实际采样的五份报告。32 项矩阵的价值是验证 shape 广度下的
稳定延迟、正确性和语义 Roofline，不冒充 32 项物理 counter 测量。

## 11. 外部优化的采用边界

### 11.1 scale load 加 shuffle 广播

在独立 worktree 中保持 `tile_k=64` 不变，只把每个 wave 的 scale load
改成 lane 0 读取后 `T.shfl_sync()` 广播。

该版本通过全部 28 项正确性测试，但性能稳定回退：

| workload | 无广播 ms | 广播 ms | 回退 |
|---|---:|---:|---:|
| rescale 128×1024 | 0.054925 | 0.061957 | 12.80% |
| rescale 128×4096 | 0.055045 | 0.066755 | 21.27% |
| rescale 1024×4096 | 0.085647 | 0.093990 | 9.74% |
| rescale-expand 512×4096 → 1024 | 0.126072 | 0.141693 | 12.39% |

结论：C500 上 shuffle 指令和依赖链的成本高于减少 scale load 指令的收益，
不应把该优化加入正式实现。

### 11.2 Expand Op-side CUDA `F.pad`

ACoolFIsh 只在 Expand 输出不是 128 整块时，在 Op 层对 CUDA position map
执行 `F.pad`，补到 128-token 对齐后再对输出切片。该做法不是其所有路径的
共同问题，但不适合当前公共 Expand 路径，原因包括：

1. 增加一个 GPU pad Kernel。
2. 增加临时 tensor 分配。
3. 输出 16 或 144 token 时仍执行完整 128-token 尾块计算。
4. Kernel 被绕过 Op 直接调用时依赖 Op-side padding 才能保证安全。

当前 Kernel 内原生边界保护更清楚，也符合迁移指南关于设备计算的要求。

### 11.3 FP32 `shared_rows=120`

ACoolFIsh 的 `shared_rows=120` 只在 FP32 且 `tile_k=128` 时启用：缓存
120 行，第二遍重新读取最后 8 行。最新 `497e723` 已将非 Expand Plain FP32
改为 `tile_k=64`，因此该路径不再使用 120 行方案；FP32 Expand 仍为
`tile_k=128`，才适用下面的限制。

该方案存在：

- 约 62 KiB shared/CTA，占 C500 64 KiB 上限的大部分。
- 额外约 6.25% 输入 HBM load。
- occupancy 风险。
- 不能与当前 tile64 register staging 的 1 KiB shared 占用相比。

当前 FP32 Plain/Expand 都使用 tile64 register staging，没有必要为扩大 tile
重新引入额外 HBM load 和接近上限的 shared 占用。

### 11.4 删除 position 上界检查

两个外部分支都没有按当前 Manifest 的失败语义完整验证。准确契约是：负数
表示合法 padding；每个非负 position 必须小于源 token 数。

```text
pos_to_token < 0 or 0 <= pos_to_token < num_tokens
```

ACoolFIsh 最新 Kernel 会把超界正索引当成 padding 零值，Op 不主动报错；这能
避免设备越界，但与当前“正 OOB 必须失败”的契约不同。augenstern Kernel 则
对未经上界检查的 token 使用 `T.assume(token < num_tokens)`，存在越界读取
风险。当前失败语义必须保留。

### 11.5 ACoolFIsh 小 Rescale FP8 vec4

`497e723` 对 `num_tokens_out <= 128` 的 Rescale 使用
`num_threads_per_token=16`、`vec_k=4`，以 4-byte FP8 访问匹配 shared bank
宽度，并增加 subgroup-aware shuffle。其提交记录小 Rescale 延迟下降 39.23%。

这是有技术依据的候选，但不直接合入：其分支同时使用动态 token、按规模切换
tile64/tile256、scale shuffle 和不同 Op 尾块策略，提交百分比不能归因到 vec4
单点。后续应在当前静态 shape、安全边界和 tile64 shared Rescale 上只切换
`threads_per_token=64→16`，通过当前 103 项正确性测试并做完整
Rescale A/B 后决定。

### 11.6 `torch.compile` 与整体接口替换

ACoolFIsh 分支还包含 custom-op/fake/meta 等 `torch.compile` 调用边界。该能力
不属于本次 Manifest #29 的迁移与性能目标，当前明确不采用，也不以此重写 Op。

不应改变：

- `QuantPerChannelCastFused*Op` 命名。
- `tileops/ops/quant/` 和 `tileops/kernels/quant/` 目录。
- Manifest 已合入的构造参数。
- 空输入语义。
- 当前独立 reference 和精确测试门禁。

## 12. 当前代码额外风险

当前 Op 使用下面的 position validation cache key：

```text
(data_ptr, tensor._version, numel, num_tokens)
```

这是一个潜在正确性风险。CUDA caching allocator 可能把同一个地址重新分配给
新的 tensor；如果新 tensor 的 `_version` 和长度也相同，缓存可能错误认为
该 position map 已经验证过，从而跳过上界检查。

建议单独修复，不能与 `tile_k` 性能提交混合。可选方案：

1. 删除缓存，每次 forward 都执行上界验证，正确性最直接，但会产生同步成本。
2. 使用 `id(tensor)` 加 `weakref.ref(tensor)`，lookup 时要求 weakref 仍指向
   同一个 tensor 对象，并同时检查 `_version`。
3. 使用有界 identity-safe cache，过期或对象释放时自动移除条目。

推荐第二或第三种方案。

## 13. 分阶段优化计划

### 13.1 固定来源基线与 C500 `tile_k=64`（已完成）

提交：`253fe14e7c33e5c9851e6120d46caf78cbe3d13b`。

已完成：

1. 官方来源固定为 TileKernels-Metax `dev@0266ab7` 的
   `tile_kernels/quant/per_channel_cast_fused_kernel.py`。
2. 固定 TileLang 基线与生产 Kernel 物理分离，防止后续优化移动基线。
3. eager PyTorch reference 固定上游路径、提交和 SHA256。
4. 生产 Kernel 四条路径统一 `_TILE_K = 64`。
5. 保持 `threads=256`、`threads_per_token=64`、dummy tensor、尾块保护、
   空输入和正 OOB 失败语义。
6. 增加 shared-memory 精确预算：FP32 shared tile128 为 67,584 B，明确
   超过 C500 65,536 B；当时 FP32 shared tile64 为 33,792 B。
7. 不扩大未经验证的 CUDA 支持声明。

### 13.2 Plain/Expand thread-local staging（已完成）

提交：`6ec6bde`。

1. 编译期增加 `register_staging`，默认值为 `not with_rescale`。
2. Plain/Expand 每线程保留 32 个原始输入值，删除输入 tile 的 shared 写/读。
3. Rescale/Rescale-Expand 保留 shared，避免输入、position 和 FP32 scale
   同时跨归约存活造成的资源压力。
4. Plain/Expand 显式 shared 降到 1,024 B；Rescale 维持 9,216 B。
5. 不改 Op、Manifest、HBM 语义字节和 Roofline 公式。

### 13.3 正确性与独立性能门禁（已完成）

提交：`3a0f2c0`。

已完成：

1. 本地契约 33 项测试通过。
2. augenstern 完整 70 项兼容矩阵通过。
3. 固定基线 7 项测试通过。
4. BF16、FP32、FP8 和四个变体全部覆盖。
5. 16/144/160 token 尾块只由安全的 TileOps Kernel 验证；固定官方式性能基线
   只允许完整 128-token block，避免掩盖其适用边界。
6. FP8 输出逐元素完全一致，scale 使用 `atol=1e-7, rtol=1e-6`。
7. benchmark 增加同 Kernel shared-staging 对照，计时前与独立 reference 比较。
8. 三次完整运行证明 Plain/Expand 五组无回退，收益 3.38%～76.78%。
9. 将 augenstern 32 项性能矩阵按当前四个 `Quant*` Op、`x_sf_invs` 和
   Op-local Roofline 完成适配，与 9 项四方 A/B 用 `__suite` 隔离。
10. C500 完整性能矩阵 `32 passed in 117.11s`；32 项 production 均快于
    eager PyTorch，原始报告和 SHA256 已归档。

### 13.4 mcProfiler 与 Roofline 验收（已完成）

提交 `ec3f87d` 固定了可复现 profiler 入口，并将目标 Kernel 与进程级计数器
分开采集。最终有效报告、counter 门禁和重新计算的 Roofline 结果已写入本文
第 9、10 节；硬件 counter 均来自精确 multi-batch 报告，没有从 Benchmark
延迟反推。

### 13.5 position validation cache 正确性修复

该阶段应为独立提交：

1. 将 raw `data_ptr` cache 改成 tensor identity-safe cache。
2. 同一个 tensor 未修改时允许复用验证结果。
3. tensor in-place 修改导致 `_version` 变化时必须重新验证。
4. 两个不同 tensor 即使 shape 相同也必须分别验证。
5. cache 条目必须有界并支持对象释放。
6. 增加测试验证 cache hit、version invalidation 和不同对象不共享结果。

### 13.6 小 Rescale FP8 vec4 实验

这是下一项性能候选，不与文档或正确性修复混合：

1. 保持当前 Op、静态 shape、tile64、shared staging 和正 OOB 语义。
2. 只把 Rescale 的 `threads_per_token` 从 64 改为 16，使 `vec_k=4`。
3. 同时正确处理 64-lane wave 内四个 16-lane subgroup 的 position/scale shuffle。
4. 运行 103 项正确性测试、7 项基线测试和全部
   Rescale/Rescale-Expand workload。
5. 对 private spill、shared conflict、wave cycles 和 occupancy 做 profiler A/B。
6. 中大型 workload 回退超过 3%，或任一路径出现 private spill，即拒绝。

### 13.7 动态 token 与完整块快速路径（降为 P2）

register staging 已使中型 Plain 追平固定 TileLang、大型 FP32 明显超过固定
TileLang，旧的“Plain 全面落后”前提不再成立。动态 token 仍可能减少 JIT
specialization，完整块快速路径仍可能改善最小 workload，但两者应分别衡量
冷启动、cache 数量和稳定态性能，且不得采用 Op-side CUDA `F.pad` 或删除尾块保护。

### 13.8 明确不做 `torch.compile`

本轮目标是 Manifest #29 对应的 TileLang 算子迁移、正确性和 C500 性能。
不增加 custom-op/fake/meta 注册，不声明 `torch.compile` 支持，也不为此改写
当前四个公开 Op。未来若需要，应以独立需求、独立契约测试和独立提交开展。

## 14. 建议提交拆分

为保证 Review 和性能归因清楚，建议拆成多个提交或 PR：

1. `optimize(quant): pin C500 tile64 baselines`（已完成，`253fe14`）
2. `docs(quant): record fixed baselines and tile64 evidence`（已完成，`e7d981b`）
3. `optimize(quant): stage plain cast inputs in registers`（已完成，`6ec6bde`）
4. `test(quant): fix staging validation and profiler entrypoints`（已完成，`3a0f2c0`）
5. `perf(quant): isolate exact profiler metrics`（已完成，`ec3f87d`）
6. `test(quant): pin rescale shared memory budget`（已完成，`c9655a2`）
7. `docs(quant): record register-staging evidence`（已完成，`dbf08f7`）
8. `test(quant): port augenstern correctness matrix`（本次，包含 70 项矩阵与统一脚本入口）
9. `docs(quant): record full correctness matrix and optimization delta`（本次）
10. `bench(quant): port augenstern performance matrix`（已完成，`d85cb35`）
11. `docs(quant): record performance matrix results`（本次）
12. `fix(quant): make position validation cache identity-safe`（后续独立提交）
13. `optimize(quant): vectorize small rescale staging`（仅在独立 A/B 通过后）

不要把 vec4、动态 shape、完整块快速路径、validation cache 或
`torch.compile` 调用边界放入同一个提交。

## 15. 最终决策表

| 项目 | 决策 | 优先级 |
|---|---|---:|
| 当前接口、目录和 Op 分层 | 保留 | P0 |
| 官方 TileLang/PyTorch 固定基线 | 已合入 | P0 |
| C500 四路径 `tile_k=64` | 已合入 | P0 |
| Plain/Expand thread-local staging | 已合入 | P0 |
| Rescale/Rescale-Expand shared staging | 保留 | P0 |
| 未经验证的 CUDA 支持 | 不扩大 | P0 |
| 正 OOB 检查和空输入 | 保留 | P0 |
| position cache identity-safe | 独立修复 | P0/P1 |
| 小 Rescale FP8 vec4 | 当前 Kernel 上独立 A/B | P1 |
| 非 Expand 完整块快速路径 | 独立 A/B | P2 |
| 动态 token Kernel | 独立评估 JIT/cache | P2 |
| `torch.compile` custom-op/fake/meta | 本轮不采用 | 拒绝 |
| 当前 64-thread 映射单独加 scale shuffle | 不采用 | 拒绝 |
| Op-side CUDA `F.pad` | 不采用 | 拒绝 |
| FP32 tile128 `shared_rows=120` | 不采用 | 拒绝 |
| 整体替换为任一外部分支 | 不采用 | 拒绝 |

## 16. 下一步建议

当前 register staging 的实现、103 项正确性、三轮 9 项四方 Benchmark 和
32 项来源性能矩阵均已完成。下一步按以下顺序推进：

1. 独立修复 position validation cache 的地址复用风险。
2. 在当前安全 Rescale Kernel 上单变量实验 FP8 vec4。
3. 只有在有明确冷启动或 cache 需求时，再评估动态 token Kernel。
4. 不开展 `torch.compile` 集成，也不重新引入已经回退的独立 scale shuffle。

每个后续点都必须保持 103 项正确性测试、7 项固定基线测试、32 项来源性能
矩阵，以及 production、同 Kernel shared 对照、固定 TileLang、eager PyTorch
的 9 项 Benchmark 可复现，并使用独立提交保证性能归因。
