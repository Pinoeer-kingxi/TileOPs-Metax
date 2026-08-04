# quant_per_channel_cast_fused 深度优化分析与开发计划

本文记录 `quant_per_channel_cast_fused` 当前 TileOPs 实现与两个外部实现的
深度对比、MetaX C500 A/B Benchmark、正确性验证、mcProfiler、Roofline、
失败实验以及后续分阶段开发计划。

本文先在独立 worktree 中形成分析结论，再把已经通过 A/B 的第一阶段最小
改动合入主分支：固定官方 TileLang/PyTorch 基线，并在 C500 四条路径统一
使用 `tile_k=64`。公开 Op 接口和计算语义没有改变。

## 1. 分析范围与版本

| 项目 | 内容 |
|---|---|
| 主仓库 | `/data/TileOPs-Metax` |
| 主分支 | `feat/quant-per-channel-cast-fused` |
| 分析基线提交 | `2b833dc` |
| 原迁移实现提交 | `967b65b` |
| 当前 `tile_k=64`/固定基线提交 | `253fe14e7c33e5c9851e6120d46caf78cbe3d13b` |
| 官方上游提交 | `0266ab740980de7dc03a828b8259cd73d100c2eb` |
| 官方 TileLang 基线 | `tile_kernels/quant/per_channel_cast_fused_kernel.py` |
| 官方 TileLang SHA256 | `64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc` |
| 官方 PyTorch reference | `tile_kernels/torch/per_channel_cast_fused.py` |
| 官方 PyTorch SHA256 | `6af7609cf7462619dd902845bc17aad5402b5fe4f3c3c8eb4a6670a4e62a481f` |
| 仓内固定 TileLang baseline blob | `4a1c4722fd6dcd3f9fbc295ed3cb2fcbdb76d652` |
| 仓内 eager PyTorch baseline blob | `adc604bcd8c10c06f01bc6f84f9bb79f582aa286` |
| ACoolFIsh 审查版本 | `34907bfc6d109ef6783e81bf3469d594fe65ecce` |
| augenstern 实现版本 | `e0d73b398a5d4793cd059088d05d6c31a74c3082` |
| augenstern 文档版本 | `05d65b4acfeb468128a6fb56365cd7292513f4fb` |
| 实测设备 | MetaX C500，25% sGPU，16000 MiB Vram Quota |
| 实测日期 | 2026-08-04 |

外部实现来源：

- ACoolFIsh：<https://gitlink.org.cn/ACoolFIsh/TileOPs-Metax/tree/feat%2Fquant-per-channel-cast-fused>
- augenstern：<https://gitlink.org.cn/augenstern/TileOPs-Metax.git>

隔离实验 worktree：

```text
/data/TileOPs-Metax-ab-small64
/data/TileOPs-Metax-ab-all64
/data/TileOPs-Metax-ab-all64-broadcast
```

以上实验 worktree 均从分析基线创建。第一阶段通过后，生产 Kernel、固定
基线、测试和 Benchmark 已作为提交 `253fe14` 合入当前分支；Op 未修改。

## 2. 最终结论

### 2.1 总体结论

不应整体替换当前实现。当前实现仍然是最合适的工程基础，原因包括：

1. 与 Manifest PR #29 的四个公开接口一致。
2. Op、Kernel、reference、测试和 Benchmark 分层清楚。
3. 正数越界索引、负数 padding、空输入和非整块尾部语义完整。
4. 29 项正确性、边界和异常测试比两个外部实现更全面。
5. FP8 输出使用逐元素完全一致的正确性门禁。
6. 另有 7 项固定基线测试，以及完整 Benchmark、mcProfiler 和 Roofline
   证据。

当前已落地的最优工程组合是：

```text
当前接口、校验、测试和文档
    +
MetaX C500 全路径 tile_k=64
    +
固定官方 TileLang 基线和 eager PyTorch 基线
```

动态 token Kernel 仍是后续独立实验，不与本轮提交混合。

### 2.2 各实现最值得保留的部分

| 实现 | 最有价值的部分 | 不应直接吸收的部分 |
|---|---|---|
| 当前实现 | 接口、失败语义、空输入、Kernel 内尾块保护、全路径 `tile_k=64`、测试和文档 | token shape 静态 specialization；完整块路径仍有边界分支成本 |
| ACoolFIsh | 小 Rescale `tile_k=64`、动态 token Kernel；另有编译器调用边界功能，但本项目明确不采用 | Host `F.pad`、正 OOB 静默处理、FP32 `shared_rows=120`、整体 Op 重写 |
| augenstern | C500 四路径统一 `tile_k=64`、MACA/CUDA lane 区分、配置化 | 接口名和目录不兼容、缺少 OOB 检查、拒绝空输入、测试不足 |

### 2.3 哪个实现更好

- 工程质量和可维护性：当前实现更好。
- C500 稳定态 Kernel 调度：augenstern 的全路径 `tile_k=64` 已选择性吸收。
- 动态 shape：ACoolFIsh 和官方 TileKernels 的动态 token 设计更完整，后续
  可独立验证。
- 编译器调用边界功能超出本项目范围，明确不采用。
- 最终方案：保持当前工程基础，只吸收有独立 C500 实测证据的优化。

## 3. 当前语义和代码边界

相关代码：

- Kernel：[`tileops/kernels/quant/per_channel_cast_fused.py`](../../../tileops/kernels/quant/per_channel_cast_fused.py)
- Op：[`tileops/ops/quant/per_channel_cast_fused.py`](../../../tileops/ops/quant/per_channel_cast_fused.py)
- 独立 reference：[`tileops/testing/per_channel_cast_fused.py`](../../../tileops/testing/per_channel_cast_fused.py)
- 测试：[`tests/ops/test_per_channel_cast_fused.py`](../../../tests/ops/test_per_channel_cast_fused.py)
- 固定基线测试：[`tests/ops/test_per_channel_cast_fused_baselines.py`](../../../tests/ops/test_per_channel_cast_fused_baselines.py)
- 固定 TileLang 基线：[`benchmarks/ops/per_channel_cast_fused_baselines.py`](../../../benchmarks/ops/per_channel_cast_fused_baselines.py)
- Benchmark：[`benchmarks/ops/bench_per_channel_cast_fused.py`](../../../benchmarks/ops/bench_per_channel_cast_fused.py)
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

每个 workgroup 处理一个 `128 × tile_k` 的 token-hidden tile。

1. 如为 Expand 变体，加载 `pos_to_token`，负数 position 表示 padding。
2. 如为 Rescale 变体，加载输入反量化 scale `x_sf_invs`。
3. 从 HBM 读取输入 `x`，写入 shared memory。
4. 对每个 hidden channel 计算 128-token block 内的绝对值最大值。
5. 将最大值截断到至少 `1e-4`，计算：

   ```text
   out_sf     = amax / 448
   out_sf_inv = 448 / amax
   ```

6. 如果 `round_sf=true`，将 scale 向上取整为 2 的幂。
7. 从 shared memory 重新读取输入，乘以 `out_sf_inv` 并转换为 FP8 E4M3。
8. 写出 FP8 `out` 和 FP32 `out_sf`。

Rescale 路径的逻辑输入是：

```text
logical_x[t, h] = x[t, h] * x_sf_invs[t, h // 128]
```

### 4.1 语义访存次数

对于每个有效输出元素：

- `x`：HBM 读取 1 次。
- `x_shared`：shared 写 1 次、shared 读 1 次。
- `out`：HBM 写 1 次。
- `out_sf`：每 128-token block、每 hidden channel 写 1 次。
- Rescale：逻辑上每 token、每 128 channel 读取一个 `x_sf_invs`。
- Expand：逻辑上每个输出 token 读取一个 `pos_to_token`。

实现中 position map 和输入 scale 会被不同 hidden tile 重复发起 load 指令，
但大部分重复读取由 VL1/L2 吸收。Manifest Roofline 使用语义最小字节数，
不使用实现的 `tile_k` 计算字节数。

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

benchmark-only 副本保持官方算法、动态 token 维度和完整块执行结构；唯一
C500 调度适配是把官方 plain/rescale 的 128/256 tile 统一改为 64。该文件
与生产 Kernel 独立，后续优化不会静默移动基线。

### 5.1 shared memory 占用

这里只统计 Kernel 中显式分配的 `x_shared` 和 `amax_shared`。

| 路径 | 原 tile | 原 shared/CTA | 当前 tile64 shared/CTA |
|---|---:|---:|---:|
| BF16 plain/expand | 128 | 34,816 B | 17,408 B |
| FP32 plain/expand | 128（官方）/64（原迁移） | 67,584 B / 33,792 B | 33,792 B |
| FP8 rescale/rescale-expand | 256 | 36,864 B | 9,216 B |

FP32 `tile_k=128` 的 67,584 B 明确超过 C500 的 65,536 B 上限；BF16
`tile_k=128` 并不会溢出，改为 64 的原因主要是并行度和性能，而不是笼统的
“所有 128 tile 都溢出”。

## 6. Benchmark 方法

正式筛选协议与仓库 `bench_kernel` 默认协议一致：

- 真实 MetaX C500，25% sGPU。
- 10 次预热。
- 每个 trial 50 次测量。
- 3 个 trial，报告 trial mean 的中位数。
- CUPTI Kernel 时间，不包含编译时间和 Host launch 时间。
- 每次测量前清空 L2。
- 输入使用 3 份 clone 轮换。
- 当前实现和候选实现使用相同随机种子和输入生成方式。
- 所有 workload 独立运行两轮。

两轮结果的最大差异小于 0.9%，说明结果具有较好的重复性。

## 7. Benchmark 实测结果

下面所有 `tile64` 数据均指不带 scale shuffle 广播的纯 `tile_k=64`
候选版本；“原迁移”指提交 `967b65b` 的 128/64/256 tile 配置。

### 7.1 最小合法 shape

| workload | 原迁移 ms | tile64 ms | 加速 | 延迟下降 |
|---|---:|---:|---:|---:|
| plain 128×128 BF16 | 0.045619 | 0.032461 | 1.405× | 28.84% |
| plain 128×256 BF16 | 0.046341 | 0.032358 | 1.432× | 30.17% |
| plain 128×512 BF16 | 0.048028 | 0.032410 | 1.482× | 32.52% |
| expand 64×128 → 16 BF16 | 0.031703 | 0.018542 | 1.710× | 41.51% |
| expand 64×128 → 144 BF16 | 0.046692 | 0.033119 | 1.410× | 29.07% |
| rescale 128×256 FP8 | 0.150441 | 0.055821 | 2.695× | 62.90% |
| rescale-expand 64×256 → 16 FP8 | 0.110300 | 0.029852 | 3.695× | 72.94% |
| rescale-expand 64×256 → 144 FP8 | 0.143775 | 0.052580 | 2.734× | 63.43% |

最小 shape 没有因 workgroup 数增加而回退，因此当前实测不支持为小 hidden
保留大 tile。

### 7.2 小型和中型 workload

| workload | 原迁移 ms | tile64 ms | 加速 | 延迟下降 |
|---|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.047903 | 0.032645 | 1.467× | 31.85% |
| plain 1024×4096 BF16 | 0.139753 | 0.070574 | 1.980× | 49.50% |
| plain 1024×4096 FP32 | 0.151181 | 0.151020 | 1.001× | 0.11% |
| expand 512×4096 → 1024 BF16 | 0.143206 | 0.075802 | 1.889× | 47.07% |
| expand 128×4096 → 144 BF16 | 0.050222 | 0.040361 | 1.244× | 19.64% |
| rescale 128×1024 FP8 | 0.154662 | 0.054925 | 2.816× | 64.49% |
| rescale 1024×4096 FP8 | 0.308659 | 0.085647 | 3.604× | 72.25% |
| rescale-expand 512×4096 → 1024 FP8 | 0.305165 | 0.126072 | 2.421× | 58.69% |

FP32 路径原迁移已经使用 `tile_k=64`，因此结果持平。

### 7.3 真实模型大 shape

| workload | 原迁移 ms | tile64 ms | 加速 | 延迟下降 |
|---|---:|---:|---:|---:|
| plain 4096×7168 BF16 | 0.834360 | 0.414592 | 2.012× | 50.31% |
| plain 8192×4096 BF16 | 0.934730 | 0.469484 | 1.991× | 49.77% |
| expand 4001×7168 → 8192 BF16 | 1.571725 | 0.852508 | 1.844× | 45.76% |
| rescale 4096×3072 FP8 | 0.611356 | 0.224118 | 2.728× | 63.34% |
| rescale-expand 4001×3072 → 8192 FP8 | 1.198172 | 0.435553 | 2.751× | 63.65% |
| rescale 128×8192 FP8 | 0.158136 | 0.059031 | 2.679× | 62.67% |

### 7.4 Benchmark 总结

- 共覆盖 22 组最小、中型、尾块和真实模型 workload。
- 21 组明显加速，收益范围为 19.64%～72.94%。
- 1 组 FP32 路径基本持平，因为原迁移配置本来就是 64。
- 未发现性能回退。
- ACoolFIsh 只在 `num_tokens_out <= 128` 时使用 64，会错失中型和大型
  Rescale 的 2.7×～3.6× 收益。

### 7.5 固定官方 TileLang 与 eager PyTorch 三方基线

提交 `253fe14` 增加了独立于生产 Kernel 的固定基线，并用统一的
10 warmup、50 repeat、3 trials 协议重新测量 9 组 Manifest workload。

| workload | TileOps ms | 固定 TileLang ms | eager PyTorch ms | TileOps/TileLang | TileOps/PyTorch |
|---|---:|---:|---:|---:|---:|
| plain 128×1024 BF16 | 0.0325 | 0.0243 | 0.0769 | 0.748× | 2.366× |
| plain 1024×4096 BF16 | 0.0705 | 0.0517 | 0.2619 | 0.733× | 3.715× |
| plain 4096×8192 FP32 | 1.2296 | 0.8020 | 1.5036 | 0.652× | 1.223× |
| expand 512×4096→1024 BF16 | 0.0756 | 0.0778 | 0.3926 | 1.029× | 5.193× |
| expand 1024×4096→2048 FP32 | 0.3310 | 0.3359 | 0.6582 | 1.015× | 1.989× |
| rescale 128×1024 FP8 | 0.0549 | 0.0525 | 0.0800 | 0.956× | 1.457× |
| rescale 1024×4096 FP8 | 0.0853 | 0.0844 | 0.3731 | 0.989× | 4.374× |
| rescale-expand 512×4096→1024 | 0.1259 | 0.1281 | 0.4482 | 1.017× | 3.560× |
| rescale-expand 1024×4096→2048 | 0.1949 | 0.1974 | 0.7917 | 1.013× | 4.062× |

结论：TileOps 9 组全部快于 eager PyTorch。Expand/Rescale-Expand 与固定
TileLang 基线持平或略快，Rescale 基本持平；Plain 吞吐仍低
25.2%～34.8%。固定基线采用官方动态 token 结构，并只允许完整 128-token
block，不含公共 Kernel 的尾块边界分支，因此下一步必须把这两个差异拆开
A/B，不能以性能差距为由删除安全检查。

原始报告：`/data/profile_run.log`，SHA256：

```text
08e2ed60869bfa2c2919e872ecf8edbddc5c80b3df78cbd78b1b8351390ec1f4
```

## 8. 正确性验证

### 8.1 完整测试套件

隔离 worktree 中的全路径 `tile_k=64` 结果：

```text
28 passed in 25.71s
```

提交 `253fe14` 增加 shared-memory 门禁后，生产测试结果为：

```text
29 passed in 26.07s
```

固定 TileLang/PyTorch 基线自身的 5 种计算路径和 2 项契约测试结果：

```text
7 passed in 41.66s
```

测试覆盖：

- BF16 和 FP32 plain 输入。
- FP8 QuantTensor/Rescale 输入。
- 四个算子变体。
- `round_sf=false/true`。
- 16、128、144、256 token 输出。
- 重复、逆序、混合和全 padding position map。
- 空输入和空 source。
- 全零、正负极值。
- 非连续输入。
- 非法 dtype、rank、hidden、token 数、scale shape。
- 正的越界 position。
- 输出 shape、dtype 和 contiguous。
- FP8 逐元素完全一致。

### 8.2 真实模型 shape 正确性

另外对两个大 shape 与独立 PyTorch reference 比较：

```text
plain 4096×7168 BF16:
  FP8 输出逐元素完全一致
  scale: atol=1e-7, rtol=1e-6

rescale-expand 4001×3072 → 8192 FP8:
  FP8 输出逐元素完全一致
  scale: atol=1e-7, rtol=1e-6
```

## 9. mcProfiler 分析

### 9.1 采样方法和无效数据排除

mcProfiler 的部分硬件计数器是进程级计数。如果 profiling 驱动使用
`torch.randn()`、`torch.rand()` 或 GPU fill 生成输入，输入初始化 Kernel
也会进入 Workgroup 和 HBM 计数。

一次无效报告中多出的约 8.39 MB HBM write，恰好等于
`1024 × 4096 × sizeof(BF16)`，证明污染来自输入随机初始化，而不是目标
Kernel。

最终有效采样采用：

1. `torch.empty()` 分配输入，避免 GPU 初始化 Kernel。
2. 目标进程只执行目标 Kernel。
3. 检查 Workgroups 是否等于理论网格，作为报告有效性门禁。
4. 对 Compute/MTE/STE、cache、shared efficiency 等关键指标关闭
   `--single-pass`，使用精确的多 event-batch 采样。
5. 不使用 single-pass 中明显超过 100% 的推断比例。

### 9.2 Plain 1024×4096 BF16

| 指标 | 原迁移 tile128 | tile64 | 变化 |
|---|---:|---:|---:|
| Workgroups | 256 | 512 | 2× |
| Waves | 1024 | 2048 | 2× |
| 显式 shared/CTA | 约 34 KiB | 约 17 KiB | -50% |
| Average wave cycles | 13174 | 7618 | -42.17% |
| Compute busy | 24.70% | 46.04% | +21.34 pp |
| MTE duty | 24.70% | 45.94% | +21.24 pp |
| STE duty | 14.34% | 11.03% | -3.31 pp |
| VL1 hit | 96.80% | 98.42% | +1.62 pp |
| L2 hit | 12.41% | 37.56% | +25.15 pp |
| HBM read | 8,404,992 B | 8,435,776 B | +0.37% |
| HBM write | 4,317,184 B | 4,329,568 B | +0.29% |
| HBM total | 12,722,176 B | 12,765,344 B | +0.34% |
| Shared efficiency | 100% | 67.78% | -32.22 pp |

### 9.3 Rescale 1024×4096 FP8

| 指标 | 原迁移 tile256 | tile64 | 变化 |
|---|---:|---:|---:|
| Workgroups | 128 | 512 | 4× |
| Waves | 512 | 2048 | 4× |
| 显式 shared/CTA | 约 36 KiB | 约 9 KiB | -75% |
| Average wave cycles | 37912 | 18551 | -51.07% |
| Compute busy | 17.70% | 50.70% | +33.00 pp |
| MTE duty | 17.66% | 50.62% | +32.96 pp |
| STE duty | 18.42% | 45.43% | +27.01 pp |
| VL1 hit | 96.13% | 98.43% | +2.30 pp |
| L2 hit | 30.90% | 73.06% | +42.16 pp |
| HBM read | 4,384,992 B | 4,427,872 B | +0.98% |
| HBM write | 4,325,792 B | 4,325,728 B | 基本不变 |
| HBM total | 8,710,784 B | 8,753,600 B | +0.49% |
| Shared efficiency | 100% | 41.19% | -58.81 pp |
| 指令数 | 8,371,120 | 9,750,278 | +16.48% |

### 9.4 性能机制结论

`tile_k=64` 没有减少语义访存，实际 HBM 流量也只增加约 0.34%～0.49%。
主要收益来自：

1. hidden 方向 workgroup 数增加。
2. 每 CTA shared memory 明显下降。
3. 平均 wave 生命周期缩短。
4. Compute、MTE 和 STE 利用率提高。
5. 更多重复 load 被更高的 VL1/L2 命中率吸收。

shared-memory efficiency 下降是一个真实现象。`tile_k=64` 时，每 lane
分别处理一个 BF16 或 FP8 元素，子字宽访问更容易产生 shared bank conflict。
但当前并行度收益显著大于 bank conflict 损失，因此不能以 shared efficiency
下降为理由退回大 tile。

### 9.5 有效报告和 SHA256

```text
33acb500f76d6d42051d99f52a22cb8f37f9b5380e2a25a26ed7ee337e59731f
  /opt/mcProfiler-ubuntu18.04/output20260804080605/1_main_kernel.txt

ba2c9873af6a2be0786e9adf0c312d1c7fd031637ace3c5ba9ecc6013d1513d3
  /opt/mcProfiler-ubuntu18.04/output20260804132836/report.txt

598bf721f99b7ce5fc26038c230d4b501d8dd9f41b6af1f48100fa425bfa97c0
  /opt/mcProfiler-ubuntu18.04/output20260804133542/report.txt

45280caf7e0dea476083267ef336f73ebd9985076df7f5b8b87e67994f42566e
  /opt/mcProfiler-ubuntu18.04/output20260804133241/report.txt
```

## 10. Roofline 分析

`tile_k` 是实现参数，不改变 Manifest 的语义 FLOPs 和最小字节数，因此
优化前后算术强度相同。

按 25% sGPU 的 HBM 峰值带宽 `460.8 GB/s` 计算：

| workload | AI | 当前效率 | tile64 效率 | tile64 语义带宽 |
|---|---:|---:|---:|---:|
| plain 1024×4096 BF16 | 0.995 FLOP/B | 19.74% | 39.10% | 180.15 GB/s |
| rescale 1024×4096 FP8 | 2.432 FLOP/B | 6.08% | 21.92% | 101.00 GB/s |
| plain 4096×7168 BF16 | 0.995 FLOP/B | 23.15% | 46.59% | 214.66 GB/s |
| rescale 4096×3072 FP8 | 2.432 FLOP/B | 9.21% | 25.13% | 115.80 GB/s |

所有 workload 的 AI 都明显低于 C500 ridge point，属于访存或延迟侧。
`tile_k=64` 提高的是有效并行度和实际数据搬运效率，而不是改变算法 AI。

小 workload 的 Roofline 效率仍然较低。例如 rescale 128×1024 从约
0.38% 提升到约 1.07%，但仍主要受启动延迟、固定调度和设备覆盖不足限制。

## 11. 已验证但不应吸收的优化

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

### 11.2 Expand Host `F.pad`

ACoolFIsh 在 Host 将 position map pad 到 128-token 对齐，再对输出切片。
不建议吸收，原因包括：

1. 增加一个 GPU pad Kernel。
2. 增加临时 tensor 分配。
3. 输出 16 或 144 token 时仍执行完整 128-token 尾块计算。
4. Kernel 被绕过 Op 直接调用时依赖 Host padding 才能保证安全。

当前 Kernel 内原生边界保护更清楚，也符合迁移指南关于设备计算的要求。

### 11.3 FP32 `shared_rows=120`

ACoolFIsh 的 FP32 方案使用 `tile_k=128`，只缓存 120 行，第二遍重新读取
最后 8 行。

该方案存在：

- 约 62 KiB shared/CTA，占 C500 64 KiB 上限的大部分。
- 额外约 6.25% 输入 HBM load。
- occupancy 风险。
- 没有独立 C500 A/B 证据。

当前 FP32 `tile_k=64` 已经安全，并且候选全路径 64 不改变 FP32 性能。

### 11.4 删除 position 上界检查

两个外部分支都没有完整验证：

```text
0 <= pos_to_token < num_tokens
```

augenstern Kernel 还对未经上界检查的 token 使用 `T.assume(token < num_tokens)`，
存在越界读取风险。当前 Manifest 明确要求非负 position 小于 source token 数，
因此当前失败语义必须保留。

### 11.5 整体替换 Op 或接口

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

### 13.1 阶段一：固定来源基线与 C500 `tile_k=64`（已完成）

提交：`253fe14e7c33e5c9851e6120d46caf78cbe3d13b`。

已完成：

1. 官方来源固定为 TileKernels-Metax `dev@0266ab7` 的
   `tile_kernels/quant/per_channel_cast_fused_kernel.py`。
2. 固定 TileLang 基线与生产 Kernel 物理分离，防止后续优化移动基线。
3. eager PyTorch reference 固定上游路径、提交和 SHA256。
4. 生产 Kernel 四条路径统一 `_TILE_K = 64`。
5. 保持 `threads=256`、`threads_per_token=64`、dummy tensor、尾块保护、
   空输入和正 OOB 失败语义。
6. 增加 shared-memory 精确预算：FP32 tile128 为 67,584 B，明确超过
   C500 65,536 B；当前 FP32 tile64 为 33,792 B。
7. 不扩大未经验证的 CUDA 支持声明。

### 13.2 阶段二：正确性和基线门禁（已完成）

已完成：

1. 主算子 29 项测试通过。
2. 固定基线 7 项测试通过。
3. BF16、FP32、FP8 和四个变体全部覆盖。
4. 16/144 token 尾块只由安全的 TileOps Kernel 验证；固定官方式性能基线
   只允许完整 128-token block，避免掩盖其适用边界。
5. FP8 输出逐元素完全一致，scale 使用 `atol=1e-7, rtol=1e-6`。
6. Manifest、Benchmark 基础测试、Ops Manifest、Ruff 和 diff check 通过。

### 13.3 阶段三：正式性能验收（已完成）

已完成两类证据：

1. 原迁移配置对全路径 tile64：22 组 workload、两轮独立运行；21 组提升
   19.64%～72.94%，FP32 持平，无回退。
2. 当前 TileOps 对固定官方式 TileLang 与 eager PyTorch：9 组统一协议；
   9 组全部快于 eager PyTorch，Expand/Rescale 基本与固定 TileLang 持平，
   Plain 尚有明确差距。

统一协议为 10 warmup、50 repeat、3 trials、L2 flush、地址轮换、CUPTI
kernel-only 时间；JIT 时间不计入稳定态延迟。

### 13.4 阶段四：mcProfiler 和 Roofline 验收（已完成）

已完成 plain 1024×4096 BF16 的 tile128/tile64，以及 rescale
1024×4096 FP8 的 tile256/tile64 精确采样。Workgroups、wave cycles、
Compute/MTE/STE、cache、HBM bytes 和 shared efficiency 均有可复核报告。

Roofline 使用 Manifest 语义字节和 25% sGPU 峰值 `460.8 GB/s`。当前
tile64 的代表性效率为：plain 39.10%，rescale 21.92%；算法仍属于
memory/latency-bound。

### 13.5 阶段五：position validation cache 正确性修复

该阶段应为独立提交：

1. 将 raw `data_ptr` cache 改成 tensor identity-safe cache。
2. 同一个 tensor 未修改时允许复用验证结果。
3. tensor in-place 修改导致 `_version` 变化时必须重新验证。
4. 两个不同 tensor 即使 shape 相同也必须分别验证。
5. cache 条目必须有界并支持对象释放。
6. 增加测试验证 cache hit、version invalidation 和不同对象不共享结果。

### 13.6 阶段六：Plain 差距拆分实验

固定 TileLang 基线表明 Plain 吞吐仍低 25.2%～34.8%。不能一次同时修改
多个因素，必须拆成两个独立实验。

实验 A：完整块快速路径。

1. 只对非 Expand 变体在编译期消除必然为真的尾块判断。
2. Expand 的 16/144 token 尾块保护完全保留。
3. 不改变正 OOB、padding、empty 和 dummy tensor 语义。
4. 对 BF16/FP32 Plain 和 FP8 Rescale 分别 A/B。
5. 任何非 Plain 路径回退超过 3% 即拒绝。

实验 B：动态 token Kernel。

1. 使用 `T.dynamic("num_tokens")` 和 `T.dynamic("num_tokens_out")`。
2. hidden、dtype、variant、rounding 和 `tile_k` 保持 specialization。
3. Kernel cache key 移除精确 token 数。
4. 同一个 Op 依次运行 128/256 token，应复用一个动态 Kernel。
5. Expand 运行 16/144/256 token，仍保留 Kernel 内边界保护。
6. 单独记录冷启动 JIT、cache 数量和稳定态性能；回退超过 3% 即拒绝。
7. 不采用 Host `F.pad`。

### 13.7 阶段七：shared-memory bank conflict 优化

Profiler 已证明 tile64 存在 shared efficiency 下降。后续可以独立探索：

1. 将 Rescale 后的 logical FP32 值暂存到 shared，避免第二遍重复 cast 和
   rescale multiply。
2. 使用 packed 32-bit shared layout，将多个 BF16/FP8 元素组合访问。
3. 调整 lane 到 token/channel 的映射。
4. 评估 swizzle 或其他 shared layout。

每个方案必须单独 A/B。不能因为理论上减少 bank conflict 就直接合入，
也不能重新加入已经实测回退的 scale shuffle 广播。

## 14. 建议提交拆分

为保证 Review 和性能归因清楚，建议拆成多个提交或 PR：

1. `optimize(quant): pin C500 tile64 baselines`（已完成，`253fe14`）
2. `docs(quant): record fixed baselines and tile64 evidence`
3. `fix(quant): make position validation cache identity-safe`
4. `optimize(quant): specialize full-block fast path`
5. `optimize(quant): reuse dynamic-token kernels`
6. `optimize(quant): reduce tile64 shared bank conflicts`，仅在独立 A/B 有收益时

不要把完整块快速路径、动态 shape、validation cache 和 shared layout 放入
同一个提交。

## 15. 最终决策表

| 项目 | 决策 | 优先级 |
|---|---|---:|
| 当前接口、目录和 Op 分层 | 保留 | P0 |
| 官方 TileLang/PyTorch 固定基线 | 已合入 | P0 |
| C500 四路径 `tile_k=64` | 已合入 | P0 |
| 未经验证的 CUDA 支持 | 不扩大 | P0 |
| 正 OOB 检查和空输入 | 保留 | P0 |
| position cache identity-safe | 独立修复 | P0/P1 |
| 非 Expand 完整块快速路径 | 独立 A/B | P1 |
| 动态 token Kernel | 独立 A/B | P1 |
| shared bank conflict 优化 | 实验后决定 | P2 |
| 编译器 custom-op/fake/meta 边界 | 不采用，超出范围 | 拒绝 |
| scale shuffle 广播 | 不采用 | 拒绝 |
| Host `F.pad` | 不采用 | 拒绝 |
| FP32 `shared_rows=120` | 默认不采用 | 拒绝 |
| 整体替换为任一外部分支 | 不采用 | 拒绝 |

## 16. 下一步建议

阶段一至阶段四已经完成。下一步按以下顺序推进：

1. 先提交并发布当前固定基线、`tile_k=64` 和文档证据。
2. 独立修复 position validation cache 的地址复用风险。
3. 只对非 Expand 完整块路径实验编译期分支消除，解释 Plain 基线差距。
4. 再独立实验动态 token Kernel，分别记录 JIT/cache 和稳定态数据。
5. 最后才评估 packed shared layout 或 swizzle；不重新引入已经回退的 scale
   shuffle 广播。

每个后续点都必须保持 29 项生产测试、7 项固定基线测试和三方 Benchmark
可复现，并使用独立提交保证性能归因。
