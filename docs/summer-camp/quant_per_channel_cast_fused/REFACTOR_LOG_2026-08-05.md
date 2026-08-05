# Quant per-channel cast fused：测试与基线重构记录（2026-08-05）

> 本文前半部分记录第一次合并时的历史状态（103项功能测试、9+32项性能入口）。
> 当日后续的提交前收敛结果以文末“提交前收敛”章节为准。

## 回退锚点

- 修改前组内基线：`main@394fc01`。
- 备份分支：`backup/quant-before-test-baseline-refactor-394fc01`。
- 开发分支：`refactor/quant-unified-tests-baselines`。

## 文件结构整理

- 将原 `tests/ops/test_per_channel_cast_fused_augenstern.py` 的70项矩阵合入 `tests/ops/test_per_channel_cast_fused.py`；主功能入口保持103项。
- 删除成员名后缀的功能测试入口；来源commit与blob常量继续保存在合并后的测试文件。
- 将原 `benchmarks/ops/bench_per_channel_cast_fused_augenstern.py` 的32项矩阵合入 `benchmarks/ops/bench_per_channel_cast_fused.py`。
- 单一benchmark文件现在收集41项：9项稳定消融与32项规模矩阵。
- PyTorch正确性oracle移入主测试文件；benchmark侧保留独立实现，避免测试oracle与性能基线耦合。
- `tests/ops/test_per_channel_cast_fused_baselines.py`继续独立存在，负责固定来源和验证比较基线，不属于production主功能入口。

## 三类外部性能基线

每个性能case在计时前先验证数值一致性，并记录：

1. `torch-eager`：benchmark本地普通PyTorch实现。
2. `torch-compile`：对同一PyTorch函数执行`torch.compile(..., fullgraph=True)`；首次编译不计入稳定态kernel时间。
3. `tilekernels-tilelang`：固定MetaX-MACA/TileKernels-Metax `dev@0266ab7`来源的TileLang实现。

Plain/Expand还保留`tileops-shared-staging`，它仅用于production消融，不属于外部基线。

## TileKernels基线回退原则

参考实现恢复官方调度与代码策略：

- Plain/Expand BF16：`tile_k=128`。
- Rescale/RescaleExpand FP8：`tile_k=256`。
- Plain/Expand FP32：仅因官方`tile_k=128`需要67,584 B显式shared memory、超过C500每workgroup 65,536 B，允许改为`tile_k=64`。
- 线程数256、`threads_per_token=64`、shared staging、两遍amax/量化、Expand的shuffle索引广播均保持官方结构。
- 接口包装、动态shape名称、输出分配与来源SHA固定属于接入适配，不计作性能优化。

这与production四条路径统一`tile_k=64`严格分离，后续可以直接衡量统一64分块的收益。

## 验证记录

- `torch.compile`最小Plain BF16：通过，与eager一致。
- `torch.compile`四变体探针：Plain/Expand/Rescale/RescaleExpand全部通过。
- 回退后的TileKernels baseline可信度：`7 passed in 37.90s`。
- 收集：主功能测试`103 tests`；统一benchmark `41 tests`（9+32）。
- 32项矩阵四变体代表case按独立进程运行，四项全部通过并归档报告。
- 资源限制：四个性能smoke在同一pytest进程累计编译时发生SIGKILL；曾有一次RescaleExpand在报告写出后的退出阶段收到SIGKILL。随后四个case分别使用独立pytest进程均干净通过。完整矩阵仍建议按case或小组分进程执行。

## 尚未完成

- 合并后的103项完整正确性回归已完成：`103 passed in 31.94s`。
- 运行9项稳定性能对比与按变体分进程的32项矩阵。
- 根据新基线结果生成加速比汇总；历史artifact使用的是旧的全路径`tile_k=64`适配baseline，不能冒充本轮原始调度结果。

## 早期四变体报告作废说明

第一次重构曾使用`Expand 17->32`和`RescaleExpand 17->32`生成四份代表性
报告。后续严格数值门禁发现，原始TileKernels的`TILE_M=128`调度没有尾块
保护，不适用于不足128行的Expand输出；因此这些报告已从工作区删除，不记录
其延迟或加速比。当前正式性能workload把Expand输出约束为128的倍数，并在
计时前同时对三类基线执行严格正确性校验。

## 最终门禁（当前工作区）

- production正确性：`103 passed in 31.94s`。
- TileKernels baseline可信度：`7 passed in 25.28s`（回退后再次运行）。
- Benchmark基础设施：`17 passed in 24.45s`。
- Ops Manifest测试：`7 passed in 23.08s`。
- strict Manifest：通过，保留10条既有synthetic shape precondition warning。
- test node delta：合并后的主文件收集103项；工具将该路径相对其默认base识别为新文件，因此报告`+103`。总节点数相对本轮修改前仍为33+70=103，没有增加production功能节点。
- `git diff --check`与Python/shell语法检查：通过。
- Ruff：当前C500环境未安装`ruff`模块或二进制，未宣称通过；统一脚本中的Ruff门禁仍保留，需在含开发依赖环境补跑。

## 提交前收敛（2026-08-05，当前工作区）

基于官方测试预算、Manifest workload职责和信任链重新整理：

- production功能测试从103项收敛到40项。删除仅改变token/hidden规模、但不触发新代码路径的70项交叉矩阵；保留四变体、BF16/FP32/FP8、rounding、tile边界、Expand索引模式、数值边界、空输入和完整异常契约。
- 在原33项契约测试上补充7项此前缺少的接口错误：`round_sf`类型，Expand输出对齐/rank/dtype，以及Rescale输入dtype/hidden对齐/scale设备。
- `tests/ops/test_per_channel_cast_fused_baselines.py`移动为`benchmarks/tests/test_per_channel_cast_fused_baseline.py`，消除`tests -> benchmarks`反向依赖；production UT现在只依赖production代码和本地PyTorch oracle。
- Manifest workload收敛为20项，Plain/Expand/Rescale/RescaleExpand各5项；覆盖小边界、中型、DeepSeek/Qwen类hidden、不同Expand比例、rounding和大token压力。
- 为保证原始TileKernels基线的比较结果可信，Expand类性能workload的输出token
  数使用128的倍数。该上游调度没有masked `TILE_M=128`尾块；production功能
  测试仍覆盖公共接口允许的16对齐与非完整128-token尾块。
- 删除个人来源的`__suite: augenstern-performance`元数据；20项全部成为中性、正式的Manifest性能workload。
- benchmark删除独立的32项矩阵入口，统一由一个20项Manifest-driven入口执行production、PyTorch eager、`torch.compile`和原始TileKernels TileLang四方对比。
- 四个公共Op构造函数以及内部共享基类改为keyword-only参数，符合当前Op接口规范。
- runner删除过时的`matrix`和`benchmark-matrix`子命令，`baselines`改为指向`benchmarks/tests/`。

当前C500验证结果：

- production功能测试：`40 passed in 236.56s`；
- 固定TileKernels baseline可信度：`7 passed in 26.31s`；
- Benchmark基础设施（不含本算子baseline测试）：`17 passed in 24.80s`；
- Ops Manifest测试：`7 passed in 23.20s`；
- strict Manifest：通过，保留10条synthetic shape precondition warning；
- 正式Benchmark：收集20项，四变体各5项；
- 四变体各抽跑1项，数值门禁、production、eager、`torch.compile`、TileKernels
  TileLang计时均完成并显示`1 passed`。C500在pytest写出报告后的退出清理阶段
  将进程SIGKILL（退出码137），因此不能表述为“进程干净退出”；完整20项建议
  使用runner按case隔离，并继续观察容器内存峰值；
- `py_compile`、shell语法和`git diff --check`通过；当前环境未安装Ruff与
  pre-commit，未宣称通过。

### 完整20项性能回归

20项随后全部按独立pytest进程完成。每项均通过四方严格数值门禁、显示
`1 passed`并写出报告；报告归档在仓库同级目录：

```text
/data/gxy/TileOPs-Metax-team-results/full-benchmark-2026-08-05/
```

`speedup = baseline_latency / production_latency`的总体结果：

| 对比对象 | workload数 | 几何平均加速 | production胜出 |
|---|---:|---:|---:|
| shared-staging消融（Plain/Expand） | 10 | 1.6506x | 10/10 |
| 原始TileKernels TileLang | 20 | 2.2470x | 20/20 |
| torch.compile | 20 | 0.6865x | 9/20 |
| PyTorch eager | 20 | 3.4424x | 20/20 |

因此当前production稳定优于eager和原始TileKernels；thread-local staging在
Plain/Expand的10项中全部优于shared-staging。`torch.compile`在小shape上
通常更快，production在中大型shape上逐渐反超，不能宣称对它全面占优。

运行限制仍存在：每个pytest在输出`1 passed`并写完报告后，解释器退出清理
阶段均被C500容器SIGKILL，shell状态137。该现象不影响已完成的数值门禁和
计时数据，但后续仍应定位退出阶段峰值内存或运行时清理问题。

### 迁移到完整C500后的复测

项目迁移到无sGPU切片的完整C500（65536 MiB显存）后，容器CPU内存上限也
从32 GiB提高到128 GiB。迁移工作区、Git状态和20份旧报告均核对完整。

- 基础GPU探针通过；
- production功能测试：`40 passed in 82.65s`；
- 正式性能测试：20/20均显示`1 passed`，逐项退出状态均为0；
- cgroup `oom_kill`始终为0；
- cgroup CPU内存峰值为68,421,550,080 bytes（约63.7 GiB）。

这直接确认旧实例的退出阶段SIGKILL来自32 GiB CPU内存上限，而非64 GB C500
显存不足或算子错误。完整卡结果归档于：

```text
/data/gxy/TileOPs-Metax-team-results/full-c500-2026-08-05/
```

完整卡的总体几何平均加速为：相对eager `3.3876x`（20/20胜出），相对原始
TileKernels `2.2433x`（20/20胜出），相对torch.compile `0.6731x`（9/20
胜出），Plain/Expand相对shared-staging `1.6457x`（10/10胜出）。

完整卡与旧切片的production延迟几何平均比值为`1.0012`，差异约0.12%，属于
正常波动；旧切片当时未表现出明显竞争干扰，但完整卡结果没有sGPU配额影响且
进程均干净退出，应作为后续正式基线。

### 组员shuffle版本完整A/B

组员仓库`main@9719985`在`6264154`中重新排列256个线程，并用4-lane
`shfl_xor`替代四条路径的`amax_shared`归约。其旧组织的103项功能测试在
完整C500得到`103 passed in 270.56s`。

随后以shuffle前`394fc01`（与本地当前Kernel字节级一致）和shuffle后
`9719985`进行同机、同脚本、同20项production-only标准协议A/B。结果为：

| 变体 | shuffle胜出 | shuffle几何平均延迟变化 |
|---|---:|---:|
| Plain | 1/5 | -3.67% |
| Expand | 0/5 | +12.29% |
| Rescale | 1/5 | +2.91% |
| RescaleExpand | 0/5 | +10.21% |
| 总体 | 2/20 | +5.24% |

shuffle仅在tiny Plain（提升30.96%）和tiny Rescale（提升8.40%）胜出；
中大型Plain、所有Expand、绝大多数Rescale和所有RescaleExpand回退。因此
不整体合并该Kernel，也不能仅凭四变体tiny smoke决定默认策略。完整逐项数据：

```text
/data/gxy/TileOPs-Metax-team-results/shuffle-ab-2026-08-05/SUMMARY.md
```

此前103项与32项结果继续作为历史演进证据，不作为当前20项四方Benchmark
完整运行的声明。重构早期`->32`代表性性能artifact已因原始TileKernels调度
缺少不足128行的Expand尾块保护而删除，不用于当前可信对比。

### 吸收TXY Rescale动态向量化

在`47ccd16`建立`backup/quant-before-txy-vectorized-47ccd16`回退锚点后，只吸收
TXY实现中与Rescale FP8访存相关的单变量优化；保留当前`tile_k=64`、混合
staging和`amax_shared`归约，不吸收此前整体回退的4-lane amax shuffle。

静态分派为：输出token不超过256时使用8 lanes/token（vec8）；普通中等规模
使用16 lanes/token（vec4）；非Expand输出至少4096或Expand输出至少2048时
使用32 lanes/token（vec2）。Expand position广播相应改为subgroup-aware源lane，
动态reduction scratch也计入shared-memory门禁。

正确性入口新增5项分派边界测试，结果为`45 passed in 78.72s`；原有40项数值
与契约覆盖全部保留。性能只对受影响的10项Rescale/RescaleExpand正式Manifest
workload执行旧/新production-only同协议A/B，未重复运行eager、`torch.compile`
和固定TileKernels三个外部基线。协议仍为10次warmup、50次repeat、3次trial。

| 变体 | 新版胜出 | 几何平均旧版/新版 | 几何平均延迟降低 |
|---|---:|---:|---:|
| Rescale | 5/5 | 1.2232x | 18.25% |
| RescaleExpand | 5/5 | 1.2107x | 17.40% |
| 总体 | 10/10 | 1.2169x | 17.83% |

逐项原始输出及汇总保存在仓库同级目录：

```text
/data/gxy/TileOPs-Metax-team-results/txy-vectorized-ab-2026-08-05/
```

### 后续消融：保持 tile64，并为 Plain/Expand 吸收 vec4

以`217d3c9`建立`backup/quant-before-dynamic-tilek-217d3c9`回退锚点，并阅读
项目同级的mcProfiler V06和mcTracer V08手册。硬件计数选择mcProfiler；
mcTracer保留用于后续Runtime API、同步和多Kernel时间线诊断。

- Rescale的vec8/vec4/vec2各完成一组mcProfiler采样，Private Read/Write均为0，
  未观察到private spill。首次采样因代理转发localhost返回502，清除代理并设置
  `NO_PROXY=127.0.0.1,localhost`后恢复。
- `tile_k=128`在Rescale/RescaleExpand 10项正式workload中胜出0/10，平均增加
  57.62%延迟，因此回退，四路径继续固定`tile_k=64`。
- Plain/Expand保持thread-local staging，仅比较64/32/16 threads/token，分别
  对应vec1/vec2/vec4。vec4在10/10项胜出，几何平均加速`1.2555x`，单项范围
  `1.1253x–1.3922x`，因此production采用16 threads/token。
- vec4没有增加每线程local元素数量（仍为32），显式reduction shared memory为
  4 KiB。完整正确性为`45 passed in 55.89s`。

三个外部性能基线的代码和结果未改变，本轮按约定不重复运行。原始日志、Profiler
报告与汇总位于：

```text
/data/gxy/TileOPs-Metax-team-results/dynamic-tilek-2026-08-05/
```

### PR A #29 原始9项兼容复测

当前20项Manifest保持不变，另以独立进程完整复测PR A `81a39fd`定义的9项
workload。本轮比较production、PyTorch eager和固定上游TileKernels TileLang，
按约定不运行`torch.compile`；协议为10次warmup、50次repeat、3次trial。

9项全部通过严格正确性门禁。production相对eager几何平均`5.2789x`，相对
TileKernels几何平均`3.2286x`，两组对比均9/9胜出。完整逐项输出和汇总位于：

```text
/data/gxy/TileOPs-Metax-team-results/pr-a9-compat-2026-08-05/
```
