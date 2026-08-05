# quant_per_channel_cast_fused：GitLink PR B 提交前自查

> 对照官方 `summer-camp-2026` 指南和算子迁移 PR 模板整理。本文用于最终
> GitLink PR B，不代表当前 GitHub 过程管理 PR 已提交到官方仓库。勾选项均有
> 当前仓库或同级结果目录中的证据；未完成项保持未勾选。

## 1. 待人工补充的课题信息

- [ ] 课题名称：待小组确认。
- [ ] 课题完整简介：待小组确认最终课题范围后填写。
- [ ] 算子认领 Issue：待填写本组 GitLink Issue 链接。
- [ ] 小组编号与成员：待填写。
- [x] GitLink PR A `#29` 已合入 `summer-camp-2026`。PR head 与最终 Manifest
  提交均为 `81a39fda24291a2c42b0e6b5c64453bb4b78774f`；已验证该提交是
  `summer-camp-2026@df672b398134cbffd73e83fb816c36602fc3f3cf` 的祖先。

算子名称为 `quant_per_channel_cast_fused`，改动类型为 `optimize`。最终 PR 标题
应采用：

```text
[quant_per_channel_cast_fused] optimize: 优化 C500 分块、staging 与向量化访存
```

## 2. 来源、范围与实现

- [x] 上游仓库：`https://github.com/MetaX-MACA/TileKernels-Metax`。
- [x] 上游提交：`0266ab740980de7dc03a828b8259cd73d100c2eb`。
- [x] 上游源码：`tile_kernels/quant/per_channel_cast_fused_kernel.py`。
- [x] 上游 Kernel SHA256：
  `64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc`。
- [x] 四个变体均实现：Plain、Expand、Rescale、RescaleExpand。
- [x] Op、Kernel、workload、Manifest、功能测试与 Benchmark 分层完整。
- [x] Manifest 为 `implemented`，四变体各有 `vars/flops/bytes` 和5个正式
  workload，共20项。
- [x] 当前 production 固定 `tile_k=64`；Plain/Expand 使用 thread-local
  staging 和 vec4；Rescale 两路使用 shared staging 和动态 vec8/vec4/vec2。
- [x] 未把 PyTorch reference 放进 production 执行路径。

## 3. 正确性证据

- [x] 独立 PyTorch eager oracle 与 production 的 FP8 输出逐元素一致；scale
  使用 `atol=1e-7, rtol=1e-6`。
- [x] 覆盖 BF16、FP32、FP8、round、128-token 边界、hidden 边界、Expand
  gather/padding/repeated/reverse、尾块、空输入、极值和异常参数。
- [x] C500 完整功能回归：`45 passed in 26.63s`（2026-08-05）。
- [x] Benchmark 基线与基础设施测试：`24 passed in 27.31s`（2026-08-05）。

复现命令：

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/gxy/TileOPs-Metax-team:$PYTHONPATH
python -m pytest -q tests/ops/test_per_channel_cast_fused.py
python -m pytest -q benchmarks/tests
```

## 4. 性能与分析证据

- [x] 正式性能矩阵共20项，四个变体各5项；每项在计时前先做严格数值门禁。
- [x] 比较 production、PyTorch eager、`torch.compile(fullgraph=True)` 和固定
  上游 TileKernels TileLang 三个外部基线。
- [x] 标准协议为10次 warmup、50次 repeat、3次 trial。
- [x] 完整C500上的上一阶段20项四方基线已归档；production相对 eager
  `3.3876x`、相对上游 TileKernels `2.2433x`，两者均20/20胜出。
- [x] Rescale动态向量化单变量A/B：10/10胜出，几何平均`1.2169x`。
- [x] Plain/Expand vec4单变量A/B：10/10胜出，几何平均`1.2555x`，单项
  `1.1253x–1.3922x`。
- [x] `tile_k=128`失败消融如实记录：0/10胜出，平均增加57.62%延迟，已回退。
- [x] PR A #29原始9项兼容集已额外复测，不改变当前20项Manifest。9项均通过
  严格正确性门禁；production相对eager几何平均`5.2789x`、相对固定上游
  TileKernels几何平均`3.2286x`，均为9/9胜出。本轮按约定未运行torch.compile。
- [x] mcProfiler已覆盖Plain/Expand staging前后和Rescale vec8/vec4/vec2；未
  观察到Private Read/Write spill，并保留shared、wave和访存计数。
- [ ] 在最终GitLink PR B代码提交上重新执行20项四方Benchmark并归档报告。
  三个外部基线实现未变化，当前优化的production-only A/B已完成；该项是最终
  提交SHA的一致性复核，不能用旧提交报告冒充。

性能结果目录（不提交进源码仓库）：

```text
/data/gxy/TileOPs-Metax-team-results/full-c500-2026-08-05/
/data/gxy/TileOPs-Metax-team-results/txy-vectorized-ab-2026-08-05/
/data/gxy/TileOPs-Metax-team-results/dynamic-tilek-2026-08-05/
/data/gxy/TileOPs-Metax-team-results/pr-a9-compat-2026-08-05/
```

## 5. 官方提交门禁

- [x] `git diff --check`：通过。
- [x] `python scripts/validate_manifest.py`：通过。全仓449条为advisory warning；
  本算子10条均为synthetic mock不满足输入前置条件，未出现blocking error。
- [x] 本算子功能测试：45项通过。
- [x] `python -m pytest -q benchmarks/tests`：24项通过。
- [x] `pre-commit`中的private-key、gitleaks、YAML、TOML、Python AST、merge
  conflict和codespell检查通过。
- [ ] `pre-commit run --all-files`整体通过。当前首轮会格式化本PR范围外的历史
  文档及5份必须保持原样的mcProfiler证据，并修复3个既有Python lint问题；
  为避免把无关机械改动混入单算子PR，自动修改已恢复。最终提交前需与维护者
  确认是基于最新官方目标分支重放，还是单独处理仓库级格式债务。

## 6. PR 模板与流程

- [x] 优化方案、正确性、性能、加速比、失败消融、mcProfiler瓶颈和复现命令
  已写入算子README与开发记录。
- [x] 临时性能日志和Profiler数据库位于仓库同级结果目录，没有进入Git索引。
- [x] 当前提交不包含密码、Token、私钥、容器地址或完整环境变量。
- [ ] 使用官方 `operator-migration.zh-CN.md` 模板创建GitLink PR B，并完整填写
  第1节中的小组信息及最终测试提交SHA。
- [ ] 在GitLink Issue #4按官方模板完成提交前检查；未完成项不得提前勾选。
