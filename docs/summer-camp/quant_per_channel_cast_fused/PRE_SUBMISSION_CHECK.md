# quant_per_channel_cast_fused：GitLink PR B 提交前自查

> 以下“自查留言”严格保持 GitLink Issue #4 的栏目、顺序和勾选项，可在补齐
> 人工信息后直接复制到该 Issue。未实际完成的项目不提前勾选。

## 自查留言

### 提交前自查

- 小组编号：待小组填写
- 课题名称：待小组确认
- 主算子名称：`quant_per_channel_cast_fused`
- 追加算子名称：无；Plain、Expand、Rescale、RescaleExpand 是主算子的四个变体
- 各算子认领 Issue：待填写本组 GitLink 算子认领 Issue
- 各算子实现 PR：待创建正式 GitLink PR B；GitHub PR #5 仅用于组内过程管理

- [ ] 算子认领及来源信息完整
- [ ] 已提交 PR，并完整填写统一 PR 模板
- [x] Manifest、算子实现、测试和 Benchmark 已完成
- [x] 正确性、边界及异常测试通过
- [ ] 已在真实 MetaX C500 上完成最终验证
- [ ] 性能数据、mcProfiler 和 Roofline 证据完整
- [ ] Review 阻塞问题已经处理
- [ ] 答辩 PPT 已完成
- [ ] PPT 包含任务介绍、实现方案、正确性、性能优化及开源价值
- [ ] 已完成答辩演练并确认时间安排
- [x] 未提交敏感信息，不存在抄袭或虚假数据

- 当前未完成项：补充小组、课题和认领 Issue；创建正式 GitLink PR B 并填写
  官方模板；在最终 GitLink 提交 SHA 上复跑20项四方 Benchmark；确认 Manifest
  workload 从 PR A 9项演进为当前20项的提交方式；处理或确认全仓 pre-commit
  历史格式问题；完成答辩 PPT 和演练。
- 需要助教协助的事项：确认当前20项 Manifest workload 能否随 PR B 提交，
  还是需要先提交 Manifest 补充 PR；确认本算子范围外的历史文档格式问题是否
  阻塞单算子 PR 的 `pre-commit run --all-files` 门禁。
- 当前结论：尚需补充
- 小组负责人：待小组填写
- 小组成员：待小组填写

## 技术证据附录

该附录用于支撑上述勾选状态，不属于 Issue #4 官方模板正文。

### Manifest 与来源

- GitLink PR A：`#29`。
- PR A head/Manifest提交：
  `81a39fda24291a2c42b0e6b5c64453bb4b78774f`。
- 已验证该提交是官方
  `summer-camp-2026@df672b398134cbffd73e83fb816c36602fc3f3cf` 的祖先。
- 上游仓库：`https://github.com/MetaX-MACA/TileKernels-Metax`。
- 上游提交：`0266ab740980de7dc03a828b8259cd73d100c2eb`。
- 上游源码：`tile_kernels/quant/per_channel_cast_fused_kernel.py`。
- 上游 Kernel SHA256：
  `64e7ad56bd8ea125c561b1726a1cdf13ce78bf722cb9bef520452026157edafc`。
- 当前 Manifest 为 `implemented`，包含四个变体和20项正式 workload；另已
  完整复测 PR A 原始9项兼容 workload。

### 正确性

- 独立 PyTorch eager oracle 与 production FP8 输出逐元素一致。
- Scale 容差：`atol=1e-7, rtol=1e-6`。
- 覆盖 BF16、FP32、FP8、round、tile/hidden边界、Expand索引与padding、
  尾块、空输入、极值和异常参数。
- 当前完整功能回归：`45 passed in 26.63s`。
- Benchmark 基线与基础设施：`24 passed in 27.31s`。

```bash
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:$PYTHONPATH
python -m pytest -q tests/ops/test_per_channel_cast_fused.py
python -m pytest -q benchmarks/tests
```

### 性能与分析

- 正式20项：四个变体各5项，10次warmup、50次repeat、3次trial。
- 上一阶段完整C500四方结果：production相对eager `3.3876x`、相对上游
  TileKernels `2.2433x`，两者均20/20胜出。
- Rescale动态向量化A/B：10/10胜出，几何平均`1.2169x`。
- Plain/Expand vec4 A/B：10/10胜出，几何平均`1.2555x`。
- PR A原始9项兼容集：全部通过正确性；production相对eager几何平均
  `5.2789x`、相对上游TileKernels `3.2286x`，均9/9胜出；未运行
  `torch.compile`。
- `tile_k=128`失败消融：0/10胜出，平均增加57.62%延迟，已回退。
- mcProfiler覆盖Plain/Expand staging前后及Rescale vec8/vec4/vec2；未观察到
  Private Read/Write spill，并保留shared、wave和访存计数。

### 门禁状态

- `git diff --check`：通过。
- `python scripts/validate_manifest.py`：通过；全仓449条均为advisory warning，
  没有blocking error。
- private-key、gitleaks、YAML、TOML、Python AST、merge conflict和codespell：
  通过。
- `pre-commit run --all-files`：尚未整体通过。它会格式化本算子PR范围外的历史
  文档和5份需保持原样的mcProfiler报告，并修复3个既有Python lint问题；自动
  修改已恢复，等待维护者确认仓库级格式债务的处理方式。
