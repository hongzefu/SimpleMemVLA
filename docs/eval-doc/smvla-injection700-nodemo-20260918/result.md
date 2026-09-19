# 实施和评估结果

## 最终结论

**700/700 条均有最终结果：455 成功、228 失败、8 次策略执行超时、9 条环境 error。
固定分母成功率为 455/700 = 65.0%，14 个任务/难度组的宏平均同为 65.0%。**
这是注入候选自定义测试集，不能标为官方 RoboMME 16×50 成绩。

按用户追加的「卡死直接 kill 并标记 error」口径，覆盖已经完成；`coverage_complete=true`，
`error_free=false`、`complete=false`。9 条 reset 错误保留在 700 分母内，没有换 seed、补样本或删掉失败。
691 条正常终态的视频全部完整解码，并在最终审计逐个复核 SHA256；共 383767 帧、772557897 字节。
BinFill 的 150 条均验证 `demo_tasks=0, demo_frames=0`。

benchmark 专用分支 `PolicyEvalThirdParty-simplememvla-0918` 已推送，子模块固定在 `b4e97f2`。
实际运行经历 `170125f`、`871406a`、`986808c` 三个控制层版本；模型、环境、权重、候选和推理参数指纹一致。
共 703 次尝试形成 700 个唯一结果，3 次重试均保留原始记录。两路分片各 350 条、交集为空、并集精确等于冻结清单。

## 分片与完成时间

| 分片 | 成功 | 失败 | 策略超时 | 环境 error | 最终恢复 step | 该 step 耗时 | 退出 |
|---|---:|---:|---:|---:|---|---|---|
| hold01 | 212 | 129 | 3 | 6 | 61495429.11 | 09:33:58 | COMPLETED 0:0 |
| hold02 | 243 | 99 | 5 | 3 | 61495430.5 | 06:16:49 | COMPLETED 0:0 |
| 合计 | 455 | 228 | 8 | 9 | — | — | 总编排退出码 0 |

底特律时间：hold02 最后一条回合于 9 月 19 日 04:07:51 结束，hold01 于 07:25:02 结束；
随后完成视频汇总验证，总结果文件于 07:29 写出。07:55 自动唤醒后再次核对完整性、视频 SHA 和来源/权重完整 SHA。
最终两个评估 step 均已退出，用户的 hold01/hold02 分配仍保留，未取消分配。
两路最终 cgroup 峰值分别为 22417723392 / 21099880448 字节，OOM 与 OOM kill 均为 0。

## 14 组成绩

| 任务 | 难度 | 成功/50 | 成功率 | 失败 | 策略超时 | error |
|---|---|---:|---:|---:|---:|---:|
| BinFill | easy | 43/50 | 86% | 7 | 0 | 0 |
| BinFill | medium | 30/50 | 60% | 18 | 2 | 0 |
| BinFill | hard | 25/50 | 50% | 24 | 1 | 0 |
| RouteStick | easy | 45/50 | 90% | 5 | 0 | 0 |
| RouteStick | medium | 42/50 | 84% | 8 | 0 | 0 |
| RouteStick | hard | 31/50 | 62% | 19 | 0 | 0 |
| RouteStick | xhard | 6/50 | 12% | 44 | 0 | 0 |
| VideoRepick | easy | 43/50 | 86% | 6 | 0 | 1 |
| VideoRepick | medium | 34/50 | 68% | 13 | 0 | 3 |
| VideoRepick | xhard | 14/50 | 28% | 33 | 0 | 3 |
| VideoUnmaskSwap | easy | 43/50 | 86% | 4 | 3 | 0 |
| VideoUnmaskSwap | medium | 49/50 | 98% | 1 | 0 | 0 |
| VideoUnmaskSwap | hard | 40/50 | 80% | 9 | 1 | 0 |
| VideoUnmaskSwap | xhard | 10/50 | 20% | 37 | 1 | 2 |

## 9 条环境错误

以下均停在 reset 阶段，执行策略步数为 0；具体异常、seed、spec 指纹及每次尝试见
[错误记录](records/final-errors.json)和[全部尝试](records/episode-attempts.jsonl)。

| 任务/难度 | episode | 分片 | 最终尝试 | reset 耗时 |
|---|---:|---|---:|---:|
| VideoRepick/easy | 189 | hold01 | 1 | 约 600 秒 |
| VideoRepick/medium | 155 | hold01 | 2 | 约 3600 秒，旧规则 |
| VideoRepick/medium | 167 | hold01 | 1 | 约 600 秒 |
| VideoRepick/medium | 172 | hold02 | 1 | 约 600 秒 |
| VideoRepick/xhard | 168 | hold02 | 1 | 约 2900 秒，修改规则时主动终止 |
| VideoRepick/xhard | 197 | hold01 | 1 | 约 600 秒 |
| VideoRepick/xhard | 201 | hold01 | 1 | 约 600 秒 |
| VideoUnmaskSwap/xhard | 139 | hold01 | 1 | 约 600 秒 |
| VideoUnmaskSwap/xhard | 164 | hold02 | 1 | 约 600 秒 |

新规则下 7 条 reset 超时均在约 600 秒记录 error，进程结束后继续后续样本，没有再次重试卡死键。
另两条保留规则变更前的真实耗时，不把历史记录改写成 600 秒。此处只确认 reset 超时，
不将未定位的原生仿真/规划根因写成已解决。

## 起跑前验证

- 9 项定向测试通过：700 条唯一键、两路平衡分片、重复/越界计划拒绝、难度隔离、
  BinFill 初始观测与 demo 区分、导入来源拒绝、恢复身份、错误保留分母、视频完整解码与损坏拒绝。
- 假环境评估验证：运行错误记 error；同样本第二次运行准确在 2000 步截断为 timeout；
  再次恢复跳过已正常完成回合，不重新加载模型。
- 真实 GPU1 环境探针：BinFill/hard episode 153、seed 19300，reset 返回 1 个正常初始帧，
  `demo_tasks=0, demo_frames=0`；图像标准差大于 1，真实 step 1 步，状态 ongoing，无运行错误。
- 上述真实探针没有加载模型，不替代随后 14 条完整闭环冒烟。

## 验收与产物

- [总结果](records/final-results.json)、[hold01](records/hold01-results.json)、[hold02](records/hold02-results.json)。
- [最终审计](records/final-audit.json)：700 唯一键、703 次尝试、691 视频及帧/字节数、来源版本和权重指纹。
- [运行日志摘要](records/campaign.summary.log)、[Slurm 最终状态](records/gl-final-status.txt)。
- 完整日志和视频保留在 `logs/robomme_sim/smvla-injection700-nodemo-20260918/` 及 `logs/setup/`，不复制大产物进 Git。
- 代码验证为 14 项定向测试通过，包含真实进程组强杀、UTF-8 错误持久化、错误不阻断后续、
  来源兼容反例、身份与分母检查；最终只追加文档和实测证据，不改变已跑代码。

## 本机完整冒烟与首轮 GL

- `170125f` 启动，本机 14/14 条正常终态：7 成功、6 失败、1 超时，无运行错误、无遗漏，
  14 个视频全部完整解码。BinFill 三组 demo_tasks/demo_frames 均为零，其余任务保留示范。
- BinFill/hard episode153、seed19300 在本机跑满 2000 步，视频 2001 帧，耗时 244.7 秒。
- GL 首轮 step 为 `61495429.7`、`61495430.3`。暂停前共有 16 条正常终态、11 条成功；
  hold01 的 RouteStick/xhard episode115 有一次 Vulkan reset 错误，hold02 的
  VideoUnmaskSwap/easy episode116 在 reset 阶段被主动暂停。两份原日志完整保留。
- 已定位控制器问题：RSS 25.2 GiB 不能当作超过 24 GiB cgroup 分配；内核实测峰值约 20.7 GiB、
  OOM 计数 0。另一个确定问题是中文异常 JSON 未指定 UTF-8，原生环境改变 locale 后触发
  UnicodeEncodeError。对应修复不改变模型/环境/评估循环，按来源兼容检查恢复。
- Vulkan 错误的根因尚未证明。此前只读内存诊断通过额外重叠 srun step 执行，可能干扰
  GPU 计算模式；暂停后直接 SSH 查询显示 Default，因此不能把该推测写成已确认根因。
- 暂停只取消上述两个评估 step；hold01、hold02 仍 Running。后续不再以重叠 GPU step 做诊断。

## 卡死超时与继续全量

9 月 18 日 21:35（底特律）检查：正常完成 134/700，87 成功、44 失败、3 超时；
hold01 正常 27 条，hold02 正常 107 条。hold01 的 VideoRepick/medium/155 两次 reset
各卡满 3600 秒，被旧控制器停止整路；hold02 的 VideoRepick/xhard/168 当时仍卡在 reset。

用户要求卡死超过阈值直接 kill 并标记 error。随后只终止本轮 step `61495430.4`，
保留两份 hold 分配；xhard/168 实测 reset 已持续约 2900 秒，记为 `error_kind=reset_timeout`。
本次暂停时共 134 条正常结果、2 条 error、564 条待跑，旧结果、旧尝试及视频全部保留。

新控制器 reset 上限 600 秒、单次推理/step 上限 300 秒；卡死不再重试或阻止剩余候选。
14 项测试通过，包括真实进程组 SIGKILL（含其 sleep 子进程）、error 持久化、停止重试卡死键、
正常长回合不被总时长误杀、覆盖完成与无错误通过分开、来源历史链不能改变模型或输入。
修订来源后恢复剩余候选，最终错误数会保留在 700 的分母中。
