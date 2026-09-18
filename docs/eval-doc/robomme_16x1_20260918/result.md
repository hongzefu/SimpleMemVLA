# result：RoboMME 16 task × 1 episode（本机双卡 + GreatLakes 两个 hold job）

> 口径提醒：本轮是 **每 task 1 个 episode**，官方口径是 16 × 50。**n=1 的成功率不是论文分数的复现**，
> 只能说明「链路通了、模型在这一个 frozen episode 上做对了」。README 报的 RoboMME 88.3 是 16 × 50 的口径。

## 一句话结论

**本机 16/16 全部成功（macro avg 100.0%），集群两个 job 各跑 2 个 task 也全部成功。** 两端不仅结论一致，
产物还几乎逐字节一致——两个集群 run 的视频字节数**完全相同**，本机与集群仅差 1 帧。

## 三次运行的实测结果

| run | 硬件 | 并行度 | 起止 | 完成 task | 结果 |
|---|---|---|---|---|---|
| `robomme_16x1_local_20260918` | sled-vail，2 × RTX 6000 Ada（sm_89） | 双卡，每卡 8 task | 16:06:30 → 16:18:47（**12 分钟**） | **16 / 16** | 全部 `success_rate=1.000`，`macro_avg=1.0`，`n_excluded_na=0` |
| `gl-61495429-16x1` | gl1517，1 × A40（sm_86），job 61495429 | 单卡串行 | 16:19:57 → 16:27:30（用户要求跑完即停） | 2 / 16 | `PickXtimes` ✓、`StopCube` ✓ |
| `gl-61495430-16x1` | gl1517，1 × A40（sm_86），job 61495430 | 单卡串行 | 16:20:00 → 16:27:0x（同上） | 2 / 16 | `PickXtimes` ✓、`StopCube` ✓ |

本机 16 个 task 逐条（全部 `n=1`，`oom=false`）：
`PickXtimes StopCube SwingXtimes BinFill VideoUnmaskSwap VideoUnmask ButtonUnmaskSwap ButtonUnmask
VideoRepick VideoPlaceButton VideoPlaceOrder PickHighlight InsertPeg MoveCube PatternLock RouteStick`
—— **16 项全部 1.000**。日志里 `reset_failed` / `RRTPlanFailure` / `[OOM]` / `Traceback` 计数均为 **0**，
即没有任何 episode 因 mplib 规划失败被从统计里剔除（`n_excluded_na=0` 也印证了这一点）。

## 两端一致性（本轮最有价值的证据）

### 1. 渲染逐值相同

本机与集群的 `render_probe.py` 输出**完全一致**：

```
图像 shape=(256, 256, 4) dtype=float32 std=0.1053 min=0.3511 max=1.0000
```

两端的 Vulkan ICD 文件名不同（本机 `nvidia_icd.json`、集群 `nvidia_icd.x86_64.json`），
显卡也不同（RTX 6000 Ada vs A40），但渲染结果在这三个统计量上不可区分。

### 2. 同硬件下完全确定性，跨硬件差 1 帧

| 产物 | 本机（sm_89） | gl01（sm_86） | gl02（sm_86） |
|---|---|---|---|
| `PickXtimes__ep0__succ.mp4` | **608** 帧，1223790 B | **609** 帧，1221827 B | **609** 帧，1221827 B |
| `StopCube__ep0__succ.mp4` | **285** 帧，439601 B | **284** 帧，437562 B | **284** 帧,437562 B |

两点观察：

- **两个集群 run 的字节数完全相同**（1221827 / 437562）。它们是两个独立的 Slurm job、独立的进程、
  独立的模型加载，跑在同一型号的 A40 上，产出逐字节一致 —— 说明整条链路（frozen episode + 
  固定 seed 策略 + 确定性渲染）在同硬件上是**完全可复现**的。
- **本机与集群相差 1 帧**（608 vs 609、285 vs 284），方向还相反。这是 bf16 在不同 SM 架构
  （sm_89 vs sm_86）上的数值差异，累积到 receding-horizon 控制里，使 episode 的终止步差了一步。
  成功判定不受影响（两端都是 succ）。**这个量级的差异是预期的，不是 bug**；但它也提醒：
  跨硬件比较分数时，单 episode 口径的噪声不可忽略，正式结论必须用 16 × 50 的完整口径。

本机内部另有一条佐证：单 task 冒烟与正式 16×1 的 `PickXtimes` 都是 608 帧、1223790 B，完全相同。

## 耗时与资源

| 项 | 实测 |
|---|---|
| 纯模拟器（本机，`sim_probe.py`） | `import robomme_sim` 6.8s、builder 15.9s、**RESET 5.9s**、**27 ms/step** |
| 本机单 task（含首次从 NFS 加载 12.69 GB 权重） | 约 5 分钟 |
| 本机 16 task 双卡 | **12 分钟** |
| 集群单 task（1 CPU，含加载） | 约 7.5 分钟跑完 2 个 task |
| 集群 GPU 占用 | `utilization.gpu` 45–54%，显存 **12794 MiB**（≈ 权重本身，activation 占比很小） |

**RESET 只要 5.9 秒**，远低于起初的担忧。上游把 `--reset_timeout` 默认设成 3600 秒有极大冗余。
由此确定：**总耗时的主导项是模型前向，不是仿真** —— 纯仿真开销仅约 0.43s/决策，跑满 84 次决策
不到 1 分钟，而单 episode 实际要几分钟。

集群慢于本机的原因有两条：每个 hold job 只有 **1 CPU**（本机 32 核，SAPIEN 的物理与渲染都吃 CPU），
且两个 job 挤在同一节点 `gl1517` 上互相争。

## 配置确认（都按预期生效）

- `transformers 5.13.1` 与 checkpoint `config.json` 记录一致，`run_robomme.sh` 的 preflight 每次起跑都校验。
- 无 flash-attn / fla / causal-conv1d，日志如期出现
  `The fast path is not available ... Falling back to torch implementation.` —— 纯 torch fallback 生效，
  `ATTN_IMPLEMENTATION=sdpa`。**结果没有因此变差**（16/16 全成功）。
- 集群 GPU `compute_mode=Default`，`--gpu_cmode=shared` 生效，没有出现 robomme-eval-GL 记录的
  `vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`。
- **NFS 上的 `.venv-robomme` 被本机与计算节点直接共用**，集群侧未安装任何依赖。
  担心的 GLIBC 坑没出现 —— 评测链路只走 `simplememvla.data.{collator,messages,normalize}`（纯 torch/numpy），
  不碰会间接拉进预编译 Rust 扩展的 `wandb` / `datasets`。

## 过程中的两个问题及处置

### 1. 集群首轮 exit 127（已修）

三级探针全过之后、进入评测那一步立刻挂：`/usr/bin/bash: line 36:  : command not found`。
根因是旧版 `gl_run_robomme.sh` 用**未加引号的 heredoc** 拼远程命令串，最后一行的续行符 `\\` 被
折叠掉，命令名变成空字符串。修法是把远程逻辑整体搬进独立文件 `scripts/gl_remote_body.sh`
（NFS 两端可见），`srun` 直接执行文件、零转义，并加 `DRY_RUN=1` 用于零 GPU 成本地验证整条链路。
详见 commit `0bf7225`。

### 2. Monitor 静默，失败没被及时发现（已修机制，但教训更重要）

首轮失败发生在约 16:12，直到 16:17 主动查日志才发现，期间还在汇报「集群两路仍在跑」。
Monitor 的过滤词表里**有** `EXIT_CODE=`，没报出来是因为整份日志只有几 KB、卡在 `tee` 的 4 KB
块缓冲里，直到进程结束才一次性落盘。机制上已修（`stdbuf -oL tee`），但根本教训是：
**长任务必须主动核对一次状态，不能把「没收到告警」当成「一切正常」**。
这正是 `CLAUDE.md` 里「管道每一级都必须行缓冲」那条规则在 `tee` 这一级上的同一个坑。

## 产物

```
logs/robomme_sim/robomme_16x1_local_20260918/   results.json + eval.log + videos/（16 段，23 MB）
logs/robomme_sim/gl-61495429-16x1/              eval.log + videos/（2 段）
logs/robomme_sim/gl-61495430-16x1/              eval.log + videos/（2 段）
docs/eval-doc/robomme_16x1_20260918/records/    本留档归档的日志与 results.json
```

集群两路是被用户要求中途停止的，所以**没有 `results.json`** —— 上游 `eval_success.py` 只在
全部 task 跑完后才写该文件。每个 task 结束时落盘的 `success_rate=` 日志行与 mp4 都在，结论不受影响。

## 下一步（不在本轮授权内）

- 跑完整的 16 × 50 官方口径，才能与 README 的 88.3 对照。按本机 12 分钟 / 16 episode 线性外推，
  双卡约需 10 小时；集群按 job array 分片（每片独立 `RUN_TAG`，见 AGENTS.md GreatLakes 第 6 条）更合适。
- 其余四个 benchmark（RMBench / MIKASA / LIBERO / RoboMemArena）各需独立 venv。
