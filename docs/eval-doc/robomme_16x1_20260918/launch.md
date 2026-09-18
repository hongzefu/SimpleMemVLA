# launch：RoboMME 16 task × 1 episode（本机双卡 + GreatLakes 两个 hold job）

## 目的

复现 SimpleMemVLA 的 RoboMME 闭环评测链路，口径为 **16 个 task × 每 task 1 个 episode**（官方口径是
16 × 50；本轮把 `EPISODES_PER_TASK` 降到 1，用于打通链路并做两端一致性对照，**不是**论文分数的复现）。
三次运行：

| run | 硬件 | 并行度 | RUN_TAG |
|---|---|---|---|
| 本机 | sled-vail，2 × RTX 6000 Ada（46 GB，sm_89） | `NUM_GPUS=2 GPUS=0,1`，task-level 分片每卡 8 个 | `robomme_16x1_local_20260918` |
| 集群 A | GreatLakes `gl1517`，1 × A40（job 61495429 / gpu-hold-01） | `NUM_GPUS=1`，16 task 串行 | `gl-61495429-16x1` |
| 集群 B | GreatLakes `gl1517`，1 × A40（job 61495430 / gpu-hold-02） | `NUM_GPUS=1`，16 task 串行 | `gl-61495430-16x1` |

## 运行环境

**环境 A（本机 sled-vail）**：判定通过 —— `hostname=sled-vail`、`/data/hongzefu` 存在、
`~/.ssh/config` 存在、2 × RTX 6000 Ada 且 compute mode 均为 `Default`、无 `sbatch`、
Vulkan ICD 为 `/usr/share/vulkan/icd.d/nvidia_icd.json`。

**环境 B（GreatLakes 计算节点）**：两个 hold job 都在 `gl1517`，各 **1 GPU / 1 CPU / 24G 内存**，
walltime `2-00:00:00`。注意这比常规评测配置紧（robomme-eval-GL 实测 MaxRSS 约 16 GB、
推荐 2 CPU / 32G），1 CPU 会拖慢 SAPIEN 的 CPU 物理仿真。spgpu 分区当前全局空闲 A40 为 0
（240 张物理卡：已分配 231、故障维护 40），353 个 job 排队 —— 所以复用已有 hold job 而不是
重新 sbatch。

## 代码与依赖锚点

- 仓库 HEAD：`794ef3a`（clean，起跑前核对）
- ManiSkill fork：`third_party/ManiSkill` @ `07be6fbc`（`YinpeiDai/ManiSkill`）
- venv：`.venv-robomme`，解释器 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10`
- 关键版本（`scripts/check/verify_env.py` 实测）：
  `torch 2.4.1+cu121` / `numpy 1.26.4` / `transformers 5.13.1` / `gymnasium 0.29.1` /
  `sapien 3.0.3` / `mani_skill 3.0.0b21`；`flash_attn` / `fla` / `causal_conv1d` **均未装**
- attention 后端：`ATTN_IMPLEMENTATION=sdpa`（本项目不装 flash-attn），linear-attention 走
  transformers 的纯 torch fallback，日志里会出现一条
  `The fast path is not available ... Falling back to torch implementation.` —— **这是预期的**

## 被评权重

`checkpoints/simplememvla_robomme`（HF `yinchenghust/simplememvla_robomme`）
- `model.safetensors` 12688542952 字节（12.69 GB，单文件无分片）
- `config.json`：`model_type=simplememvla`、`action_horizon=30`、`action_dim=8`、`use_proprio=True`、
  `transformers_version=5.13.1`（与环境一致，已 preflight 校验）
- `stats.json` 存在（action 反归一化参考）
- 落点必须在 NFS 仓库内 —— GreatLakes 计算节点看不见 `/home` 与 `/data`

## 本轮代码改动

只加不改：新增 `AGENTS.md` / `CLAUDE.md`（commit `c55e206`）、`scripts/check/{verify_env,render_probe,sim_probe}.py`
与 `scripts/{run_robomme,gl_run_robomme}.sh`、`pyproject.toml` 的三段 uv 声明、`.gitignore` 两条
（commit `794ef3a`）。**上游的 `scripts/eval_robomme.sh` 与 `robomme_sim/` 一行未动。**

## 评测配置

`scripts/run_robomme.sh` 固化的口径（其余沿用上游 `scripts/eval_robomme.sh` 默认）：

| 参数 | 取值 | 说明 |
|---|---|---|
| `EPISODES_PER_TASK` | 1 | 官方 test split 是 50，本轮只取第 0 个 |
| `GROUP_SIZE` | 1 | 上游 `group_size = min(g, n_req)`，n_req=1 时写多少都会压到 1 |
| `DATASET_SPLIT` | test | frozen 列表，metadata 已 vendored 在 `robomme_sim/robomme/env_metadata/test/` |
| `MAX_STEPS` | 1300 | 上游默认（official MME-VLA setting），`hard_bound = ceil(1300/16)+2 = 84` 次决策 |
| `EXECUTE_HORIZON` | 16 | receding horizon |
| `NUM_DENOISING_STEPS` | 10 | DiT flow-matching |
| `MAX_SUBTASK_TOKENS` | 64 | |
| `COMPUTE_DTYPE` | bfloat16 | |
| `ATTN_IMPLEMENTATION` | sdpa | |
| `PIPELINED` | 0 | |
| `VIDEO_MAX_PER_TASK` | 1 | 每 task 留 1 成功 + 1 失败 |

16 个 task（`robomme_sim/eval_success.py` 的 `DEFAULT_TASKS`）：
`PickXtimes StopCube SwingXtimes BinFill VideoUnmaskSwap VideoUnmask ButtonUnmaskSwap ButtonUnmask
VideoRepick VideoPlaceButton VideoPlaceOrder PickHighlight InsertPeg MoveCube PatternLock RouteStick`

本机双卡的 `_partition` 轮询分桶（`i % 2`）：
- worker 0 / GPU 0：`PickXtimes SwingXtimes VideoUnmaskSwap ButtonUnmaskSwap VideoRepick VideoPlaceOrder InsertPeg PatternLock`
- worker 1 / GPU 1：`StopCube BinFill VideoUnmask ButtonUnmask VideoPlaceButton PickHighlight MoveCube RouteStick`

## 执行顺序与命令

```bash
# 前置校验（两端都跑，全绿才往下）
PYTHONPATH=$PWD .venv-robomme/bin/python scripts/check/verify_env.py checkpoints/simplememvla_robomme
PYTHONPATH=$PWD .venv-robomme/bin/python scripts/check/render_probe.py
PYTHONPATH=$PWD .venv-robomme/bin/python scripts/check/sim_probe.py PickXtimes 16

# 单 task 冒烟
TASKS=PickXtimes NUM_GPUS=1 GPUS=0 RUN_TAG=smoke_pickxtimes bash scripts/run_robomme.sh

# 本机 16×1 双卡
NUM_GPUS=2 GPUS=0,1 RUN_TAG=robomme_16x1_local_20260918 bash scripts/run_robomme.sh

# 集群（登录节点执行，先只跑探针，全绿再跑评测）
PROBE_ONLY=1 bash scripts/gl_run_robomme.sh 61495429 probe-61495429
bash scripts/gl_run_robomme.sh 61495429 gl-61495429-16x1
bash scripts/gl_run_robomme.sh 61495430 gl-61495430-16x1
```

## 产物路径

- `logs/robomme_sim/<RUN_TAG>/eval.log`、`results.json`、`videos/<task>/<task>__ep0__{succ,fail}.mp4`
- 本留档：`docs/eval-doc/robomme_16x1_20260918/`（`launch.md` / `result.md` / `records/`）

## 盯盘项（Monitor 过滤词见 CLAUDE.md）

1. **`success_rate` 全 0 而没有任何报错** → transformers 版本不对（本轮已 preflight 钉死 5.13.1，应不会发生）。
2. **`reset_failed:` / `RRTPlanFailure`** → mplib 规划的固有随机性，该 episode 记为 `None` 从统计里剔除，
   **不算失败**。`results.json` 里会表现为 `success_rate: null, n: 0`，最终报告必须单列、不能混进 macro average。
3. **`ErrorInitializationFailed` / `EXCLUSIVE_PROCESS`** → 只可能出现在集群侧，说明 `--gpu_cmode=shared`
   没生效，停下来报告。
4. **静默空转** → `--reset_timeout` 默认 3600 秒，卡在 RRT* 重试里时进程活着、日志不出错也不产出。
   本机实测 RESET 仅 5.9 秒，所以任何 task 超过 10 分钟没有新的 `success_rate=` 行就要介入。
5. **视频黑屏** → Vulkan 选错设备。渲染探针已在两端各跑一次把关。

## 耗时预期（据实测修正）

`sim_probe.py` 实测（本机，PickXtimes）：`import robomme_sim` 6.8s、builder 就绪 15.9s、
**RESET 5.9s**、16 步仿真 0.43s（**27 ms/step**）。即纯仿真开销仅约 0.43s/决策，跑满 84 次决策
的纯仿真时间不到 1 分钟 —— **总耗时的主导项是模型前向，不是仿真**，这推翻了起初「reset 可能要
数分钟」的担忧（`--reset_timeout 3600` 的默认值有很大冗余）。

真实的每次决策耗时由单 task 冒烟给出，正式跑的预期在 `result.md` 里据此修正。

## 本轮 tmux 会话清单

`smvla-msclone`（ManiSkill clone）、`smvla-ckpt`（checkpoint 下载）、`smvla-uvsync`（uv sync）、
`smvla-simprobe`（纯模拟器冒烟）、`smvla-smoke`（单 task 冒烟）、`smvla-16x1-local`（本机正式跑）、
`smvla-16x1-gl01` / `smvla-16x1-gl02`（集群两次）。

⚠ 本机同时跑着其他项目的 tmux 会话（`site*` / `ensite*` / `unisite` / `v3site`），
清理只能 `tmux kill-session -t '=<确切名>'`，禁止 `kill-server` / `kill-session -a` / `pkill`。
