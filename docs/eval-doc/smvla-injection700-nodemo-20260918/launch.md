# SimpleMemVLA 注入候选 700 条评估

## 目的与用户决定

使用现有 SimpleMemVLA 权重评估 MotionJEPA `v2-eval-0917` 所用的冻结 test/primary 700 条，
不称为官方 RoboMME 16×50 分数。用户要求「所有700个 本机冒烟后都在gl上跑」「binfill任务不送demo」
「注意只能使用hold1和2」，并确认 2000 步、全部回合视频和本运行名。
分支最终更正为 `PolicyEvalThirdParty-simplememvla-0918`，推送到用户的 benchmark 仓库。

## 代码、环境与权重

- 主仓库启动提交由运行根 `manifest.json` 的 `source_commit` 记录；起跑必须主仓库及子模块干净。
- 子模块：`third_party/robomme_benchmark`，仓库 `hongzefu/robomme_benchmark_MotionJEPA`，
  分支 `PolicyEvalThirdParty-simplememvla-0918`，提交 `b4e97f22fe007078e297205898c07c1acbc69165`。
- 参考策略仓库 `v2-eval-0917`：`5a746c6cc1cb16536d49b33466404fc0a1d19805`。
- 工作根 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA`，本机与 GL 共用。
- 复用 `.venv-robomme`、Python 3.10.19、Torch 2.4.1、transformers 5.13.1、SAPIEN 3.0.3，
  ManiSkill 固定现有 fork。benchmark 通过固定源码路径导入，不安装其冲突的整套依赖。
- 权重 `checkpoints/simplememvla_robomme`，来源 HF `yinchenghust/simplememvla_robomme`，
  本轮不换权重。各权重/配置完整 SHA256 在准备时记录、每路启动时复核；各批次检查大小与首尾指纹，
  后者不能检测只修改中间内容且保持长度的变化。运行期间权重冻结。
- 核心改动：候选构造、逐回合账本、流式录像、恢复、进程重启和两路 srun。
  原官方 metadata 评估默认行为保留；受保护目录无改动。

## 固定测试集及配置

唯一入口是子模块 `artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl`。
文件 SHA256：`32bdbdb4931bf32ab5e37613da86ac9f68ab42eb7aec8d9a8d80103af9481809`；
身份 SHA256：`3b4de03a0b4639181f9c68086aea11e9b51ffcb65bbed701a44f573e74e0beb7`。
筛选 `split=test, role=primary`，14 组各 50 条，共 700 条，唯一键 `(task,difficulty,episode)`。

| 任务 | 难度 | 数量 |
|---|---|---:|
| BinFill | easy、medium、hard | 150 |
| RouteStick | easy、medium、hard、xhard | 200 |
| VideoUnmaskSwap | easy、medium、hard、xhard | 200 |
| VideoRepick | easy、medium、xhard | 150 |

`max_steps=2000, execute_horizon=16, num_denoising_steps=10, eval_temperature=1.0, max_subtask_tokens=64`，
`compute_dtype=bfloat16, attn_implementation=sdpa, group_size=1, num_gpus=1`。
每回合在 reset 完成后设置策略随机种子 0；环境 seed/spec 完整读取候选，不换样本。
BinFill 示范任务及示范帧必须为零，正常初始帧仍进入模型；其他任务保留示范。

## 执行顺序与资源

1. 本机 GPU1：BinFill/hard 第一条真实闭环，然后其他 13 组各一条。14 条正常终态且视频可解码才放行 GL。
2. hold-01 `61495429`、hold-02 `61495430`：各 1 GPU/1 CPU/24 GB，使用已有分配，不另交 sbatch。
   每组按 episode 排序后偶数序号给 hold01、奇数序号给 hold02，各 350 条。
3. 每路 25 批，每批 14 组各一条。每批新进程；首批正常完成且内核无新增 OOM kill 后继续。
   RSS 包含共享映射页，不能直接与 cgroup 限额比较；记录 cgroup 的真实上限、峰值与 OOM 事件。
4. 进程启动前设置 `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536`。
   reset 超时 3600 秒、单次推理/step 超时 300 秒，监督进程额外留 30 秒收尾；挂死时终止本轮批次进程组。
5. `srun --overlap --exact --account=chaijy2 --partition=spgpu --gpu_cmode=shared`，
   `/usr/bin/env`、`/usr/bin/bash` 使用绝对路径。保留 Slurm GPU 映射，禁止使用其他 hold。

起跑命令（本机，tmux 会话 `smvla-candidates-0918`）：

```bash
cd /nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA
tmux new-session -d -s smvla-candidates-0918 \
  'set -o pipefail; bash scripts/run_candidate_campaign.sh 2>&1 | stdbuf -oL tee logs/setup/smvla-injection700-nodemo-20260918.campaign.log; echo EXIT_CODE=$? | tee -a logs/setup/smvla-injection700-nodemo-20260918.campaign.log'
```

唯一主编排脚本是 `scripts/run_candidate_campaign.sh`，每路执行器为 `scripts/run_robomme_candidates.sh`。
同名运行根禁止覆盖。中断恢复使用现有计划，显式 `RESUME=1`；正常失败和超时不能挑样本重跑，
运行错误最多同身份尝试两次，仍有错误就停止该路。恢复要重新核对来源/权重指纹。

## 产物及验收

根目录 `logs/robomme_sim/smvla-injection700-nodemo-20260918/`：
`manifest.json`、`plans/`、`local-smoke/`、`hold01/`、`hold02/`、总 `results.json`。
每条在分片目录 `episodes/<task>/<difficulty>/<episode>/attempt-<n>.json` 保存结果，
对应 mp4 保存完整 front+wrist 回放，错误重跑使用新的 attempt 编号，不覆盖旧证据。
各记录含状态、seed、spec 指纹、PID、策略种子、步数、示范帧数、耗时、峰值 RSS、视频 SHA/帧数。

`COVERAGE_PASS planned=700` 且两路退出码 0、700 条正常终态、700 个视频完整解码才算完成。
`success/fail/timeout` 都是正常终态，`error` 与 missing 单列且不缩小分母；完整性通过不等于策略成功。
报告给出每组成功数/50、总成功数/700、宏平均、错误、重跑及耗时。本机冒烟不加入正式统计。
运行中冻结源码与权重；结束后补写 result.md 并归档轻量原始证据，不归档代码、脚本或权重副本。

## 首轮控制器修订与恢复

启动提交为 `170125f`。首轮本机 14 条完整通过；GL 的 16 条正常结果保留，另有一次 Vulkan
reset 错误和一次为修复而主动暂停的在途回合。两路暂停时只取消本轮 step，未取消 hold 分配。

两处控制层修复：JSON 文件显式使用 UTF-8，避免原生库更改 locale 后中文异常写盘失败；
用已核实的 GL cgroup v2 job 层计数取代不正确的 RSS 阈值。内核实测 hold01
`memory.max=25769803776, memory.peak=22243323904, oom=0, oom_kill=0`。
不再用额外重叠 GPU step 做运行中的诊断，避免计算模式干扰；Vulkan 错误根因尚未证明。

恢复只允许控制器、候选持久化模块、编排 shell 和定向测试四个文件变化；模型、环境适配器、
评估循环、权重、候选、配置任一变化都会拒绝沿用结果。完整旧 manifest 单独备份并内嵌到新版，
旧计划和旧回合文件保持原字节，旧/新来源 SHA 均可追溯。并发运行器持锁时禁止修订。

提交修复并确认所有本轮 step 退出后，在本机执行：

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-robomme" UV_LINK_MODE=copy UV_CACHE_DIR="$HOME/.cache/uv"
uv run --frozen --no-sync python -m robomme_sim.candidate_plan revise-controller \
  logs/robomme_sim/smvla-injection700-nodemo-20260918
tmux new-session -d -s smvla-candidates-0918-r1 \
  'set -o pipefail; RESUME=1 bash scripts/run_candidate_campaign.sh 2>&1 | stdbuf -oL tee logs/setup/smvla-injection700-nodemo-20260918.campaign-r1.log; echo EXIT_CODE=$? | tee -a logs/setup/smvla-injection700-nodemo-20260918.campaign-r1.log'
```

恢复跳过全部正常终态；中断回合记入第一次尝试，运行错误仍最多尝试两次，不重置计数。
恢复的 GL 包装日志使用 `-r1` 后缀，不覆盖首轮日志。只跟踪本轮两个 step，不释放 hold。
