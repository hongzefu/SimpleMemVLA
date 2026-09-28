# SimpleMemVLA test-hard 评估（2026-09-28）：启动与配置

## 目的

在 benchmark 拆包后的 `robomme_hard`（`dataset="test-hard"`）上评估官方 SimpleMemVLA，用与上次评估（本仓库 `aab093f` 留档 `docs/eval-doc/v6xhard-0927/`）完全相同的 1100 个身份和按档步数上限，便于直接对比（benchmark 0927 计划 U-5、U-6、U-8、U-9）。

## 版本与代码状态

- 本仓库：分支 `testhard-eval-0928-0134`，从官方 `OpenBMB/SimpleMemVLA@c564c17d276d7294200122b286c21901a3bfb99f` 切出；接入提交 `f229e6a`，`--resume` 修复 `3534e2d`（两轮均在 `3534e2d` 上跑）。
- 子模块 `third_party/robomme_benchmark` → `hongzefu/robomme_benchmark_MotionJEPA@31e61259f9885f7dba8a321d0b6e699eb5061866`（benchmark 12.210，分支 `PolicyEvalThirdParty-simplememvla-0928-0311`）。
- 权重：`checkpoints/simplememvla_robomme`（`config.json` sha256 `5c5c1506cd7b6120c2e82a456b509e0ff92ab02862f188b22317b5aa8719bc5d`），启动时解析为绝对路径显式传入。
- 环境：`.venv-robomme`（cpython 3.10.19），benchmark 经 `sys.path` 引入。

## 身份、轮次与分片

- 身份由 `robomme_hard` builder 给出：每任务 episode 0..79（xhard4-only 的 StopCube／InsertPeg／MoveCube 为 0..19），档序主序、档内 candidate 升序。
- 第一轮取每格 candidate 最小的 10 局，第二轮取其余 10 局：55 格 × 10 局 × 2 轮 = 1100。
- `--shard i/10`：该轮身份按 `(task, episode)` 排序后下标 ≡ i (mod 10)，每片 55 局。
- 步数上限按档 `TIER_MAX_STEPS = {xhard1: 1500, xhard2: 1700, xhard3: 2000, xhard4: 2600}`，逐局核「环境阈值 = 服务返回 = 策略循环上限 = 落盘值」。

## 硬件与命令

GreatLakes spgpu，10 个评估占位 job `62177614`～`62177623`（各 1 GPU A40／1 CPU／32 G／48 h），片 i 在 job `62177614+i`：

```bash
# 登录节点 tmux：hs-eval-smvla-r<轮>-s<i>
bash scripts/gl_run_testhard.sh <62177614+i> <i>/10 testhard-0928 --round <1|2>
# 内部：srun --jobid=<job> --gpu_cmode=shared --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
#        run_testhard.sh --resume（最多 3 遍；每片每轮基础设施重跑上限 55）
```

冒烟（BinFill@xhard1 候选 0，job 62177614）：`EVAL_SMOKE=PASS policy=simplememvla … status=fail steps=1014 max_steps=1500 wall_s=161.0 rss_gb=25.4 binding_available=1 injected_mismatch=0`。

## 时间线（EDT）

第一轮 03:35～05:25；第二轮 07:02～08:56。
