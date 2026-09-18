#!/usr/bin/env bash
# 在 GreatLakes 一个**已经 Running** 的分配内起 RoboMME 评测（在登录节点执行）。
# 本脚本只负责 srun 交棒，计算节点上真正跑的是 scripts/gl_remote_body.sh。
#
# 为什么是 srun --overlap 而不是 sbatch：本轮要复用用户已经占住的 gpu-hold-* job，
# 不另行排队（spgpu 当前空闲 A40 为 0，重新提交会 PENDING 很久）。
#
# 三个必须照做的点（全部来自 robomme-eval-GL 的实测，见 AGENTS.md「GreatLakes 评测的增量规则」）：
#   1. --account / --partition / --gpu_cmode 要在 srun 上**重复写一遍**，否则这个 step 拿不到
#      shared 计算模式，SAPIEN 会以 vk::PhysicalDevice::createDeviceUnique:
#      ErrorInitializationFailed 崩掉（A40 默认 Exclusive_Process）。
#   2. env / bash 写全路径 /usr/bin/env、/usr/bin/bash —— srun 会把 env 解析成登录节点的
#      ~/.local/bin/env，计算节点上报 Permission denied、退出码 13。
#   3. --cpus-per-task / --gpus-per-node 不能超过目标 job 自己的分配（hold job 是 1 GPU / 1 CPU），
#      超了 srun 会一直等不到资源。
#
# 用法（在登录节点）：
#   bash scripts/gl_run_robomme.sh <JOBID> <RUN_TAG>
#   PROBE_ONLY=1 bash scripts/gl_run_robomme.sh <JOBID> probe   # 只跑三级探针
#   DRY_RUN=1    bash scripts/gl_run_robomme.sh <JOBID> dry     # 只验证 srun 链路
set -euo pipefail

JOBID="${1:?必须指定已 Running 的 JOBID}"
RUN_TAG="${2:?必须指定 RUN_TAG}"

REPO="${REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
CPUS="${CPUS:-1}"                    # hold job 只分到 1 CPU，不能要更多
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
LOG="$REPO/logs/setup/${RUN_TAG}.gl.log"
mkdir -p "$(dirname "$LOG")"

echo "[gl_run] jobid=$JOBID run_tag=$RUN_TAG log=$LOG"
set +e
# stdbuf -oL 让 tee 行缓冲——否则整份日志（只有几 KB）会卡在 tee 的 4KB 块缓冲里，
# 直到进程结束才一次性落盘，盯日志的 Monitor 全程静默。
srun --jobid="$JOBID" --account="$ACCOUNT" --partition="$PARTITION" --gpu_cmode=shared \
     --overlap --exact --nodes=1 --ntasks=1 \
     --cpus-per-task="$CPUS" --gpus-per-node="$GPUS_PER_NODE" \
     --chdir="$REPO" \
     /usr/bin/env \
       REPO="$REPO" \
       RUN_TAG="$RUN_TAG" \
       PROBE_ONLY="${PROBE_ONLY:-0}" \
       DRY_RUN="${DRY_RUN:-0}" \
       TASKS="${TASKS:-}" \
       EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}" \
     /usr/bin/bash "$REPO/scripts/gl_remote_body.sh" 2>&1 | stdbuf -oL tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT_CODE=$status" | tee -a "$LOG"
exit "$status"
