#!/usr/bin/env bash
# 在已 Running 的占位 job 里塞一片 V6 快照评估（登录节点执行，建议放在登录节点 tmux 里）。
# 用法：bash scripts/gl_run_v6spec.sh <JOBID> <i/n> <RUN_TAG> [额外参数，如 --limit 1 / --resume]
# 规格与 gl_run_robomme.sh 相同：1 CPU、1 GPU、--gpu_cmode=shared、--overlap --exact。
set -euo pipefail
JOBID="${1:?必须指定已 Running 的 JOBID}"
SHARD="${2:?必须指定分片 i/n}"
RUN_TAG="${3:?必须指定 RUN_TAG}"
shift 3
REPO="${REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
BENCH="$REPO/third_party/robomme_benchmark"
SPECS="${SPECS:-$BENCH/scripts/configs/newtask-v6/smvla-0927/xhard1/specs.reselected.jsonl}"
OUT="$REPO/logs/v6spec/$RUN_TAG"
i="${SHARD%/*}"
LOG="$OUT/shard$(printf %02d "$i").log"
mkdir -p "$OUT"
echo "[gl_run_v6spec] jobid=$JOBID shard=$SHARD specs=$SPECS log=$LOG" | tee -a "$LOG"
set +e
# 片内 error 身份会在下一轮 --resume 里重跑（v6spec_eval 按 --max_attempts 截止）；进程级崩溃同样靠这一轮续上
for round in 1 2 3; do
echo "V6SPEC_ROUND $round" | tee -a "$LOG"
srun --jobid="$JOBID" --account=chaijy2 --partition=spgpu --gpu_cmode=shared \
     --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-node=1 \
     --chdir="$REPO" \
     /usr/bin/env SPECS="$SPECS" BENCH="$BENCH" OUT="$OUT" SHARD="$SHARD" \
     /usr/bin/bash "$REPO/scripts/run_v6spec.sh" --resume "$@" 2>&1 | stdbuf -oL tee -a "$LOG"
status=${PIPESTATUS[0]}
[ "$status" = 0 ] && break
done
echo "EXIT_CODE=$status" | tee -a "$LOG"
exit "$status"
