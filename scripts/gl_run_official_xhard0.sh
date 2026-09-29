#!/usr/bin/env bash
# 在已 Running 的占位 job 里塞一片官方路线 xhard0 评估（登录节点执行，放在登录节点 tmux 里）。
# 用法：
#   MANIFEST=<清单.jsonl> [VIDEO_DIR=<录像目录>] \
#     bash scripts/gl_run_official_xhard0.sh <JOBID> <i/n> <OUTDIR> [额外参数，透传给 eval_success]
# - MANIFEST（必填）：每行 {task, source_episode, seed, shard}；本片取 shard==i 的行。
# - OUTDIR：逐局日志 episodes-shard<ii>of<nn>.jsonl 与控制台日志 shard<ii>.log 所在目录；
#   录像默认 $OUTDIR/videos，VIDEO_DIR 可改，设为 none 关闭。路径都在登录节点解析成绝对路径后经 /usr/bin/env 显式传入 srun。
# 规格：4 CPU、1 GPU、--gpu_cmode=shared、--overlap --exact。最多 3 遍，每遍带 --resume：
# 逐局日志里已有终态（success/fail/timeout）的 (task, source_episode) 跳过，只重跑 error 局。
# 退出码 0 全部有终态即停；2 为配置错误（参数、清单、种子核对不过）即停，不再续。
set -euo pipefail
JOBID="${1:?必须指定已 Running 的 JOBID}"
SHARD="${2:?必须指定分片 i/n}"
OUTDIR="${3:?必须指定 OUTDIR}"
shift 3
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${MANIFEST:?必须设 MANIFEST=<xhard0 清单 jsonl>}"
MANIFEST="$(readlink -f "$MANIFEST")"
[ -f "$MANIFEST" ] || { echo "清单不存在：$MANIFEST"; exit 2; }
OUTDIR="$(readlink -m "$OUTDIR")"
VIDEO_DIR="${VIDEO_DIR:-$OUTDIR/videos}"
if [ "$VIDEO_DIR" = none ]; then VIDEO_DIR=""; else VIDEO_DIR="$(readlink -m "$VIDEO_DIR")"; fi
i="${SHARD%/*}"
LOG="$OUTDIR/shard$(printf %02d "$i").log"
mkdir -p "$OUTDIR"
echo "[gl_run_official_xhard0] jobid=$JOBID shard=$SHARD repo=$REPO manifest=$MANIFEST video_dir=${VIDEO_DIR:-<关闭>} log=$LOG" | tee -a "$LOG"
set +e
for attempt in 1 2 3; do
  echo "XHARD0_PASS $attempt" | tee -a "$LOG"
  srun --jobid="$JOBID" --gpu_cmode=shared --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
       --chdir="$REPO" /usr/bin/env MANIFEST="$MANIFEST" SHARD="$SHARD" OUTDIR="$OUTDIR" VIDEO_DIR="$VIDEO_DIR" \
       /usr/bin/bash "$REPO/scripts/run_official_xhard0.sh" --resume "$@" 2>&1 | stdbuf -oL tee -a "$LOG"
  status=${PIPESTATUS[0]}
  [ "$status" = 0 ] && break
  [ "$status" = 2 ] && break
done
echo "EXIT_CODE=$status" | tee -a "$LOG"
exit "$status"
