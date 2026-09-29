#!/usr/bin/env bash
# 在已 Running 的占位 job 里塞一片 test-hard 评估（登录节点执行，放在登录节点 tmux 里）。v7 口径。
# 用法：
#   IDENTITIES=<身份清单.jsonl> [VIDEO_DIR=<录像目录>] \
#     bash scripts/gl_run_testhard.sh <JOBID> <i/10> <RUN_TAG> --round <1|2> [额外参数，如 --limit 1]
# - IDENTITIES（必填）：v7 身份清单，按 round/shard 选局；透传为 testhard_eval --identities。
# - VIDEO_DIR（可选，默认 $OUT/videos）：每局录像目录，透传为 --video_dir；设为 none 可关闭录像。
#   两者都在登录节点解析成绝对路径后经 /usr/bin/env 显式传进 srun 与 run_testhard.sh，不依赖 srun 的环境继承。
# 规格：4 CPU、1 GPU、--gpu_cmode=shared、--overlap --exact。片内 error 身份靠下一遍 --resume 重跑，最多 3 遍，
# 受 testhard_eval 的每片每轮 55 次重跑上限约束（用满打 RETRY_CAP_HIT 退出码 4，不再续）；
# 退出码 2（参数错误、身份清单或 builder 身份核对不过）同样不再续。
set -euo pipefail
JOBID="${1:?必须指定已 Running 的 JOBID}"
SHARD="${2:?必须指定分片 i/10}"
RUN_TAG="${3:?必须指定 RUN_TAG}"
shift 3
[ "${1:-}" = "--round" ] || { echo "第四个参数必须是 --round <1|2>"; exit 2; }
ROUND="$2"; shift 2
REPO="${REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
OUT="$REPO/logs/testhard/$RUN_TAG"
: "${IDENTITIES:?必须设 IDENTITIES=<v7 身份清单 jsonl>}"
IDENTITIES="$(readlink -f "$IDENTITIES")"
[ -f "$IDENTITIES" ] || { echo "身份清单不存在：$IDENTITIES"; exit 2; }
VIDEO_DIR="${VIDEO_DIR:-$OUT/videos}"
if [ "$VIDEO_DIR" = none ]; then VIDEO_DIR=""; else VIDEO_DIR="$(readlink -m "$VIDEO_DIR")"; fi
i="${SHARD%/*}"
LOG="$OUT/r${ROUND}-shard$(printf %02d "$i").log"
mkdir -p "$OUT"
echo "[gl_run_testhard] jobid=$JOBID round=$ROUND shard=$SHARD identities=$IDENTITIES video_dir=${VIDEO_DIR:-<关闭>} log=$LOG" | tee -a "$LOG"
set +e
for attempt in 1 2 3; do
  echo "TESTHARD_PASS $attempt" | tee -a "$LOG"
  srun --jobid="$JOBID" --gpu_cmode=shared --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
       --chdir="$REPO" /usr/bin/env OUT="$OUT" ROUND="$ROUND" SHARD="$SHARD" \
       IDENTITIES="$IDENTITIES" VIDEO_DIR="$VIDEO_DIR" \
       /usr/bin/bash "$REPO/scripts/run_testhard.sh" --resume "$@" 2>&1 | stdbuf -oL tee -a "$LOG"
  status=${PIPESTATUS[0]}
  [ "$status" = 0 ] && break
  [ "$status" = 4 ] && break
  [ "$status" = 2 ] && break
done
echo "EXIT_CODE=$status" | tee -a "$LOG"
exit "$status"
