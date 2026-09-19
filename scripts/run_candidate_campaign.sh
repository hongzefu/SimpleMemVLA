#!/usr/bin/env bash
# 已批准的完整执行顺序：本机 14 条冒烟 → 两个已有 hold → 700 条合并验收。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ "$(hostname)" = sled-vail ] || { echo "必须在 sled-vail 发起"; exit 2; }
command -v uv >/dev/null
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-robomme" UV_LINK_MODE=copy UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export RUN_TAG=smvla-injection700-nodemo-20260918
export SUITE="$REPO/logs/robomme_sim/$RUN_TAG"
export RESUME="${RESUME:-0}"
case "$RESUME" in 0|1) ;; *) echo "RESUME 只能为 0 或 1"; exit 2 ;; esac
RESTART_SUFFIX=""
if [ "$RESUME" = 1 ]; then RESTART_SUFFIX="-resume-$(date +%Y%m%dT%H%M%S)"; fi
ssh -O check greatlakes
if [ "$RESUME" = 0 ]; then
    uv run --frozen --no-sync python -m robomme_sim.candidate_plan prepare "$SUITE"
fi
LANE=local-smoke GPUS=1 bash scripts/run_robomme_candidates.sh
uv run --frozen --no-sync python -m robomme_sim.candidate_plan summarize "$SUITE" --lane local-smoke
# 两路各启动一个 srun step。路径和名称固定，无用户文本拼接到远程 shell。
ssh greatlakes "/usr/bin/env RESUME=$RESUME CANDIDATE_SUITE=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/logs/robomme_sim/smvla-injection700-nodemo-20260918 CANDIDATE_LANE=hold01 /usr/bin/bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/scripts/gl_run_robomme.sh 61495429 smvla-injection700-nodemo-20260918-hold01$RESTART_SUFFIX" &
HOLD01_PID=$!
ssh greatlakes "/usr/bin/env RESUME=$RESUME CANDIDATE_SUITE=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/logs/robomme_sim/smvla-injection700-nodemo-20260918 CANDIDATE_LANE=hold02 /usr/bin/bash /nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/scripts/gl_run_robomme.sh 61495430 smvla-injection700-nodemo-20260918-hold02$RESTART_SUFFIX" &
HOLD02_PID=$!
CODE01=0
CODE02=0
wait "$HOLD01_PID" || CODE01=$?
wait "$HOLD02_PID" || CODE02=$?
echo "HOLD_EXIT hold01=$CODE01 hold02=$CODE02"
uv run --frozen --no-sync python -m robomme_sim.candidate_plan summarize "$SUITE"
[ "$CODE01" = 0 ] && [ "$CODE02" = 0 ]
echo "CAMPAIGN_COMPLETE total=700（环境error保留在结果中，不等于无错误通过）"
