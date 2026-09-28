#!/usr/bin/env bash
# test-hard 评估的两端共用包装（本机 sled-vail / GreatLakes 计算节点）。环境口径同上次 run_v6spec.sh
# （venv、ICD、sdpa、清兼容开关），入口换成 robomme_sim.testhard_eval；权重显式传绝对路径（benchmark 计划 Codex #4）。
# 用法：OUT=<目录> ROUND=<1|2> SHARD=<i/10> bash scripts/run_testhard.sh [额外参数，如 --resume / --limit 1]
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
PYBIN="${PYBIN:-$REPO/.venv-robomme/bin/python}"
CHECKPOINT="$(readlink -f "${CHECKPOINT:-$REPO/checkpoints/simplememvla_robomme}")"
BENCH="${BENCH:-$REPO/third_party/robomme_benchmark}"
: "${OUT:?}" "${ROUND:?}"
SHARD="${SHARD:-0/10}"
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for cand in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$cand" ] && { export VK_ICD_FILENAMES="$cand"; break; }
  done
fi
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export MPLBACKEND=Agg MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
unset SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER || true
echo "TESTHARD_PREFLIGHT host=$(hostname) bench=$BENCH round=$ROUND shard=$SHARD ckpt=$CHECKPOINT ckpt_config_sha256=$(sha256sum "$CHECKPOINT/config.json" | cut -c1-64)"
nvidia-smi --query-gpu=index,name,compute_mode --format=csv,noheader
exec "$PYBIN" -m robomme_sim.testhard_eval --benchmark_root "$BENCH" --out "$OUT" --round "$ROUND" \
  --shard "$SHARD" --pretrained_checkpoint "$CHECKPOINT" --attn_implementation sdpa --group_size 1 "$@"
