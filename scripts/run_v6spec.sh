#!/usr/bin/env bash
# V6 冻结快照评估的两端共用包装（本机 sled-vail / GreatLakes 计算节点）。
# 环境口径与 scripts/run_robomme.sh 相同（venv、ICD、sdpa、清兼容开关），入口换成 robomme_sim.v6spec_eval。
# 用法：SPECS=<specs.jsonl> BENCH=<benchmark 根> OUT=<目录> SHARD=0/12 bash scripts/run_v6spec.sh [额外参数]
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
PYBIN="${PYBIN:-$REPO/.venv-robomme/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/simplememvla_robomme}"
: "${SPECS:?}" "${BENCH:?}" "${OUT:?}"
SHARD="${SHARD:-0/1}"
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for cand in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$cand" ] && { export VK_ICD_FILENAMES="$cand"; break; }
  done
fi
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export MPLBACKEND=Agg MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
unset SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER || true
echo "V6SPEC_PREFLIGHT host=$(hostname) specs=$SPECS bench=$BENCH shard=$SHARD"
nvidia-smi --query-gpu=index,name,compute_mode --format=csv,noheader
exec "$PYBIN" -m robomme_sim.v6spec_eval --specs "$SPECS" --benchmark_root "$BENCH" --out "$OUT" \
  --shard "$SHARD" --pretrained_checkpoint "$CHECKPOINT" --attn_implementation sdpa \
  --group_size 1 "$@"
