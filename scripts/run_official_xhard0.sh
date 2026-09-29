#!/usr/bin/env bash
# 官方评估路线（robomme_sim.eval_success，--dataset_split test，vendored 官方 robomme，不含 robomme_hard）跑 xhard0 清单的一片。
# 计算节点上由 gl_run_official_xhard0.sh 经 srun 调用，也可在本机直接跑。
# 用法：MANIFEST=<清单.jsonl> SHARD=<i/n> OUTDIR=<目录> [VIDEO_DIR=<录像目录>] bash scripts/run_official_xhard0.sh [额外参数，如 --resume]
# 推理参数取官方 scripts/eval_robomme.sh 默认值（execute_horizon 16、max_steps 1300、去噪 10 步、温度 1.0、
# 子任务 64 token、bfloat16），组大小固定为 1（清单模式逐局评估，--group_size 1 只作显式标注）；
# 注意力后端默认 sdpa（.venv-robomme 未装 flash_attn，且与 test-hard 路线一致），可用 ATTN_IMPLEMENTATION 覆盖。
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# 本检出（git worktree）不带 venv 与权重：默认借主检出的
MAIN_REPO="${MAIN_REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
PYBIN="${PYBIN:-$MAIN_REPO/.venv-robomme/bin/python}"
CHECKPOINT="$(readlink -f "${CHECKPOINT:-$MAIN_REPO/checkpoints/simplememvla_robomme}")"
: "${MANIFEST:?}" "${SHARD:?}" "${OUTDIR:?}"
VIDEO_DIR="${VIDEO_DIR-$OUTDIR/videos}"
i="${SHARD%/*}"; n="${SHARD#*/}"
EPISODE_LOG="$OUTDIR/episodes-shard$(printf %02d "$i")of$(printf %02d "$n").jsonl"
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for cand in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$cand" ] && { export VK_ICD_FILENAMES="$cand"; break; }
  done
fi
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export MPLBACKEND=Agg MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# 以下沿用官方 eval_robomme.sh 的环境设定
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/torchinductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$OUTDIR"
unset __EGL_VENDOR_LIBRARY_DIRS SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER || true
echo "XHARD0_PREFLIGHT host=$(hostname) repo=$REPO head=$(git -C "$REPO" rev-parse --short HEAD) shard=$SHARD manifest=$MANIFEST ckpt=$CHECKPOINT ckpt_config_sha256=$(sha256sum "$CHECKPOINT/config.json" | cut -c1-64) log=$EPISODE_LOG video_dir=${VIDEO_DIR:-<关闭>}"
nvidia-smi --query-gpu=index,name,compute_mode --format=csv,noheader || true
exec "$PYBIN" -m robomme_sim.eval_success \
  --pretrained_checkpoint "$CHECKPOINT" \
  --dataset_split test \
  --group_size 1 \
  --execute_horizon "${EXECUTE_HORIZON:-16}" \
  --max_steps "${MAX_STEPS:-1300}" \
  --num_denoising_steps "${NUM_DENOISING_STEPS:-10}" \
  --eval_temperature "${EVAL_TEMPERATURE:-1.0}" \
  --max_subtask_tokens "${MAX_SUBTASK_TOKENS:-64}" \
  --compute_dtype "${COMPUTE_DTYPE:-bfloat16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --num_gpus 1 \
  --episode_manifest "$MANIFEST" --shard "$SHARD" --episode_log "$EPISODE_LOG" \
  ${VIDEO_DIR:+--video_dir "$VIDEO_DIR"} \
  "$@"
