#!/usr/bin/env bash
# 固定候选评估：本机冒烟或已批准的两路 hold，禁止覆盖历史结果。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
command -v uv >/dev/null
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-robomme"
export UV_LINK_MODE=copy UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536
export XDG_CACHE_HOME="$REPO/logs/cache/candidates/xdg"
export HF_HOME="$REPO/logs/cache/candidates/huggingface"
export TORCH_HOME="$REPO/logs/cache/candidates/torch"
export MS_ASSET_DIR="$REPO/logs/cache/candidates/maniskill"
export CUDA_CACHE_PATH="$REPO/logs/cache/candidates/cuda"
export TRITON_CACHE_DIR="$REPO/logs/cache/candidates/triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO/logs/cache/candidates/inductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER
for candidate_icd in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$candidate_icd" ]; then export VK_ICD_FILENAMES="$candidate_icd"; break; fi
done
SUITE="${SUITE:?必须提供冻结计划目录 SUITE}"
LANE="${LANE:?必须指定 local-smoke、hold01 或 hold02}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    case "$SLURM_JOB_ID:$LANE" in
        61495429:hold01|61495430:hold02) ;;
        *) echo "未经批准的作业/分片组合"; exit 2 ;;
    esac
    [ "${CUDA_VISIBLE_DEVICES:-}" != "" ] || { echo "Slurm 未提供 GPU 映射"; exit 2; }
else
    [ "$LANE" = local-smoke ] || { echo "正式 700 条只能在 GL 运行"; exit 2; }
    export CUDA_VISIBLE_DEVICES="${GPUS:?本机冒烟必须指定一张空闲 GPU}"
fi
# 仅检查分配给本进程的卡，不能把同节点其他作业的 compute mode 当成本任务状态。
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,compute_mode,memory.total --format=csv,noheader
modes="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=compute_mode --format=csv,noheader | sort -u)"
[ "$modes" = Default ] || { echo "GPU 必须是 Default 模式"; exit 2; }
apps="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
[ -z "$apps" ] || { echo "指定 GPU 已有计算进程：$apps"; exit 2; }
uv run --frozen --no-sync python scripts/check/verify_env.py "$REPO/checkpoints/simplememvla_robomme"
uv run --frozen --no-sync python scripts/check/render_probe.py
extra=()
if [ "${RESUME:-0}" = 1 ]; then extra+=(--resume); fi
exec uv run --frozen --no-sync python -m robomme_sim.candidate_launcher "$SUITE" "$LANE" "${extra[@]}"
