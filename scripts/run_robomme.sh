#!/usr/bin/env bash
# RoboMME 闭环评测的两端共用包装（本机 sled-vail / GreatLakes 计算节点）。
#
# 只做三件事，随后把活交给上游的 scripts/eval_robomme.sh：
#   1. 固化本项目的运行口径（venv、PYTHONPATH、ATTN_IMPLEMENTATION=sdpa 等），避免每次
#      手敲十几个环境变量时漏掉一个；
#   2. 抹平两端差异——Vulkan ICD 文件名在本机是 nvidia_icd.json、在 GreatLakes 是
#      nvidia_icd.x86_64.json（SAPIEN 只查固定文件名，所以在集群会误报 ICD 缺失）；
#   3. 起跑前跑一遍 preflight：transformers 版本必须与 checkpoint 的 config.json 一致，
#      以及 GPU compute mode 必须是 Default。前者装错不报错、只静默掉分；后者不对时
#      SAPIEN 会以 vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed 崩掉
#      （解药是 Slurm 侧的 --gpu_cmode=shared，见 AGENTS.md「GreatLakes 评测的增量规则」）。
#
# 用法（全部靠环境变量，与上游脚本的契约一致）：
#   RUN_TAG=robomme_16x1_local NUM_GPUS=2 GPUS=0,1 bash scripts/run_robomme.sh
#   TASKS=PickXtimes NUM_GPUS=1 GPUS=0 RUN_TAG=smoke bash scripts/run_robomme.sh
# 默认口径就是 16 task × 1 episode。
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

VENV="${VENV:-$REPO/.venv-robomme}"
PYBIN="${PYBIN:-$VENV/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO/checkpoints/simplememvla_robomme}"

[ -x "$PYBIN" ] || { echo "找不到解释器：$PYBIN（先建 .venv-robomme）"; exit 2; }
[ -d "$CHECKPOINT" ] || { echo "找不到 checkpoint：$CHECKPOINT"; exit 2; }

# ---- 两端差异：Vulkan ICD 文件名 ----
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for cand in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
              /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$cand" ] && { export VK_ICD_FILENAMES="$cand"; break; }
  done
fi
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export MPLBACKEND="${MPLBACKEND:-Agg}"       # robomme 的 env 模块顶部 import pyplot
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# 上一轮诊断遗留的兼容开关一律不继承（robomme-eval-GL 的教训：这些开关单独用是死路）
unset SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER || true

# ---- 本项目的固定口径 ----
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"   # 本项目不装 flash-attn
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
export GROUP_SIZE="${GROUP_SIZE:-1}"                        # n_req=1 时上游也会压到 1，显式写让日志自解释
export DATASET_SPLIT="${DATASET_SPLIT:-test}"
export NUM_GPUS="${NUM_GPUS:-1}"
export GPUS="${GPUS:-0}"
export RUN_TAG="${RUN_TAG:-robomme_$(date +%Y%m%d_%H%M%S)}"
export CHECKPOINT PYBIN

echo "========== preflight =========="
hostname
nvidia-smi --query-gpu=index,name,compute_mode,memory.total --format=csv,noheader
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<未设>}"
echo "venv=$VENV"
echo "checkpoint=$CHECKPOINT"
echo "RUN_TAG=$RUN_TAG  NUM_GPUS=$NUM_GPUS  GPUS=$GPUS  EPISODES_PER_TASK=$EPISODES_PER_TASK  TASKS=${TASKS:-<全部 16 个>}"

# GPU compute mode：Exclusive_Process 下 SAPIEN 起不来（仅在 Slurm 环境内断言，本机不强制）
if [ -n "${SLURM_JOB_ID:-}" ]; then
  modes="$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u)"
  [ "$modes" = "Default" ] || {
    echo "PREFLIGHT_FAIL: GPU compute mode 为 '$modes'，SAPIEN 需要 Default。"
    echo "  在 srun/salloc/sbatch 上显式加 --gpu_cmode=shared。"
    exit 2
  }
  echo "compute mode: Default (--gpu_cmode=shared 已生效)"
fi

# transformers 版本 vs checkpoint（上游 eval_robomme.sh 没有这个 preflight）
"$PYBIN" - "$CHECKPOINT" <<'PY' || exit 2
import json, sys
import transformers
want = json.load(open(f"{sys.argv[1]}/config.json")).get("transformers_version")
got = transformers.__version__
print(f"transformers: ckpt 要求 {want}，环境为 {got}")
if want and want != got:
    print("PREFLIGHT_FAIL: transformers 版本与 checkpoint 不一致 —— prompt 渲染版本相关，"
          "评测不会报错、只会静默掉分。")
    sys.exit(2)
PY
echo "========== preflight 通过，启动评测 =========="

exec bash "$REPO/scripts/eval_robomme.sh" "$@"
