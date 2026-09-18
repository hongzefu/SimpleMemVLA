#!/usr/bin/env bash
# 在 GreatLakes 计算节点上真正执行的运行器（由 scripts/gl_run_robomme.sh 经 srun 调起）。
#
# 为什么单独成文件而不是塞进 srun 的命令串：命令串要经过 heredoc 拼接 → srun 参数传递 →
# bash -c 解析三道关，任何一处的引号/续行符转义都会变成难查的 exit 127。脚本文件在 NFS 上
# 两端可见，srun 直接执行，零转义；也便于单独 bash -n 和 DRY_RUN 验证。
# 这也是 AgentMetaRules 的 job.sbatch 模板的思路：提交层只做判定 + 交棒给运行器。
#
# 入参全走环境变量：
#   RUN_TAG    必填，评测的 run 名
#   PROBE_ONLY 非 0 时只跑三级探针，不跑评测
#   DRY_RUN    非 0 时只打印将要执行的评测命令，不真跑（用于快速验证 srun 链路）
#   TASKS / EPISODES_PER_TASK 等透传给 run_robomme.sh
set -euo pipefail

REPO="${REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
RUN_TAG="${RUN_TAG:?必须指定 RUN_TAG}"
cd "$REPO"

# 候选路径有自己的来源、资源和原生渲染检查，保留 Slurm 的 GPU 映射。
if [ -n "${CANDIDATE_SUITE:-}" ]; then
  case "$(hostname)" in gl*) ;; *) echo "不是 GL 计算节点"; exit 2 ;; esac
  [ ! -e /data/hongzefu ] || { echo "计算节点环境判据冲突"; exit 2; }
  export SUITE="$CANDIDATE_SUITE" LANE="${CANDIDATE_LANE:?必须指定分片}"
  exec /usr/bin/bash "$REPO/scripts/run_robomme_candidates.sh"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg
export MS_ASSET_DIR="$HOME/.maniskill"
# 集群的系统 ICD 带架构后缀，SAPIEN 的默认文件名检查识别不了，必须显式指定
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
fi
# 不继承上一轮诊断遗留的兼容开关（robomme-eval-GL 的教训）
unset SAPIEN_DISABLE_RAY_TRACING ROBOMME_GPU_RASTER || true

echo "===== 一、运行环境判定（AGENTS.md 第 0 条）====="
hostname
echo "repo=$REPO"
for p in /nfs/turbo/coe-chaijy-unreplicated/hongzefu /data/hongzefu; do
  printf '%s: %s\n' "$p" "$([ -e "$p" ] && echo 存在 || echo 不存在)"
done
nvidia-smi --query-gpu=name,compute_mode,memory.total --format=csv,noheader
ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo "（无 ICD 目录）"

case "$(hostname)" in
  gl*) ;;
  *) echo "判定失败：主机名不是 gl*，这不是 GreatLakes 计算节点"; exit 2 ;;
esac
[ ! -e /data/hongzefu ] || { echo "判定失败：计算节点不该看得见 /data/hongzefu"; exit 2; }
modes="$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u)"
[ "$modes" = "Default" ] || {
  echo "判定失败：GPU compute mode=$modes，SAPIEN 需要 Default。"
  echo "  在 srun/salloc/sbatch 上显式加 --gpu_cmode=shared。"
  exit 2
}
echo "环境 B：GreatLakes 计算节点，compute mode=Default ✓"

echo "===== 二、依赖复用校验（NFS 上的 .venv-robomme 在计算节点是否可用）====="
command -v uv >/dev/null
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-robomme" UV_LINK_MODE=copy UV_CACHE_DIR="$HOME/.cache/uv"
uv run --frozen --no-sync python "$REPO/scripts/check/verify_env.py" \
  "$REPO/checkpoints/simplememvla_robomme"

echo "===== 三、SAPIEN 离屏渲染探针 ====="
uv run --frozen --no-sync python "$REPO/scripts/check/render_probe.py"

if [ "${PROBE_ONLY:-0}" != 0 ]; then
  echo "PROBE_ONLY=1，三级探针全部通过，不跑评测。"
  exit 0
fi

echo "===== 四、RoboMME 评测（单卡串行，RUN_TAG=$RUN_TAG）====="
if [ "${DRY_RUN:-0}" != 0 ]; then
  echo "DRY_RUN=1，将要执行："
  echo "  RUN_TAG=$RUN_TAG NUM_GPUS=1 GPUS=0 bash $REPO/scripts/run_robomme.sh"
  exit 0
fi

export RUN_TAG NUM_GPUS=1 GPUS=0
exec /usr/bin/bash "$REPO/scripts/run_robomme.sh"
