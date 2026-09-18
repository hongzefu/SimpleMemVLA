#!/usr/bin/env bash
# 在 GreatLakes 一个**已经 Running** 的分配内起 RoboMME 评测（在登录节点执行）。
#
# 为什么是 srun --overlap 而不是 sbatch：本轮要复用用户已经占住的 gpu-hold-* job，
# 不另行排队（spgpu 当前空闲 A40 为 0，重新提交会 PENDING 很久）。
#
# 三个必须照做的点（全部来自 robomme-eval-GL 的实测，见 AGENTS.md「GreatLakes 评测的增量规则」）：
#   1. --account / --partition / --gpu_cmode 要在 srun 上**重复写一遍**，否则这个 step 拿不到
#      shared 计算模式，SAPIEN 会以 vk::PhysicalDevice::createDeviceUnique:
#      ErrorInitializationFailed 崩掉（A40 默认 Exclusive_Process）。
#   2. env / bash 写全路径 /usr/bin/env、/usr/bin/bash —— srun 会把 env 解析成登录节点的
#      ~/.local/bin/env，计算节点上报 Permission denied、退出码 13。
#   3. --cpus-per-task / --mem 不能超过目标 job 自己的分配（hold job 是 1 GPU / 1 CPU / 24G），
#      超了 srun 会一直等不到资源。
#
# 用法（在登录节点）：
#   bash scripts/gl_run_robomme.sh <JOBID> <RUN_TAG> [额外传给 run_robomme.sh 的参数...]
# 例：
#   bash scripts/gl_run_robomme.sh 61495429 gl-61495429-16x1
#   PROBE_ONLY=1 bash scripts/gl_run_robomme.sh 61495429 probe   # 只跑三级探针，不跑评测
set -euo pipefail

JOBID="${1:?必须指定已 Running 的 JOBID}"
RUN_TAG="${2:?必须指定 RUN_TAG}"
shift 2 || true

REPO="${REPO:-/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA}"
ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
CPUS="${CPUS:-1}"          # hold job 只分到 1 CPU，不能要更多
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
LOG="$REPO/logs/setup/${RUN_TAG}.gl.log"
mkdir -p "$(dirname "$LOG")"

# 计算节点侧要跑的整段脚本。先三级探针，全绿才跑评测。
REMOTE_CMD=$(cat <<REMOTE
set -o pipefail
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg
export MS_ASSET_DIR="\$HOME/.maniskill"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
[ -f "\$VK_ICD_FILENAMES" ] || unset VK_ICD_FILENAMES

echo "===== 一、运行环境判定（AGENTS.md 第 0 条）====="
hostname
echo "repo=\$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null)"
for p in /nfs/turbo/coe-chaijy-unreplicated/hongzefu /data/hongzefu; do
  printf '%s: %s\n' "\$p" "\$([ -e "\$p" ] && echo 存在 || echo 不存在)"
done
nvidia-smi --query-gpu=name,compute_mode,memory.total --format=csv,noheader
ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo "（无 ICD 目录）"
case "\$(hostname)" in gl*) ;; *) echo "判定失败：主机名不是 gl*"; exit 2;; esac
[ ! -e /data/hongzefu ] || { echo "判定失败：计算节点不该看得见 /data/hongzefu"; exit 2; }
modes="\$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u)"
[ "\$modes" = "Default" ] || { echo "判定失败：compute mode=\$modes，需要 --gpu_cmode=shared"; exit 2; }
echo "环境 B：GreatLakes 计算节点，compute mode=Default ✓"

echo "===== 二、依赖复用校验（NFS 上的 .venv-robomme 在计算节点是否可用）====="
"$REPO/.venv-robomme/bin/python" "$REPO/scripts/check/verify_env.py" \
  "$REPO/checkpoints/simplememvla_robomme" || exit 3

echo "===== 三、SAPIEN 离屏渲染探针 ====="
"$REPO/.venv-robomme/bin/python" "$REPO/scripts/check/render_probe.py" || exit 4

if [ "\${PROBE_ONLY:-0}" != 0 ]; then
  echo "PROBE_ONLY=1，三级探针全部通过，不跑评测。"
  exit 0
fi

echo "===== 四、RoboMME 16 task × 1 episode（单卡串行）====="
RUN_TAG="$RUN_TAG" NUM_GPUS=1 GPUS=0 \\
  /usr/bin/bash "$REPO/scripts/run_robomme.sh" $*
REMOTE
)

echo "[gl_run] jobid=$JOBID run_tag=$RUN_TAG log=$LOG"
set +e
srun --jobid="$JOBID" --account="$ACCOUNT" --partition="$PARTITION" --gpu_cmode=shared \
     --overlap --exact --nodes=1 --ntasks=1 \
     --cpus-per-task="$CPUS" --gpus-per-node="$GPUS_PER_NODE" \
     --chdir="$REPO" \
     /usr/bin/env PROBE_ONLY="${PROBE_ONLY:-0}" \
     /usr/bin/bash -c "$REMOTE_CMD" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT_CODE=$status" | tee -a "$LOG"
exit "$status"
