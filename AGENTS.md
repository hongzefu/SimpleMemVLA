# AGENTS.md

本文件规定 agent 在本仓库（SimpleMemVLA）中的工作方式。所有仓库任务都必须先遵守本文件，再结合用户当前明确指令确定本轮范围。

**通用规则引用**：本仓库的通用工作规则以 [`AgentMetaRules-hongzefu`](https://github.com/hongzefu/AgentMetaRules-hongzefu) 的 `AGENTS.md` 为正本，引用 commit `a57506e35b509651720a35a13f75fb4dfb2813a5`（回流时更新）。本文件只写：① 项目专属的判据表与占位符取值；② 对正本条目的**显式覆盖项**（按正本条号引用）；③ 项目 scope 与规则来源。优先级：系统 / 开发者 / 用户当前指令 > 本文件明确写出的覆盖项 > 正本；本文件未覆盖处以正本为准。

## 本仓库是什么

SimpleMemVLA = Qwen3.5 VLM（产出 sub-task 文本）+ DiT flow-matching action head，带长视觉历史窗口，在五个 memory-centric 机器人操作 benchmark 上训练与评测：

| benchmark | 目录 | 仿真栈 | 官方口径 |
|---|---|---|---|
| RMBench | `rmbench_sim/` | SAPIEN `3.0.0b1` + CuRobo | 10 task × 100 seed |
| RoboMME | `robomme_sim/` | ManiSkill fork + SAPIEN stable `3.0.x` | **16 task × 50 episode** |
| MIKASA-Robo | `mikasa_sim/` | mani_skill `3.0.0b15` + SAPIEN `3.0.0b1` | 5 task × 100 episode |
| RoboMemArena | `robomemarena_sim/` | robosuite `1.4.0` + mujoco `2.3.2` + LIBERO fork | 26 task × 51 trial |
| LIBERO | `libero_sim/` | robosuite + mujoco + 上游 LIBERO | 4 suite × 10 task × 50 init state |

**一个 benchmark 一个 venv**，这不是洁癖而是硬冲突：RMBench/MIKASA 要 SAPIEN beta `3.0.0b1`，RoboMME 要 stable `3.0.x`；LIBERO 与 RoboMemArena 需要两份互斥的 `libero` 包（前者往 site-packages 写 `zz_libero_repo.pth` 指向 `third_party/LIBERO`，后者硬校验 `evaluation_benchmark/libero_fork`）。venv 命名 `.venv-<benchmark>`。

## 运行环境判定（每次开工第一步）

按正本第 0 条执行。判定命令（只读，可整段贴）：

```bash
echo "repo=$(git rev-parse --show-toplevel 2>/dev/null)"
hostname
for p in /nfs/turbo/coe-chaijy-unreplicated/hongzefu /data/hongzefu ~/.ssh/config; do
  printf '%s: %s\n' "$p" "$([ -e "$p" ] && echo 存在 || echo 不存在)"
done
nvidia-smi --query-gpu=name,compute_mode --format=csv,noheader 2>/dev/null | sort | uniq -c
command -v sbatch >/dev/null && echo "sbatch: 有" || echo "sbatch: 无"
ls /usr/share/vulkan/icd.d/ 2>/dev/null
```

判据表（同一行的两列互斥；口径日期 2026-09-18）：

| 判据 | 环境 A：本机 sled-vail | 环境 B：GreatLakes 计算节点 |
|---|---|---|
| 主机名（`hostname` 前缀） | `sled-vail` | `gl*`（如 `gl1519.arc-ts.umich.edu`） |
| 仓库根 | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA` | 同左（**两端共用同一份副本**） |
| 共享存储路径 `/nfs/turbo/coe-chaijy-unreplicated/hongzefu` | 存在 | 存在（**唯一**可见共享路径） |
| 本机盘路径 `/data/hongzefu` | 存在 | **不存在** |
| `~/.ssh/config`（集群 ControlMaster） | 存在 | 不存在 |
| GPU | 2 × RTX 6000 Ada Generation（46 GB，sm_89） | NVIDIA A40（40 GB，spgpu 分区） |
| GPU compute mode | `Default` | 默认 `Exclusive_Process`，**须 `--gpu_cmode=shared`** |
| Slurm / 集群提交 | 不可用（无 `sbatch`） | 可用 |
| Vulkan ICD 文件名 | `/usr/share/vulkan/icd.d/nvidia_icd.json` | `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json`（**带架构后缀**） |
| conda / micromamba | conda 无、micromamba 有（不用） | 都没有 |
| 原始数据 / 权重落点 | `checkpoints/`、`logs/`（均在 NFS 仓库内） | 同左 |
| 可做的事 | 改代码、git、建环境、下权重、提交 Slurm、跑评测 | **只跑作业**，不安装依赖、不 git |

**冲突即停**（正本第 0 条）。判定输出与上表任一行不符、或出现不属于这两列的第三套硬件，一律停下来把原始输出交用户裁决，不得自行挑一列往下走，也不得按「多数判据像 A」推断。

环境 B 的红线：不在计算节点安装任何依赖（只 `cd` 进来跑）；不在计算节点做 git 操作（集群侧无凭据，git 一律在本机发起）；不写 `/home`、`/data`（集群上不存在）。

## 占位符取值

| 占位符 | 本仓库取值 |
|---|---|
| `<WORK_ROOT>` | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA`（本机与集群共用这一份，从根上不存在两份副本的同步问题） |
| `<ARCHIVE_ROOT>` | 无 |
| `<STORE_ROOT>` | `logs/`（评测产物：`eval.log` / `results.json` / `videos/`）、`checkpoints/`（模型权重）、`data/datasets/`（LeRobot 数据集）、`third_party/`（install 脚本 clone 的外部源码）——全部已在 `.gitignore` |
| `<DOC_ROOT>` | `docs/eval-doc/`（评测留档；`launch.md` / `result.md` / `records/`） |
| `<FAST_LOCAL_CACHE_ROOT>` | `$HOME/.cache/uv`（`UV_CACHE_DIR` **必须显式设**，见下方覆盖第 3 条） |
| `<PY_INTERPRETER>` | `/nfs/turbo/coe-chaijy-unreplicated/hongzefu/uv-python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10` |
| `<GL_ACCOUNT>` / `<GL_PARTITION>` / `<SSH_HOST>` | `chaijy2` / `spgpu` / `greatlakes` |
| `<GL_REPO>` | 同 `<WORK_ROOT>` |
| `<GL_SUBMIT>` | 无专用提交器；按正本第 8 条以 `greatlakes.md` 为权威源 |
| `<PROTECTED_DIRS>` | `lerobot/`（vendored LeRobot v0.5.1 副本）、`third_party/MIKASA-Robo/`（vendored 上游）、`evaluation_benchmark/`（RoboMemArena 官方 scorer 与 bddl）——任何改动与覆盖须用户逐个批准 |
| `<COMMIT_SUBJECT_STYLE>` | 仓库原只有一个 `init` commit，无既有体例。新建体例：`<模块>: <中文描述>`，模块取 `docs` / `env` / `eval` / `fix` 之一 |
| `<PLAN_EXEMPLAR>` | `docs/eval-doc/` 下最近一次评测的 `launch.md` |

## 对正本的覆盖项（按条号；未列出的条目按正本执行）

- **覆盖第 3 条（uv）**：
  - venv 名为 `.venv-<benchmark>`（如 `.venv-robomme`），**不用裸 `.venv`**——见开头的「一个 benchmark 一个 venv」。用 `UV_PROJECT_ENVIRONMENT` 显式指定，不靠 uv 的默认落点。
  - 解释器钉 `<PY_INTERPRETER>` 那个 NFS 绝对路径，**不用系统 python**。理由：本机系统 python 是 3.12、集群是另一个补丁号，而 venv 里编译好的扩展模块对 ABI 敏感；钉在 NFS 同一个二进制上两端才可复现。禁止手动重链或改 `pyvenv.cfg` 修补。
  - NFS 上必须 `export UV_LINK_MODE=copy`（NFS 不支持 hardlink）。
  - **`UV_CACHE_DIR` 必须显式设回 `$HOME/.cache/uv`**，不能靠「不设」——uv 缓存目录遵循 XDG，一旦设了 `XDG_CACHE_HOME` 指向仓库内，uv cache 会被一起拖到 NFS（2026-09-17 在 uv 0.10.2 实测）。
  - 仓库本体**不需要** `pip install -e .`：所有 `scripts/*.sh` 都 `export PYTHONPATH="$REPO_ROOT"`，`simplememvla` / `lerobot` / `*_sim` 靠它导入。
- **覆盖第 4 条（改动后验证）**：
  - 核心短测：对应 benchmark 的单 task 单 episode 冒烟，如 `TASKS=PickXtimes EPISODES_PER_TASK=1 GROUP_SIZE=1 NUM_GPUS=1 GPUS=0 ATTN_IMPLEMENTATION=sdpa bash scripts/eval_robomme.sh`。
  - 依赖层改动：跑版本+editable 指向校验（torch / numpy / transformers / sapien / mani_skill 版本，且 `mani_skill.__file__` 指向 `third_party/ManiSkill`）。
  - 仿真栈改动：跑 SAPIEN 离屏渲染探针，断言出图且 `img.std() > 1`（**不是黑屏**）。
  - 纯文档改动按纯文档口径：`git diff --check` + 相对链接可解析。
- **覆盖第 6 条（run_name）**：评测的 run 名由 `RUN_TAG` 决定，落点 `logs/<bench>_sim/<RUN_TAG>/`。**MIKASA 与 RoboMemArena 的同目录重跑会 resume 而不是覆盖**（前者按 `episodes/<env_id>/<seed>.json` 跳过，后者按 `shards/` 跳过并用 `run_args.json` 里的 `checkpoint_fingerprint` 拒绝混权重）；LIBERO / RoboMME / RMBench 无 resume，同名重跑会覆盖 `results.json`。
- **覆盖第 21 条（受保护目录）**：`<PROTECTED_DIRS>` 见上表。默认冻结项：`lerobot/` 全目录，验证命令 `git diff --quiet HEAD -- lerobot/`。
- **不采用第 5 条**（patch 级热力图最近邻放大）：本仓库无 patch 级特征可视化脚本。
- **不采用第 10 条**（训练超参落点确认）、**第 13 条**（数据集构建留档）、**第 18 条**（训练链路前后链路图）：当前工作范围只有评测复现，无训练链路改动。将来接手训练时恢复这三条。
- **不采用第 20 条**（Codex `apply_patch` 无管理员权限回退）：该条只对 Codex 生效。

## 本仓库特有的坑（进入实施前必须核实的只读事实）

1. **`transformers` 必须精确 `5.13.1`**。checkpoint 的 `config.json` 外层与 `backbone_config` 都写着 `"transformers_version": "5.13.1"`；prompt 由 `processor.apply_chat_template(..., enable_thinking=False)` 渲染，版本相关。`scripts/eval_robomemarena.sh` 有硬 preflight（不等于 5.13.1 就 exit，理由原文 "The rendered prompt is version-dependent"），**但其余四个 eval 脚本没有**——而 `requirements.txt` 只写 `>=5.11`。装错版本不报错、只静默掉分。已在 `pyproject.toml` 的 `[tool.uv] constraint-dependencies` 钉死。
2. **checkpoint 与 venv 都必须落在 NFS 仓库内**，不能放 `/home` 或 `/data`——集群计算节点看不见那两个盘。
3. **`huggingface_hub>=1.0` 已移除 `huggingface-cli` 入口**，README 里的 `huggingface-cli download ...` 会 `command not found`，改用 `hf download`。
4. **不装 flash-attn 时必须显式传 `ATTN_IMPLEMENTATION=sdpa`**——所有 eval 脚本默认 `flash_attention_2`，缺包会 fail-fast（这是好事，不是坑）。`install_fast_path.sh` 开头硬检查 `nvcc`，本机没有 nvcc，整个脚本跑不了；但 flash-attn 本身是预编译 wheel、不需要 nvcc，只有 causal-conv1d 需要编译（训练必需，eval 可选）。
5. **`flash-linear-attention` 用 `--no-deps` 装是半残的**：`is_flash_linear_attention_available()` 只查 `find_spec("fla")` 返回 True，随后 `from fla.modules import ...` 会因缺 `einops` 抛 `ModuleNotFoundError` 且不在 try 内，导致整个 `modeling_qwen3_5` import 失败。**半装比不装更糟**。不装 fla 时 transformers 5.13.1 有完整纯 torch fallback（`torch_chunk_gated_delta_rule` 等），只打一条 `warning_once`，数学结果不变。
6. **`uv sync` 默认 `--exact`**，会卸掉不在 lock 里的包——手工 `pip install -e third_party/ManiSkill` 装的 ManiSkill 会被下一次 `uv sync` 干掉。所以 ManiSkill 必须走 `[tool.uv.sources]` 的 path+editable 声明。
7. **editable 的实际指向肉眼不可见**（藏在 site-packages 的 `.pth` / `__editable__*` 里），装错了不报错、只是跑的是另一份代码。每次评测前核对 `mani_skill.__file__`。
8. **闭环评测不需要下 LeRobot 数据集**。RoboMME 的 frozen episode metadata 已 vendored 在 `robomme_sim/robomme/env_metadata/`（1.7 MB），conditioning demo 在 env reset 里由仿真器回放。数据集只有 open-loop eval 和训练才用。
9. **集群计算节点 glibc 比本机老**。`robomme-eval-GL` 踩过 `cryptography/_rust.abi3.so` 要求 `GLIBC_2.30` 而节点不满足的坑。本仓库依赖里 `wandb` / `datasets` / `deepspeed` 都可能间接拉进预编译扩展；好在评测链路只走 `simplememvla.data.{collator,messages,normalize}`（纯 torch/numpy），不碰 `simplememvla/data/dataset.py`（那个才 import `lerobot`）。
10. **`scripts/eval_*.sh` 里的 `export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}..."` 在无 conda 环境下导出空串**，是 no-op、无害。真正找 `libvulkan.so.1` 的是 `simplememvla/sapien_runtime.py::prepare_sapien_runtime()`。

## GreatLakes 评测的增量规则

正本第 8 条指向的权威源 `greatlakes.md`（account / partition / 资源包络 / qos / PENDING 读法 / Okta / ControlMaster）不在本仓库，以 MotionJEPA 仓库根目录那份为准。本节只写它没有覆盖的评测增量：

1. **A40 上 SAPIEN 跑不通的根因是 GPU compute mode，不是 ICD 缺失。** 失败特征是 `[svulkan2] [warning] CUDA device 0 is in EXCLUSIVE or EXCLUSIVE_PROCESS mode` followed by `RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`。同时出现的 `Failed to find Vulkan ICD file` 是**红鲱鱼**——节点上 ICD 存在，只是文件名带架构后缀 `nvidia_icd.x86_64.json`，SAPIEN 只查固定名字所以误报。
   - 解药：`srun` / `salloc` / `sbatch` 都支持插件选项 `--gpu_cmode=<shared|exclusive|prohibited>`（**默认 exclusive**）。加 `--gpu_cmode=shared` 后 A40 实测变为 `Default`，原生渲染立刻通过（`robomme-eval-GL` 实测：4 张 256×256×3 图像与本机逐值相同，`max_diff=0`）。
   - 起跑前断言 `nvidia-smi --query-gpu=compute_mode --format=csv,noheader` 为 `Default`，不是就停。
2. **在已 Running 的分配内起任务**用 `srun --jobid=<ID> --overlap --exact`，且 **`--account` / `--partition` / `--gpu_cmode` 必须在 `srun` 上重复写一遍**，否则 step 拿不到 shared 模式。
3. **`env` 和 `bash` 写全路径 `/usr/bin/env`、`/usr/bin/bash`**。`srun` 会把 `env` 解析成登录节点的 `~/.local/bin/env`，计算节点上报 `Permission denied`、退出码 13（`sacct` 里是 `<jobid>.<step>|env|FAILED|13:0`）。
4. **渲染探针失败不自动回退软渲染**。节点上 `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`（llvmpipe）是存在的，退到它能「跑通」但慢几十倍、分数无参考价值。探针失败就记录失败并定位原因，把原始输出交用户处置。
5. **内存 32 G 起步**。`robomme-eval-GL` 实测 MaxRSS 约 16 GB，16 G 分配在渲染阶段就挂。
6. 按 job array 分片跑时，**每片必须用独立的 `RUN_TAG` / `RUN_DIR`**——多片并发写同一目录会互相覆盖 `results.json`。

## 项目 scope（未来工作，不代表当前实施授权）

- 当前范围：复现 RoboMME 闭环评测（本机 2 卡 + GreatLakes），建立可复现的环境与留档体例。
- 未来可能：其余四个 benchmark 的评测复现（各自独立 venv）、open-loop 代理指标、训练链路。**这些都不是当前授权**。
- 明确不做：不改 vendored 上游源码（`lerobot/`、`third_party/MIKASA-Robo/`、`evaluation_benchmark/`），除非用户逐个批准。

## 规则来源

通用规则引用自 `AgentMetaRules-hongzefu` 的 `AGENTS.md` @ `a57506e35b509651720a35a13f75fb4dfb2813a5`。**未采用的条目及原因**：

- 第 5 条（patch 热力图最近邻放大）：本仓库无 patch 级特征可视化。
- 第 10 条（训练超参落点确认）：当前范围无训练链路改动。
- 第 13 条（数据集构建留档）：当前范围不构建数据集。
- 第 18 条（训练链路前后链路图）：同第 10 条。
- 第 20 条（Codex `apply_patch` 回退）：只对 Codex 生效。

GreatLakes 评测的增量规则参考自 [`robomme-eval-GL`](https://github.com/hongzefu/robomme-eval-GL) 分支 `v2-vail-eval-0917` 的 `AGENTS.md` 第 14 条与 `docs/eval-doc/framesamp-modul-smoke/result.md`。

Claude Code 独有机制（Workflow 与 Agent 模型、Monitor 工具、Skill 调用、plan mode）不写在本文件，见同目录 `CLAUDE.md`。两份文件冲突时**一律以本文件为准**。
