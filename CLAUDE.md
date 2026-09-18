# CLAUDE.md

本文件只写 **Claude Code 独有机制**的约定。仓库通用工作规则的唯一来源是 `AGENTS.md`，本文件不复述其任何内容。

@AGENTS.md

## 规则来源与优先级

- `AGENTS.md` 是本仓库通用规则的唯一来源（通用条目引用 [`AgentMetaRules-hongzefu`](https://github.com/hongzefu/AgentMetaRules-hongzefu) 正本 @ `a57506e35b509651720a35a13f75fb4dfb2813a5`）。上面的 `@AGENTS.md` 会把它的正文内联进上下文。
- **即便该导入失效，动手前也必须先完整读一遍 `AGENTS.md`。** 实测 Claude Code 的 memory 发现列表只含 `CLAUDE.md` / `CLAUDE.local.md` / `.claude/rules`，**不含 `AGENTS.md`**——没有本文件的显式导入时，`AGENTS.md` 不会进入会话上下文。这一条是兜底，不可省略。
- **动手前必须先做运行环境判定**：执行 `AGENTS.md`「运行环境判定」一节的判定命令，并在当轮第一条回复里写明结论是 **环境 A：本机 sled-vail** 还是 **环境 B：GreatLakes 计算节点**。判据互相矛盾、或出现第三套硬件时，把原始输出交用户裁决，不得自行挑一套往下走。
- **两份文件冲突时一律以 `AGENTS.md` 为准**，本文件仅在 `AGENTS.md` 未规定处生效；两份仓库文档都服从系统、开发者及用户当前指令。

## Claude Code 独有机制

Workflow 与 Agent 模型、Monitor 工具、Skill 调用、plan mode 与计划文件四块，全部以 `AgentMetaRules-hongzefu` 的 [`CLAUDE.md`](https://github.com/hongzefu/AgentMetaRules-hongzefu/blob/main/CLAUDE.md) 正本 @ `a57506e35b509651720a35a13f75fb4dfb2813a5` 为准。本文件只写项目专属补充。

### Monitor 过滤词表（正本 Monitor 第 5、7 条）

一份日志挂一个 Monitor，管道每一级都行缓冲（`stdbuf -oL tr`，只给 `grep --line-buffered` 不够——中间的 `tr` 默认 4 KB 块缓冲会把结束时的 `EXIT_CODE=` 行永远卡在缓冲区里，任务跑完了 Monitor 却静默）。

**致命（立即告警）**：

```
EXIT_CODE=  Traceback  crashed  CUDA out of memory  OOM even at group=1
Checkpoint weight mismatch  Cannot find a valid renderer device
vk::enumeratePhysicalDevices  ErrorInitializationFailed  EXCLUSIVE_PROCESS
libvulkan.so.1: cannot open  libGL.so.1: cannot open  GLIBC_2.  No module named
AttributeError: module 'sapien'  flash_attn is not installed
subprocess-exited-with-error  Failed to build
single-patch video preprocessing does not reproduce
```

**警告（记录，不中断）**：`reset_failed:` / `env reset(s) failed` / `RRTPlanFailure` / `[OOM]` / `sub-task decode hit the` / `still waiting on N worker`

**进度心跳**：`[eval] started worker` / `success_rate=` / `wrote results JSON` / `OVERALL (macro avg)`

**白名单（预期出现，看到反而说明 fallback 正常生效，不告警）**：

```
The fast path is not available because one of the required library is not installed
Falling back to torch implementation
```

**本仓库的「静默空转」风险点**（正本 Monitor 第 7 条）：RoboMME 的 `--reset_timeout` 默认 3600 秒、LIBERO 的 `--worker_timeout` 默认 14400 秒、MIKASA 默认 28800 秒。reset 会用 mplib motion planner 回放整段 conditioning demo，卡在 RRT* 重试里时进程活着、日志不出错、也不产出。所以过滤器必须同时覆盖 `reset_failed:` / `RRTPlanFailure` / `still waiting on N worker`，不能只盯 `EXIT_CODE=`。评测起跑前先记下预期耗时，超过 3 倍就介入排查。

### Skill 调用

- 本仓库当前环境（环境 A：本机 sled-vail）**有**集群访问：`~/.ssh/config` 存在、`ssh greatlakes` 可用。查 `chaijy2` 账户占用（GPU / 内存 / CPU 配额余量、谁在用、我的 job、PENDING、spgpu 全局 A40 占用）**一律先调全局 skill `greatlakes-usage`**，不手搓 `ssh` + `squeue` / `sacctmgr` 拼答案。
- **发起 ssh 登录前必须先问用户用哪种验证方式**（6 位 TOTP 码填 `Okta passcode` prompt，或留空触发 push + 数字匹配），不默认、不复用上次选择。**但先跑 `ssh -O check greatlakes`**：ControlMaster 存活时直接复用、零认证，不必问。
- 提交作业、建 ControlMaster、Okta 验证方式等集群操作细节，按 `AGENTS.md` 的 GreatLakes 一节与正本第 8 条以 `greatlakes.md` 为权威源。

### plan mode 只读核实清单

正本要求「在只读阶段把事实核实清楚」。本仓库进入实施前必须只读核实的项：

1. `.venv-<benchmark>` 里 `transformers.__version__` 是否等于 checkpoint `config.json` 的 `transformers_version`（**装错不报错、只静默掉分**）。
2. `mani_skill.__file__` / `libero.__file__` 的 editable 实际指向——藏在 site-packages 的 `.pth` 里、肉眼不可见，装错了跑的是另一份代码。
3. `sapien.__version__` 与所在 benchmark 要求的 beta / stable 线是否一致。
4. 两端 Vulkan ICD 文件名差异（本机 `nvidia_icd.json`、集群 `nvidia_icd.x86_64.json`）。
5. 集群 job 的 GPU compute mode 是否为 `Default`（不是就说明 `--gpu_cmode=shared` 没生效）。
6. checkpoint 目录是否含 `stats.json`（action 反归一化参考，缺了 RoboMemArena 直接 `FileNotFoundError`，其余 benchmark 表现为动作尺度错乱）。

### 长任务与 tmux 会话命名

本仓库的 tmux 会话名统一带 `smvla-` 前缀（如 `smvla-uvsync` / `smvla-ckpt` / `smvla-16x1-local` / `smvla-16x1-gl01`），并在留档 `launch.md` 里记下本轮会话清单。死活判断用 `tmux has-session -t '=<确切名>'`（`=` 前缀是精确匹配；不带 `=` 的 `-t` 做前缀匹配，`-t smvla` 会命中所有 `smvla` 开头的会话）。

**清理红线**：本机跑着其他项目的 tmux 会话（`site*` / `ensite*` / `unisite` / `v3site` 等）。任何情况下禁止 `tmux kill-server`、`tmux kill-session -a`、`pkill -f tmux`、`killall tmux`。唯一允许的清理方式是 `tmux kill-session -t '=<确切会话名>'`，一次只杀一个、名字写全，禁止通配与 `xargs` 批量；删前删后各跑一次 `tmux ls` 对比。
