# v6xhard-0927：SimpleMemVLA 在 RoboMME V6 xhard1–4 上的评估（每格 20 条）

## ① 一句话结论与指标速览

SimpleMemVLA（`checkpoints/simplememvla_robomme`）在 V6 新值档 xhard1–4 共 55 格、每格 20 条（1100 条，全部生成出 h5）上评估，
**0 条未解决 error**；总成功率 xhard1 35.0%（91/260）、xhard2 18.1%（47/260）、xhard3 11.5%（30/260）、xhard4 16.2%（52/320）。
低成功率格（≤20%）用 20 条新 seed 补测后合并 40 条，结论不变：PickHighlight、PickXtimes 在四档全部 0/40，
SwingXtimes、RouteStick、VideoRepick 在 xhard2 起（RouteStick、VideoRepick 自 xhard3 起）0/40；步数上限翻倍后 PickHighlight 仍 0/20。
低分不是样本太少造成的。

| 档 | 格数 | 本体成功 | 补测格（新 seed 20 条） | 补测成功 |
|---|---|---|---|---|
| xhard1 | 13 | 91/260 = 35.0% | 5 | 13/100 = 13.0% |
| xhard2 | 13 | 47/260 = 18.1% | 8 | 14/160 = 8.8% |
| xhard3 | 13 | 30/260 = 11.5% | 10 | 9/200 = 4.5% |
| xhard4 | 16 | 52/320 = 16.2% | 11 | 10/220 = 4.5% |

逐格全表见 `records/summary.txt`（本体、补测、加长步数、本体+补测 40 条合并四张表）。

## ② 版本与代码状态

- 本仓库分支 `eval-v6xhard-0927-0146`（自 `main@bde7fc1` 切出）；评估入口 `robomme_sim/v6spec_eval.py`，
  包装 `scripts/run_v6spec.sh`、`scripts/gl_run_v6spec.sh`、补测串行链 `scripts/gl_chain_v6supp_0927.sh`，
  回放核对 `scripts/check/v6_h5_replay.py`。
- benchmark：`hongzefu/robomme_benchmark_MotionJEPA` 分支 `PolicyEvalThirdParty-simplememvla-0927-0146`，
  自 12.191（`57fe972`）切出，源码零改动（`source_fingerprint` 不变），只新增 specs：12.192–12.201。
  submodule `third_party/robomme_benchmark` 随每次新增快照前移，只加文件，不影响在跑评估。
- 权重 `checkpoints/simplememvla_robomme`（transformers 5.13.1），`--attn_implementation sdpa`、`group_size 1`、policy_seed 0。

## ③ 启动与配置还原

- **生成（本机 sled-vail，2×RTX 6000 Ada）**：`scripts/parity/v5_generation.py pipeline --release newtask-v6 --seed-profile v6`；
  本体 xhard1 `--run-id smvla-0927 --candidates-per-env 25 --select 0..19`；xhard2–4 `--candidates-per-env 30 --max-reset-attempts 200 --select 0..29`；
  补抽 `--run-id smvla-0927-fill`（xhard1 三格、xhard4 InsertPeg）；补测 `--run-id smvla-0927-more --candidates-per-env 60 --max-reset-attempts 400 --select 30..59`。
  每轮后 `v4_specs reselect` 只把生成出 h5 的局标为 selected，评估读 `specs.reselected.jsonl`。
- **评估（GreatLakes）**：12 个 48h 占位 job（`--account=chaijy2 --partition=spgpu --gres=gpu:1 --gpu_cmode=shared --cpus-per-task=1 --mem=32G`），
  JobID 62049974–62049985；每片 `srun --jobid --overlap --exact --ntasks=1 --cpus-per-task=1`，身份按 (task, episode) 排序轮转切 12 片。
  步数上限按档：xhard1 1500 / xhard2 1700 / xhard3 2000 / xhard4 2600（同 benchmark `v4_eval.py::NEWVALUE_MAX_STEPS`），演示帧不计。
- **本轮 tmux 会话**：本机 `gen-xhard1`、`gen-fill234`、`gen-fillx4-more`、`gen-more-x23`、`gen-more-x4`、`v6-h5replay`、`v6-illdef`（用户纠正后停掉）、
  `gen-xhard234`/`gen-more-x1`（改序时停掉）；GL 登录节点 `v6-smoke`、`v6x1-s*`、`v6x1f-s*`、`v6x2-s*`、`v6x3-s*`、`v6x4-s*`、`v6x4f-s*`、
  `v6x1L-s*`（未开跑即撤）、`v6sup-s*`、`v6sup3-s*`、`v6sup4-s*`，均已结束。

## ④ 数据集与划分口径

- 格子：xhard1–3 各 13 个有梯度 task；xhard4 另含 MoveCube、InsertPeg、StopCube，共 16 个。
- 每格 20 条、全部带 h5（生成侧 `results.jsonl` 中 `ok=true` 且 h5 存在）。缺口处理：xhard1 BinFill/VideoPlaceOrder/VideoRepick 以补抽快照 12.193 为准；
  xhard4 InsertPeg 取 12.196 的 14 条加 12.197 的 6 条新 seed（`--exclude_specs` 去重）。
- 补测：本体成功率 ≤20% 的格子另取 20 条新 seed（`--exclude_specs` 与本体 seed 交集为 0，四档均核对 `seed_overlap 0`）。
- 加长步数：本体 timeout ≥5/20 的格子（PickHighlight×4 档、BinFill xhard1–3、MoveCube/InsertPeg xhard4）同 20 条，步数上限翻倍。

## ⑤ 关键超参

`execute_horizon 16`、`num_denoising_steps 10`、`max_subtask_tokens 64`、bf16；基础设施 error 每身份最多 3 次尝试（`--max_attempts 3`），
正常执行未完成（fail/timeout）如实记录，不为挑成功重跑。

## ⑥ 硬件与耗时

- 本机评估 smoke：PickXtimes/0 850 步 92.5 s，MaxRSS 26.4 GB（因此占位由 24G 改 32G）。
- GL A40 单条约 1.5–3 min；本体 1100 条约 4 h（12 片），补测与加长步数约 4 h。

## ⑦ 过程行为

- 评测接口一致性：xhard1 13 task 各 1 条经 `SimEnvService`（V6 快照模式）回放 h5 非演示 `joint_action`，13/13 到达 success；
  reset 帧逐像素 MAD 0.00–0.02，全程 mad_mean 0.48–1.73、mad_max ≤5.40（0–255），对照图目视一致。
- 集群 smoke（62049974，1 条）：BinFill/0 fail 1014 步 161.4 s，RSS 26.4 GB。

## ⑧ 评估结果

见 ① 与 `records/summary.txt`。补充观察（仅陈述数字）：
- PickHighlight 以 timeout 为主（本体 11–19/20），步数翻倍后 timeout 降到 7–12，成功仍 0；
- BinFill xhard1 翻倍后 timeout 7→0，成功 6→6；MoveCube/InsertPeg 翻倍后结果与本体相同（12/20、9/20）。
- SwingXtimes xhard1 16/20，xhard2 起 0/20（全 fail）。

## ⑨ 用户决策记录（原话）

1. 「给出方案 生成每个task*每个难度 20个episode 然后用/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA进行eval 都checkout新的branch来实现 benchmarkrepo 使用 PolicyEvalThirdParty-simplememvla加上时间 用gl实现 8个48小时job 尽可能压低cpu mem」
2. 「生成20episode可以在本机跑」；拍板难度 = V6 xhard1–4，benchmark 基底 = /data 12.191；「controlmaster如果断开只能停了」
3. 「优先测试完毕xhard1」「提到12卡」「同意这条任务 请求12张卡！」
4. 「开跑后在本机测试一下 目视检查和数据生成过程中是否一样」「benchmark我可能没说清楚 你还需要都能生成h5作为能做完的判据」
   「你还需要测试一下 走评测接口 目视检查和数据生成过程中是否一样 输入h5的动作是否可以正常继续进入下一步 做小规模的测试 不要影响12卡起泡」
5. 「12 片评估直接跑会被 OOM 杀掉， 这个问题你要优先处理啊优先重新跑job」
6. 「你这个H5声称失败的你要补充把它填满了 补充满直到生成二十个episode都能够H5。 评估需要保证都有20个 不够的补跑」
7. 「全部跑完后记得把十二个GPU全释放你要确保你所有的都跑完了」
8. 「如果成功率比较低的task可以再生成一些再试一下」→ 选「两者都做」（新 seed 加 20 条 + 加长步数重测）
9. 「确保task没有ill defined的意思是 确保不是因为种类太小而是没有办法完成的不是说让你重新检查task」
10. 「先把本体全测完了XHARD1234然后再做补测」

## ⑩ 计划外事件与处置

- 首次 `sbatch` 被 Claude Code 自动权限审查拦下，用户明确同意后提交。
- 24G 占位 OOM 风险：重提 32G，旧 12 个（62049039–62049050）逐个 scancel。
- 抽签总尝试上限默认 30 导致 xhard1 三格不足 20：补抽并把 xhard2–4 改为 200 次上限、30 候选全实跑。
- xhard4 InsertPeg 仅 14 条 h5：补抽新 seed；递补用到 0–29 号候选使 5 条同 seed，评估按 seed 去重。
- 回放复核（ill-defined 误解）起跑后按用户纠正停掉；加长步数排队一度排在 xhard3/4 本体前，按用户指令撤回改序。
- 生成报告告警 `demo_frames_out_of_band=40`（xhard1）未处置，待用户判读。

## ⑪ 结论与下一步

本体与补测均 0 未解决 error；12 个占位 job 已按清单逐个 scancel（删前删后 `squeue` 仅少这 12 个，`62018665` 非本轮未动）。
下一步由用户决定：是否判读 `demo_frames_out_of_band` 告警、是否对 0/40 的格子做失败模式分析。

## ⑫ 归档文件清单

- `records/summary.txt`：四张汇总表。
- `records/<run>/results-shardXXofNN.jsonl`：16 个评估目录的逐条结果（含 key、seed、spec_sha256、status、steps、max_steps、rss）。
