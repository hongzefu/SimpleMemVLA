#!/usr/bin/env bash
# 0927 补测串行链（本体 xhard1–4 全部测完后执行）：每个占位 job 按序跑
#   1. xhard1 低成功率 5 格新 seed 各 20 条；2. xhard2 低成功率 8 格新 seed 各 20 条；
#   3. timeout ≥5/20 的格子用原 20 条、步数上限翻倍重测（单独报告）。
# 用法：bash scripts/gl_chain_v6supp_0927.sh <JOBID> <i>
set -uo pipefail
J="$1"; i="$2"; N=12
REPO=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA
C=$REPO/third_party/robomme_benchmark/scripts/configs/newtask-v6
R() { SPECS="$1" bash "$REPO/scripts/gl_run_v6spec.sh" "$J" "$i/$N" "${@:2}"; }
R $C/smvla-0927-more/xhard1/specs.reselected.jsonl more-xhard1-0927 --exclude_specs $C/smvla-0927/xhard1/specs.reselected.jsonl,$C/smvla-0927-fill/xhard1/specs.reselected.jsonl
R $C/smvla-0927-more/xhard2/specs.reselected.jsonl more-xhard2-0927 --exclude_specs $C/smvla-0927/xhard2/specs.reselected.jsonl
# 加长步数：各档上限翻倍（1500/1700/2000/2600 → 3000/3400/4000/5200）
R $C/smvla-0927/xhard1/specs.reselected.jsonl long-xhard1-main-0927 --only_tasks PickHighlight --max_steps_override 3000
R $C/smvla-0927-fill/xhard1/specs.reselected.jsonl long-xhard1-fill-0927 --only_tasks BinFill --max_steps_override 3000
R $C/smvla-0927/xhard2/specs.reselected.jsonl long-xhard2-0927 --only_tasks BinFill,PickHighlight --max_steps_override 3400
R $C/smvla-0927/xhard3/specs.reselected.jsonl long-xhard3-0927 --only_tasks BinFill,PickHighlight --max_steps_override 4000
R $C/smvla-0927/xhard4/specs.reselected.jsonl long-xhard4-main-0927 --only_tasks PickHighlight,MoveCube,InsertPeg --max_steps_override 5200
R $C/smvla-0927-fill/xhard4/specs.reselected.jsonl long-xhard4-fill-0927 --only_tasks InsertPeg --per_task 6 --exclude_specs $C/smvla-0927/xhard4/specs.reselected.jsonl --max_steps_override 5200
echo "CHAIN_DONE job=$J shard=$i"
