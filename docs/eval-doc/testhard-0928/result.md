# SimpleMemVLA test-hard 评估（2026-09-28）：结果

## 一句话结论

1100 局（xhard1/2/3 各 13 任务 × 20 局 + xhard4 16 任务 × 20 局）全部正常终态、零基础设施重跑；逐局回注绑定全部通过；与上次在 `robomme`（benchmark 12.191）上评的同一 1100 个身份相比，**1100 局终态全部相同、1099 局步数也相同**。

## 判定行（`records/smvla-gates.txt`）

```text
EVAL_ROUND1=PASS policy=simplememvla episodes=550 normal=550 error_left=0 retries=0/55(max_per_shard=0) shards=10 per_shard=55-55 shape=55x10
EVAL_ROUND2=PASS policy=simplememvla episodes=550 normal=550 error_left=0 retries=0/55(max_per_shard=0) shards=10 per_shard=55-55 shape=55x10
EVAL_IDENTITY_SET=PASS policy=simplememvla rounds=2 shards=10 per_shard=55 episodes=1100 missing=0 extra=0 dup=0
EVAL_BINDING=PASS policy=simplememvla episodes=1100 replay=1100 injected_mismatch=0 recorded_drift=62 max_abs=1.2e-07 unused=0
EVAL_TIER_CAP=PASS policy=simplememvla episodes=1100 mismatch=0
EVAL_DEMO_FRAMES=INFO policy=simplememvla episodes=1100 exact=1054 within_5=1096 max_diff=19
```

## 分档成功率

| 档 | 局数 | success | fail | timeout | 成功率 | 上次（aab093f） |
|---|---|---|---|---|---|---|
| xhard1 | 260 | 91 | 140 | 29 | 35.0% | 35.0%（91/140/29） |
| xhard2 | 260 | 47 | 183 | 30 | 18.1% | 18.1%（47/183/30） |
| xhard3 | 260 | 30 | 202 | 28 | 11.5% | 11.5%（30/202/28） |
| xhard4 | 320 | 52 | 236 | 32 | 16.3% | 16.3%（52/236/32） |

逐格明细 `records/smvla-cells.json`；逐局结果 `records/results-r<轮>-shard<i>of10.jsonl`。

## 与上次评估的差异

- 同身份、同按档上限；差异来源只剩 benchmark 由 `robomme`（12.191）换成 `robomme_hard`（12.210）与演示回放的 RRT 墙钟噪声。
- 逐局比对：终态 1100/1100 相同；步数 1099/1100 相同，唯一不同的是 VideoPlaceOrder／xhard3／seed 13100901（两次均 fail，步数 463 → 494）。
- `recorded_drift=62`（最大 1.2e-7）全部在只记录不回注的观测点上（方案甲 ≤ 1e-5），回注点零差。
- 本次未录评估视频。
