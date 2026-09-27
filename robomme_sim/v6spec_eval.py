"""按 benchmark V6 冻结快照（specs.jsonl）评估 SimpleMemVLA。

seed / difficulty / sampling_config / 规格全部取自快照，经
``BenchmarkEnvBuilder.from_v4_specs`` 建环境；episode 按 (task, episode) 排序后轮转切片，
每条结果即时追加到 ``<out>/results.jsonl``，``--resume`` 跳过已有结果的身份。
策略侧与 ``candidate_eval`` 同一套推理循环；正常执行未完成任务如实记 fail，不为挑成功重试。

用法：
    python -m robomme_sim.v6spec_eval --specs <specs.jsonl> --benchmark_root <benchmark> \
        --out <目录> --shard 0/12 --per_task 20 --pretrained_checkpoint <ckpt> [--resume]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# 与 benchmark scripts/eval/v4_eval.py::NEWVALUE_MAX_STEPS 同值（演示帧不计入）
NEWVALUE_MAX_STEPS = {"xhard1": 1500, "xhard2": 1700, "xhard3": 2000, "xhard4": 2600}
NORMAL = ("success", "fail", "timeout")


def _own_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n，按排序后身份轮转切片")
    ap.add_argument("--per_task", type=int, default=20, help="每个 task 取前 N 个 selected 身份")
    ap.add_argument("--limit", type=int, default=0, help="本片只跑前 N 条（smoke 用）")
    ap.add_argument("--only_tasks", default="", help="逗号分隔，只评这些 task（补抽快照按 task 分开评）")
    ap.add_argument("--max_steps_override", type=int, default=0,
                    help="非 0 时覆盖按档取的步数上限（timeout 格加长步数重测，结果须单独报告）")
    ap.add_argument("--exclude_specs", default="",
                    help="另一份快照：其 selected 行的 seed 一律排除（补抽快照与本体快照去重）")
    ap.add_argument("--max_attempts", type=int, default=3, help="基础设施 error 的最多尝试次数；fail/timeout 不重跑")
    return ap.parse_known_args()


def _selected_seeds(specs_path: str) -> set:
    seeds = set()
    for line in Path(specs_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("record") != "header" and rec.get("selected"):
                seeds.add(int(rec["seed"]))
    return seeds


def _identities(specs_path: str, per_task: int, exclude: set = frozenset()):
    header, rows = None, []
    for line in Path(specs_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record") == "header":
            header = rec
        elif rec.get("selected") and int(rec["seed"]) not in exclude:
            rows.append(rec)
    by_task: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: (r["task"], int(r["episode"]))):
        by_task.setdefault(r["task"], []).append(r)
    picked = [r for t in sorted(by_task) for r in by_task[t][:per_task]]
    return header, picked


def main() -> int:
    own, rest = _own_args()
    sys.argv = [sys.argv[0], *rest]
    from robomme_sim.eval_success import build_policy, parse_args
    args = parse_args()
    if not args.benchmark_root:
        raise ValueError("必须提供 --benchmark_root（specs 生成时的 benchmark 源码根）")

    exclude = set()
    for path in filter(None, own.exclude_specs.split(",")):  # 逗号分隔多份快照
        exclude |= _selected_seeds(path)
    header, picked = _identities(own.specs, own.per_task, exclude)
    if own.only_tasks:
        keep = set(own.only_tasks.split(","))
        picked = [r for r in picked if r["task"] in keep]
    difficulty = header["difficulty"]
    max_steps = own.max_steps_override or NEWVALUE_MAX_STEPS.get(difficulty, args.max_steps)
    args.max_steps = max_steps
    i, n = (int(x) for x in own.shard.split("/"))
    mine = picked[i::n]
    out = Path(own.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / f"results-shard{i:02d}of{n:02d}.jsonl"
    # 每个身份：已有正常终态即完成；只有 error 且次数未满才重跑（不为挑成功重试）
    done, tries = set(), {}
    if results.exists():
        if not args.resume:
            raise FileExistsError(f"{results} 已存在，只能显式 --resume 恢复")
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                tries[rec["key"]] = tries.get(rec["key"], 0) + 1
                if rec["status"] in NORMAL:
                    done.add(rec["key"])
    pending = [r for r in mine if f"{r['task']}/{r['episode']}" not in done
               and tries.get(f"{r['task']}/{r['episode']}", 0) < own.max_attempts]
    if own.limit:
        pending = pending[: own.limit]
    print(f"V6EVAL_START difficulty={difficulty} max_steps={max_steps} shard={i}/{n} "
          f"total={len(picked)} mine={len(mine)} pending={len(pending)}", flush=True)

    expect = Path(args.benchmark_root).resolve() / "src" / "robomme"
    batched, buffer_factory, normalizer = build_policy(args)
    from robomme_sim.inproc_pool import InProcSimPool

    def new_pool():
        pool = InProcSimPool(1, max_steps=max_steps, benchmark_root=args.benchmark_root,
                             v6_specs=own.specs, reset_timeout=args.reset_timeout,
                             step_timeout=args.step_timeout)
        pool.start()
        return pool

    pool = new_pool()
    for row in pending:
        key = f"{row['task']}/{row['episode']}"
        started = time.time()
        record = dict(key=key, task=row["task"], difficulty=row["difficulty"], episode=row["episode"],
                      seed=row["seed"], spec_sha256=row["spec_sha256"], status="error", steps=0,
                      max_steps=max_steps, checkpoint=str(Path(args.pretrained_checkpoint).resolve()),
                      policy_seed=0, demo_frames=None, demo_tasks=None, error=None)
        print(f"EPISODE_START key={key}", flush=True)
        try:
            reset = pool.reset([dict(task=row["task"], episode=row["episode"])])[0]
            if not reset or not reset.get("ok"):
                raise RuntimeError(f"reset 错误：{reset}")
            loaded = Path(sys.modules["robomme"].__file__).resolve().parent
            if loaded != expect:
                raise RuntimeError(f"robomme 来源不符：{loaded}，期望 {expect}")
            record.update(demo_frames=reset["demo_frames"], demo_tasks=reset["demo_tasks"])
            buffer = buffer_factory()
            buffer.reset()
            cam_map = {k.split(".")[-1]: k for k in buffer.image_keys}

            def observe(frames):
                for frame in frames:
                    buffer.observe({cam_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()})

            observe(reset["frames"])
            state = reset["states"][-1]
            torch.manual_seed(0)
            np.random.seed(0)
            while record["steps"] < max_steps:
                normalized = None
                if normalizer is not None:
                    normalized = normalizer.normalize(torch.from_numpy(np.asarray(state, dtype=np.float32)))
                    normalized = normalized.unsqueeze(0).to(batched.device)
                actions, _subtask = batched.generate_batch([buffer._prepare_inputs(reset["instruction"])], [normalized])[0]
                count = min(args.execute_horizon, max_steps - record["steps"])
                chunk = np.asarray(actions[:count])
                if chunk.shape != (count, 8) or not np.isfinite(chunk).all():
                    raise ValueError(f"模型动作形状或数值非法：{chunk.shape}")
                response = pool.step([chunk], [0])[0]
                if (not isinstance(response, dict) or response.get("error") or
                        response.get("error_message") or response.get("status") == "error"):
                    detail = ({k: response.get(k) for k in ("status", "error", "error_message")}
                              if isinstance(response, dict) else response)
                    raise RuntimeError(f"step 错误：{detail}")
                record["steps"] += response["consumed"]
                observe(response["frames"])
                if response["states"]:
                    state = response["states"][-1]
                if response["done"]:
                    if response["status"] not in NORMAL:
                        raise RuntimeError(f"未知终态：{response['status']}")
                    record["status"] = response["status"]
                    break
            else:
                record["status"] = "timeout"
        except Exception:
            record.update(status="error", error=traceback.format_exc()[-2000:])
            print(record["error"], flush=True)
        record.update(elapsed_s=round(time.time() - started, 1),
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"EPISODE_END key={key} status={record['status']} steps={record['steps']} "
              f"elapsed={record['elapsed_s']}s rss_kib={record['max_rss_kib']}", flush=True)
        if record["status"] == "error":
            # 卡死线程无法取消：丢弃旧池重建，下一条从干净环境开始
            try:
                pool.close()
            except Exception:
                pass
            pool = new_pool()
        else:
            pool.services[0].close()
    pool.close()
    recs = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()] if results.exists() else []
    final = {}
    for r in recs:
        if r["key"] not in final or final[r["key"]]["status"] not in NORMAL:
            final[r["key"]] = r
    ok = sum(r["status"] == "success" for r in final.values())
    valid = sum(r["status"] in NORMAL for r in final.values())
    err = len(mine) - valid
    print(f"V6EVAL_DONE shard={i}/{n} identities={len(mine)} valid={valid} success={ok} unresolved_errors={err}", flush=True)
    return 0 if err == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
