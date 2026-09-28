"""按 benchmark ``dataset="test-hard"`` 评估 SimpleMemVLA（benchmark 0927 计划 §8.1、§3.3）。

身份由 ``robomme_hard`` 的 builder 给出：每任务 episode 0..79（xhard4-only 的 StopCube/InsertPeg/MoveCube 为 0..19），
档序主序、档内 candidate 升序；``resolve_identity`` 返回 ``{tier, candidate, seed, spec_sha256}``。
``--round 1`` 取每格 candidate 最小的前 10 局、``--round 2`` 取后 10 局；``--shard i/10`` 取该轮身份按
``(task, episode)`` 排序后下标 ≡ i (mod 10) 的局。每条结果即时追加到 ``<out>/results-shard<i>of<n>.jsonl``，
``--resume`` 按身份续评。

逐局核「环境阈值 = 服务返回 = 策略循环上限 = 落盘值」（按档步数上限 1500/1700/2000/2600），以及回注绑定
``spec_binding``：``available`` 且 ``mode=replay``、``spec_sha256`` 等于该身份、``value_points>0``；不等即该局记 error。
正常执行未完成任务如实记 fail／timeout，不为挑成功重试；只有基础设施 error 重跑，每身份最多 ``--max_attempts`` 次，
每片每轮合计最多 ``--retry_cap`` 次（默认 55），用满即打 ``RETRY_CAP_HIT`` 退出。

用法：
    python -m robomme_sim.testhard_eval --benchmark_root <benchmark> --out <目录> --round 1 --shard 0/10 \
        --pretrained_checkpoint <ckpt 绝对路径> [--resume] [--limit 1]
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

NORMAL = ("success", "fail", "timeout")
ALL_TASKS = ("PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap", "VideoUnmask",
             "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
             "PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick")


def _own_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    ap.add_argument("--shard", default="0/10", help="i/n：该轮身份按 (task, episode) 排序后下标 ≡ i (mod n)")
    ap.add_argument("--per_round", type=int, default=10, help="每格每轮局数（第一轮前 N、第二轮后 N）")
    ap.add_argument("--limit", type=int, default=0, help="本片只跑前 N 条（smoke 用）")
    ap.add_argument("--only", default="", help="逗号分隔的 task/episode，只评这些（smoke 用）")
    ap.add_argument("--max_attempts", type=int, default=3, help="单身份基础设施 error 的最多尝试次数")
    ap.add_argument("--retry_cap", type=int, default=55, help="本片本轮基础设施 error 重跑合计上限")
    return ap.parse_known_args()


def identities(benchmark_root: str, per_round: int = 10) -> list[dict]:
    """全部 1100 个身份（含 builder episode 号），每格按 candidate 升序标轮次。"""
    src = str(Path(benchmark_root).resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder, TIER_MAX_STEPS

    out = []
    for task in ALL_TASKS:
        builder = BenchmarkEnvBuilder(task, dataset="test-hard")
        cells: dict[str, list] = {}
        for episode in range(builder.get_episode_num()):
            ident = builder.resolve_identity(episode)
            cells.setdefault(ident["tier"], []).append(ident)
        for tier, rows in cells.items():
            rows.sort(key=lambda r: r["candidate"])
            for rank, ident in enumerate(rows):
                out.append(dict(task=task, key=f"{task}/{ident['episode']}", round=1 if rank < per_round else 2,
                                max_steps=TIER_MAX_STEPS[tier], identity=ident, **ident))
    return out


def main() -> int:
    own, rest = _own_args()
    sys.argv = [sys.argv[0], *rest]
    from robomme_sim.eval_success import build_policy, parse_args
    args = parse_args()
    if not args.benchmark_root:
        raise ValueError("必须提供 --benchmark_root（含 robomme_hard 的 benchmark 检出）")
    args.dataset_split = "test-hard"
    ckpt = str(Path(args.pretrained_checkpoint).resolve())
    all_ids = identities(args.benchmark_root, own.per_round)
    picked = sorted((r for r in all_ids if r["round"] == own.round), key=lambda r: (r["task"], r["episode"]))
    i, n = (int(x) for x in own.shard.split("/"))
    mine = picked[i::n]
    if own.only:
        keep = set(own.only.split(","))
        mine = [r for r in all_ids if r["key"] in keep]
    out = Path(own.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / f"results-r{own.round}-shard{i:02d}of{n:02d}.jsonl"
    done, tries, infra = set(), {}, 0
    if results.exists():
        if not args.resume:
            raise FileExistsError(f"{results} 已存在，只能显式 --resume 恢复")
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                tries[rec["key"]] = tries.get(rec["key"], 0) + 1
                if rec["status"] in NORMAL:
                    done.add(rec["key"])
                else:
                    infra += 1
    pending = [r for r in mine if r["key"] not in done and tries.get(r["key"], 0) < own.max_attempts]
    if own.limit:
        pending = pending[: own.limit]
    print(f"TESTHARD_START round={own.round} shard={i}/{n} total={len(picked)} mine={len(mine)} "
          f"pending={len(pending)} infra_so_far={infra} ckpt={ckpt}", flush=True)

    expect = Path(args.benchmark_root).resolve() / "src"
    batched, buffer_factory, normalizer = build_policy(args)
    from robomme_sim.inproc_pool import InProcSimPool

    def new_pool():
        pool = InProcSimPool(1, dataset_split="test-hard", max_steps=args.max_steps,
                             benchmark_root=args.benchmark_root, reset_retries=0,
                             reset_timeout=args.reset_timeout, step_timeout=args.step_timeout)
        pool.start()
        return pool

    pool = new_pool()
    code = 0
    for row in pending:
        if tries.get(row["key"], 0) > 0 and infra >= own.retry_cap:
            print(f"RETRY_CAP_HIT round={own.round} shard={i}/{n} infra={infra} cap={own.retry_cap}", flush=True)
            code = 4
            break
        started = time.time()
        record = dict(key=row["key"], task=row["task"], episode=row["episode"], tier=row["tier"],
                      candidate=row["candidate"], seed=row["seed"], spec_sha256=row["spec_sha256"],
                      identity=row["identity"], round=own.round, max_steps=row["max_steps"], status="error",
                      task_success=False, error_class=None, steps=0, attempt=tries.get(row["key"], 0) + 1,
                      checkpoint=ckpt, policy_seed=0, spec_binding=None, demo_frames=None, demo_tasks=None, error=None)
        print(f"EPISODE_START key={row['key']} tier={row['tier']} max_steps={row['max_steps']}", flush=True)
        try:
            reset = pool.reset([dict(task=row["task"], episode=row["episode"])])[0]
            if not reset or not reset.get("ok"):
                raise RuntimeError(f"reset 错误：{reset}")
            for name in ("robomme", "robomme_hard"):
                loaded = Path(sys.modules[name].__file__).resolve().parent.parent
                if loaded != expect:
                    raise RuntimeError(f"{name} 来源不符：{loaded}，期望 {expect}")
            binding = reset.get("spec_binding") or {}
            record.update(demo_frames=reset.get("demo_frames"), demo_tasks=reset.get("demo_tasks"), spec_binding=binding)
            # 四者一致：环境阈值（builder 按档）= 服务返回 = 策略循环上限 = 落盘值
            if reset.get("max_steps") != row["max_steps"] or reset.get("tier") != row["tier"]:
                raise RuntimeError(f"步数上限或档位不一致：服务 {reset.get('max_steps')}/{reset.get('tier')}，"
                                   f"身份 {row['max_steps']}/{row['tier']}")
            if reset.get("identity", {}).get("spec_sha256") != row["spec_sha256"]:
                raise RuntimeError("服务返回的身份与本局不符")
            if not (binding.get("available") and binding.get("mode") == "replay" and binding.get("value_points", 0) > 0
                    and binding.get("spec_sha256") == row["spec_sha256"]):
                raise RuntimeError(f"回注绑定不合格：{binding}")
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
            cap = row["max_steps"]
            while record["steps"] < cap:
                normalized = None
                if normalizer is not None:
                    normalized = normalizer.normalize(torch.from_numpy(np.asarray(state, dtype=np.float32)))
                    normalized = normalized.unsqueeze(0).to(batched.device)
                actions, _subtask = batched.generate_batch([buffer._prepare_inputs(reset["instruction"])], [normalized])[0]
                count = min(args.execute_horizon, cap - record["steps"])
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
            record["task_success"] = record["status"] == "success"
        except Exception:
            record.update(status="error", error_class="infra", error=traceback.format_exc()[-2000:])
            infra += 1
            print(record["error"], flush=True)
        record.update(elapsed_s=round(time.time() - started, 1),
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"EPISODE_END key={row['key']} status={record['status']} steps={record['steps']} "
              f"elapsed={record['elapsed_s']}s rss_kib={record['max_rss_kib']}", flush=True)
        if record["status"] == "error":
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
    valid = sum(r["status"] in NORMAL for r in final.values())
    ok = sum(r["status"] == "success" for r in final.values())
    err = len(mine) - valid
    print(f"TESTHARD_DONE round={own.round} shard={i}/{n} identities={len(mine)} valid={valid} success={ok} "
          f"unresolved_errors={err} infra={infra}/{own.retry_cap}", flush=True)
    return code or (0 if err == 0 else 3)


if __name__ == "__main__":
    raise SystemExit(main())
