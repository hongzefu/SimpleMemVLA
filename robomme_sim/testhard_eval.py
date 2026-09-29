"""按 benchmark ``dataset="test-hard"`` 评估 SimpleMemVLA（v7：xhard0～xhard4，按身份清单分轮分片）。

v7 口径（``--identities <jsonl>`` 必传）：身份清单每行一局
``{task, episode, tier, seed, candidate, source_episode, round, shard}``；``episode`` 是 test-hard builder 下标。
``--round r --shard i/n`` 取清单里 ``round==r`` 且 ``shard==i`` 的行，片内按 ``(task, episode)`` 排序；
``n`` 必须等于清单该轮的分片数（分片号须为 0..n-1）。开跑前逐行核 ``builder.resolve_identity(episode)`` 与清单一致
（tier、seed；xhard0 核 source_episode 且 candidate 为 null，其他档核 candidate），不符即整片退出码 2。

回注绑定按档分类（结果行 ``binding_class``）：
- xhard0（官方 test 原局）→ ``export``：``available`` 且 ``mode=export``、``spec_kind=native-parity/1``、``injected_mismatch==0``；
- xhard1～4 → ``replay``：``available`` 且 ``mode=replay``、``injected_mismatch==0``、``value_points>0``、
  ``spec_sha256`` 等于 builder 身份给出的值，且（若有该键）``layout_drift==0``。
不合格即该局记 error。步数上限按档取 ``TIER_MAX_STEPS``（xhard0 为 1300），逐局核「环境阈值 = 服务返回 = 策略循环上限 = 落盘值」。

``--video_dir`` 给出时每局录像（front|wrist 横拼、含示范帧，同官方 ``run_group``），终态确定后存为
``{task}_{tier}_{episode}_{seed}.mp4``，路径写入结果行 ``video``；存盘失败只记 ``video_error``，不影响该局终态。
error 局重跑时同名录像被覆盖。

正常执行未完成任务如实记 fail／timeout，不为挑成功重试；只有基础设施 error 重跑，每身份最多 ``--max_attempts`` 次，
每片每轮合计最多 ``--retry_cap`` 次（默认 55），用满即打 ``RETRY_CAP_HIT`` 退出码 4。
不给 ``--identities`` 时退回 v6 旧口径（按 candidate 升序、``--per_round`` 分轮），v7 不走这条路。

用法：
    python -m robomme_sim.testhard_eval --benchmark_root <benchmark> --identities <清单.jsonl> --out <目录> \
        --round 1 --shard 0/10 --video_dir <录像目录> --pretrained_checkpoint <ckpt 绝对路径> [--resume] [--limit 1]
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

NORMAL = ("success", "fail", "timeout")
ALL_TASKS = ("PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap", "VideoUnmask",
             "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
             "PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick")
MANIFEST_KEYS = ("task", "episode", "tier", "seed", "candidate", "source_episode", "round", "shard")
EXPORT_TIER = "xhard0"


def _own_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    ap.add_argument("--shard", default="0/10", help="i/n：取身份清单里该轮 shard==i 的行（n 须等于该轮清单分片数）")
    ap.add_argument("--identities", default="", help="v7 身份清单 jsonl（每行一局，含 round/shard）；v7 必传")
    ap.add_argument("--video_dir", default="", help="每局录像目录，文件名 {task}_{tier}_{episode}_{seed}.mp4")
    ap.add_argument("--per_round", type=int, default=10, help="仅旧口径（无 --identities）：每格每轮局数")
    ap.add_argument("--limit", type=int, default=0, help="本片只跑前 N 条（smoke 用）")
    ap.add_argument("--only", default="", help="逗号分隔的 task/episode，只评这些（smoke 用）")
    ap.add_argument("--max_attempts", type=int, default=3, help="单身份基础设施 error 的最多尝试次数")
    ap.add_argument("--retry_cap", type=int, default=55, help="本片本轮基础设施 error 重跑合计上限")
    ap.add_argument("--resume", action="store_true", help="按结果文件里已有正常终态的身份续评（官方 parse_args 无此参数）")
    return ap.parse_known_args()


# ---------- 纯函数（不导入仿真与 torch，单测直接调用） ----------

def parse_shard(text: str) -> tuple[int, int]:
    """解析 ``i/n``，要求 0<=i<n。"""
    i, n = (int(x) for x in str(text).split("/"))
    if not (n > 0 and 0 <= i < n):
        raise ValueError(f"--shard 非法：{text}")
    return i, n


def load_manifest(path: str) -> list[dict]:
    """读身份清单：逐行校验字段齐全与类型，(task, episode) 不得重复；每行补 key=task/episode。"""
    rows, seen = [], set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in MANIFEST_KEYS if k not in row]
        if missing:
            raise ValueError(f"身份清单第 {lineno} 行缺字段 {missing}")
        if row["task"] not in ALL_TASKS:
            raise ValueError(f"身份清单第 {lineno} 行任务未知：{row['task']}")
        for k in ("episode", "seed", "round", "shard"):
            if not isinstance(row[k], int) or isinstance(row[k], bool):
                raise ValueError(f"身份清单第 {lineno} 行 {k} 须为整数：{row[k]!r}")
        if row["round"] not in (1, 2) or row["shard"] < 0:
            raise ValueError(f"身份清单第 {lineno} 行 round/shard 非法：{row['round']}/{row['shard']}")
        key = (row["task"], row["episode"])
        if key in seen:
            raise ValueError(f"身份清单重复身份：{key}")
        seen.add(key)
        rows.append(dict(row, key=f"{row['task']}/{row['episode']}"))
    if not rows:
        raise ValueError(f"身份清单为空：{path}")
    return rows


def select_rows(rows: list[dict], round_: int, shard_i: int, shard_n: int) -> list[dict]:
    """取 round==round_ 且 shard==shard_i 的行，按 (task, episode) 排序；该轮清单分片号须恰为 0..n-1。"""
    in_round = [r for r in rows if r["round"] == round_]
    shards = sorted({r["shard"] for r in in_round})
    if shards != list(range(shard_n)):
        raise ValueError(f"--shard 的 n={shard_n} 与清单第 {round_} 轮分片 {shards} 不符")
    return sorted((r for r in in_round if r["shard"] == shard_i), key=lambda r: (r["task"], r["episode"]))


def check_identity(row: dict, ident: dict) -> None:
    """清单行与 builder.resolve_identity 逐项核对，不符即抛 ValueError。"""
    diffs = []
    if ident.get("episode", row["episode"]) != row["episode"]:
        diffs.append(("episode", row["episode"], ident.get("episode")))
    for k in ("tier", "seed"):
        if ident.get(k) != row[k]:
            diffs.append((k, row[k], ident.get(k)))
    if row["tier"] == EXPORT_TIER:
        if ident.get("source_episode") != row["source_episode"]:
            diffs.append(("source_episode", row["source_episode"], ident.get("source_episode")))
        if row["candidate"] is not None or ident.get("candidate") is not None:
            diffs.append(("candidate", row["candidate"], ident.get("candidate")))
    elif ident.get("candidate") != row["candidate"]:
        diffs.append(("candidate", row["candidate"], ident.get("candidate")))
    if diffs:
        raise ValueError(f"身份清单与 builder 不符 {row['task']}/{row['episode']}：(字段, 清单, builder) {diffs}")


def classify_binding(tier: str, binding: dict | None, expected_sha: str | None) -> str:
    """按档核回注绑定并返回 binding_class（export|replay）；不合格抛 RuntimeError。"""
    b = binding or {}
    if tier == EXPORT_TIER:
        ok = (bool(b.get("available")) and b.get("mode") == "export" and b.get("spec_kind") == "native-parity/1"
              and b.get("injected_mismatch") == 0)
        cls = "export"
    else:
        ok = (bool(b.get("available")) and b.get("mode") == "replay" and b.get("injected_mismatch") == 0
              and (b.get("value_points") or 0) > 0 and expected_sha is not None
              and b.get("spec_sha256") == expected_sha and b.get("layout_drift", 0) == 0)
        cls = "replay"
    if not ok:
        raise RuntimeError(f"回注绑定不合格（{tier} 应为 {cls}）：{b}")
    return cls


def video_name(task: str, tier: str, episode: int, seed: int) -> str:
    return f"{task}_{tier}_{episode}_{seed}.mp4"


def save_video(recorder, video_dir: str, row: dict) -> tuple[str | None, str | None]:
    """终态确定后存录像；返回 (路径, 错误)，任何失败都不抛。"""
    if recorder is None:
        return None, None
    path = str(Path(video_dir).resolve() / video_name(row["task"], row["tier"], row["episode"], row["seed"]))
    try:
        if len(recorder) == 0:
            return None, "无帧可存（首帧之前即失败）"
        if recorder.save(path):
            return path, None
        return None, "RolloutVideoRecorder.save 返回 False（imageio 与 cv2 均未写出）"
    except Exception as exc:  # 录像失败不影响该局终态
        return None, f"{type(exc).__name__}: {exc}"


# ---------- 身份来源（惰性导入 robomme_hard） ----------

def _import_hard(benchmark_root: str):
    src = str(Path(benchmark_root).resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder, TIER_MAX_STEPS
    return BenchmarkEnvBuilder, TIER_MAX_STEPS


def identities(benchmark_root: str, per_round: int = 10) -> list[dict]:
    """旧口径（v6，无 --identities）：全部身份每格按 candidate 升序标轮次。"""
    BenchmarkEnvBuilder, TIER_MAX_STEPS = _import_hard(benchmark_root)
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


def manifest_identities(benchmark_root: str, rows: list[dict]) -> list[dict]:
    """v7：对清单行逐一取 builder 身份并核对，补 identity／spec_sha256（取 builder 值）／max_steps。"""
    BenchmarkEnvBuilder, TIER_MAX_STEPS = _import_hard(benchmark_root)
    builders, out = {}, []
    for row in rows:
        builder = builders.get(row["task"])
        if builder is None:
            builder = builders[row["task"]] = BenchmarkEnvBuilder(row["task"], dataset="test-hard")
        ident = builder.resolve_identity(row["episode"])
        check_identity(row, ident)
        out.append(dict(row, identity=ident, spec_sha256=ident.get("spec_sha256"),
                        max_steps=TIER_MAX_STEPS[row["tier"]]))
    return out


def main() -> int:
    own, rest = _own_args()
    sys.argv = [sys.argv[0], *rest]
    import torch

    from robomme_sim.eval_success import build_policy, parse_args
    args = parse_args()
    if not args.benchmark_root:
        raise ValueError("必须提供 --benchmark_root（含 robomme_hard 的 benchmark 检出）")
    args.dataset_split = "test-hard"
    ckpt = str(Path(args.pretrained_checkpoint).resolve())
    i, n = parse_shard(own.shard)
    try:
        if own.identities:
            manifest = load_manifest(own.identities)
            if own.only:
                keep = set(own.only.split(","))
                chosen = sorted((r for r in manifest if r["key"] in keep), key=lambda r: (r["task"], r["episode"]))
            else:
                chosen = select_rows(manifest, own.round, i, n)
            total = sum(r["round"] == own.round for r in manifest)
            mine = manifest_identities(args.benchmark_root, chosen)
        else:
            all_ids = identities(args.benchmark_root, own.per_round)
            picked = sorted((r for r in all_ids if r["round"] == own.round), key=lambda r: (r["task"], r["episode"]))
            total = len(picked)
            mine = picked[i::n]
            if own.only:
                keep = set(own.only.split(","))
                mine = [r for r in all_ids if r["key"] in keep]
    except ValueError as exc:  # 清单或身份核对不过：配置错误，整片退出码 2（启动脚本据此不再重跑）
        print(f"TESTHARD_CONFIG_ERROR {exc}", flush=True)
        return 2
    out = Path(own.out)
    out.mkdir(parents=True, exist_ok=True)
    video_dir = Path(own.video_dir).resolve() if own.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    results = out / f"results-r{own.round}-shard{i:02d}of{n:02d}.jsonl"
    done, tries, infra = set(), {}, 0
    if results.exists():
        if not own.resume:
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
    print(f"TESTHARD_START round={own.round} shard={i}/{n} total={total} mine={len(mine)} "
          f"pending={len(pending)} infra_so_far={infra} ckpt={ckpt} manifest={own.identities or '<v6 旧口径>'} "
          f"video_dir={video_dir or '<关闭>'}", flush=True)

    expect = Path(args.benchmark_root).resolve() / "src"
    batched, buffer_factory, normalizer = build_policy(args)
    from robomme_sim.inproc_pool import InProcSimPool
    recorder_cls, video_fps = None, 20.0
    if video_dir is not None:
        from robomme_sim.video_writer import RolloutVideoRecorder as recorder_cls
        # 帧率取策略的原生控制频率，同官方 run_group
        video_fps = float(getattr(buffer_factory(), "native_fps", 0) or 0) or 20.0

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
                      candidate=row["candidate"], source_episode=row.get("source_episode"), seed=row["seed"],
                      spec_sha256=row["spec_sha256"], identity=row["identity"], round=own.round,
                      max_steps=row["max_steps"], status="error", task_success=False, error_class=None, steps=0,
                      attempt=tries.get(row["key"], 0) + 1, checkpoint=ckpt, policy_seed=0, spec_binding=None,
                      binding_class=None, demo_frames=None, demo_tasks=None, error=None, video=None, video_error=None)
        recorder = recorder_cls(fps=video_fps) if recorder_cls is not None else None
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
            served = reset.get("identity") or {}
            for k in ("tier", "seed", "candidate", "source_episode", "spec_sha256"):
                if served.get(k) != row["identity"].get(k):
                    raise RuntimeError(f"服务返回的身份与本局不符：{k} 服务 {served.get(k)!r}，身份 {row['identity'].get(k)!r}")
            record["binding_class"] = classify_binding(row["tier"], binding, row["identity"].get("spec_sha256"))
            buffer = buffer_factory()
            buffer.reset()
            cam_map = {k.split(".")[-1]: k for k in buffer.image_keys}

            def observe(frames):
                for frame in frames:
                    buffer.observe({cam_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()})
                    if recorder is not None:  # 录像用原始 front/wrist 帧，同官方 run_group
                        recorder.add(frame)

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
        # 终态已定，再存录像；失败只记 video_error
        if recorder is not None:
            record["video"], record["video_error"] = save_video(recorder, str(video_dir), row)
            recorder = None
        record.update(elapsed_s=round(time.time() - started, 1),
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"EPISODE_END key={row['key']} status={record['status']} steps={record['steps']} "
              f"binding={record['binding_class']} video={record['video'] or record['video_error']} "
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
    mine_keys = {r["key"] for r in mine}
    final = {}
    for r in recs:
        if r["key"] not in mine_keys:
            continue
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
