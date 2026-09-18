"""候选运行器：跨批次重启、同身份最多两次尝试、进程级超时和完整汇总。"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from robomme_sim.candidate_plan import (
    CONFIG, NORMAL, atomic_json, attempts, digest, load_plan, read_json,
    result_dir, summarize, verify_manifest, read_candidates, partition, key, manifest_tokens,
)


def slurm_memory_snapshot():
    """读取已核实的 GL cgroup v2 作业内存，避免把共享映射 RSS 当成实际收费内存。"""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return {}
    membership = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    relative = next(line.split(":", 2)[2] for line in membership if line.startswith("0::"))
    current = Path("/sys/fs/cgroup") / relative.lstrip("/")
    root = next(p for p in (current, *current.parents) if p.name == f"job_{job}")
    events = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
    snapshot = {"limit_bytes": int((root / "memory.max").read_text()),
                "peak_bytes": int((root / "memory.peak").read_text()),
                "oom_kill": int(events["oom_kill"]), "oom": int(events["oom"])}
    if snapshot["limit_bytes"] != 24 * 1024 ** 3:
        raise RuntimeError(f"已有 hold 内存上限与批准的 24 GiB 不符：{snapshot}")
    return snapshot


def record_crash(run, manifest, message):
    path = run / "active.json"
    if not path.exists():
        return False
    record = read_json(path)
    task, difficulty, episode = record["key"]
    row = dict(task=task, difficulty=difficulty, episode=episode)
    target = result_dir(run, row) / f"attempt-{record['attempt']}.json"
    if not target.exists():
        record.update(status="error", error=message, elapsed_s=time.time() - record["started"])
        atomic_json(target, record)
    path.unlink()
    return True


def launch(suite, lane, resume=False):
    suite = Path(suite).resolve()
    run = suite / lane
    if run.exists() and not resume:
        raise FileExistsError(f"运行目录存在，须显式恢复：{run}")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_manifest(suite, full=True)
        lane_plan = read_json(suite / "plans" / f"{lane}.json")
        if lane_plan["manifest_sha256"] not in manifest_tokens(manifest):
            raise ValueError("分片目录指纹不符")
        _, all_rows = read_candidates(manifest["benchmark_root"])
        expected_chunks = partition(all_rows)[lane]
        actual_chunks = []
        for relative in lane_plan["batches"]:
            _, _, batch_rows = load_plan(suite / relative, manifest)
            actual_chunks.append([key(r) for r in batch_rows])
        if actual_chunks != expected_chunks:
            raise ValueError("分片计划与批准的偶数/奇数分配或冒烟顺序不同")
        expected = os.environ.get("SLURM_JOB_ID")
        if lane.startswith("hold") and expected != {"hold01": "61495429", "hold02": "61495430"}[lane]:
            raise ValueError("正式分片只能在对应已批准 hold 内执行")
        if lane == "local-smoke" and expected:
            raise ValueError("本机冒烟不能在 GL 执行")
        record_crash(run, manifest, "上次运行中断；保留原候选和 seed")
        memory_before = slurm_memory_snapshot()
        for relative in lane_plan["batches"]:
            plan_path = suite / relative
            _, _, rows = load_plan(plan_path, manifest)
            while True:
                records = [attempts(run, r, manifest) for r in rows]
                pending = [r for r, rs in zip(rows, records)
                           if not rs or (rs[-1]["status"] == "error" and len(rs) < CONFIG["max_attempts"])]
                if not pending:
                    if any(not rs or rs[-1]["status"] not in NORMAL for rs in records):
                        raise RuntimeError("固定候选两次尝试仍有运行错误，停止该路后续批次")
                    break
                count_before = sum(len(rs) for rs in records)
                command = ["uv", "run", "--frozen", "--no-sync", "python", "-m", "robomme_sim.eval_success",
                           "--benchmark_root", manifest["benchmark_root"], "--episode_plan", str(plan_path),
                           "--run_dir", str(run), "--resume", "--pretrained_checkpoint", manifest["checkpoint"]]
                for name in ("max_steps", "execute_horizon", "num_denoising_steps", "eval_temperature",
                             "max_subtask_tokens", "compute_dtype", "attn_implementation", "group_size", "num_gpus"):
                    command.extend([f"--{name}", str(CONFIG[name])])
                print(f"BATCH_START lane={lane} plan={plan_path.name} pending={len(pending)}", flush=True)
                child = subprocess.Popen(command, start_new_session=True)
                last_active = time.time()
                reason = None
                try:
                    while child.poll() is None:
                        active = run / "active.json"
                        try:
                            deadline = read_json(active)["deadline"]
                            last_active = time.time()
                        except FileNotFoundError:
                            deadline = last_active + 1800
                        if time.time() > deadline:
                            reason = "回合阶段超时，已终止整个批次进程组"
                            os.killpg(child.pid, signal.SIGKILL)
                            break
                        time.sleep(2)
                    code = child.wait()
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                record_crash(run, manifest, reason or f"批次进程异常退出：{code}")
                count_after = sum(len(attempts(run, r, manifest)) for r in rows)
                print(f"BATCH_EXIT lane={lane} code={code} new_attempts={count_after - count_before}", flush=True)
                if count_after == count_before:
                    raise RuntimeError("批次没有产生回合记录，不能将导入/模型启动失败当作完成")
            latest = [attempts(run, r, manifest)[-1] for r in rows]
            peak = max(r["max_rss_kib"] for r in latest)
            memory_after = slurm_memory_snapshot()
            if memory_after and memory_after["oom_kill"] > memory_before["oom_kill"]:
                raise RuntimeError(f"本次运行出现 cgroup OOM kill：{memory_after}")
            print(f"BATCH_PASS lane={lane} plan={plan_path.name} episodes={len(rows)} "
                  f"peak_rss_kib={peak} cgroup={memory_after}", flush=True)
        if not summarize(suite, lane)["complete"]:
            raise RuntimeError("分片完整性验收失败")


def main():
    def stop(_signum, _frame):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("lane", choices=["local-smoke", "hold01", "hold02"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    launch(args.suite, args.lane, args.resume)


if __name__ == "__main__":
    main()
