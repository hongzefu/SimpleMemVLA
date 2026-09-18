"""700 条候选的固定计划、逐回合账本和完整性汇总。准备阶段仅在本机执行 Git。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
BENCHMARK_COMMIT = "b4e97f22fe007078e297205898c07c1acbc69165"
CANDIDATES = "artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl"
CANDIDATES_SHA = "32bdbdb4931bf32ab5e37613da86ac9f68ab42eb7aec8d9a8d80103af9481809"
GROUPS = {
    "BinFill": ("easy", "medium", "hard"),
    "RouteStick": ("easy", "medium", "hard", "xhard"),
    "VideoUnmaskSwap": ("easy", "medium", "hard", "xhard"),
    "VideoRepick": ("easy", "medium", "xhard"),
}
CONFIG = dict(max_steps=2000, execute_horizon=16, num_denoising_steps=10,
              eval_temperature=1.0, max_subtask_tokens=64, compute_dtype="bfloat16",
              attn_implementation="sdpa", group_size=1, num_gpus=1,
              policy_seed=0, binfill_demo=False, max_attempts=2, chunk_size=14)
NORMAL = {"success", "fail", "timeout"}
# 只允许修复调度/持久化控制层；环境、模型、评估循环和推理配置不能沿用旧结果。
CONTROLLER_FILES = {"robomme_sim/candidate_launcher.py", "robomme_sim/candidate_plan.py",
                    "scripts/run_candidate_campaign.sh", "tests/test_candidate_eval.py"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def manifest_tokens(manifest):
    tokens = {digest(manifest)}
    ignored = {"source_commit", "source_files", "previous_manifests"}
    for old in manifest.get("previous_manifests", []):
        if old.get("previous_manifests"):
            raise ValueError("不支持嵌套来源修订")
        if {k: v for k, v in old.items() if k not in ignored} != {
                k: v for k, v in manifest.items() if k not in ignored}:
            raise ValueError("来源修订改变了权重、候选或评估配置")
        old_files, new_files = old["source_files"], manifest["source_files"]
        if old_files.keys() != new_files.keys():
            raise ValueError("来源修订改变了运行文件集合")
        changed = {p for p in old_files if old_files[p] != new_files[p]}
        if not changed or not changed <= CONTROLLER_FILES:
            raise ValueError(f"来源修订超出控制层：{sorted(changed)}")
        tokens.add(digest(old))
    return tokens


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cheap_file(path):
    """批次检查大小及首尾；完整 SHA 在每个运行器启动时检查，不能以此替代。"""
    p = Path(path)
    with p.open("rb") as f:
        first = f.read(1024 * 1024)
        f.seek(max(0, p.stat().st_size - 1024 * 1024))
        tail = f.read()
    return {"size": p.stat().st_size, "edges_sha256": hashlib.sha256(first + tail).hexdigest()}


def atomic_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(row):
    return row["task"], row["difficulty"], row["episode"]


def read_candidates(root):
    root = Path(root).resolve()
    path = root / CANDIDATES
    if sha_file(path) != CANDIDATES_SHA:
        raise ValueError("候选文件与批准版本不同")
    # 不注册环境，不导入 CUDA；验证器本身随固定子模块保存。
    spec = importlib.util.spec_from_file_location("_frozen_candidates", root / "src/robomme/injection_candidates.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    header, rows = reader.load_candidates(path, repo_root=root)
    selected = [r for r in rows if r["split"] == "test" and r["role"] == "primary"]
    by_group = defaultdict(list)
    for row in selected:
        by_group[key(row)[:2]].append(row)
    expected = {(task, diff) for task, diffs in GROUPS.items() for diff in diffs}
    if set(by_group) != expected or any(len(rs) != 50 for rs in by_group.values()):
        raise ValueError("候选必须为指定的 14 组 × 50")
    if len({key(r) for r in selected}) != 700:
        raise ValueError("候选存在重复键")
    return header, sorted(selected, key=key)


def partition(rows):
    groups = defaultdict(list)
    for row in sorted(rows, key=key):
        groups[key(row)[:2]].append(row)
    lanes = {}
    for lane, parity in (("hold01", 0), ("hold02", 1)):
        lanes[lane] = [[key(groups[g][2 * i + parity]) for g in sorted(groups)] for i in range(25)]
    smoke = [key(groups[g][0]) for g in sorted(groups)]
    first = next(k for k in smoke if k[:2] == ("BinFill", "hard"))
    lanes["local-smoke"] = [[first], [k for k in smoke if k != first]]
    a = {k for chunk in lanes["hold01"] for k in chunk}
    b = {k for chunk in lanes["hold02"] for k in chunk}
    if len(a) != 350 or len(b) != 350 or a & b or a | b != {key(r) for r in rows}:
        raise ValueError("两路分片不是互斥的 350 + 350")
    return lanes


def git(*args, root=REPO):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def prepare(output, benchmark_root, checkpoint):
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("准备计划和 Git 校验只能在本机执行")
    output, root, ckpt = map(lambda p: Path(p).resolve(), (output, benchmark_root, checkpoint))
    if output.exists():
        raise FileExistsError(f"运行根已存在，不能覆盖：{output}")
    if git("status", "--porcelain") or git("status", "--porcelain", root=root):
        raise RuntimeError("正式计划必须从主仓库及子模块的干净提交生成")
    if git("rev-parse", "HEAD", root=root) != BENCHMARK_COMMIT:
        raise ValueError("benchmark 提交不符")
    header, rows = read_candidates(root)
    files = {}
    # 整个运行依赖的源码锁定；计算节点只读并核验这些字节，不调用 Git。
    for tree in (REPO, root, REPO / "third_party/ManiSkill"):
        for name in git("ls-files", root=tree).splitlines():
            p = tree / name
            if p.is_file() and (p.suffix in {".py", ".sh", ".toml", ".lock", ".json", ".jinja"}
                                or p.name == ".gitmodules"):
                files[str(p.relative_to(REPO))] = sha_file(p)
    assets = {p.name: dict(sha256=sha_file(p), **cheap_file(p))
              for p in sorted(ckpt.iterdir()) if p.is_file()}
    manifest = dict(schema=1, source_commit=git("rev-parse", "HEAD"),
                    benchmark_commit=BENCHMARK_COMMIT, benchmark_root=str(root),
                    candidates_sha256=CANDIDATES_SHA, identity_sha256=header["identity_sha256"],
                    checkpoint=str(ckpt), assets=assets, source_files=files, config=CONFIG)
    token = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    for lane, chunks in partition(rows).items():
        paths = []
        for i, keys in enumerate(chunks):
            path = output / "plans" / lane / f"batch-{i:02d}.json"
            atomic_json(path, dict(manifest_sha256=token, keys=keys, lane=lane, batch=i))
            paths.append(str(path.relative_to(output)))
        atomic_json(output / "plans" / f"{lane}.json", dict(manifest_sha256=token, batches=paths))
    print(f"PLAN_PASS total=700 hold01=350 hold02=350 smoke=14 identity={header['identity_sha256']}", flush=True)


def verify_manifest(suite, full=False):
    manifest = read_json(Path(suite) / "manifest.json")
    manifest_tokens(manifest)
    if manifest["config"] != CONFIG or manifest["benchmark_commit"] != BENCHMARK_COMMIT:
        raise ValueError("运行配置或 benchmark 版本不符")
    for relative, expected in manifest["source_files"].items():
        if sha_file(REPO / relative) != expected:
            raise ValueError(f"运行源码发生变化：{relative}")
    for name, expected in manifest["assets"].items():
        p = Path(manifest["checkpoint"]) / name
        if cheap_file(p) != {k: expected[k] for k in ("size", "edges_sha256")}:
            raise ValueError(f"权重/配置指纹不符：{name}")
        if full and sha_file(p) != expected["sha256"]:
            raise ValueError(f"权重完整 SHA 不符：{name}")
    return manifest


def load_plan(path, manifest):
    plan = read_json(path)
    keys = [tuple(k) for k in plan["keys"]]
    if plan["manifest_sha256"] not in manifest_tokens(manifest) or not 0 < len(keys) <= 14 or len(set(keys)) != len(keys):
        raise ValueError("批次计划指纹、长度或唯一键不符")
    header, rows = read_candidates(manifest["benchmark_root"])
    lookup = {key(r): r for r in rows}
    if any(k not in lookup for k in keys):
        raise ValueError("计划引用了 test/primary 以外的候选")
    return plan, header, [lookup[k] for k in keys]


def result_dir(run_dir, candidate):
    task, diff, ep = key(candidate)
    return Path(run_dir) / "episodes" / task / diff / str(ep)


def attempts(run_dir, candidate, manifest):
    paths = sorted(result_dir(run_dir, candidate).glob("attempt-*.json"))
    records = [read_json(p) for p in paths]
    for i, r in enumerate(records, 1):
        if (tuple(r["key"]) != key(candidate) or r["spec_sha256"] != candidate["spec_sha256"]
                or r["seed"] != candidate["seed"] or r["manifest_sha256"] not in manifest_tokens(manifest)
                or r["attempt"] != i or r["status"] not in NORMAL | {"error"}):
            raise ValueError(f"回合结果身份、状态或尝试顺序不符：{paths[i - 1]}")
    if len(records) > 2 or any(r["status"] in NORMAL for r in records[:-1]):
        raise ValueError("出现超过两次尝试或对正常终态重跑")
    return records


def check_normal(record, run_dir, decode=False):
    if record["status"] not in NORMAL:
        return False
    p = Path(run_dir) / record["video"]
    if not record.get("video_frames") or sha_file(p) != record["video_sha256"]:
        raise ValueError(f"视频缺失或被修改：{p}")
    if decode:
        from robomme_sim.video_writer import verify_video
        verify_video(p, record["video_frames"])
    return True


def summarize(suite, lane=None, decode=True):
    suite = Path(suite)
    manifest = verify_manifest(suite)
    _, rows = read_candidates(manifest["benchmark_root"])
    lookup = {key(r): r for r in rows}
    lanes = partition(rows)
    names = [lane] if lane else ["hold01", "hold02"]
    groups = defaultdict(lambda: defaultdict(int))
    errors, missing = [], []
    for name in names:
        run = suite / name
        for k in [k for c in lanes[name] for k in c]:
            g = "/".join(k[:2])
            groups[g]["planned"] += 1
            records = attempts(run, lookup[k], manifest)
            if not records:
                groups[g]["missing"] += 1
                missing.append(k)
                continue
            r = records[-1]
            if check_normal(r, run, decode):
                groups[g][r["status"]] += 1
            else:
                groups[g]["error"] += 1
                errors.append(k)
    for stats in groups.values():
        stats["success_rate"] = stats["success"] / stats["planned"]
    total = sum(g["planned"] for g in groups.values())
    successes = sum(g["success"] for g in groups.values())
    payload = dict(complete=not errors and not missing, planned=total, successes=successes,
                   micro_avg=successes / total, macro_avg=sum(g["success_rate"] for g in groups.values()) / len(groups),
                   groups=dict(groups), errors=errors, missing=missing, manifest_sha256=digest(manifest))
    atomic_json((suite / lane if lane else suite) / "results.json", payload)
    print(f"COVERAGE_{'PASS' if payload['complete'] else 'FAIL'} planned={total} success={successes} "
          f"errors={len(errors)} missing={len(missing)}", flush=True)
    return payload


def revise_controller(suite):
    """停机后的窄范围修订：保存完整旧指纹，旧结果与旧计划不改一个字节。"""
    if os.environ.get("SLURM_JOB_ID") or git("status", "--porcelain"):
        raise RuntimeError("控制器修订须在本机干净提交上执行")
    suite = Path(suite)
    locks = []
    try:
        for lane in ("local-smoke", "hold01", "hold02"):
            lock = (suite / lane / "runner.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(suite / "manifest.json")
        if old.get("previous_manifests"):
            raise ValueError("本入口只允许一次明确的控制层修订")
        updated = copy.deepcopy(old)
        updated["source_commit"] = git("rev-parse", "HEAD")
        updated["source_files"] = {p: sha_file(REPO / p) for p in old["source_files"]}
        updated["previous_manifests"] = [old]
        manifest_tokens(updated)
        for name, expected in old["assets"].items():
            if sha_file(Path(old["checkpoint"]) / name) != expected["sha256"]:
                raise ValueError(f"权重发生变化：{name}")
        backup = suite / "manifest.before-controller-revision.json"
        if backup.exists():
            raise FileExistsError(backup)
        atomic_json(backup, old)
        atomic_json(suite / "manifest.json", updated)
        print(f"CONTROLLER_REVISION_PASS old={digest(old)} new={digest(updated)} "
              f"source_commit={updated['source_commit']}", flush=True)
    finally:
        for lock in locks:
            lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("suite")
    p.add_argument("--benchmark_root", default=str(REPO / "third_party/robomme_benchmark"))
    p.add_argument("--checkpoint", default=str(REPO / "checkpoints/simplememvla_robomme"))
    p = sub.add_parser("summarize")
    p.add_argument("suite")
    p.add_argument("--lane", choices=["local-smoke", "hold01", "hold02"])
    p = sub.add_parser("revise-controller")
    p.add_argument("suite")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.suite, args.benchmark_root, args.checkpoint)
    elif args.command == "revise-controller":
        revise_controller(args.suite)
    else:
        raise SystemExit(0 if summarize(args.suite, args.lane)["complete"] else 1)


if __name__ == "__main__":
    main()
