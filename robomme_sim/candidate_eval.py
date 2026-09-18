"""单进程最多 14 条候选；复用 SimpleMemVLA 模型与观测缓冲，不改官方评估行为。"""
from __future__ import annotations

import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch

from robomme_sim.candidate_plan import (
    CONFIG, NORMAL, atomic_json, attempts, check_normal, digest, key,
    load_plan, result_dir, sha_file, verify_manifest,
)
from robomme_sim.inproc_pool import InProcSimPool
from robomme_sim.video_writer import StreamingVideoRecorder, verify_video


def evaluate_candidates(args):
    from robomme_sim.eval_success import build_policy
    if not args.run_dir or not args.benchmark_root:
        raise ValueError("候选模式必须提供 --run_dir 和 --benchmark_root")
    run = Path(args.run_dir).resolve()
    manifest = verify_manifest(run.parent)
    plan, header, rows = load_plan(args.episode_plan, manifest)
    if plan["lane"] != run.name or Path(args.benchmark_root).resolve() != Path(manifest["benchmark_root"]):
        raise ValueError("候选计划与运行目录或 benchmark 不匹配")
    if Path(args.pretrained_checkpoint).resolve() != Path(manifest["checkpoint"]):
        raise ValueError("权重路径与固定计划不同")
    if args.pipelined or args.tasks:
        raise ValueError("候选模式不接受官方任务列表或流水线推理")
    for field in ("max_steps", "execute_horizon", "num_denoising_steps", "eval_temperature",
                  "max_subtask_tokens", "compute_dtype", "attn_implementation", "group_size", "num_gpus"):
        if getattr(args, field) != CONFIG[field]:
            raise ValueError(f"固定推理参数不符：{field}")
    pending = []
    for row in rows:
        records = attempts(run, row, manifest)
        if records and not args.resume:
            raise FileExistsError("已有结果，只能显式 --resume 恢复")
        if records and check_normal(records[-1], run):
            continue
        if len(records) >= CONFIG["max_attempts"]:
            continue
        pending.append(row)
    if not pending:
        return 0 if all(attempts(run, r, manifest)[-1]["status"] in NORMAL for r in rows) else 1

    batched, buffer_factory, normalizer = build_policy(args)
    pool = InProcSimPool(1, max_steps=args.max_steps, benchmark_root=args.benchmark_root,
                         reset_timeout=args.reset_timeout, step_timeout=args.step_timeout)
    pool.start()
    token = digest(manifest)
    for row in pending:
        number = len(attempts(run, row, manifest)) + 1
        started = time.time()
        record = dict(key=key(row), seed=row["seed"], spec_sha256=row["spec_sha256"],
                      manifest_sha256=token, attempt=number, status="error", steps=0,
                      started=started, pid=os.getpid(), policy_seed=0, video=None,
                      demo_frames=None, demo_tasks=None)
        active = run / "active.json"
        atomic_json(active, dict(record, phase="reset", deadline=time.time() + args.reset_timeout + 30))
        print(f"EPISODE_START key={key(row)} attempt={number} pid={os.getpid()}", flush=True)
        recorder = None
        try:
            reset = pool.reset([dict(task=row["task"], episode=row["episode"], candidate=row,
                                     sampling=header["sampling_config"])])[0]
            if not reset or not reset.get("ok"):
                raise RuntimeError(f"reset 错误：{reset}")
            record.update(demo_frames=reset["demo_frames"], demo_tasks=reset["demo_tasks"])
            print(f"DEMO_CHECK key={key(row)} tasks={record['demo_tasks']} frames={record['demo_frames']}", flush=True)
            buffer = buffer_factory()
            buffer.reset()
            cam_map = {k.split(".")[-1]: k for k in buffer.image_keys}
            video = result_dir(run, row) / f"attempt-{number}.mp4"
            recorder = StreamingVideoRecorder(video, fps=float(buffer.native_fps or 20))

            def observe(frames):
                for frame in frames:
                    buffer.observe({cam_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()})
                    recorder.add(frame)

            observe(reset["frames"])
            state = reset["states"][-1]
            torch.manual_seed(0)
            np.random.seed(0)
            while record["steps"] < args.max_steps:
                atomic_json(active, dict(record, phase="decision", deadline=time.time() + args.step_timeout + 30))
                normalized = None
                if normalizer is not None:
                    normalized = normalizer.normalize(torch.from_numpy(np.asarray(state, dtype=np.float32)))
                    normalized = normalized.unsqueeze(0).to(batched.device)
                actions, subtask = batched.generate_batch([buffer._prepare_inputs(reset["instruction"])], [normalized])[0]
                count = min(args.execute_horizon, args.max_steps - record["steps"])
                chunk = np.asarray(actions[:count])
                if chunk.shape != (count, 8) or not np.isfinite(chunk).all():
                    raise ValueError(f"模型动作形状或数值非法：{chunk.shape}")
                atomic_json(active, dict(record, phase="step", deadline=time.time() + args.step_timeout + 30))
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
            atomic_json(active, dict(record, phase="video", deadline=time.time() + 300))
            recorder.close()
            frames = verify_video(video, recorder.count)
            record.update(video=str(video.relative_to(run)), video_frames=frames, video_sha256=sha_file(video))
        except Exception:
            record.update(status="error", error=traceback.format_exc())
            print(record["error"], flush=True)
        finally:
            if recorder is not None:
                try:
                    recorder.close()
                except Exception:
                    record.update(status="error", error=traceback.format_exc())
            record.update(elapsed_s=time.time() - started,
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            path = result_dir(run, row) / f"attempt-{number}.json"
            if path.exists():
                raise FileExistsError(path)
            atomic_json(path, record)
            active.unlink(missing_ok=True)
        print(f"EPISODE_END key={key(row)} status={record['status']} steps={record['steps']} "
              f"elapsed={record['elapsed_s']:.1f}s rss_kib={record['max_rss_kib']}", flush=True)
        if record["status"] == "error":
            # Future 超时不能取消运行中的线程；退出整个进程后外层才允许恢复。
            return 1
        pool.services[0].close()
    pool.close()
    return 0
