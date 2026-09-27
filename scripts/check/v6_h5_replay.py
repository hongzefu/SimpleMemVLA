"""走评测接口回放生成侧 h5 的动作，核对能否推进到终态、画面是否与生成时一致。

reset 与 step 都经 ``robomme_sim.robomme_env.SimEnvService``（V6 快照模式，与 v6spec_eval 同一通路）；
动作取 h5 里 ``is_video_demo=False`` 的 ``joint_action``，逐条送入。
输出：每条一行 ``H5REPLAY key=… status=… steps=… h5_steps=… reset_mad=… mad_mean=… mad_max=…``，
以及 ``<out>/<key>.png``（上排 h5 帧、下排评测帧，取 5 个等分时刻，便于目视）。

用法：python scripts/check/v6_h5_replay.py --specs <specs.jsonl> --benchmark_root <bench> \
        --rollout <rollout/run1> --out <目录> [--keys PickXtimes/0,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from robomme_sim.robomme_env import SimEnvService


def _h5_steps(path: Path):
    with h5py.File(path, "r") as f:
        ep = f[next(iter(f.keys()))]
        names = sorted((k for k in ep.keys() if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1]))
        rows = []
        for k in names:
            t = ep[k]
            rows.append(dict(demo=bool(t["info/is_video_demo"][()]),
                             action=np.asarray(t["action/joint_action"][()], dtype=np.float64),
                             front=np.asarray(t["obs/front_rgb"][()], dtype=np.uint8)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--benchmark_root", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--max_steps", type=int, default=1500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [json.loads(l) for l in (Path(args.rollout) / "results.jsonl").read_text().splitlines() if l.strip()]
    wanted = set(filter(None, args.keys.split(",")))
    bench = Path(args.benchmark_root)
    svc = SimEnvService(max_steps=args.max_steps, benchmark_root=str(bench), v6_specs=args.specs)
    worst = 0
    for r in results:
        key = f"{r['task']}/{r['episode']}"
        if not r.get("ok") or (wanted and key not in wanted):
            continue
        rows = _h5_steps(bench / r["h5"])
        live = [x for x in rows if not x["demo"]]
        reset = svc.reset({"task": r["task"], "episode": r["episode"]})
        if not reset.get("ok"):
            print(f"H5REPLAY key={key} status=reset_error detail={reset.get('reason')}", flush=True)
            worst = 1
            continue
        eval_frames = [reset["frames"][-1]["front"]]
        status, steps = "ongoing", 0
        # h5 第 i 个非演示时刻的 obs 对应动作执行前的画面；动作逐条送入
        for x in live:
            resp = svc.step({"action_chunk": x["action"][None, :]})
            steps += resp["consumed"]
            if resp["frames"]:
                eval_frames.append(resp["frames"][-1]["front"])
            status = resp["status"]
            if resp["done"]:
                break
        n = min(len(eval_frames), len(live))
        mads = [float(np.abs(eval_frames[i].astype(np.int16) - live[i]["front"].astype(np.int16)).mean()) for i in range(n)]
        picks = sorted(set(int(round(q * (n - 1))) for q in (0, .25, .5, .75, 1.0)))
        top = np.concatenate([live[i]["front"] for i in picks], axis=1)
        bot = np.concatenate([eval_frames[i] for i in picks], axis=1)
        cv2.imwrite(str(out / f"{key.replace('/', '_')}.png"), cv2.cvtColor(np.concatenate([top, bot], axis=0), cv2.COLOR_RGB2BGR))
        print(f"H5REPLAY key={key} status={status} steps={steps} h5_steps={len(live)} "
              f"reset_mad={mads[0]:.2f} mad_mean={np.mean(mads):.2f} mad_max={np.max(mads):.2f}", flush=True)
        worst |= status != "success"
        svc.close()
    return int(worst)


if __name__ == "__main__":
    raise SystemExit(main())
