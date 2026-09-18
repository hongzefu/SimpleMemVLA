#!/usr/bin/env python3
"""RoboMME 纯模拟器冒烟：不加载模型，只测仿真侧能不能跑、以及 reset 到底多慢。

这一步把「SAPIEN / ManiSkill / mplib 能不能跑」与「模型能不能跑」彻底解耦，是整条链路上
性价比最高的一次测量。RoboMME 的 reset 会用 mplib motion planner 逐步执行并渲染整段
conditioning demo（60 秒历史窗口 @ 20 Hz 意味着数百步），eval_success.py 把
`--reset_timeout` 默认设成 3600 秒本身就是「reset 很慢」的强信号——先把这个数测出来，
才能给正式评测一个靠谱的耗时预期和 Monitor 告警阈值。

用法：
    PYTHONPATH=<repo> python scripts/check/sim_probe.py [task_name] [n_steps]
默认 task 取 PickXtimes（DEFAULT_TASKS 的第一个），默认走 16 步（一个 execute_horizon）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "PickXtimes"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    if not os.environ.get("VK_ICD_FILENAMES"):
        for cand in ("/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
                     "/usr/share/vulkan/icd.d/nvidia_icd.json"):
            if os.path.isfile(cand):
                os.environ["VK_ICD_FILENAMES"] = cand
                break
    os.environ.setdefault("MPLBACKEND", "Agg")  # robomme 的 env 模块顶部 import pyplot

    t0 = time.time()
    import robomme_sim  # 触发 fla torch shim + prepare_sapien_runtime
    from robomme_sim.robomme_env import RoboMMESimEnv
    print(f"[sim_probe] import robomme_sim 用时 {time.time() - t0:.1f}s")

    t0 = time.time()
    env = RoboMMESimEnv(task, dataset_split="test", max_steps=1300)
    print(f"[sim_probe] builder 就绪 用时 {time.time() - t0:.1f}s  "
          f"task={task}  test split 共 {env.num_episodes} 个 episode")

    t0 = time.time()
    obs, info = env.reset(0)
    reset_sec = time.time() - t0
    print(f"[sim_probe] RESET 用时 {reset_sec:.1f}s   <- 这是总耗时估算的主要变量")
    if isinstance(obs, dict):
        print(f"[sim_probe] obs keys: {sorted(obs.keys())}")
        for k, v in obs.items():
            if isinstance(v, (list, tuple)):
                head = np.asarray(v[0]).shape if len(v) else "空"
                print(f"[sim_probe]   {k}: list(len={len(v)}) 首元素 shape={head}")
            elif hasattr(v, "shape"):
                print(f"[sim_probe]   {k}: shape={tuple(np.asarray(v).shape)}")
    if isinstance(info, dict):
        print(f"[sim_probe] info: { {k: info[k] for k in list(info)[:6]} }")

    t0 = time.time()
    zero_action = np.zeros(8, dtype=np.float64)
    for _ in range(n_steps):
        env.step_one(zero_action)
    step_sec = time.time() - t0
    print(f"[sim_probe] {n_steps} 步仿真用时 {step_sec:.2f}s  "
          f"({step_sec / n_steps * 1000:.0f} ms/step)  status={env.status}")

    # 按 hard_bound = ceil(1300/16) + 2 = 84 次决策给出一个粗估
    per_decision_sim = step_sec / n_steps * 16
    print(f"[sim_probe] 推算：每次决策的仿真开销 ≈ {per_decision_sim:.2f}s，"
          f"跑满 84 次决策的纯仿真时间 ≈ {per_decision_sim * 84 / 60:.1f} min"
          f"（不含模型前向），单 episode 下限 ≈ {(reset_sec + per_decision_sim * 84) / 60:.1f} min")

    env.close()
    print("SIM_PROBE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
