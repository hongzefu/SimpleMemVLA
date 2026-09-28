
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robomme_sim._sapien_env import prepare_sapien_runtime

prepare_sapien_runtime()

VENDORED_SIM_DIR = os.path.dirname(os.path.abspath(__file__))

CAM_FRONT = "front"
CAM_WRIST = "wrist"


def _setup_robomme_path(benchmark_root: str | None = None) -> None:
    source = Path(benchmark_root).resolve() / "src" if benchmark_root else Path(VENDORED_SIM_DIR)
    loaded = sys.modules.get("robomme")
    if loaded is not None and Path(loaded.__file__).resolve().parent != source / "robomme":
        raise RuntimeError(f"robomme 已从另一来源导入：{loaded.__file__}，期望 {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def pin_worker_gpu(gpu_id: int) -> str:
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    dev = visible[gpu_id] if gpu_id < len(visible) else str(gpu_id)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    return dev


def _to_uint8_hwc(img) -> np.ndarray:
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.uint8, copy=False)


def _to_f32(vec) -> np.ndarray:
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def encode_frames(obs: dict) -> list[dict[str, np.ndarray]]:
    fronts = obs["front_rgb_list"]
    wrists = obs["wrist_rgb_list"]
    return [
        {CAM_FRONT: _to_uint8_hwc(f), CAM_WRIST: _to_uint8_hwc(w)}
        for f, w in zip(fronts, wrists)
    ]


def encode_states(obs: dict) -> list[np.ndarray]:
    joints = obs["joint_state_list"]
    grippers = obs["gripper_state_list"]
    states = []
    for j, g in zip(joints, grippers):
        jv = _to_f32(j)[:7]
        gv = _to_f32(g)
        g0 = gv[:1] if gv.size else np.zeros(1, dtype=np.float32)
        states.append(np.concatenate([jv, g0]).astype(np.float32))
    return states


def _scalar(x) -> float:
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            return float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).reshape(-1)[0])
    return float(x)


class RoboMMESimEnv:

    def __init__(
        self,
        task_name: str,
        dataset_split: str = "test",
        max_steps: int = 1300,
        action_space: str = "joint_angle",
        benchmark_root: str | None = None,
    ):
        _setup_robomme_path(benchmark_root)
        import robomme.robomme_env
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        if dataset_split == "test-hard":  # test-hard 只在 robomme_hard 里：同 id 环境由它接管
            from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder, TIER_MAX_STEPS, spec_binding
            self._hard = (TIER_MAX_STEPS, spec_binding)

        self.task_name = task_name
        self.dataset_split = dataset_split
        self.max_steps = int(max_steps)
        self.builder = BenchmarkEnvBuilder(
            env_id=task_name,
            dataset=dataset_split,
            action_space=action_space,
            max_steps=self.max_steps,
        )
        n = self.builder.get_episode_num()
        if n <= 0:
            raise RuntimeError(
                f"RoboMME metadata lists no '{dataset_split}' episodes for task "
                f"'{task_name}' (env_metadata missing?)."
            )
        self.num_episodes = n
        self.env = None
        self._status = "ongoing"
        self._done = False

    def reset(self, episode_idx: int) -> tuple[dict, dict]:
        self.close()
        hard = getattr(self, "_hard", None)
        if hard is None:
            env = self.builder.make_env_for_episode(int(episode_idx))
        else:  # 按档步数上限逐局传入；reset 之后取回注绑定摘要
            _seed, tier = self.builder.resolve_episode(int(episode_idx))
            env = self.builder.make_env_for_episode(int(episode_idx), max_steps=hard[0][tier])
        obs, info = env.reset()
        self.env = env
        if hard is not None:
            info = dict(info, tier=tier, max_steps=hard[0][tier], spec_binding=hard[1](env),
                        identity=self.builder.resolve_identity(int(episode_idx)),
                        demo_frames=max(0, len(obs["front_rgb_list"]) - 1),
                        demo_tasks=sum(bool(t.get("demonstration", False)) for t in env.unwrapped.task_list))
        self._status = str(info.get("status", "ongoing")) if isinstance(info, dict) else "ongoing"
        self._done = False
        return obs, info

    def step_one(self, action: np.ndarray) -> tuple[dict | None, bool, bool, dict]:
        if self.env is None:
            raise RuntimeError("call reset() before step_one()")
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if act.shape[0] < 8:
            raise ValueError(f"Expected an 8-dim joint_angle action, got shape {act.shape}")
        obs, _reward, terminated, truncated, info = self.env.step(act[:8])
        term = bool(_scalar(terminated)) if terminated is not None else False
        trunc = bool(_scalar(truncated)) if truncated is not None else False
        status = str(info.get("status", "ongoing")) if isinstance(info, dict) else "ongoing"
        self._status = status
        self._done = term or trunc or status in ("success", "fail", "timeout", "error")
        return obs, term, trunc, info if isinstance(info, dict) else {}

    @property
    def status(self) -> str:
        return self._status

    @property
    def success(self) -> bool:
        return self._status == "success"

    @property
    def done(self) -> bool:
        return self._done

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


class SimEnvService:

    def __init__(self, dataset_split: str = "test", max_steps: int = 1300,
                 reset_retries: int = 2, benchmark_root: str | None = None):
        self.benchmark_root = benchmark_root
        self.dataset_split = dataset_split
        self.max_steps = int(max_steps)
        self.reset_retries = int(reset_retries)
        self.env: RoboMMESimEnv | None = None
        self.task_name: str | None = None

    def _ensure_env(self, task_name: str):
        if self.env is None or self.task_name != task_name:
            if self.env is not None:
                self.env.close()
            self.env = RoboMMESimEnv(
                task_name, dataset_split=self.dataset_split, max_steps=self.max_steps,
                benchmark_root=self.benchmark_root,
            )
            self.task_name = task_name

    def reset(self, payload: dict) -> dict:
        task_name = payload["task"]
        episode = int(payload["episode"])
        last_err = "unknown"
        for attempt in range(self.reset_retries + 1):
            try:
                self._ensure_env(task_name)
                obs, info = self.env.reset(episode)
                frames = encode_frames(obs)
                states = encode_states(obs)
                if not frames or not states:
                    raise RuntimeError("reset returned no frames")
                task_goal = info.get("task_goal")
                if isinstance(task_goal, (list, tuple)) and task_goal:
                    instruction = str(task_goal[0])
                else:
                    instruction = str(task_goal) if task_goal else f"Complete the {task_name} task."
                return {
                    "ok": True,
                    "episode": episode,
                    "instruction": instruction,
                    "frames": frames,
                    "states": states,
                    "max_steps": info.get("max_steps", self.max_steps),
                    **{k: info[k] for k in ("tier", "identity", "spec_binding", "demo_frames", "demo_tasks") if k in info},
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                try:
                    if self.env is not None:
                        self.env.close()
                except Exception:
                    pass
                self.env = None
                self.task_name = None
        return {"ok": False, "reason": f"reset_failed: {last_err}"}

    def step(self, payload: dict) -> dict:
        action_chunk = np.asarray(payload["action_chunk"], dtype=np.float64)
        frames, states, consumed = [], [], 0
        error = None
        for action in action_chunk:
            obs, _term, _trunc, info = self.env.step_one(action)
            consumed += 1
            if self.env.status == "error" or obs is None:
                error = str(info.get("error_message", "env step error"))
                break
            frames.extend(encode_frames(obs))
            states.extend(encode_states(obs))
            if self.env.done:
                break
        return {
            "frames": frames,
            "states": states,
            "consumed": consumed,
            "done": bool(self.env.done or error is not None),
            "success": bool(self.env.success),
            "status": self.env.status,
            "error_message": error,
        }

    def obs(self, _payload=None) -> dict:
        raise NotImplementedError("RoboMME frames are consumed from reset/step returns")

    def close(self, _payload=None) -> dict:
        if self.env is not None:
            self.env.close()
            self.env = None
        return {"ok": True}
