
from __future__ import annotations

import os

import numpy as np

CAMERA_ORDER = ("front", "wrist")


class StreamingVideoRecorder:
    """逐帧编码；成功关闭并完整解码后，调用方才发布 episode 结果。"""

    def __init__(self, path, fps=20.0):
        from pathlib import Path
        import imageio.v2 as imageio
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.writer = imageio.get_writer(
            str(self.path), format="FFMPEG", mode="I", fps=fps,
            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_params=["-threads", "1"],
        )
        self.count = 0

    def add(self, frame):
        tiled = tile_frame(frame)
        if tiled is None:
            raise ValueError("视频帧缺少相机图像")
        self.writer.append_data(tiled)
        self.count += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def verify_video(path, expected_frames=None):
    """完整逐帧解码；只看文件大小无法识别截断视频。"""
    import av
    with av.open(str(path)) as container:
        count = sum(1 for _ in container.decode(video=0))
    if count <= 0 or (expected_frames is not None and count != expected_frames):
        raise ValueError(f"视频帧数不符：{path}, 实际 {count}, 期望 {expected_frames}")
    return count


def tile_frame(frame: dict) -> np.ndarray | None:
    imgs = []
    for key in CAMERA_ORDER:
        v = frame.get(key)
        if v is None:
            continue
        a = np.asarray(v)
        if a.ndim != 3 or a.shape[2] < 3:
            continue
        imgs.append(a[:, :, :3].astype(np.uint8))
    if not imgs:
        return None
    h = max(a.shape[0] for a in imgs)
    padded = []
    for a in imgs:
        if a.shape[0] < h:
            pad = np.zeros((h - a.shape[0], a.shape[1], 3), dtype=np.uint8)
            a = np.concatenate([a, pad], axis=0)
        padded.append(a)
    return np.concatenate(padded, axis=1)


class RolloutVideoRecorder:

    def __init__(self, fps: float = 16.6667):
        self.fps = float(fps)
        self.frames: list[np.ndarray] = []

    def add(self, frame: dict) -> None:
        tiled = tile_frame(frame)
        if tiled is not None:
            self.frames.append(tiled)

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, out_path: str) -> bool:
        if not self.frames:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        arr = np.stack(self.frames, axis=0)
        ok = _write_imageio(arr, out_path, self.fps)
        if not ok:
            ok = _write_cv2(arr, out_path, self.fps)
        self.frames = []
        return ok


def _write_imageio(arr: np.ndarray, out_path: str, fps: float) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        try:
            import imageio
        except Exception:
            return False
    try:
        writer = imageio.get_writer(
            out_path, format="FFMPEG", mode="I", fps=fps,
            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        )
        try:
            for f in arr:
                writer.append_data(f)
        finally:
            writer.close()
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _write_cv2(arr: np.ndarray, out_path: str, fps: float) -> bool:
    try:
        import cv2
    except Exception:
        return False
    try:
        n, h, w, _ = arr.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if not vw.isOpened():
            return False
        for f in arr:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False
