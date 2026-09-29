"""testhard_eval v7 纯函数自检（仅 CPU、假数据，不导入仿真）。

运行：.venv-robomme/bin/python -m pytest robomme_sim/test_testhard_v7_select.py -q
.venv-robomme 未装 pytest 时直接跑脚本（内置极简替身）：.venv-robomme/bin/python -m robomme_sim.test_testhard_v7_select
"""
from __future__ import annotations

import contextlib
import importlib
import json
import sys

try:
    import pytest
except ImportError:  # 无 pytest 时的极简替身，只覆盖本文件用到的三个接口
    class _Skip(Exception):
        pass

    class _PytestShim:
        Skip = _Skip

        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"未抛出 {exc.__name__}")

        @staticmethod
        def importorskip(name):
            try:
                return importlib.import_module(name)
            except ImportError as err:
                raise _Skip(str(err))

        @staticmethod
        def skip(reason):
            raise _Skip(reason)

    pytest = _PytestShim()

from robomme_sim import testhard_eval as th


def _manifest(tmp_path, rows):
    path = tmp_path / "ids.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _rows():
    # 两个任务、两轮、两片；xhard0 行 candidate 为 null，其他档 source_episode 为 null
    out = []
    for task in ("BinFill", "StopCube"):
        for ep in range(8):
            tier = "xhard0" if ep < 4 else "xhard1"
            out.append(dict(task=task, episode=ep, tier=tier, seed=1000 + ep,
                            candidate=None if tier == "xhard0" else ep * 10,
                            source_episode=ep + 20 if tier == "xhard0" else None,
                            round=1 if ep % 2 == 0 else 2, shard=(ep // 2) % 2))
    return out


def test_module_import_is_light():
    # 纯函数模块导入时不应拉起仿真
    assert "robomme_hard" not in sys.modules
    assert "sapien" not in sys.modules


def test_parse_shard():
    assert th.parse_shard("3/10") == (3, 10)
    for bad in ("10/10", "-1/10", "0/0"):
        with pytest.raises(ValueError):
            th.parse_shard(bad)


def test_select_and_sort(tmp_path):
    rows = th.load_manifest(_manifest(tmp_path, list(reversed(_rows()))))
    got = th.select_rows(rows, 1, 0, 2)
    assert [(r["task"], r["episode"]) for r in got] == [("BinFill", 0), ("BinFill", 4), ("StopCube", 0), ("StopCube", 4)]
    assert all(r["round"] == 1 and r["shard"] == 0 for r in got)
    assert got[0]["key"] == "BinFill/0"
    # 各轮各片并起来恰好覆盖全清单、互不重叠
    union = [r["key"] for rd in (1, 2) for s in (0, 1) for r in th.select_rows(rows, rd, s, 2)]
    assert sorted(union) == sorted(r["key"] for r in rows) and len(union) == len(set(union))


def test_shard_count_must_match(tmp_path):
    rows = th.load_manifest(_manifest(tmp_path, _rows()))
    with pytest.raises(ValueError):
        th.select_rows(rows, 1, 0, 10)
    with pytest.raises(ValueError):
        th.select_rows(rows, 1, 0, 1)


def test_manifest_validation(tmp_path):
    rows = _rows()
    with pytest.raises(ValueError):
        th.load_manifest(_manifest(tmp_path, rows + [rows[0]]))  # 重复身份
    bad = dict(rows[0])
    del bad["shard"]
    with pytest.raises(ValueError):
        th.load_manifest(_manifest(tmp_path, [bad]))
    with pytest.raises(ValueError):
        th.load_manifest(_manifest(tmp_path, [dict(rows[0], round=3)]))
    with pytest.raises(ValueError):
        th.load_manifest(_manifest(tmp_path, [dict(rows[0], task="Nope")]))


def test_check_identity():
    x0 = dict(task="BinFill", episode=0, tier="xhard0", seed=7, candidate=None, source_episode=3)
    th.check_identity(x0, dict(episode=0, tier="xhard0", candidate=None, seed=7, source_episode=3, spec_sha256=None))
    for bad in (dict(seed=8), dict(source_episode=4), dict(tier="xhard1"), dict(candidate=0), dict(episode=1)):
        ident = dict(episode=0, tier="xhard0", candidate=None, seed=7, source_episode=3)
        ident.update(bad)
        with pytest.raises(ValueError):
            th.check_identity(x0, ident)
    x2 = dict(task="BinFill", episode=40, tier="xhard2", seed=9, candidate=12, source_episode=None)
    th.check_identity(x2, dict(episode=40, tier="xhard2", candidate=12, seed=9, spec_sha256="ab"))
    with pytest.raises(ValueError):
        th.check_identity(x2, dict(episode=40, tier="xhard2", candidate=13, seed=9))


def test_classify_binding_export():
    good = dict(available=True, mode="export", spec_kind="native-parity/1", injected_mismatch=0, spec_sha256=None)
    assert th.classify_binding("xhard0", good, None) == "export"
    for bad in (dict(available=False), dict(mode="replay"), dict(spec_kind="x"), dict(injected_mismatch=1),
                dict(injected_mismatch=None)):
        with pytest.raises(RuntimeError):
            th.classify_binding("xhard0", dict(good, **bad), None)
    with pytest.raises(RuntimeError):
        th.classify_binding("xhard0", None, None)


def test_classify_binding_replay():
    good = dict(available=True, mode="replay", injected_mismatch=0, value_points=5, spec_sha256="aa",
                layered=True, layout_hit=3, layout_paths_expected=3, layout_overridden=0, layout_drift=0)
    assert th.classify_binding("xhard3", good, "aa") == "replay"
    no_layout = {k: v for k, v in good.items() if not k.startswith("layout")}
    assert th.classify_binding("xhard1", no_layout, "aa") == "replay"  # 无 layout_drift 键时不核
    for bad in (dict(mode="export"), dict(injected_mismatch=2), dict(value_points=0), dict(spec_sha256="bb"),
                dict(layout_drift=1), dict(available=False)):
        with pytest.raises(RuntimeError):
            th.classify_binding("xhard3", dict(good, **bad), "aa")
    with pytest.raises(RuntimeError):
        th.classify_binding("xhard3", good, None)  # 期望 sha 缺失不放行


class _FakeRecorder:
    def __init__(self, n, ok=True, boom=False):
        self.n, self.ok, self.boom, self.saved = n, ok, boom, None

    def __len__(self):
        return self.n

    def save(self, path):
        if self.boom:
            raise OSError("disk full")
        self.saved = path
        return self.ok


def test_save_video(tmp_path):
    row = dict(task="BinFill", tier="xhard0", episode=3, seed=77)
    assert th.video_name("BinFill", "xhard0", 3, 77) == "BinFill_xhard0_3_77.mp4"
    rec = _FakeRecorder(5)
    path, err = th.save_video(rec, str(tmp_path), row)
    assert err is None and path == str(tmp_path.resolve() / "BinFill_xhard0_3_77.mp4") == rec.saved
    assert th.save_video(None, str(tmp_path), row) == (None, None)
    assert th.save_video(_FakeRecorder(0), str(tmp_path), row)[1]
    assert th.save_video(_FakeRecorder(5, ok=False), str(tmp_path), row)[1]
    path, err = th.save_video(_FakeRecorder(5, boom=True), str(tmp_path), row)
    assert path is None and "disk full" in err


def test_real_recorder_writes_mp4(tmp_path):
    # 真实 RolloutVideoRecorder：front|wrist 横拼后写出 mp4（imageio/cv2 均不可用则跳过）
    np = pytest.importorskip("numpy")
    from robomme_sim.video_writer import RolloutVideoRecorder

    rec = RolloutVideoRecorder(fps=10)
    for k in range(6):
        rec.add({"front": np.full((32, 32, 3), k * 20, np.uint8), "wrist": np.zeros((32, 32, 3), np.uint8)})
    assert rec.frames[0].shape == (32, 64, 3)
    path, err = th.save_video(rec, str(tmp_path), dict(task="BinFill", tier="xhard1", episode=40, seed=5))
    if err and "返回 False" in err:
        pytest.skip("本机无 imageio-ffmpeg 与 cv2")
    assert err is None and (tmp_path / "BinFill_xhard1_40_5.mp4").stat().st_size > 0


if __name__ == "__main__":
    # 无 pytest 时的直跑入口：逐个执行 test_*，tmp_path 用临时目录
    import inspect
    import tempfile
    from pathlib import Path

    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        kwargs = {}
        with tempfile.TemporaryDirectory() as tmp:
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(tmp)
            try:
                fn(**kwargs)
                print(f"PASS {name}")
            except Exception as exc:  # 替身的 Skip 与 pytest 的 Skipped 在这里分流
                if type(exc).__name__ in ("_Skip", "Skipped"):
                    print(f"SKIP {name}: {exc}")
                else:
                    failed += 1
                    print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"SELFTEST={'PASS' if failed == 0 else 'FAIL'} failed={failed}")
    sys.exit(1 if failed else 0)
