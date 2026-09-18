#!/usr/bin/env python3
"""依赖版本与 editable 指向校验（本机与 GreatLakes 计算节点共用）。

对应 AGENTS.md「覆盖第 4 条」的依赖层验证口径。校验三类事实：
  1. 关键包的版本（torch / numpy / transformers / gymnasium / sapien / mani_skill）；
  2. editable 包的**实际指向**——`pip install -e` 的指向藏在 site-packages 的 .pth /
     __editable__ 里、肉眼不可见，装错了不报错、只是跑的是另一份代码；
  3. transformers 版本是否与 checkpoint 的 config.json 记录的一致——RoboMME 的 prompt 由
     processor.apply_chat_template 渲染、版本相关，而 eval_robomme.sh 没有版本 preflight，
     装错版本不报错、只静默掉分。

用法：
    PYTHONPATH=<repo> python scripts/check/verify_env.py [checkpoint 目录]

任何一项不符即退出码 1，并把不符项逐条打出来。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 期望值。None 表示只打印、不断言（如 sapien 允许 3.0.x 任一补丁号）。
EXPECTED = {
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "transformers": "5.13.1",
    "gymnasium": "0.29.1",
}
NUMPY_RANGE = (1, 26)  # numpy 必须 >=1.26 且 <2（mujoco/robosuite/sapien 都是 numpy1 ABI）


def main() -> int:
    failures: list[str] = []

    import numpy
    import torch
    import torchvision
    import transformers

    print(f"torch        {torch.__version__}  cuda={torch.version.cuda}  "
          f"available={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}      {torch.cuda.get_device_name(i)}  sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}")
    print(f"torchvision  {torchvision.__version__}")
    print(f"numpy        {numpy.__version__}")
    print(f"transformers {transformers.__version__}")

    for mod, want in EXPECTED.items():
        try:
            got = __import__(mod).__version__.split("+")[0]
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod}: 导入失败 {exc!r}")
            continue
        if got != want:
            failures.append(f"{mod}: 期望 {want}，实际 {got}")

    nv = tuple(int(x) for x in numpy.__version__.split(".")[:2])
    if not (NUMPY_RANGE <= nv < (2, 0)):
        failures.append(f"numpy: 期望 >=1.26,<2，实际 {numpy.__version__}")

    # ---- 仿真栈（可能未装，按需报告）----
    for mod in ("gymnasium", "sapien", "mani_skill"):
        if importlib.util.find_spec(mod) is None:
            print(f"{mod:12s} 未安装")
            continue
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"{mod:12s} {ver}   <- {m.__file__}")

    # ---- editable 指向：mani_skill 必须来自本仓库的 third_party/ManiSkill ----
    if importlib.util.find_spec("mani_skill") is not None:
        import mani_skill
        actual = Path(mani_skill.__file__).resolve()
        expect_root = (REPO / "third_party" / "ManiSkill").resolve()
        if expect_root not in actual.parents:
            failures.append(
                f"mani_skill 的 editable 指向不对：实际 {actual}，期望在 {expect_root} 之下")
        else:
            print(f"editable 指向 OK：mani_skill <- {expect_root}")

    # ---- 快路径内核（本项目刻意不装，打印以确认 fallback 生效）----
    for mod in ("flash_attn", "fla", "causal_conv1d"):
        print(f"{mod:12s} {'已装' if importlib.util.find_spec(mod) is not None else '未装（预期，走纯 torch fallback）'}")

    # ---- checkpoint 的 transformers_version 比对 ----
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "checkpoints" / "simplememvla_robomme"
    cfg = ckpt / "config.json"
    if cfg.is_file():
        want = json.loads(cfg.read_text()).get("transformers_version")
        got = transformers.__version__
        print(f"checkpoint {ckpt.name}: config.json 记录 transformers_version={want}，环境为 {got}")
        if want and want != got:
            failures.append(
                f"transformers 版本与 checkpoint 不一致（ckpt {want} vs env {got}）—— "
                "prompt 渲染版本相关，评测不会报错、只会静默掉分")
        if not (ckpt / "stats.json").is_file():
            failures.append(f"{ckpt}/stats.json 缺失——action 反归一化参考未知")
    else:
        print(f"checkpoint 未就绪（{cfg} 不存在），跳过版本比对")

    print()
    if failures:
        print("VERIFY_ENV_FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFY_ENV_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
