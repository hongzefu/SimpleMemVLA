#!/usr/bin/env python3
"""SAPIEN 离屏渲染探针（本机与 GreatLakes 计算节点共用）。

对应 AGENTS.md「覆盖第 4 条」的仿真栈验证口径。起 policy server / 加载 12.7 GB 权重之前
先花 30 秒证明渲染能出图，省掉大量白烧的 walltime。

三个判据：
  1. 能创建 Vulkan RenderSystem（A40 上若 compute mode 是 Exclusive_Process，这一步会抛
     `vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`，解药是 Slurm 的
     `--gpu_cmode=shared`，见 AGENTS.md「GreatLakes 评测的增量规则」第 1 条）；
  2. 能 take_picture 拿到图；
  3. 图像 std > 1 —— 单纯「不报错」不够，选错渲染设备时会静默出黑屏。

两端的 Vulkan ICD 文件名不同（本机 nvidia_icd.json，集群 nvidia_icd.x86_64.json，SAPIEN 只查
固定文件名所以在集群会误报 "Failed to find Vulkan ICD file"），这里自动探测并显式设置。
**不自动回退 llvmpipe 软渲染**——节点上 lvp_icd 是存在的，退到它能「跑通」但慢几十倍、
分数没有参考价值（AGENTS.md GreatLakes 第 4 条）。

用法：
    PYTHONPATH=<repo> python scripts/check/render_probe.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ICD_CANDIDATES = (
    "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",  # GreatLakes 计算节点
    "/usr/share/vulkan/icd.d/nvidia_icd.json",          # 本机 sled-vail
)


def main() -> int:
    print(f"hostname   {os.uname().nodename}")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_mode,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False).stdout.strip()
        print("nvidia-smi:")
        for line in out.splitlines():
            print(f"  {line}")
        if "Exclusive" in out:
            print("  ⚠ compute mode 不是 Default —— SAPIEN 的 Vulkan 设备初始化会失败。"
                  "Slurm 侧需要 --gpu_cmode=shared。")
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi 不可用：{exc!r}")

    if not os.environ.get("VK_ICD_FILENAMES"):
        for cand in ICD_CANDIDATES:
            if Path(cand).is_file():
                os.environ["VK_ICD_FILENAMES"] = cand
                break
    print(f"VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '<未设，用 loader 默认搜索>')}")
    icd_dir = Path("/usr/share/vulkan/icd.d")
    if icd_dir.is_dir():
        print(f"ICD 目录内容：{sorted(p.name for p in icd_dir.iterdir())}")

    # prepare_sapien_runtime 会清掉 conda 注入的 __EGL_VENDOR_LIBRARY_DIRS 污染，
    # 并把找到 libvulkan.so.1 的目录 prepend 进 LD_LIBRARY_PATH。
    from simplememvla.sapien_runtime import prepare_sapien_runtime
    prepare_sapien_runtime()

    import sapien
    print(f"sapien     {sapien.__version__}")

    scene = sapien.Scene()
    scene.add_ground(0.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0.0, 0.5, -1.0], [1.0, 1.0, 1.0])
    # 放一个有颜色的盒子，保证画面不是纯色地面（纯色也会让 std 偏低）
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[1.0, 0.2, 0.2])
    box = builder.build_static(name="probe_box")
    box.set_pose(sapien.Pose([0.0, 0.0, 0.1]))

    cam = scene.add_camera("probe", 256, 256, 1.0, 0.01, 100.0)
    cam.set_local_pose(sapien.Pose([-0.8, 0.0, 0.6], [0.9239, 0.0, 0.3827, 0.0]))
    scene.update_render()
    cam.take_picture()
    img = cam.get_picture("Color")

    std = float(img.std())
    print(f"图像 shape={tuple(img.shape)} dtype={img.dtype} std={std:.4f} "
          f"min={float(img.min()):.4f} max={float(img.max()):.4f}")
    if std <= 1e-2:
        print("RENDER_PROBE_FAIL  画面近乎纯色（多半是黑屏 / 选错渲染设备）")
        return 1
    print("RENDER_PROBE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
