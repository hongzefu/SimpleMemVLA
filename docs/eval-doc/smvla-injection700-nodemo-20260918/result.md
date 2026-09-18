# 实施和评估结果

## 当前状态

代码适配已完成，本机 14 条已通过完整性验收，正式 700 条尚未完成。
benchmark 专用分支 `PolicyEvalThirdParty-simplememvla-0918` 已推送，子模块固定在 `b4e97f2`。

## 起跑前验证

- 9 项定向测试通过：700 条唯一键、两路平衡分片、重复/越界计划拒绝、难度隔离、
  BinFill 初始观测与 demo 区分、导入来源拒绝、恢复身份、错误保留分母、视频完整解码与损坏拒绝。
- 假环境评估验证：运行错误记 error；同样本第二次运行准确在 2000 步截断为 timeout；
  再次恢复跳过已正常完成回合，不重新加载模型。
- 真实 GPU1 环境探针：BinFill/hard episode 153、seed 19300，reset 返回 1 个正常初始帧，
  `demo_tasks=0, demo_frames=0`；图像标准差大于 1，真实 step 1 步，状态 ongoing，无运行错误。
- 上述真实探针没有加载模型，不替代随后 14 条完整闭环冒烟。

## 待回填

GL 两路剩余回合和总 700 条的视频验收，待实跑后按原始记录回填。

## 本机完整冒烟与首轮 GL

- `170125f` 启动，本机 14/14 条正常终态：7 成功、6 失败、1 超时，无运行错误、无遗漏，
  14 个视频全部完整解码。BinFill 三组 demo_tasks/demo_frames 均为零，其余任务保留示范。
- BinFill/hard episode153、seed19300 在本机跑满 2000 步，视频 2001 帧，耗时 244.7 秒。
- GL 首轮 step 为 `61495429.7`、`61495430.3`。暂停前共有 16 条正常终态、11 条成功；
  hold01 的 RouteStick/xhard episode115 有一次 Vulkan reset 错误，hold02 的
  VideoUnmaskSwap/easy episode116 在 reset 阶段被主动暂停。两份原日志完整保留。
- 已定位控制器问题：RSS 25.2 GiB 不能当作超过 24 GiB cgroup 分配；内核实测峰值约 20.7 GiB、
  OOM 计数 0。另一个确定问题是中文异常 JSON 未指定 UTF-8，原生环境改变 locale 后触发
  UnicodeEncodeError。对应修复不改变模型/环境/评估循环，按来源兼容检查恢复。
- Vulkan 错误的根因尚未证明。此前只读内存诊断通过额外重叠 srun step 执行，可能干扰
  GPU 计算模式；暂停后直接 SSH 查询显示 Default，因此不能把该推测写成已确认根因。
- 暂停只取消上述两个评估 step；hold01、hold02 仍 Running。后续不再以重叠 GPU step 做诊断。
