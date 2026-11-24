---
title: 下一步行动
type: actions
updated: 2026-11-07
---

# Next Actions

> 这些是**待执行**动作，不是完成记录。完成后把证据链接写到本表，并同步 [`Current_Status.md`](Current_Status.md)。  
> 顺序原则：先环境事实，再阅读，再独立小代码；不先下载大权重。
>
> **临时安排（2026-11-07 至 11-17）**：主时段用于 [MIT 18.06 十天冲刺](../01_Roadmap/MIT_18.06_十天冲刺计划.md)（A 线，6–7 h/天）。下列 N01–N08 顺延；每天只保留 N01（环境事实）与 N03（SmolVLA 阅读）各 30 分钟，不并行装环境。11-18 复核后恢复原顺序。

## 最近 7 天（建议 2026-11-07 至 11-13）

| 顺序 | 动作 | 完成证据 | 依赖/边界 |
|---|---|---|---|
| 1 | 记录本机环境事实：`nvidia-smi`、Python、`import torch`、`nvcc`、磁盘余量 | 一个 `environment_probe.txt` 或日志片段；含日期、命令、原始输出 | 不安装、不下载、不改系统环境 |
| 2 | 确认 A100 40GB 是否可访问、配额、单次时长、存储与网络 | 导师/管理员书面答复或登录记录；没有就写“未确认” | 不猜测服务器配置；不保存 SSH 密钥/密码 |
| 3 | 读 SmolVLA 论文摘要/方法/实验 + LeRobot LIBERO 文档；画一页推理链路 | 一页 Markdown：输入、预处理、模型、动作块、执行、评价；标注未核实处 | 不改仓库、不下载权重；读不懂的公式保留原文 |
| 4 | 用纯 Python 写 `latency_summary.py`：读一组延迟数，输出 n/mean/p50/p95，处理空输入和坏值 | 代码 + 3 类测试输出 + 一段解释 | 不给完整答案；先自己写 30 分钟 |
| 5 | 用 NumPy 重做同一统计，和纯 Python 结果对比；记录 shape 与 dtype | 对比表 + 一个广播/空数组测试 | 只查 API；不引入 pandas |
| 6 | 决定是否安装 PyTorch：独立环境、Python 版本、CPU/CUDA wheel、记录理由 | 环境决策记录；若安装，附 `pip freeze` 与一个最小 tensor 测试 | 不在系统 Python 里装；不因安装阻塞阅读 |
| 7 | 用 30 分钟周复盘：实际做了什么、证据在哪、下周第一件事 | [`Weekly_Review`](../07_Templates/Weekly_Review.md) 填好；不虚构 | 没有记录就写“没有”，不补造 |

## 当前优先级队列

| ID | 动作 | 状态 | 完成标准 | 关联 |
|---|---|---|---|---|
| N01 | 本机环境证据包 | 待执行 | 命令 + 原始输出 + 日期；GPU 被识别但计算栈状态明确 | [`Tools_and_Environment`](../06_Resources/Tools_and_Environment.md) |
| N02 | A100 可用性确认 | 待确认 | 有配额/访问证据，或明确写“无” | [`Baseline_Plan`](../04_Projects/Undergraduate_Thesis/Baseline_Plan.md) |
| N03 | SmolVLA/LIBERO 阅读笔记 | 待执行 | 一页推理链路 + 3 个风险 + 3 个待问导师问题 | [`VLA_Inference_Track`](../01_Roadmap/VLA_Inference_Track.md) |
| N04 | Python 延迟统计小工具 | 待执行 | 空/正常/坏输入测试通过；能解释 p50/p95 | [`Python 笔记`](../02_Learning/Python/Python_基础学习笔记.md) |
| N05 | NumPy 统计对照 | 待执行 | 与纯 Python 数值一致；shape/dtype 说明正确 | [`NumPy 笔记`](../02_Learning/Machine_Learning/NumPy_Matplotlib_学习笔记.md) |
| N06 | PyTorch 环境决策 | 待执行 | 决策记录 + 版本/来源/回滚方式 | [`Foundation_Plan`](../01_Roadmap/Foundation_Plan.md) |
| N07 | 第一轮论文筛选 | 待执行 | 从候选表选 2 篇，与 RQ 明确关联；不按“名校/热度”全收 | [`Papers_Index`](../06_Resources/Papers_Index.md) |
| N08 | 导师问题清单 | 待执行 | 5 个需要确认的问题，带默认方案与影响 | [`Advisor_Feedback`](../04_Projects/Undergraduate_Thesis/Advisor_Feedback.md) |

## 停止规则

- 环境安装超过 **1 小时**没有可复现进展：停止，记录错误，先用 CPU/阅读推进；另约时间。
- 模型/权重下载前没有确定版本、存储位置和哈希方案：不下载。
- 基线没有跑通：不做量化/编译对比，不写“优化有效”。
- 连续 2–3 天没有最小可运行结果：缩小任务，不增加新课程/新模型。
- 任何“已复现/已部署/已加速”的表述，必须有原始日志、命令、版本和评价分母。

## 不建议现在做

- 不同时搭 SmolVLA 和 OpenVLA-OFT 两套环境。
- 不先做 INT4 量化、CUDA kernel 或 Triton 重写。
- 不把世界模型、Agent、车臂协同架构加入本科主线。
- 不为“看起来完整”预生成未来 30 天的日志。
