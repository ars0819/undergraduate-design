---
title: 本科毕设项目入口
type: project
updated: 2026-11-06
status: 方案/资料准备；未批准、未运行
---

# Undergraduate Thesis

> **当前状态**：只有方案、资料与讨论稿；没有导师/学院批准，没有模型运行、没有 GPU 日志。  
> 本目录不是代码仓库，不复制/修改/启动既有毕设项目中的代码。

## 1. 当前方向（用户转述）

- 具身模型轻量化、高效推理、实际部署。
- 本科允许复现已有工作；不把 Agent、Robot-Use Agent、世界模型、机器人大脑作为主目标。
- 近期候选：**SmolVLA + LIBERO**。
- 备用：**OpenVLA-OFT + LIBERO**。
- 未来应用：OpenFleX 轮式双臂；当前平台可用性未验证。

## 2. 六个入口

| 文件 | 用途 |
|---|---|
| [Research_Question](Research_Question.md) | 工作版问题、范围、假设、边界 |
| [Baseline_Plan](Baseline_Plan.md) | 两条候选路线、门槛、指标、退路 |
| [Experiment_Index](Experiment_Index.md) | 真实运行索引；当前为空 |
| [Issues](Issues.md) | 已知问题/阻塞；不隐藏 |
| [Advisor_Feedback](Advisor_Feedback.md) | 导师方向转述、待确认问题 |
| [SmolVLA_Reproduction](../SmolVLA_Reproduction/README.md) | 首选复现项目入口（无代码） |

## 3. 外部既有工作区（只读索引）

- `D:\Desktop\bishe\本科毕设_轻量化高效部署_V2`
  - 已有任务书讨论稿、实验基线/复现规划、文献矩阵、两周任务清单。
  - 其 2026-11-06 文档明确：源码静态核查完成；Ubuntu 安装、模型下载、GPU 推理、任务评测**均未执行**。
  - 本库只引用其路径，不复制/改动代码，不把其中的计划当本库结果。
- `D:\Desktop\bishe\本科毕设_正式任务书填写_20261106`
  - 任务书草稿/格式材料；未见正式批准。
- `D:\Desktop\openflex`
  - OpenFleX 说明资料；平台可用性未验证。

> 绝对路径是机器相关的外部引用；不作为可移植 Markdown 链接。

## 4. 当前最小目标（未完成）

1. 一页推理链路 + 风险清单（阅读阶段）。
2. 本机/服务器环境事实（不写“可用”除非有日志）。
3. 一个完整 LIBERO episode 可重跑（模型加载→动作→终止）。
4. 原始基线的延迟/显存/任务指标。
5. 一种优化（compile 或 num_steps）的同条件对照。
6. 复跑包与失败分析。

## 5. 不能写的话

- “已复现 SmolVLA/OpenVLA-OFT” → 除非有完整日志与评价。
- “已部署 OpenFleX” → 当前没有平台/接口/真机证据。
- “量化加速” → 除非有真实内核、同条件延迟和任务结果。
- “成功率 100%” → 除非分母、重复和协议支持。
- “导师已批准” → 除非有书面确认。

## 6. 下一步

见 [`Next_Actions.md`](../../00_Home/Next_Actions.md) 与 [`Baseline_Plan.md`](Baseline_Plan.md)。当前先做阅读与环境事实，不下载权重。
