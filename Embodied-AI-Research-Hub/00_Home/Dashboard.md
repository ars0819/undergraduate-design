---
title: 研究仪表盘
type: dashboard
updated: 2026-11-06
source_of_truth: 00_Home/Current_Status.md
---

# Dashboard

> **这是视图，不是状态源。** 任何完成状态只在 [`Current_Status.md`](Current_Status.md) 修改；本页若与它冲突，以状态页为准。
> 最近同步：2026-11-06。下一次真实学习/实验记录后同步。

## 一眼状态

| 项目 | 当前内容 |
|---|---|
| 研究方向 | Efficient AI / ML Systems for Embodied Intelligence；VLA 轻量化、高效推理、部署评测 |
| 当前阶段 | 本科毕设准备 + 基础补强；尚无 VLA 运行记录 |
| 当前重点学习 | Python 独立程序诊断；MIT 18.01 复习；吴恩达机器学习当前章节（P 数待确认） |
| 当前课程 | MIT 18.01（复习/自测）；吴恩达 Machine Learning Specialization；不新开完整课程 |
| 本周最重要任务 | 完成 SmolVLA + LIBERO 推理链路的一页纸与风险清单；把本地 GPU/Python 环境变成有日志的事实 |
| 当前本科实验项目 | [`Undergraduate_Thesis`](../04_Projects/Undergraduate_Thesis/README.md)；状态：方案与资料准备，未复现、未测速 |
| 最近一次真实学习证据 | 2026-10-06（数学笔记；Python/ML 段无独立日期） |
| 最近一次旧计划快照 | 2026-10-13（Excel；不是学习证据） |
| 最近一次知识库更新 | 2026-11-06（本库建立；不是学习完成） |
| 当前卡点 | torch/CUDA 计算栈未安装；A100 可用性未验证；Python 独立水平缺自测；无任何实验日志 |
| 下一步可执行动作 | 按 [`Next_Actions.md`](Next_Actions.md) 的 N01–N04 顺序执行；先证据，后安装/下载 |

## 进度口径

- **0 项**被标记为“已通过独立验收”。这不是悲观，而是目前没有自测/运行证据。
- 有笔记 ≠ 已掌握；有课程截图 ≠ 已完成课程；有模型计划 ≠ 已复现。
- 具体逐项状态见 [`Current_Status.md#学习状态`](Current_Status.md#学习状态)。

## 本周三个优先级

1. **环境事实**：记录本机 GPU、驱动、Python、torch/CUDA 是否存在；A100 只写“待确认”，不写“可用”。
2. **读懂再跑**：读 SmolVLA 论文/模型卡与 LeRobot LIBERO 文档，画出“观测→预处理→模型→动作块→执行”的推理链路；此阶段不下载权重。
3. **独立小代码**：用纯 Python 写一个延迟统计小工具（n、mean、p50、p95），带正常/空/坏输入测试；这是推理评测的最小基线能力。

## 不要做

- 不要因为旧计划说“第1周”就认为现在仍在第1周。
- 不要先装一大堆包、下载权重，再补测量协议。
- 不要把 OpenVLA-OFT 与 SmolVLA 同时搭成两套主线。
- 不要把 LIBERO 回合成功写成 OpenFleX 真机部署。
- 不要用 AI 生成的日志、指标或“已运行”文字填充本库。
