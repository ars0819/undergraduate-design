---
title: SmolVLA 复现项目入口
type: project
updated: 2026-11-06
status: 未开始；本目录无代码
---

# SmolVLA Reproduction

> 这里只放**项目入口、检查表和记录**，不是代码仓库。  
> 没有复制、修改或启动任何既有毕设代码；没有下载权重/数据；没有 GPU 日志。

## 1. 外部资源

- 论文：[arXiv 2506.01844](https://arxiv.org/abs/2506.01844)
- 代码：[huggingface/lerobot](https://github.com/huggingface/lerobot)
- LIBERO checkpoint：[lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero)
- 数据：[lerobot/libero](https://huggingface.co/datasets/lerobot/libero)
- 评测环境：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- 研究计划：[`VLA_Inference_Track.md`](../../01_Roadmap/VLA_Inference_Track.md)

## 2. 阶段（未执行）

| 阶段 | 任务 | 证据 | 状态 |
|---|---|---|---|
| R0 | 读论文/模型卡/文档；画推理链路 | 一页图 + 风险清单 | 未开始 |
| R1 | 环境与版本锁定 | commit、权重 revision、freeze、GPU | 未开始 |
| R2 | 单次前向 | 输入 schema、动作 shape/范围、显存 | 未开始 |
| R3 | 单回合 | task0 episode、视频/终止 | 未开始 |
| R4 | 原始基线 | 新块 p50/p95、成功率 | 未开始 |
| R5 | 一种优化 | 同条件对照 | 未开始 |

## 3. 目录边界

- **不要**把权重、数据、虚拟环境、缓存放在这里。
- 外部工作区路径待用户确定；确定后只在本页记录路径，不复制大文件。
- 代码改动应在外部仓库进行，保留 commit/diff；本库只记录实验索引与结论。

## 4. 第一轮只读问题

1. 输入图像键、分辨率、颜色顺序？state 维度/单位？
2. 动作维度、范围、归一化/反归一化？
3. 动作块长度、执行长度、控制频率？
4. 动作头是什么（flow/diffusion/回归/token）？迭代步数？
5. 官方 LIBERO 评测入口、任务套件、成功判据？
6. 12GB 是否足够？需要什么精度/batch？
7. 哪些结论是论文证明的，哪些是 demo/推测？

答案写入 [`Baseline_Plan.md`](../Undergraduate_Thesis/Baseline_Plan.md) 或独立论文笔记，不写在本页当已验证事实。

## 5. 完成定义

只有同时满足以下条件，才可在实验索引写“复现完成”：

- 固定版本/权重/环境/数据；
- 一个完整 episode 可重跑；
- 原始延迟/显存/任务指标有日志；
- 失败与分母保留；
- 别人能按 README 复跑。

当前：**0 项满足**。
