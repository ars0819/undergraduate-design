---
title: 实验记录区
type: experiments
updated: 2026-11-06
---

# Experiments

> **当前没有任何运行。** 不要为未来实验预建大量空目录；真实运行时再建记录。  
> 大文件（权重、数据、视频、缓存、环境）放外部实验目录，本库只放小记录/索引。

## 规则

1. 每个真实运行一个 Run ID：`YYYYMMDD_模型_目的_vNN`。
2. `04_Projects/Undergraduate_Thesis/Experiment_Index.md` 加一行。
3. 用 [`Experiment_Record.md`](../../07_Templates/Experiment_Record.md) 记录；原始日志路径写绝对/外部路径。
4. 失败/中止也记录；不删除分母。
5. 不在本目录复制代码仓库；代码放外部项目，记录 commit/diff。

## 最小记录

- 假设/目的
- 代码/权重/数据/环境版本
- 命令/配置/seed
- 原始指标/日志
- 结果与限制
- 是否可复跑

## 目录

- `records/`：未来按 Run ID 建 Markdown；当前不建空文件。
- 图片/小附件可放 `99_Assets/`；视频/大文件放外部。
