---
title: <Run ID>
type: experiment
date: <YYYY-MM-DD>
status: <计划中/运行中/完成/失败中止/无效>
project: <毕设/其他>
---

# Experiment：<Run ID>

## 1. 假设/目的

- 假设：
- 竞争解释：
- 本次只改的一个因素：
- 停止条件：

## 2. 版本与环境

- 代码 commit/diff：
- 权重 revision/hash：
- 数据/任务/初态 hash：
- Python/torch/CUDA/驱动/GPU：
- 环境 freeze 路径：
- 容器/主机：

## 3. 命令/配置

```bash
# exact command
```

- config diff：
- seed/noise：
- batch/精度/执行长度：
- 输入 schema：

## 4. 原始结果

- 日志/CSV/JSONL 路径：
- 延迟 p50/p95：
- 峰值显存：
- 动作误差：
- 成功率（成功/总数）：
- 完成时间：
- 失败分类：

## 5. 解释

- 支持/不支持假设：
- 混杂因素：
- 与基线差异：
- 不能证明什么：

## 6. 复跑

- 新进程复跑命令：
- 是否通过：
- 已知问题：

## 7. 结论状态

- [ ] 仅计划
- [ ] 运行完成，结果待分析
- [ ] 结果可支持有限结论
- [ ] 结果否定假设
- [ ] 无效/中止

> 不要用本模板生成“已完成”数据；数字必须来自原始日志。
