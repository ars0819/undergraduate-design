# SmolVLA 学习材料

第一次读源码可以从第 1 课开始。研究模型设计或复现条件时，可直接查阅论文笔记和已有实验记录。

## 1. 论文精读 + LIBERO 复现 —— [`smolvla-精读复现/`](smolvla-精读复现/README.md)

- 6 篇精读笔记：导读与背景 / 方法逐节精读 / 实验与消融 / 批判性阅读 / 代码对照与复现 / 自测题
- 3 张自绘教学图（数据流、注意力掩码、流匹配）+ 论文图 1–4
- 复现证据包括 3 段视频、运行日志、指标 JSON、复跑一致性与执行协议探针。`n_action_steps=1` 的探查是单样本验证。
- 重跑手册 + 结构核对脚本（参数量 / 视觉 token 数）
- 入口：[`smolvla-精读复现/README.md`](smolvla-精读复现/README.md)

## 2. 源码导读与练习

- [零基础快速开始与五阶段路线](smolvla-reading/GUIDE.md) 从终端命令追到 Python 函数，并提供工作区、LeRobot 目录和 Python 包的速查表。
- [第 1 课练习](smolvla-reading/tutorials/01_imports/README.md) 用纯 Python 学习 `import`、模块和函数。练习目录包含可运行的示例、填空模板和单独的参考答案。
- [加过中文注释的 LeRobot 文件](lerobot/) 保留原本的 `.py` 和 `pyproject.toml` 路径。搜索 `中文 L0256` 可以查找原提交第 256 行。
- [早期阅读副本](smolvla-reading/annotated/) 包括 checkpoint 配置和安装依赖的对照说明。

---

论文原文请从 [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) 获取（仓库不含 PDF，版权原因）。
