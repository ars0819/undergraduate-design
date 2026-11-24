---
title: 候选 VLA 模型入口
type: index
updated: 2026-11-06
status: 未运行
---

# VLA Models

| 模型 | 角色 | 官方入口 | 当前状态 |
|---|---|---|---|
| SmolVLA | 本科首选候选 | [arXiv 2506.01844](https://arxiv.org/abs/2506.01844)；[LeRobot](https://github.com/huggingface/lerobot)；[LIBERO checkpoint](https://huggingface.co/lerobot/smolvla_libero) | 未读全文/未下载/未运行 |
| OpenVLA-OFT | 备用候选 | [arXiv 2502.19645](https://arxiv.org/abs/2502.19645)；[官方代码](https://github.com/moojink/openvla-oft)；[LIBERO checkpoint](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial) | 未读全文/未下载/未运行 |
| LIBERO | 评测环境/任务套件 | [arXiv 2306.03310](https://arxiv.org/abs/2306.03310)；[官方代码](https://github.com/Lifelong-Robot-Learning/LIBERO) | 未安装 |

## 说明

- 这里只放入口和状态，不复制模型卡/论文全文。
- 具体架构、输入输出、动作头、动作块、训练数据，必须读原文/代码后写；不要从模型名推断。
- 版本/权重/数据/环境放外部实验目录；本库只存记录与结果索引。
- 复现入口：[`04_Projects/SmolVLA_Reproduction/README.md`](../../04_Projects/SmolVLA_Reproduction/README.md)。
- 研究计划：[`VLA_Inference_Track.md`](../../01_Roadmap/VLA_Inference_Track.md)。
