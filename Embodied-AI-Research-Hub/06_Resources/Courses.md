---
title: 课程与资源索引
type: resources
updated: 2026-11-07
checked_source_date: 2026-10-13
---

# Courses

> **不是完成记录。** 课程列表来自 V3.0/Excel（B 站目录核对截止 2026-10-13），只代表“资源存在”。  
> 平台目录可能变化；看课前先核对标题/UP 主/P 数。当前只把吴恩达 ML 作为主课，其余按需。

## 1. 当前课程策略

- 同一时间只保留一门主课：吴恩达 Machine Learning Specialization。2026-11-08 至 11-17 临时改为 MIT 18.06，见 [`MIT_18.06_十天冲刺计划.md`](../01_Roadmap/MIT_18.06_十天冲刺计划.md)；11-18 复核后恢复。
- 数学用短复习/自测插入，不串行刷完 18.01/18.02/18.06/6.041SC。
- CS231n/CS285/DLS 是“按需资源”，不是开学前置。
- 有代码问题时，优先查官方文档/最小示例，不先找一套新课。

## 2. B 站合集（旧 V3 核对）

| 课程 | BV 号 | P 数（2026-10-13） | 用途 | 当前优先级 |
|---|---|---|---:|---|
| 吴恩达机器学习 MLS（新版中英） | `BV16jyuBBEom` | 142 | 监督学习→神经网络→RL 概念 | 主课 |
| NumPy（莫烦 Numpy&Pandas） | `BV1Ex411L7oT` | 18 | 数组基础；补官方文档 | 按需 |
| MIT 18.01 单变量微积分 | `BV1T4411v7iP` | 35 | 导数/链式/近似/极值/Newton | 复习 |
| MIT 18.02 多变量微积分 | `BV1GY4y1B7x6` | 35 | 偏导/梯度/链式 | 按需 |
| MIT 18.06 线性代数 | `BV19g411v7gD` | 35 | 矩阵/投影/最小二乘/SVD | 10 天主课（11-08 至 11-17） |
| MIT 6.041SC 概率 | `BV1NCfTBVEts` | 76 | 概率基础；混排讲座/习题 | 按需 |
| Matplotlib（莫烦） | `BV1Jx411L7LU` | 19 | 损失曲线/散点 | 按需 |
| PyTorch（小土堆） | `BV1hE411t7RN` | 33 | 训练链/DataLoader/保存 | 环境就绪后 |
| CS231n 2017 | `BV1Dx411n7UE` | 16 | softmax/两层网络/CNN | 按需 |
| CS285 2023 | `BV1BK411a7pf` | 99 | BC/PG/DQN/模型规划 | 扩展/按需 |
| DLS 旧版 183 讲 | `BV1Qu4y1U7p7` | 183 | 反传/优化/误差分析/时序 | 只补难点 |

B 站链接（统一格式）：`https://www.bilibili.com/video/BV...`。  
例：[吴恩达 MLS](https://www.bilibili.com/video/BV16jyuBBEom)、[MIT 18.01](https://www.bilibili.com/video/BV1T4411v7iP)、[PyTorch 小土堆](https://www.bilibili.com/video/BV1hE411t7RN)。

> 旧 V3 还记录了“P 数”和章节取舍；完整表在 [`V3.0_完整机械提取.md`](../90_Archive/Old_Roadmaps/V3.0_完整机械提取.md) 与 [`Excel_七工作表结构化提取.md`](../90_Archive/Old_Roadmaps/Excel_七工作表结构化提取.md)。新路线不要求按旧 P 号执行。

## 3. 官方入口（未逐页复核）

| 资源 | 入口 | 用途 |
|---|---|---|
| MIT 18.01 OCW | https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/ | 讲义/习题/考试 |
| MIT 18.02 OCW | https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/ | 多变量微积分 |
| MIT 18.06 OCW | https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/ | 线性代数 |
| MIT 6.041SC OCW | https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/ | 概率 |
| CS231n 2017 | https://cs231n.stanford.edu/2017/ | 视觉/作业 |
| CS285 | https://rail.eecs.berkeley.edu/deeprlcourse/ | 深度 RL（年份以页面为准） |
| PyTorch Learn the Basics | https://pytorch.org/tutorials/beginner/basics/intro.html | tensor/autograd/训练 |
| NumPy beginner | https://numpy.org/doc/stable/user/absolute_beginners.html | 数组/广播 |
| Matplotlib tutorials | https://matplotlib.org/stable/tutorials/index.html | 画图 |
| Python docs | https://docs.python.org/3/ | 标准库/语法 |
| Git docs | https://git-scm.com/doc | 版本管理 |
| LeRobot | https://github.com/huggingface/lerobot | SmolVLA/评测入口 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 任务套件 |
| Hugging Face | https://huggingface.co/ | 模型/数据（下载前先规划） |

> 这些官方链接没有在 2026-11-06 逐页抓取核验；打开后以实际页面为准。不要在未读情况下引用具体课程结论。

## 4. 仍待核验

- DLS 新版 Transformer 的可靠 B 站合集/分 P：本库不编造。
- 16-831 机器人学习、MIT 6.5940 高效计算、CS294-158 生成模型的可靠同版本 B 站合集：未核实。
- 这些主题保留官方/搜索入口，遇到具体问题再查；不作为当前前置。

## 5. 资源使用规则

1. 先确定当前问题，再找资源；不为了“收藏完整”刷目录。
2. 看视频前先做 3–5 题诊断；已会则跳。
3. 每看一段，产出一个可运行例子/一张图/一段解释；否则不算学习。
4. 不把 P 数当进度；不把“看完”当掌握。
5. 安装/下载前先记录版本、空间、来源；不把环境放进本库。
