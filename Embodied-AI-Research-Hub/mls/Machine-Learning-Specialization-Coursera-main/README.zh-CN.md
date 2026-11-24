# 机器学习专项课程中文使用手册

本仓库整理了 Andrew Ng 在 Coursera 上的 **Machine Learning Specialization（机器学习专项课程）** 学习资料，包括课程周次索引、可选实验、编程作业和练习测验参考。

> 这不是一个需要运行的完整软件项目，而是一套以 Jupyter Notebook 为主的学习仓库。建议先学习课程视频，再使用 Notebook 动手实验，最后完成编程作业和练习测验。

## 一、项目整体结构

```text
Machine-Learning-Specialization-Coursera-main/
├── README.md                                      英文总目录
├── README.zh-CN.md                                本中文手册
├── CODE_OF_CONDUCT.md                             社区行为规范
├── CONTRIBUTING.md                                贡献说明
├── LICENSE                                        许可证
├── .github/ISSUE_TEMPLATE/                        Issue 模板
├── resources/                                     图片等资源
├── C1 - Supervised Machine Learning .../          课程 1
├── C2 - Advanced Learning Algorithms/             课程 2
└── C3 - Unsupervised Learning, Recommenders .../ 课程 3
```

每门课程下面按照 `week1`、`week2` 等周次继续划分。周次目录通常包含以下三类内容：

| 名称 | 作用 |
| --- | --- |
| `README.md` 或 `Readme.md` | 该课程或该周的内容索引 |
| `Optional Labs` / `optional-labs` | 可选实验，用 Notebook 演示课程概念和代码 |
| `C?W?A?` | 编程作业目录，通常包含需要运行和理解的 Notebook |
| `Practice quiz...` | 练习测验、题目截图或参考答案 |

目录大小写和命名并不完全统一，例如 `Optional Labs`、`optional-labs`、`README.md`、`Readme.md` 都可能出现，这是原仓库的组织方式，不影响内容使用。

## 二、推荐学习顺序

1. 先按照 Coursera 课程视频学习当前周的理论。
2. 阅读对应周的 `README`，确认实验和作业位置。
3. 先运行 `Optional Labs`，用小例子理解公式、图像和代码。
4. 再完成 `C?W?A?` 编程作业。重点是理解每个函数，而不是只运行出结果。
5. 最后查看 `Practice quiz...`，把它当作复习和自测材料。
6. 遇到不会的题目，回到对应实验和课程视频，不建议一开始直接看答案。

课程之间的关系是：

```text
课程 1：监督学习基础
    ↓
课程 2：神经网络、模型评估和决策树
    ↓
课程 3：聚类、异常检测、推荐系统和强化学习
```

## 三、课程 1：监督学习、回归与分类

目录：`C1 - Supervised Machine Learning - Regression and Classification/`

课程 1 建立机器学习基础，主要研究带有标签数据的学习问题。输入特征为 `x`，目标值为 `y`，模型通过训练数据学习参数，然后对新数据进行预测。

### Week 1：线性回归和梯度下降

目录：`C1.../week1/`

主要内容：

- 监督学习与无监督学习的区别
- 线性回归和模型表示
- 损失函数、代价函数
- 梯度下降和参数更新
- 回归问题的基本建模流程

主要子目录：

- `Practice quiz - Regression`：回归基础练习
- `Practice quiz - Supervised vs unsupervised learning`：监督学习与无监督学习练习
- `Practice quiz - Train the model with gradient descent`：梯度下降练习
- `Optional Labs/`：模型表示、代价函数、梯度下降 Notebook

### Week 2：多元线性回归

目录：`C1.../week2/`

主要内容：

- 多个特征的线性回归
- NumPy 向量化
- 特征缩放和学习率
- 特征工程与多项式回归
- 使用 Scikit-learn 实现回归

主要子目录：

- `Practice quiz - Gradient descent in practice`
- `Practice quiz - Multiple linear regression`
- `Optional Labs/`：向量化、多元回归、特征缩放、特征工程和 Scikit-learn 实验
- `C1W2A1/`：线性回归编程作业
  - `C1_W2_Linear_Regression.ipynb`

### Week 3：逻辑回归

目录：`C1.../week3/`

主要内容：

- 分类问题
- Sigmoid 函数
- 决策边界
- 逻辑损失和代价函数
- 过拟合与正则化
- 使用 Scikit-learn 完成逻辑回归

主要子目录：

- `Practice quiz - Cost function for logistic regression`
- `Practice quiz - Gradient descent for logistic regression`
- `Optional Labs/`：分类、Sigmoid、决策边界、逻辑损失、梯度下降、过拟合和正则化
- `C1W3A1/`：逻辑回归编程作业
  - `C1_W3_Logistic_Regression.ipynb`

## 四、课程 2：高级学习算法

目录：`C2 - Advanced Learning Algorithms/`

课程 2 从传统机器学习进入神经网络，并介绍机器学习项目中的模型选择、偏差方差和决策树。

### Week 1：神经网络基础

目录：`C2.../week1/`

主要内容：

- 人工神经元和神经网络层
- 前向传播
- 神经网络的二分类
- TensorFlow/Keras 基本写法
- 使用 NumPy 手动实现神经网络部分计算

主要子目录：

- `Practice quiz - Neural networks intuition`
- `Practice quiz - Neural network model`
- `Practice quiz - TensorFlow implementation`
- `Practice-Quiz-Neural-Networks-Implementation-in-python`
- `optional-labs/`：神经元、网络层和咖啡烘焙示例
- `C2W1A1/`：神经网络二分类编程作业
  - `C2_W1_Assignment.ipynb`

### Week 2：神经网络训练和多分类

目录：`C2.../week2/`

主要内容：

- ReLU 等激活函数
- Softmax
- 多分类问题
- 交叉熵损失
- TensorFlow 神经网络训练
- 反向传播的基本思想

主要子目录：

- `Practice-Quiz-Neural-Network-Training`
- `Practice-Quiz-Activation-Functions`
- `Practice-quiz-Multiclass-Classification`
- `Practice-Quiz-Additional-Neural-Network-Concepts`
- `optional-labs/`：ReLU、Softmax、多分类和反向传播实验
- `C2W2A1/`：手写数字多分类编程作业
  - `C2_W2_Assignment.ipynb`

### Week 3：机器学习项目实践

目录：`C2.../week3/`

主要内容：

- 训练集、交叉验证集和测试集
- 偏差与方差
- 欠拟合与过拟合
- 误差分析
- 数据增强和迁移学习思路
- 如何选择改进模型的方向

主要子目录：

- `Practice-Quiz-Advice-for-applying-machine-learning`
- `practice-quiz-bias-and-variance`
- `practice-quiz-machine-learning-development-process`
- `C2W3A1/`：Applied Machine Learning 建议编程作业
  - `C2_W3_Assignment.ipynb`

### Week 4：决策树和树集成

目录：`C2.../week4/`

主要内容：

- 决策树的节点、分裂和叶节点
- 熵和信息增益
- 连续特征的分裂
- 随机森林和树集成
- 结构化数据与非结构化数据的模型选择

主要子目录：

- `practice-quiz-decision-trees`
- `practice-quiz-decision-tree-learning`
- `practice-quiz-tree-ensembles`
- `optional labs/`：决策树和集成树实验
- `C2W4A1/`：决策树编程作业
  - `C2_W4_Decision_Tree_with_Markdown.ipynb`

## 五、课程 3：无监督学习、推荐系统与强化学习

目录：`C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/`

课程 3 介绍没有人工标签时如何发现数据结构，并进一步学习推荐系统和强化学习。

### Week 1：聚类和异常检测

目录：`C3.../week1/`

主要内容：

- 无监督学习
- K-means 聚类
- 聚类中心和聚类分配
- 肘部法选择聚类数量
- 高斯分布与异常检测
- 选择阈值识别异常样本

主要子目录：

- `Practice Quiz - Clustering`
- `Practice Quiz - Anomaly Detection`
- `C3W1A/`
  - `C3W1A1/`：K-means 编程作业
  - `C3W1A2/`：异常检测编程作业

### Week 2：推荐系统

目录：`C3.../week2/`

主要内容：

- 内容过滤
- 协同过滤
- 用户和物品特征向量
- 评分矩阵
- 推荐系统的代价函数
- 使用神经网络实现推荐系统

主要子目录：

- `Practice Quiz - Collaborative Filtering`
- `Practice Quiz - Content-based filtering`
- `Practice Quiz - Recommender systems implementation`
- `C3W2/`
  - `C3W2A1/`：协同过滤推荐系统
  - `C3W2A2/`：神经网络推荐系统

### Week 3：强化学习

目录：`C3.../week3/`

主要内容：

- 状态、动作、奖励和环境
- 马尔可夫决策过程
- 状态价值函数和状态-动作价值函数
- 折扣因子和回报
- 连续状态空间
- Deep Q-Learning

主要子目录：

- `Practice Quiz - Reinforcement learning introduction`
- `Practice Quiz - State-action value function`
- `Practice Quiz - Continuous state spaces`
- `optional-labs/`：强化学习和价值函数实验
- `C3W3A1/`：Lunar Lander 深度 Q 学习作业
  - `C3_W3_A1_Assignment.ipynb`

## 六、如何运行 Notebook

### 1. 安装 Python

建议使用 Python 3.9 或更高版本，并使用虚拟环境隔离依赖。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 阻止激活脚本，可以直接在 VS Code 中选择 Python 解释器，或使用命令提示符激活：

```bat
.venv\Scripts\activate.bat
```

### 2. 安装常用依赖

仓库没有统一的依赖配置文件。大多数实验会使用以下科学计算和 Notebook 库：

```powershell
python -m pip install jupyter notebook numpy matplotlib pandas scikit-learn tensorflow
```

如果某个 Notebook 报缺少其他模块，按照错误信息单独安装对应包即可。TensorFlow 在不同 Python 版本和操作系统上的支持情况可能不同，遇到安装问题时应优先查看 TensorFlow 官方兼容性说明。

### 3. 在 VS Code 中打开

1. 用 VS Code 打开仓库根目录。
2. 安装 Microsoft 的 Python 和 Jupyter 扩展。
3. 打开任意 `.ipynb` 文件。
4. 在右上角选择 `.venv` 对应的 Python Kernel。
5. 按顺序运行单元格，不要一开始只运行最后一个单元格。
6. 修改代码后重新运行受影响的单元格；必要时使用“重启 Kernel 并运行全部单元格”。

### 4. 在命令行中启动 Jupyter

在仓库根目录执行：

```powershell
jupyter notebook
```

然后在浏览器中打开目标课程、周次和 Notebook。

## 七、如何选择不同类型的内容

- 想看概念和公式：先看课程视频，再打开 `Optional Labs`。
- 想练习考试知识点：打开对应的 `Practice quiz...` 目录。
- 想学习完整实现：打开 `C?W?A?` 下的编程作业 Notebook。
- 想比较 NumPy 和 TensorFlow：优先查看课程 2 Week 1、Week 2 的实验。
- 想从基础开始：按 `C1 Week 1 -> C1 Week 2 -> C1 Week 3 -> C2 -> C3` 顺序学习。

名称中的字母和数字通常表示：

- `C1`、`C2`、`C3`：第 1、2、3 门课程
- `W1`、`W2` 等：第几周
- `A1`、`A2`：该周第几个编程作业
- `Soln`：solution，表示参考解答版本
- `Lab`：实验
- `RecSys`：Recommender System，推荐系统
- `TF`：TensorFlow
- `NN`：Neural Network，神经网络

## 八、常见问题

### 为什么 Notebook 打不开？

确认已安装 Jupyter 扩展，并且当前 VS Code 选择了正确的 Python Kernel。也可以在终端执行 `jupyter notebook` 检查 Jupyter 是否安装成功。

### 为什么导入 TensorFlow 失败？

通常是 Python 版本、TensorFlow 版本或环境选择不匹配。确认 VS Code 使用的是安装 TensorFlow 的同一个解释器，并根据操作系统安装兼容版本。

### 为什么运行顺序很重要？

Notebook 的变量会保存在 Kernel 内存中。后面的单元格可能依赖前面单元格创建的变量、函数或数据，因此最好从上到下运行。

### 这些目录里的答案可以直接提交吗？

本仓库主要用于学习和复习。编程作业中的参考实现应当先理解，再自己重新编写和验证；提交课程作业时请遵守 Coursera 的课程规则和学术诚信要求。

### 这个仓库有统一的启动命令吗？

没有。它不是 Web 应用，也没有统一的训练脚本或服务入口。使用方式是选择对应的 Notebook，在 Jupyter 或 VS Code 中运行。

## 九、建议的最短入门路线

如果你刚开始学习，可以按下面的路线完成第一个闭环：

1. `C1.../week1/Optional Labs/` 中的模型表示。
2. `C1.../week1/Optional Labs/` 中的代价函数。
3. `C1.../week1/Optional Labs/` 中的梯度下降。
4. `C1.../week2/C1W2A1/` 的线性回归作业。
5. `C1.../week3/C1W3A1/` 的逻辑回归作业。
6. 再进入课程 2 的神经网络和课程 3 的推荐系统、强化学习。

这样可以依次理解：数据和模型表示 -> 代价函数 -> 参数优化 -> 回归 -> 分类 -> 神经网络 -> 更复杂的机器学习系统。

## 十、相关原始文档

- [英文总 README](README.md)
- [课程 1 README](C1%20-%20Supervised%20Machine%20Learning%20-%20Regression%20and%20Classification/README.md)
- [课程 2 README](C2%20-%20Advanced%20Learning%20Algorithms/Readme.md)
- [课程 3 README](C3%20-%20Unsupervised%20Learning%2C%20Recommenders%2C%20Reinforcement%20Learning/Readme.md)
- [Coursera 机器学习专项课程](https://www.coursera.org/specializations/machine-learning-introduction)
