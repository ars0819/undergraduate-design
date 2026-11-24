---
title: NumPy 与 Matplotlib 学习笔记
type: course-note
source: 926150.docx
source_sha256: 05c1e7311cbc24a274dbb2fae12bf405b5e30ffae9145bb6f0ffc6286536d1e0
source_paragraphs: P0443-P0472
original_date: 无法从源文档确定
status: 已接触但待复习
updated: 2026-11-06
---

# NumPy 与 Matplotlib 学习笔记

> **来源**：`D:\Desktop\926150.docx`（SHA256 `05c1e731…6d1e0`），段落 P0443–P0472。  
> **原始日期**：无独立日期。  
> **性质**：零散代码摘录，不是完整 NumPy/Matplotlib 课程；不能据此认定已掌握。

## 1. 源笔记内容

【原始笔记】P0443–P0446：

- `dangp`（无法辨认，保留）
- `Numpy￥￥￥matplotlab`（标题/分隔，原文保留）

### 1.1 数组 shape 与样本数

【原始笔记】P0447–P0454：

- NumPy 数组有 `.shape` 属性；`x_train.shape` 返回 Python 元组，每一项对应一个维度。
- 例：`x_train.shape` 返回 `(2,)`。
- `x_train` 是一维数组，只有 1 个维度；元组里的 `2` 表示有 2 个元素。
- 样本数可以写 `m = len(x_train)`，或 `m = x_train.shape[0]`。

```python
m = len(x_train)
print(f"Number of training examples is: {m}")

m = x_train.shape[0]
print(f"Number of training examples is: {m}")
```

【AI解释】`(2,)` 是“1 维、长度 2”，不是“2 行 1 列”。如果是二维数组 `(N,D)`，`shape[0]` 是行数（样本数），`shape[1]` 是特征数。`len(x_train)` 对二维数组返回行数，但显式写 `shape[0]` 更清楚。

### 1.2 Matplotlib：scatter 与 plot

【原始笔记】P0455–P0467：

- 用 `plt.scatter()` 画两个点；参数 `marker`、`c` 可把点显示为红色叉号（默认蓝色圆点）。
- `plt.plot()` 画折线/直线，默认按 x 顺序连线。
- `plt.scatter()` 画散点，只画孤立数据点，不自动连线。

```python
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```

> 【待核实】源笔记没有给出 `import matplotlib.pyplot as plt` 和 `x_train/y_train` 的构造；这段代码不能单独运行。不要把它当完整脚本。

### 1.3 parameter 与 argument

【原始笔记】P0468–P0471：

- parameter（形参）：函数定义时写在括号里的变量名。
- argument（实参）：调用函数时传入的具体数值/变量。

```python
def func(x, w, b):  # x, w, b 是 parameters（形参）
    ...

# x_train, 100, 100 是 arguments（实参）的示例
```

> 【待核实】P0471 的原文在“是”后中断；这里只保留可确定的部分。

### 1.4 `np.zeros()`

【原始笔记】P0472：

- `np.zeros()` 是 NumPy 函数，用来创建全部元素为 0 的数组。

【AI解释】默认 dtype 通常是 `float64`；可传 `dtype=` 指定。`np.zeros((2,3))` 创建 2×3 数组。

## 2. 用户当时的理解

源笔记没有明确的第一人称总结；以上是代码/术语摘录。不能据此认定已经会 NumPy 或 Matplotlib。

## 3. 待解决问题

1. `(N,)`、`(N,1)`、`(N,D)` 的 shape 分别什么意思？
2. `*` 与 `@` 的结果 shape 有何不同？什么时候发生错误广播？
3. 如何只用训练集估计均值/标准差，再标准化训练/验证/测试？
4. `plt.plot` 与 `plt.scatter` 的选择标准？
5. `np.zeros` 的默认 dtype 是什么？什么时候要指定？
6. 如何用 Matplotlib 画训练/验证损失曲线并保存？

## 4. 自测（先不给答案）

1. 构造 `x = np.arange(6).reshape(2,3)`；说出 `x.shape`、`x.shape[0]`、`len(x)`。
2. 构造 `a` 形状 `(3,)`、`b` 形状 `(3,1)`；预测 `a+b` 的 shape。再试 `a @ b`。
3. 给一个 `(N,D)` 数组，按列标准化；只用训练集统计量，写出代码。
4. 用 `plt.scatter` 画 5 个点，加标题/轴标签/图例，保存 PNG。
5. 解释 `parameter` 与 `argument` 的区别，各举一个例子。
6. 用 `np.zeros((2,3))` 创建数组，打印 shape/dtype。

**通过标准**：1、2、3 不看答案能写并解释；4 能生成可打开图片；5、6 能口述。通过后才考虑改状态。

## 5. 关联

- [Python 基础学习笔记](../Python/Python_基础学习笔记.md)：函数、列表、循环。
- [吴恩达机器学习笔记](吴恩达机器学习笔记.md)：训练集、`m`、损失。
- 旧 V3 第 3–4 周：NumPy、Matplotlib、线性回归。
- 后续：PyTorch tensor 的 shape 与 NumPy 类似但语义不同（尚未学习）。
