# 第 1 课练习 · 从调用找到定义

**任务**　完成一个小脚本。直接运行时显示一条回合标签，导入时可以使用函数。

**前置知识**　能读懂变量、`print()`、函数调用和缩进。先在 [example.py](example.py) 和 [task_labels.py](task_labels.py) 中对照一次 `import`，再打开 [starter.py](starter.py)。

## 练习 1 · 找函数并调用

打开 `task_labels.py`，找一个同时接收任务编号和回合编号的函数。在 `starter.py` 的第一个 TODO 处添加导入语句，在 `main()` 内的 TODO 处调用它并打印返回值。保留题目给定的编号。完成后，可以先从终端导入模块，再手动调用 `main()` 验证结果。

```powershell
py -c "import starter; starter.main()"
```

预期输出一行。

```text
任务 2，第 3 回合
```

## 练习 2 · 只在直接运行时启动

在 `starter.py` 底部的 TODO 处写入口判断。让直接运行脚本自动调用 `main()`，同时保留练习 1 中可由导入方手动调用的能力。在 PowerShell 中逐条执行，比较两次输出。

```powershell
py .\starter.py
py -c "import starter; print('导入完成')"
```

第一条预期只输出 `任务 2，第 3 回合`。第二条预期只输出 `导入完成`。

以上命令都在本文件所在目录执行。Windows 上如果只有 `python` 命令，可以用它替换 `py`。完成后，试着口头说出 Python 是从哪个文件拿到标签函数的，以及两种运行方式分别从哪里开始执行。

[做完再看参考答案](reference.py) · [返回指南](../../GUIDE.md)
