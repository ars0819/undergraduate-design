---
title: 工具与环境事实
type: environment
updated: 2026-11-06
status: 本机探测；A100 未验证
---

# Tools and Environment

> 本页记录**实际探测到的事实**，不是“应该可用”。硬件被识别不等于计算栈可用；一次成功 forward 也不等于任务可复现。  
> 探测日期：2026-11-06。硬件/驱动/磁盘会变，使用前重测。

## 1. 本机事实

| 项目 | 探测结果 | 解释 |
|---|---|---|
| OS | Windows 11，版本 10.0.26200 | `platform.platform()` |
| CPU | Intel64 Family 6 Model 183 Stepping 1；20 logical CPUs | 实验室/桌面机 |
| Python | 3.12.10，`C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe` | 系统 Python；不是项目环境承诺 |
| NumPy | 2.5.4 | 已安装 |
| Matplotlib | 3.11.2 | 已安装 |
| PyTorch | **未安装**（`ModuleNotFoundError`） | 不能做 tensor/GPU 测试 |
| pandas / transformers / accelerate | 未安装 | 不要在系统 Python 里随手装 |
| CUDA toolkit | `nvcc` **未找到** | 没有系统 CUDA 编译工具证据 |
| Git | 2.55.0.windows.5 | 可用 |
| conda | **未找到** | 如需独立环境，先决定方案 |
| GPU | NVIDIA GeForce RTX 5070，12227 MiB，驱动 617.14，compute capability 12.0 | `nvidia-smi` 可识别；未做 CUDA 计算测试 |
| D: 剩余 | 720,264,830,976 bytes（约 670.8 GiB） | 2026-11-06 快照 |
| C: 剩余 | 86,630,133,760 bytes（约 80.7 GiB） | 2026-11-06 快照 |

## 2. 未验证

- **A100 40GB**：没有登录、配额、单次时长、存储、网络、独占/计费证据。不得写“可用”。
- **OpenFleX**：没有 SDK/ROS/急停/相机/自由度/控制周期证据。不得写“已具备平台”。
- **CUDA 计算**：torch 未安装；没有 `torch.cuda.is_available()`、kernel 或 profiler 记录。
- **12GB 是否够跑 SmolVLA/OpenVLA-OFT**：未实测；不以模型规模猜结论。
- **网络下载**：没有 Hugging Face/GitHub 大文件下载记录；不要在无版本计划时下载。

## 3. 环境探测命令

把输出保存到外部实验目录（不要只截图）：

```powershell
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

python --version
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

git --version
nvcc --version
```

如果 `torch` 不存在，记录 `ModuleNotFoundError`，不要安装后再补写“原本可用”。

## 4. 项目环境放哪里

- 本库**不保存** Conda/venv、CUDA 安装、模型权重、数据集、缓存、日志大文件。
- 建议在独立实验目录（由用户选择，例如 `D:\vla-workspaces\...`）建立环境；本库只记录路径、版本、命令和结果索引。
- 不同模型/不同 CUDA 版本用独立环境；不要把一个环境改来改去后还比较结果。
- 每次环境变更记录：日期、命令、Python/torch/CUDA/驱动、成功/失败、回滚方式。

## 5. 安全边界

- 不把 SSH 私钥、API token、密码、学校服务器配置、Cookie 放进本库。
- 不在 Markdown 里写真实密钥；用环境变量/凭据管理器。
- 不上传/同步/公开私人文件；不初始化远程仓库/push，除非用户明确授权。
- 不运行来源不明的安装脚本；不把 `curl | bash` 当默认步骤。

## 6. 环境验收（未来）

一个环境至少通过：

1. `nvidia-smi` 与 `torch.cuda.is_available()` 一致；
2. 一个 tensor 在 CPU/GPU 间移动；
3. 一个小模型 forward/backward；
4. 保存/加载；
5. 一个真实数据 batch 的 shape/精度检查；
6. 记录 `pip freeze` 与驱动版本。

当前：**未通过**（torch 未安装）。

## 7. 现有资源

见 [`Existing_Resources_Index.md`](Existing_Resources_Index.md)：Zotero、Obsidian Vault、既有文献库、毕设工作区、OpenFleX 资料。全部只读引用，不修改。
