# LeRobot 目录与 Python 包指南

## 零基础快速开始

第一次打开项目时，从下面的第 1 课开始。先做一次「在文件 A 看见一个名字，到文件 B 找到它」，就能用同样的方法追踪 `lerobot-eval`。本课只用纯 Python，源码只需在编辑器里打开。

### 五节入门课

| 课次 | 主题 | 读完能做什么 |
|---|---|---|
| 1 | Python 项目结构、`import`、模块和函数 | 从命令名找到模块，再从调用找到函数定义。 |
| 2 | 类、对象、继承与 `dataclass` | 区分配置类和一个实际配置对象，读懂字段和父类。 |
| 3 | NumPy / PyTorch Tensor、`shape` 与索引 | 为批次、时间和特征标记各自的轴，预测索引后的形状。 |
| 4 | SmolVLA 配置与模型加载 | 追踪源码默认值、checkpoint 配置、命令行参数和权重加载。 |
| 5 | `select_action()`、Action Chunk 与动作队列 | 手推几次调用，指出何时生成一块动作以及何时取走一个动作。 |

### 第 1 课 · Python 项目结构、import、模块和函数

**本课目标**　把 `lerobot-eval`、`lerobot.scripts.lerobot_eval` 和 `main()` 分别对应到命令、模块与函数，再找一次跨文件导入。

#### 必须掌握的知识

- **模块和包**　一个 `.py` 文件可以是一个模块。包用点号组织模块。`lerobot.configs.eval` 中前两段是包路径，最后的 `eval` 对应 `eval.py`。
- **导入**　`from A import B` 在模块或包 `A` 中查找名字 `B`，并让当前文件使用这个名字。被导入的名字也可能由 `A` 从另一个文件引入。
- **函数**　`def main():` 定义函数，缩进处是函数体。`main()` 才是一次调用。看到调用时，可先记下函数名，再搜索它的定义或导入语句。
- **直接执行与导入**　直接运行 `example.py` 时，文件里的 `__name__` 是 `"__main__"`。用 `import example` 导入它时，`__name__` 是 `"example"`。文件底部的判断让导入者可以使用函数，由直接运行这个文件的人启动示例。

#### 需要阅读的真实代码

下面的行号取自锁定提交 `83883ae60a8e1682552b6853e4170c74d83a4525`。原文件如果已插入中文注释，可以搜 `中文 L0414` 一类的标记。打开 `.txt` 对照副本时搜 `# L0414`。

| 顺序 | 打开文件并定位 | 只读这一小段 |
|---|---|---|
| 1 | [pyproject.toml](../lerobot/pyproject.toml) 原第 403、414、443 行 | `[project.scripts]` 中找到 `lerobot-eval="lerobot.scripts.lerobot_eval:main"`。冒号左边是模块路径，右边是其中的函数。第 443 行的 `where = ["src"]` 解释源码为什么能在 `src/` 下找到。 |
| 2 | [lerobot_eval.py](../lerobot/src/lerobot/scripts/lerobot_eval.py) 原第 1245–1252 行 | 找到 `def main():`，看到函数体先设置日志、登记插件，再调用 `eval_main()`。最下面还有直接运行该文件时调用 `main()` 的判断。 |
| 3 | [lerobot_eval.py](../lerobot/src/lerobot/scripts/lerobot_eval.py) 原第 76 行和 [configs/eval.py](../lerobot/src/lerobot/configs/eval.py) 原第 30 行 | 从 `from lerobot.configs.eval import EvalPipelineConfig` 追到 `configs/eval.py`。看到名字定义就停，类的写法留到第 2 课。 |

阅读评测入口时，先沿着磁盘上的对应关系找到 `main()`。

```text
终端命令 lerobot-eval
    ↓ pyproject.toml 的入口配置
Python 模块 lerobot.scripts.lerobot_eval
    ↓ 工作区里的 lerobot/src/lerobot/scripts/lerobot_eval.py
入口函数 main() → eval_main()
```

#### 运行一遍纯 Python 示例

示例有两个小文件。[example.py](tutorials/01_imports/example.py) 负责组织任务编号和打印，[task_labels.py](tutorials/01_imports/task_labels.py) 负责把编号变成可读文字。看到 `format_task(task_id)` 时，先在示例顶部找到 `import`，再打开定义该函数的文件。`f"任务 {task_id}"` 会把花括号里的值填入字符串，`return` 把结果交给调用者。

在 Windows PowerShell 中从本指南所在的 `smolvla-reading/` 目录执行下面的命令。如果系统只有 `python` 命令，把 `py` 换成 `python`。

```powershell
Set-Location .\tutorials\01_imports
py .\example.py
py -c "import example; print(example.__name__)"
```

运行脚本会按行打印 `任务 0` 和 `任务 1`。`py -c` 那一行只打印 `example`。两次 Python 运行都会定义函数。直接执行脚本时，文件底部的条件成立并调用 `main()`。

#### 两道练习

1. **跨文件调用**　打开 `starter.py`，从 `task_labels.py` 导入适合生成回合标签的函数，再用题目给出的任务编号和回合编号打印一条标签。
2. **控制程序入口**　补上文件底部的入口判断。用直接运行和导入两种方式验证，检查哪一次会打印任务标签。

题目、前置知识和完整预期输出在练习页。先自行填写 `starter.py`，再从练习页底部打开独立的参考答案。 [第 1 课练习](tutorials/01_imports/README.md)

**学完应该能解释**　哪个文件决定 `lerobot-eval` 导入 `main`？拿到一条 `from ... import ...` 后如何找定义？为什么 `py example.py` 会打印任务，而导入 `example` 时由调用者决定什么时候运行函数？

## 五阶段学习路线

入门课提供读代码需要的 Python 和张量基础。五阶段路线按项目功能组织，带你从入口读到完整的仿真闭环。

| 阶段 | 主要入口 | 学习目标 |
|---|---|---|
| 1. 认识项目结构 | `pyproject.toml`、`scripts/lerobot_eval.py`、`envs/__init__.py` | 用项目目录、模块路径和导入关系找到代码。 |
| 2. 理解 SmolVLA 配置 | `configuration_smolvla.py`、checkpoint 的 `config.json` | 分清动作维度、`chunk_size`、`n_action_steps` 和 `num_steps`，核对生效值。 |
| 3. 理解加载与推理入口 | `policies/factory.py`、`policies/pretrained.py`、`modeling_smolvla.py` | 跟踪策略创建、权重加载和 `select_action()`，判断何时触发完整采样。 |
| 4. 理解模型内部 | `smolvlm_with_expert.py`、`VLAFlowMatching.sample_actions()`、`euler_integrate()` | 从观测编码跟到 Action Expert 和动作块，读到对应操作时再补注意力和 Flow Matching。 |
| 5. 理解 LIBERO 闭环 | `rollout()`、策略前后处理、`LocalVectorEnv.step()`、`LiberoEnv.step()` | 从相机、状态和任务文字一路追踪到仿真动作，再读回下一步观测。 |

实验记录中的 `probe-n1/` 是 **单样本验证**，`n_action_steps=1` 表示每块取 1 步，`n_episodes=1` 表示只测了 1 个回合，记录为 0/1。原始 `n_action_steps=50` 的 1/1 结果也只来自单个回合。两个记录都只能描述各自的那一次运行。

## 目录与包速查

这份指南回答「这个文件夹放什么」和「我该去哪个包找代码」。路径以本机复现所用的 **LeRobot 0.7.1**、提交 `83883ae60a8e1682552b6853e4170c74d83a4525` 为准。包的职责来自这份源码，涉及到其他模型的目录也保留了定位说明。

## 1. 打开工作区时先认这两层

**`lerobot-smolvla-repro/` 是实验工作区。** **`lerobot/` 是其中克隆的 LeRobot 仓库。** 例如磁盘路径 `lerobot/src/lerobot/envs/factory.py` 中，第一个 `lerobot` 是仓库目录，第二个是导入名为 `lerobot` 的 Python 包。

| 工作区目录或文件 | 用途 |
|---|---|
| `lerobot/` | 官方仓库的本地检出。Python 源码在 `lerobot/src/lerobot/`。 |
| `libero-assets/` | LIBERO 的场景、模型和纹理。 `scenes/` 放场景，`articulated_objects/` 放有关节的物体，`stable_hope_objects/` 和 `stable_scanned_objects/` 放两类物体资源，`textures/` 放贴图。`serving_region.xml` 是仿真资源文件。 |
| `libero-config/` | 指定本机 LIBERO 数据位置的 `config.yaml`。 |
| `hf-cache/` | Hugging Face 下载缓存。`hub/` 包含 checkpoint 快照与文件缓存，`xet/` 是下载机制使用的缓存。 |
| `driver-runtime-580.178.04/` | 复现用的 NVIDIA 用户态运行库与驱动配置，目录结构来自提取的系统包。 |
| `wheels/` | 复现时下载的 Python wheel 安装文件。 |
| `logs/` | 安装、下载、硬件检查和评测日志。 |
| `outputs/` | 评测结果。`spatial-task0/` 是原始运行，`spatial-task0-rerun/` 是复跑，`probe-n1/` 是单步出队设置的探查。结果通常包含 JSON 和 `videos/`。 |
| `smolvla-reading/` | 本指南。 `annotated/` 中是此前的逐行对照副本，也保留了外部 Conda 依赖和 checkpoint JSON 的阅读副本。 |
| `.vscode/` | 当前工作区的 VS Code 设置。 |
| `.agents/`、`.codex/`、`.aws/`、`.git/` | 工作区的工具或环境元数据目录。LeRobot 仓库自己的 Git 目录是 `lerobot/.git/`。 |
| `AGENTS.md`、`reproduction_notes.md` | 工作区协作提醒，以及原评测命令、配置、环境和结果的复现记录。 |

GitHub 仓库 `ars0819/git-demo` 的 `smolvla/` 收纳学习资料。`smolvla-精读复现/` 内的 `paper/` 存论文资料与提取工具，`notes/` 存精读笔记，`figures/` 存论文和复现用图，`repro/` 存复现证据。`smolvla/smolvla-reading/` 存本目录指南和旧对照副本，`smolvla/lerobot/` 存本次加注释的源码文件。完整上游目录可以对照固定提交。 [LeRobot 固定提交](https://github.com/huggingface/lerobot/tree/83883ae60a8e1682552b6853e4170c74d83a4525)

## 2. LeRobot 仓库的一级目录

本节路径都相对于 **LeRobot 仓库根目录**，也就是工作区里的第一个 `lerobot/`。

| 路径 | 用途 |
|---|---|
| `src/` | Python 发行包的源码根目录。`src/lerobot/` 是实际导入的顶层包，`src/lerobot.egg-info/` 是本地安装时生成的发行元数据。 |
| `benchmarks/` | 基准任务集与运行矩阵的配置，例如 `benchmarks.yaml`。 |
| `docs/` | 文档项目。`docs/source/` 是教程与参考内容，`docs/source/api/` 是 API 参考。 |
| `examples/` | 面向使用者的完整示例，涵盖训练、录制、遥操作、部署和推理。 |
| `tests/` | 自动测试。很多子目录名字与源码包对应，适合查一个接口的调用实例。 |
| `docker/` | 不同硬件和用途的容器镜像。`docker/sim/` 放仿真服务依赖与 Compose 文件，`docker/sim/configs/` 放具体仿真服务配置。 |
| `scripts/` | 仓库维护和 CI 驱动脚本。`scripts/ci/` 负责基准调度、指标解析与仿真兼容性检查。命令 `lerobot-eval` 的入口在 `src/lerobot/scripts/`。 |
| `media/` | 文档与项目展示媒体。`media/readme/` 保存 README 使用的媒体。 |
| `utils/` | 仓库开发工具，例如检查配置和文档字符串。`src/lerobot/utils/` 才是运行时可导入的工具包。 |
| `.github/` | GitHub 项目配置。`.github/workflows/` 放测试和发布工作流，`.github/ISSUE_TEMPLATE/` 放问题模板。 |
| `.git/` | 这个 LeRobot 检出的提交、引用和仓库配置。 |

仓库根部的 [pyproject.toml](../lerobot/pyproject.toml) 声明包、依赖和命令行入口。`uv.lock` 固定依赖解析结果，`Makefile` 收录开发和测试命令，`AGENTS.md`、`CONTRIBUTING.md` 与 `README.md` 介绍项目和协作方式。

## 3. `src/lerobot/` 下的 Python 包与资源目录

下面的相对路径都以 **`lerobot/src/lerobot/`** 为起点。Python 代码里的 `import lerobot.envs.factory` 对应 `envs/factory.py`。带 `__init__.py` 的目录是普通包，个别目录可通过命名空间方式提供模块。YAML、提示词和文档模板所在的目录主要用来保存资源。

| 一级目录 | 主要职责 |
|---|---|
| `annotations/` | 用模型为数据集添加文本、任务阶段和问答等标注。 |
| `cameras/` | 相机抽象、相机配置与不同设备的取图驱动。 |
| `common/` | 高层工作流共用的训练、机器人控制和 Trackio 辅助。 |
| `configs/` | 训练、评测和策略的通用配置类，以及命令行解析。 |
| `controllers/` | 机器人控制器，负责把高层命令转成硬件能执行的控制。 |
| `data_processing/` | 数据标注等离线加工逻辑。 |
| `datasets/` | `LeRobotDataset`、元数据、加载保存、采样与数据集编辑。 |
| `distributed/` | 多进程或多设备训练的并行组织、分布式配置和 checkpoint。 |
| `env_server/` | 将仿真环境作为独立服务运行，以及连接、收发观测和动作的客户端。 |
| `envs/` | 环境配置、环境工厂、观察协议转换和 checkpoint 兼容检查。 |
| `inference/` | 部署推理的执行、动作块调度、同步或 RTC、延迟与运行接口。 |
| `jobs/` | 数据、标注等作业的提交与运行辅助，包括 Hugging Face Jobs。 |
| `model/` | 通用机器人运动学等模型工具，例如 `kinematics.py`。 |
| `motors/` | 电机总线和电机驱动，给具体机器人提供读写、校准等底层接口。 |
| `optim/` | 优化器、学习率调度和相应配置工厂。 |
| `policies/` | 策略模型及其配置和数据处理，含通用预训练类和模型工厂。 |
| `processor/` | 把观测或动作依序交给多个步骤，负责改名、分词、归一化和设备转换等。 |
| `remote_inference/` | 策略远程执行的服务端、客户端、编码和部署配置。 |
| `rewards/` | 从轨迹和观测估计成功、进度或奖励的模型。 |
| `rl/` | 强化学习算法和数据来源，含在线与离线样本组合。 |
| `robots/` | 受控机器人的接口和具体设备实现，负责读状态与发动作。 |
| `rollout/` | 机器人执行会话的控制、停止、缓冲、记录、接管及策略选择。 |
| `scripts/` | 安装后的终端命令入口，例如评测、训练、录制和校准。 |
| `sim/` | 仿真运行协议、批量环境和各仿真器的适配层。 |
| `streaming/` | 按回合流式读取视频、Parquet 与缓存索引，可按字节范围拉取数据。 |
| `teleoperators/` | 主动提供人工控制命令的输入设备和遥操作接口。 |
| `templates/` | 策略和奖励模型的模型卡 Markdown 模板。 |
| `transforms/` | 图像等训练数据的变换和旧接口兼容。 |
| `transport/` | 服务与设备通信使用的消息、编解码和传输工具，含 Zenoh 和 gRPC 相关代码。 |
| `utils/` | 运行时共享工具，涵盖设备、随机种子、旋转、日志、统计、可视化等。 |

几个相近的名字容易混淆。 `policies/` 负责算动作，`processor/` 负责把数据变成策略和仿真双方能理解的样子，`inference/` 负责在部署时调度策略。 `envs/` 决定如何创建环境和检查数据接口，`sim/` 负责实际接入仿真器。 `robots/` 接收控制指令，`teleoperators/` 从人手输入取得指令。

### 数据、硬件与部署的子目录

| 子目录 | 用途 |
|---|---|
| `annotations/steerable_pipeline/` | 可配置的数据标注流程，包含读帧、调用模型、校验和写回。 |
| `annotations/steerable_pipeline/modules/` | 任务分解、视觉问答、插话与语音等标注步骤。 |
| `annotations/steerable_pipeline/prompts/` | 上述标注流程读取的提示词模板。 |
| `cameras/opencv/` | OpenCV 相机取图。 |
| `cameras/reachy2_camera/` | Reachy 2 相机接口。 |
| `cameras/realsense/` | Intel RealSense 的相机接口。 |
| `cameras/zmq/` | 通过 ZeroMQ 接收相机帧。 |
| `configs/recipes/` | 可复用的训练等配置配方。 |
| `controllers/unitree_g1/` | Unitree G1 的运动和全身控制器，例如 SONIC、Holosoma 和 GR00T 运动控制。 |
| `data_processing/sarm_annotations/` | 为 SARM 创建子任务和阶段标注。 |
| `motors/damiao/` | 达妙电机通信与配置。 |
| `motors/dynamixel/` | Dynamixel 电机通信与配置。 |
| `motors/feetech/` | Feetech 舵机通信与配置。 |
| `motors/robstride/` | RobStride 电机通信与配置。 |
| `rewards/classifier/` | 用分类模型提供奖励或成功信号。 |
| `rewards/robometer/` | Robometer 进度与成功奖励模型及数据处理。 |
| `rewards/sarm/` | 对阶段和阶段进度建模的 SARM 奖励模型，含训练样本加权工具。 |
| `rewards/topreward/` | 利用模型 token 概率估计进度的 TOPReward 实现。 |
| `rl/algorithms/` | 强化学习算法的基类、配置和具体实现。 |
| `rl/algorithms/sac/` | Soft Actor-Critic 的配置和优化过程。 |
| `rl/data_sources/` | 强化学习样本来源和在线、离线数据混合。 |
| `rollout/strategies/` | 回合执行与记录策略，例如接管、DAgger 或事件保留。 |
| `sim/integrations/` | LIBERO、LIBERO-Plus、MetaWorld 等模拟器的具体适配。 |
| `transport/protocol/` | 通信协议的消息外壳、编解码和客户端。 |

### 机器人与人机输入的具体实现

| 子目录 | 用途 |
|---|---|
| `robots/so_follower/`、`robots/bi_so_follower/` | 单臂与双臂 SO 系列从动端。 |
| `robots/openarm_follower/`、`robots/bi_openarm_follower/` | 单臂与双臂 OpenArm 从动端。 |
| `robots/rebot_b601_follower/`、`robots/bi_rebot_b601_follower/` | 单臂与双臂 ReBot B601 从动端。 |
| `robots/yam_follower/`、`robots/bi_yam_follower/` | 单臂与双臂 YAM 从动端。 |
| `robots/koch_follower/` | Koch 从动臂。 |
| `robots/omx_follower/` | OMX 从动臂。 |
| `robots/earthrover_mini_plus/` | EarthRover Mini Plus 移动机器人。 |
| `robots/hope_jr/` | HopeJR 机器人。 |
| `robots/lekiwi/` | LeKiwi 移动操作机器人。 |
| `robots/reachy2/` | Reachy 2 机器人。 |
| `robots/sim_robot/` | 通过仿真提供机器人接口。 |
| `robots/unitree_g1/` | Unitree G1 机器人接口。 |
| `teleoperators/so_leader/`、`teleoperators/bi_so_leader/` | 单臂与双臂 SO 主动端。 |
| `teleoperators/openarm_leader/`、`teleoperators/bi_openarm_leader/` | 单臂与双臂 OpenArm 主动端。 |
| `teleoperators/openarm_mini/`、`teleoperators/bi_openarm_mini/` | OpenArm Mini 的单臂与双臂控制输入。 |
| `teleoperators/rebot_102_leader/`、`teleoperators/bi_rebot_102_leader/` | 单臂与双臂 ReBot 102 主动端。 |
| `teleoperators/koch_leader/` | Koch 主动臂。 |
| `teleoperators/omx_leader/` | OMX 主动臂。 |
| `teleoperators/gamepad/`、`teleoperators/keyboard/`、`teleoperators/phone/` | 用手柄、键盘或手机采集人工控制命令。 |
| `teleoperators/homunculus/` | Homunculus 遥操作设备的控制输入。 |
| `teleoperators/reachy2_teleoperator/` | Reachy 2 遥操作输入。 |
| `teleoperators/unitree_g1/` | Unitree G1 遥操作输入。 |

### 策略的具体实现

在策略目录里，通常先认三个文件名。`configuration_*.py` 定义模型设置，`modeling_*.py` 定义 PyTorch 模型和策略接口，`processor_*.py` 定义与模型配套的输入输出变换。当前的 SmolVLA 就采用这一结构。

| 子目录 | 用途 |
|---|---|
| `policies/smolvla/` | SmolVLA 配置、预处理、动作队列和生成模型，以及 SmolVLM 骨干与 Action Expert。 |
| `policies/common/` | 多种策略共用的 VLA 工具、Flow Matching 采样和特征处理。 |
| `policies/rtc/` | Real-Time Chunking 的配置和推理辅助。 |
| `policies/act/` | Action Chunking Transformer 策略。 |
| `policies/diffusion/` | Diffusion Policy，将扩散模型用于连续动作生成。 |
| `policies/tdmpc/` | 基于学习到的动力学模型做动作规划的 TD-MPC。 |
| `policies/vqbet/` | 用向量量化表示动作的 VQ-BeT 策略。 |
| `policies/gaussian_actor/` | 输出高斯动作分布的 actor，包括策略和训练配置。 |
| `policies/multi_task_dit/` | 多任务 Diffusion Transformer 策略。 |
| `policies/pi0/`、`policies/pi05/`、`policies/pi0_fast/` | π₀、π₀.₅ 和 π₀-FAST 的 LeRobot 实现。 |
| `policies/fineart_vla/` | 基于 π₀.₅ 的 FineART-VLA，提供分层语言推理与动作生成。 |
| `policies/cosmos3/` | Cosmos3 策略的模型、配置和环境相关处理适配。 |
| `policies/dm05/` | DM05 的策略包装、权重读取与推理。`policies/dm05/core/` 放核心模型结构。 |
| `policies/eo1/` | EO1 模型、训练和推理的 LeRobot 接口。 |
| `policies/evo1/` | Evo1 策略的配置、模型与处理流水线。 |
| `policies/fastwam/` | FastWAM 视频动作策略。`policies/fastwam/wan/` 放 Wan 相关模型实现。 |
| `policies/flux3/` | FLUX 3 Action 的视频和动作联合去噪。`policies/flux3/f3/` 放 FLUX 3 骨干相关模块。 |
| `policies/g05/` | OpenGalaxea G0.5 VLA，包含视觉语言模型、动作专家和解码。 |
| `policies/groot/` | NVIDIA GR00T 的 LeRobot 策略接口。`policies/groot/action_head/` 放动作头。 |
| `policies/lawam/` | LaWAM 策略包装、模型和数据适配。 |
| `policies/lawam/lam_core/` | LaWAM 的潜在动作模型代码。 |
| `policies/lawam/lam_core/core/` | 潜在动作模型、量化与视觉编码。 |
| `policies/lawam/lam_core/core/utils/` | 潜在动作的编码、解码、位置嵌入等构件。 |
| `policies/lawam/latent_world/` | 潜在世界模型的输入、视觉语言接入及批次辅助。 |
| `policies/lawam/latent_world/runtime/` | 潜在世界模型的运行辅助和策略冻结。 |
| `policies/lawam/vlas/` | LaWAM 视觉语言动作模型与 Flow Matching 动作专家。 |
| `policies/lingbot_va/` | 以 Wan 为基础的自回归视频动作世界模型 LingBot-VA。 |
| `policies/lingbot_vla_v2/` | LingBot-VLA v2 策略和处理器。`policies/lingbot_vla_v2/model_core/` 放核心层与张量运算。 |
| `policies/longwam/` | Long-WAM 视频动作模型与 Flow Matching 实现。 |
| `policies/molmoact2/` | MolmoAct2 的 LeRobot 适配。`policies/molmoact2/molmoact2_hf_model/` 放模型的 Hugging Face 实现。 |
| `policies/praxis/` | Praxis-1 视频动作模型的策略适配。 |
| `policies/praxis/praxis_core/` | Praxis-1 骨干、VAE、动作专家和采样。 |
| `policies/praxis/praxis_core/kernels/` | Praxis-1 的 Triton 等计算内核。 |
| `policies/praxis/praxis_core/train/` | Praxis-1 的训练目标、数据与优化组件。 |
| `policies/vla_jepa/` | VLA-JEPA 策略的模型、配置与处理实现。 |
| `policies/wall_oss_05/` | Wall-OSS-0.5 的策略、配置和预处理。 |
| `policies/wall_x/` | Wall-X 策略。`policies/wall_x/qwen_model/` 放对其 Qwen2.5-VL 骨干的扩展。 |
| `policies/xvla/` | X-VLA 策略及其动作表示和处理适配。 |

## 4. 仓库示例与测试目录怎么找

### 示例目录

| 路径 | 用途 |
|---|---|
| `examples/backward_compatibility/` | 旧版接口的回放示例。 |
| `examples/dataset/` | 数据集构建、修改和加载示例。 |
| `examples/flux3/`、`examples/praxis/` | FLUX 3 与 Praxis 的模型导出、配置或运行示例。 |
| `examples/isaac_teleop_to_so101/` | 从 Isaac 遥操作到 SO-101 的示例。`examples/isaac_teleop_to_so101/isaac_teleop/` 放设备接入实现。 |
| `examples/lekiwi/`、`examples/omx/` | LeKiwi、OMX 的录制、回放、执行或遥操作示例。 |
| `examples/notebooks/` | 可逐单元阅读的快速入门 notebook。 |
| `examples/phone_to_so100/`、`examples/so100_to_so100_EE/` | 用手机或末端执行器控制方式操纵 SO-100。 |
| `examples/port_datasets/` | 将外部数据集转换、分片和上传为 LeRobot 数据。 |
| `examples/remote_inference/` | 远程推理部署配置。`examples/remote_inference/zenoh/` 放路由、服务端和机器人端配置。 |
| `examples/rtc/` | 用数据集试验 RTC。 |
| `examples/training/` | 使用 LeRobot 训练接口的示例。 |
| `examples/tutorial/` | 入门教程使用的配套代码。 |
| `examples/tutorial/act/`、`examples/tutorial/diffusion/`、`examples/tutorial/pi0/`、`examples/tutorial/smolvla/` | 分别展示四种策略的使用。 |
| `examples/tutorial/rl/` | 强化学习入门示例。 |

### 测试目录

测试通常按源码包命名。表里的同一行收录同类测试，每个目录内的 `test_*.py` 表明具体验证对象。

| 路径 | 用途 |
|---|---|
| `tests/annotations/`、`tests/datasets/`、`tests/processor/` | 标注、数据集读取与编辑、处理流水线。 |
| `tests/cameras/`、`tests/controllers/`、`tests/motors/`、`tests/robots/`、`tests/teleoperators/` | 相机、控制器、电机、机器人和遥操作接口。 |
| `tests/envs/`、`tests/sim/`、`tests/sim/integrations/`、`tests/env_server/` | 环境工厂、仿真协议、单个仿真接入和环境服务通信。 |
| `tests/inference/`、`tests/remote_inference/`、`tests/transport/` | 推理调度、远程推理和数据传输协议。 |
| `tests/common/`、`tests/configs/`、`tests/utils/`、`tests/model/` | 共享工作流、配置解析、工具和运动学等通用实现。 |
| `tests/ci/`、`tests/distributed/`、`tests/jobs/` | CI 辅助、多设备逻辑和作业接口。 |
| `tests/optim/`、`tests/rewards/`、`tests/rl/`、`tests/training/` | 优化器、奖励、强化学习与训练流程。 |
| `tests/scripts/` | 命令行入口和脚本行为。 |
| `tests/fixtures/`、`tests/mocks/` | 测试样例数据，以及替代硬件或依赖的测试对象。 |
| `tests/policies/`、`tests/policies/common/` | 策略接口与模型公共组件测试。 |
| `tests/policies/cosmos3/`、`tests/policies/dm05/`、`tests/policies/eo1/`、`tests/policies/evo1/`、`tests/policies/fastwam/`、`tests/policies/fineart_vla/`、`tests/policies/flux3/`、`tests/policies/g05/` | 各目录对应同名策略的功能和兼容性测试。 |
| `tests/policies/groot/`、`tests/policies/groot/utils/`、`tests/policies/lawam/`、`tests/policies/lingbot_va/`、`tests/policies/lingbot_vla_v2/`、`tests/policies/longwam/`、`tests/policies/molmoact2/`、`tests/policies/multi_task_dit/` | GR00T 的模型及辅助测试，以及其他同名策略测试。 |
| `tests/policies/pi0_fast/`、`tests/policies/pi0_pi05/`、`tests/policies/pi0_pi05/openpi_pytorch/`、`tests/policies/pi0_pi05/utils/` | π₀ 系列测试，含 OpenPI 的 PyTorch 对照实现与测试辅助。 |
| `tests/policies/praxis/`、`tests/policies/praxis/checkpoint_configs/` | Praxis 测试及其 checkpoint 配置样例。 |
| `tests/policies/rtc/`、`tests/policies/smolvla/`、`tests/policies/vla_jepa/`、`tests/policies/wall_oss_05/`、`tests/policies/wall_x/`、`tests/policies/xvla/` | RTC 机制测试和各同名策略测试。 |
| `tests/artifacts/`、`tests/artifacts/cameras/`、`tests/artifacts/encoded_videos/`、`tests/artifacts/image_transforms/` | 测试读取的文件样本，分别含相机数据、编码视频和图像变换样例。 |
| `tests/artifacts/datasets/`、`tests/artifacts/datasets/lerobot/` | LeRobot 数据集测试样例的入口和数据目录。 |
| `tests/artifacts/datasets/lerobot/aloha_sim_insertion_human/`、`tests/artifacts/datasets/lerobot/pusht/`、`tests/artifacts/datasets/lerobot/xarm_lift_medium/` | 三个用于测试的数据集样例。 |
| `tests/artifacts/policies/`、`tests/artifacts/policies/aloha_sim_insertion_human_act_/`、`tests/artifacts/policies/pusht_diffusion_/` | 测试策略保存的产物，含 Aloha 的 ACT 和 PushT 的 Diffusion 样例。 |
| `tests/artifacts/policies/xarm_lift_medium_tdmpc_use_mpc/`、`tests/artifacts/policies/xarm_lift_medium_tdmpc_use_policy/` | XArm Lift 的 TD-MPC 测试产物，分别对应规划和直接策略执行。 |

## 5. 作为 SmolVLA 读者，打开哪些原文件

已加注释的文件是原路径下的 `.py` 与 `pyproject.toml`。**搜索 `中文 L0256` 可以定位原提交第 256 行的注释**。编辑器当前显示的行号会随新增注释移动。GitHub 的 `lerobot/` 收录加注释的文件与许可证，完整包结构对应固定的上游提交。

| 想找什么 | 直接打开 |
|---|---|
| 终端命令映射、启动与闭环 | [pyproject.toml](../lerobot/pyproject.toml) · [lerobot_eval.py](../lerobot/src/lerobot/scripts/lerobot_eval.py) |
| 命令行、评测配置和 checkpoint 配置 | [parser.py](../lerobot/src/lerobot/configs/parser.py) · [eval.py](../lerobot/src/lerobot/configs/eval.py) · [configs/policies.py](../lerobot/src/lerobot/configs/policies.py) |
| 选策略、加载权重 | [policies/factory.py](../lerobot/src/lerobot/policies/factory.py) · [pretrained.py](../lerobot/src/lerobot/policies/pretrained.py) |
| SmolVLA 设置、动作队列和模型 | [configuration_smolvla.py](../lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py) · [modeling_smolvla.py](../lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py) · [smolvlm_with_expert.py](../lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py) |
| 模型前后处理与采样工具 | [processor_smolvla.py](../lerobot/src/lerobot/policies/smolvla/processor_smolvla.py) · [processor/factory.py](../lerobot/src/lerobot/processor/factory.py) · [pipeline.py](../lerobot/src/lerobot/processor/pipeline.py) · [flow_matching.py](../lerobot/src/lerobot/policies/common/flow_matching.py) |
| LIBERO 的创建、观测和动作 | [envs/factory.py](../lerobot/src/lerobot/envs/factory.py) · [envs/utils.py](../lerobot/src/lerobot/envs/utils.py) · [local_vector_env.py](../lerobot/src/lerobot/sim/local_vector_env.py) · [libero.py](../lerobot/src/lerobot/sim/integrations/libero.py) |
| checkpoint JSON 与已安装依赖里的参考实现 | [checkpoint 配置](annotated/checkpoint/config.json.txt) · [前处理配置](annotated/checkpoint/policy_preprocessor.json.txt) · [后处理配置](annotated/checkpoint/policy_postprocessor.json.txt) · [hf-libero 环境包装](annotated/libero-dependency/env_wrapper.py.txt) |

上次运行的主线可以沿路径快速找一遍。

```text
lerobot-eval
  → scripts/lerobot_eval.py      解析配置、创建环境和策略、开始回合
  → envs/ → sim/integrations/    产生两路相机图、机器人状态和任务描述
  → processor/                  处理观测并送到策略设备
  → policies/smolvla/            预测动作块或取出队列里的下一步动作
  → processor/ → sim/            恢复动作尺度，推进仿真，获得下一步观测
```
