> 📌 **阅读提示**：本文是 SmolVLA 复现过程在机器本地（`/home/lxy/lerobot-smolvla-repro`）的原始记录，文中相对链接指向那个目录，在本仓库中不可点击。对应的可公开证据已归档在 [`repro/artifacts/`](.)；整理版复现说明见 [`notes/05-代码对照与复现.md`](../../notes/05-代码对照与复现.md)。

# SmolVLA + LIBERO-Spatial task 0 复现记录

日期为 2026-11-20（Asia/Shanghai）。目标是在 Ubuntu 22.04 和 RTX 5070 12 GB 上，用公开权重做本地推理，运行 LIBERO-Spatial 的 task 0，单回合、batch size 1。

## 1. 宿主机只读检查

| 项目 | 实测 |
| --- | --- |
| 操作系统和内核 | Ubuntu 22.04.5 LTS，`6.8.0-149-generic` |
| 显卡和驱动 | RTX 5070，12227 MiB，NVIDIA Open Kernel Module `580.178.04` |
| 三处 CUDA 版本 | `nvidia-smi` 报告 13.0，系统 `nvcc` 为 12.2.140，原 PyTorch wheel 运行时为 12.8 |
| Conda | 25.5.1，原有 `base`、`embodied`、`ml-basic`、`trt` |
| Python | 系统 3.10.12，base 3.13.5，embodied 3.10.20，ml-basic 3.11.18，trt 3.10.15 |
| 原 PyTorch | `ml-basic` 中 `torch 2.11.0+cu128` 和 `torchvision 0.26.0+cu128` |
| 原 LeRobot | 上述解释器中没有 LeRobot 或 LIBERO 包，在用户目录也未找到相关源码目录 |

主要只读命令如下。第一条和最后的 CUDA 张量测试分别在宿主机复核。

```bash
nvidia-smi
cat /etc/os-release
uname -r
lspci -nnk -d 10de:
/usr/local/cuda/bin/nvcc --version
cat /proc/driver/nvidia/version
/home/lxy/miniconda3/bin/conda info --envs
/home/lxy/miniconda3/bin/conda --version
find /home/lxy -iname '*lerobot*' -o -iname '*libero*'
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/ml-basic/bin/python -B -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list()); print((torch.tensor([2], device="cuda")+1).item())'
```

沙箱内缺少 `/dev/nvidia*`，`nvidia-smi` 报 `couldn't communicate with the NVIDIA driver`，PyTorch 报 CUDA 不可用。宿主机同类检查成功，PyTorch 报 capability `(12, 0)`、架构列表包含 `sm_120`、张量结果为 `3`。驱动报告的 CUDA 能力、本地编译器、wheel 的运行时分别属于三层。验证现成 wheel 最直接的方法是实际执行一个 GPU 算子。

## 2. 代码、权重和依赖选择

- LeRobot `0.7.1`，代码提交 `83883ae60a8e1682552b6853e4170c74d83a4525`。本机原先没有可核对的 LeRobot 版本，因此固定联网检查时的官方 HEAD。该提交要求 Python ≥3.12、PyTorch ≥2.7 且 <2.16、torchvision ≥0.22 且 <0.30、NumPy ≥2.0 且 <2.3。只选 `evaluation,smolvla,libero` 三组 extras。LIBERO extra 在 Linux 上使用 `hf-libero>=0.1.5,<0.2.0`，SmolVLA 和 LIBERO 共用 Transformers ≥5.22 且 <5.23。[固定提交的 pyproject.toml](https://github.com/huggingface/lerobot/blob/83883ae60a8e1682552b6853e4170c74d83a4525/pyproject.toml)
- 安装文档推荐 Python 3.12 和 Conda 中的 FFmpeg。选择 PyTorch `2.11.0+cu128`、torchvision `0.26.0+cu128` 复用已验证的 RTX 5070 构建组合，另配 TorchCodec `0.12.0+cpu` 处理视频依赖，经由 Conda 安装 FFmpeg 7.1.1。[Installation](https://huggingface.co/docs/lerobot/installation) · [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) · [LIBERO](https://huggingface.co/docs/lerobot/libero)
- 主权重为 `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`，已核对 Hub 的 Git ref。其配置引用的 SmolVLM2 骨干缓存为 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@4cde9d208e19519c36917aa170fffc7e381aa0ad`。评测入口是 `lerobot-eval`，会将策略设为评测模式并在 `torch.no_grad()` 下运行。[Simulation](https://huggingface.co/docs/lerobot/simulation) · [checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d)

## 3. 下载和安装命令

工作目录为 `/home/lxy/lerobot-smolvla-repro`。环境 `lerobot-smolvla-repro` 独立于 `ml-basic`。源码和模型缓存都在复现项目下，Conda 包与 pip wheel 下载还使用各自的包缓存。

```bash
mkdir -p /home/lxy/lerobot-smolvla-repro/logs
cd /home/lxy/lerobot-smolvla-repro
git ls-remote https://github.com/huggingface/lerobot.git HEAD
git ls-remote https://huggingface.co/lerobot/smolvla_libero refs/heads/main
git clone --no-checkout https://github.com/huggingface/lerobot.git lerobot > logs/clone.log 2>&1
git -C lerobot switch --detach 83883ae60a8e1682552b6853e4170c74d83a4525
/home/lxy/miniconda3/bin/conda create -y -n lerobot-smolvla-repro --override-channels -c conda-forge python=3.12 pip ffmpeg=7.1.1 > logs/conda-create.log 2>&1

env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/ml-basic/bin/python -B -m pip download --no-deps --only-binary=:all: --python-version 3.12 --index-url https://download.pytorch.org/whl/cu128 --dest wheels torch==2.11.0+cu128 torchvision==0.26.0+cu128 > logs/download-torch.log 2>&1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip download --no-deps --index-url https://download.pytorch.org/whl/cpu --dest wheels torchcodec==0.12.0 > logs/download-torchcodec.log 2>&1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip install 'huggingface-hub>=1.31.0,<3.0.0' > logs/install-hub.log 2>&1

export HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/hf download lerobot/smolvla_libero --revision a24f1f380461b665158d52acd140152c79e8b36d --max-workers 2 > logs/download-checkpoint.log 2>&1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct config.json generation_config.json model.safetensors preprocessor_config.json processor_config.json tokenizer.json tokenizer_config.json added_tokens.json special_tokens_map.json chat_template.json merges.txt vocab.json --max-workers 2 > logs/download-vlm-retry.log 2>&1

cd /home/lxy/lerobot-smolvla-repro/lerobot
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip install --find-links /home/lxy/ml-basic-workspace/.wheelhouse ../wheels/*.whl -e '.[evaluation,smolvla,libero]' > ../logs/install-lerobot.log 2>&1
```

`--find-links` 只读复用原来下载的通用 CUDA 依赖 wheel。若另一台机器没有该目录，可在新环境下先从 `cu128` 索引安装对应 PyTorch 构建，并保留实际解析出的版本清单。此次骨干下载按默认分支解析，日志记录了实际 revision。回放时可先确认缓存的 `models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main` 指向上节提交，再离线评测。

## 4. 评测参数和源码阅读路线

| 参数 | 采用值和验证依据 |
| --- | --- |
| 套件和回合 | `libero_spatial`、`task_ids=[0]`、`n_episodes=1`、`batch_size=1`、`max_parallel_tasks=1` |
| 相机对应 | 主相机 `agentview_image` → `observation.images.image` → `observation.images.camera1`；腕部 `robot0_eye_in_hand_image` → `observation.images.image2` → `observation.images.camera2`。checkpoint 的 `train_config.json` 与官方示例一致 |
| 观测和动作 | checkpoint 接受两路 RGB 和 8 维机器人状态，输出一条 7 维动作。前三维为末端平移、接着三维为旋转，最后一维控制夹爪 |
| 控制语义 | `relative` 与 `train_config.json` 一致。环境 reset 后对控制器设 `use_delta=True` |
| 动作生成 | `num_steps=10` 给动作生成中的 Euler 积分，`chunk_size=50` 规定生成长度，`n_action_steps=50` 规定动作队列消耗长度。队列空时才根据最新观测重新生成 |
| 初始状态 | 使用官方示例对应版本的环境默认频率 20 Hz、固定初态和 hard reset，seed 1000。原训练配置另记载 30 Hz |
| 执行方式 | 本地 EGL 离屏渲染。单环境选择同步模式，便于在同一进程看异常栈 |

从 [lerobot_eval.py](lerobot/src/lerobot/scripts/lerobot_eval.py) 的评测入口追踪到模拟器创建、观测校验、策略加载和回合循环。在 [libero.py](lerobot/src/lerobot/sim/integrations/libero.py) 找相机改名、状态拼接、reset 和 step。在 [modeling_smolvla.py](lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py) 看 `select_action` 如何取队列头部，以及 `sample_actions` 如何调用 Euler 积分。成功判定来自 LIBERO 环境的 `check_success()`，单回合只产生一个成功或失败样本。

## 5. LIBERO 素材与项目内配置

按固定代码提交中的 [Dockerfile.sim](lerobot/docker/Dockerfile.sim) 选用 `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`。这是几何模型、纹理和场景素材。任务定义（BDDL）和初始状态由 `hf-libero` wheel 携带。将 [config.yaml](libero-config/config.yaml) 放在进程的 `LIBERO_CONFIG_PATH` 下，避免首次运行询问路径。`hf-libero 0.1.5` 的对象加载器优先找包内 `assets`，因此在新环境内将那个位置链接到项目素材目录，沿用包自己的加载逻辑。

```bash
cd /home/lxy/lerobot-smolvla-repro
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/hf download lerobot/libero-assets --repo-type dataset --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --local-dir libero-assets --max-workers 4 > logs/download-libero-assets.log 2>&1
ln -s /home/lxy/lerobot-smolvla-repro/libero-assets /home/lxy/miniconda3/envs/lerobot-smolvla-repro/lib/python3.12/site-packages/libero/libero/assets
sha256sum hf-cache/hub/models--*/snapshots/*/model.safetensors > logs/model-sha256.txt
```

## 6. 已观察到的下载问题

- 沙箱内的 `git ls-remote https://github.com/huggingface/lerobot.git HEAD` 报 `Could not resolve host: github.com`。授权在宿主机运行同一命令成功。
- 首次 Hub 主 checkpoint 下载有短暂 TLS 握手中断，日志中的 `UNEXPECTED_EOF_WHILE_READING` 通过客户端自动重试恢复。骨干第一次下载退出码 1，有相同的 TLS 错误，还报告忽略 `--exclude` 的解析警告。原命令为 `hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --max-workers 2 --exclude '*.md' '*.mp4' '*.png' '*.gguf'`，重试命令见上。两份主 safetensors 均已与 Hub LFS SHA-256 核对一致。详见 [主权重日志](logs/download-checkpoint.log)、[骨干首试日志](logs/download-vlm.log)、[骨干重试日志](logs/download-vlm-retry.log) 和 [校验记录](logs/model-sha256.txt)。

## 7. 驱动更新中断与恢复

安装期间，系统的 dpkg 日志记录 NVIDIA 软件包在约 19:49 更新到 `580.196.07`，`/var/run/reboot-required` 同期出现。已加载的内核模块仍为 `580.178.04`，`modinfo -F version nvidia` 则报告磁盘上的模块为 `580.196.07`。

- 宿主机 `nvidia-smi` 退出码 18，报 `Failed to initialize NVML: Driver/library version mismatch`。新环境的 [预检日志](logs/preflight.log) 报 `Error 804: forward compatibility was attempted on non supported HW`，先前通过的 `ml-basic` 复测也报同样的错误。
- 用 `LD_DEBUG=libs` 复测，确认 PyTorch 加载了系统的 `libcuda.so.580.196.07`。详见 [动态加载日志](logs/cuda-loader.log)。
- 从现有 Ubuntu APT 缓存解出与已加载内核模块相配的五个 amd64 包，只写入复现目录。CUDA 和 EGL 的库搜索路径只设在本次子进程。匹配的 [nvidia-smi](logs/nvidia-scoped.log) 成功识别 RTX 5070，[CUDA 张量测试](logs/preflight-scoped.log) 恢复并返回 `3`。EGL 预检能创建 context，但用 PyOpenGL 查询的 renderer 字符串为空，因此用后续成功渲染并解码的视频验证实际 EGL 路径。

```bash
cd /home/lxy/lerobot-smolvla-repro
cat /proc/driver/nvidia/version
modinfo -F version nvidia
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
mkdir -p driver-runtime-580.178.04
for package in libnvidia-compute-580 libnvidia-gl-580 libnvidia-cfg1-580 libnvidia-extra-580 nvidia-utils-580; do
    dpkg-deb -x "/var/cache/apt/archives/${package}_580.178.04-0ubuntu0.22.04.1_amd64.deb" driver-runtime-580.178.04 || exit
done
```

## 8. 实际执行的评测命令

在可访问 `/dev/nvidia*` 的宿主机运行。下列两项库路径用于复现本次驱动更新窗口内的组合。重启后如果系统驱动的内核模块和用户态库已一致，从命令去掉 `LD_LIBRARY_PATH=...` 和 `__EGL_VENDOR_LIBRARY_FILENAMES=...` 这两项，使用系统驱动。重跑时给 `--output_dir` 和日志改用新路径。

```bash
cd /home/lxy/lerobot-smolvla-repro
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  PYTHONNOUSERSITE=1 PATH="/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin:$PATH" \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  LIBERO_CONFIG_PATH=/home/lxy/lerobot-smolvla-repro/libero-config \
  HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH=/home/lxy/lerobot-smolvla-repro/driver-runtime-580.178.04/usr/lib/x86_64-linux-gnu \
  __EGL_VENDOR_LIBRARY_FILENAMES=/home/lxy/lerobot-smolvla-repro/driver-runtime-580.178.04/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d \
  --policy.device=cuda --policy.num_steps=10 --policy.n_action_steps=50 \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --env.control_mode=relative --env.max_parallel_tasks=1 \
  --env.init_states=true --env.hard_reset=true \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false --seed=1000 \
  --output_dir=outputs/spatial-task0 > logs/eval-spatial-task0.log 2>&1
```

全量素材下载进度停滞时，另从同一 revision 优先补齐 task 0 的物体和纹理。该批下载 57 个文件成功，实际评测复用了全量下载已落盘的场景和道具。评测成功后结束了仍在等待的全量下载。详见 [任务素材日志](logs/download-task0-assets.log) 和 [全量素材日志](logs/download-libero-assets.log)。

```bash
cd /home/lxy/lerobot-smolvla-repro
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -B -c 'from huggingface_hub import snapshot_download; print(snapshot_download("lerobot/libero-assets", repo_type="dataset", revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6", local_dir="libero-assets", allow_patterns=["stable_scanned_objects/akita_black_bowl/*", "stable_scanned_objects/plate/*", "stable_scanned_objects/glazed_rim_porcelain_ramekin/*", "stable_hope_objects/cookies/*", "textures/*"], max_workers=6))' > logs/download-task0-assets.log 2>&1
```

## 9. 已完成的验证和结果

| 项目 | 结果 |
| --- | --- |
| 实际任务 | `libero_spatial` task 0，把盘子和小烤盅之间的黑碗放到盘子上 |
| 设备和配置 | RTX 5070，CUDA，seed 1000，batch size 1，单回合，`relative`，`num_steps=10`，`n_action_steps=50` |
| 退出状态 | 0，日志末尾为 `End of eval` |
| 完成判定 | `success: true`，成功 1/1，累计奖励和最高单步奖励均为 1.0 |
| 用时 | 评测聚合计时约 8.90 秒，计时范围为评测回合与相关输出处理 |
| 视频 | H.264，360×360，20 fps，4 秒，80 帧。源码在 reset 后保存一帧，之后每次 step 保存一帧，本回合执行了 79 个动作 |

- [评测日志](logs/eval-spatial-task0.log) · [指标 JSON](outputs/spatial-task0/eval_info.json) · [任务说明](outputs/spatial-task0/task_descriptions.json) · [视频](outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4) · [视频探测](logs/video-probe.json)。使用环境内 FFmpeg 解码整段视频，退出码 0。
- 实际环境为 Python 3.12.16、FFmpeg 7.1.1、LeRobot 0.7.1、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、TorchCodec 0.12.0+cpu、Transformers 5.22.2、Hub 2.4.0、Accelerate 1.16.0、Gymnasium 1.4.0、hf-libero 0.1.5、robosuite 1.4.0、MuJoCo 3.8.1、NumPy 2.2.6、SciPy 1.18.1、PyAV 15.1.0。完整版本见 [pip 清单](logs/pip-versions.txt) 和 [Conda 包构建清单](logs/conda-explicit.txt)。[pip check](logs/pip-check.log) 通过。
- 运行日志记录了 robosuite 缺少可选的私有宏、缺少可选 OpenGL 加速模块、Transformers 的 `torch_dtype` 弃用提示。这些提示在本次运行中没有中断评测。固定提交的 `git status --porcelain` 为空，官方模型源码未修改。

## 10. 范围与下一步

本次完成了指定任务的单回合 GPU 推理。未开展训练、微调、多 seed 测试或完整 suite 评测，也未完成所有 LIBERO 素材下载。单次成功用于验证本地链路，样本量不足以估计成功率。宿主机的待重启状态仍在，复现命令运行时通过项目内库绕过了这个窗口。

建议在方便维护机器时重启，让已安装的驱动模块与用户态库对齐。先复跑宿主机 `nvidia-smi` 和 CUDA 张量测试，然后去掉临时库路径，用新输出目录复跑同一个 seed。学习代码时可对照 `rollout()` 的 reset、观测预处理、`select_action()` 和 `env.step()`，从视频确认抓起与放置动作，再从 `eval_info.json` 检查环境给出的成功标志。如果之后要报告成功率，再补齐素材并扩大任务和 seed 范围。
