# 复现证据核对：SmolVLA + LIBERO-Spatial task 0（单回合 GPU 推理）

> **核对对象**：`/home/lxy/lerobot-smolvla-repro` 下已完成的 SmolVLA + LIBERO-Spatial task 0 单回合、batch size 1 的 GPU 推理（`reproduction_notes.md` 记录，声称 success 1/1）。
>
> **核对方式**：只读读取该项目全部产物、日志、HF 权重缓存与源码；在本会话内用环境内 ffmpeg 7.1.1 / ffprobe 7.1.1 重新探测并解码视频、用 `sha256sum` 重算两份 `model.safetensors`、用 `readlink`/`/proc`/`dpkg` 复核当前驱动现场。
>
> **核对时间**：2026-11-20 20:15–20:25（Asia/Shanghai），紧接原复现（当日 19:42–20:08）之后。
>
> **只读声明**：未修改 `/home/lxy/lerobot-smolvla-repro` 内任何文件（只读访问）；本核对新增内容仅写入 `notes/_raw/repro-evidence.md` 与 `figures/repro/`。
>
> **环境限制**：本会话文件沙箱为 `workspace-write`，`/dev/nvidia*` 不存在，无法在本会话重跑 GPU 评测。GPU 相关结论来自只读现场检查 + 当日日志，不是本会话实测出的推理结果。

---

## 0. 一句话结论

**单回合链路已在当日宿主机 GPU 上跑通**：`logs/eval-spatial-task0.log` 以 `End of eval` 结束、无异常栈，`eval_info.json` 记录 `success=true`（1/1，`sum_reward=max_reward=1.0`，`eval_s=8.8968 s`），视频为 H.264 360×360 / 20 fps / 80 帧、整段解码 exit 0，两份权重 sha256 本次实测与记录一致。

**但这只是 n=1、单 task（10 个中的第 1 个）、单 seed（1000）、单固定初态的一次成功，只能证明「本地推理链路连通」，不能作为成功率证据。** LeRobot 打印的 95% Wilson 区间 `[20.7%, 100.0%]` 是 n=1 下的二项抽样区间，宽到没有估计意义。

---

## 1. 成功标志与关键指标（可直接引用）

| 指标 | 核对值 | 证据位置 |
| --- | --- | --- |
| 完成标志 | 日志最后一行：`INFO 2026-11-20 20:05:13 bot_eval.py:942 End of eval`（全 224 行，无 Traceback） | `logs/eval-spatial-task0.log:224` |
| 退出码 | 记录声称 0；**项目内无独立记录，无法复算**（详见 §5.1） | `reproduction_notes.md` §9 |
| 成功判定 | `successes=[true]` → `n_success=1`、`pc_success=100.0` | `outputs/spatial-task0/eval_info.json` |
| 回合奖励 | `sum_reward=1.0`、`max_reward=1.0`；聚合 `avg_sum_reward=avg_max_reward=1.0` | 同上 |
| 95% 区间 | `pc_success_ci95=[20.654931437723747, 100.0]`（Wilson score interval，n=1） | 同上 + 日志第 214 行 |
| 评测计时 | `eval_s=8.896803617477417` s，`eval_ep_s=8.896803855895996`；`Stepping through eval batches` 8.63 s/it | `eval_info.json` + 日志第 210 行 |
| 动作步数 | **79**（rollout 进度到 79/280 后 running_success_rate=100%，触发完成提前退出） | 日志第 209–210 行 |
| 视频 | 85 754 B；H.264（High profile, avc1, yuv420p）360×360、20/1 fps、4.000000 s、**80 帧** | 本次 ffprobe 复核 + `logs/video-probe.json` |
| 视频解码 | 整段解码退出码 0（本次复核，`-f null -`） | 本次核对 |
| 评测配置 | seed 1000；batch_size 1；n_episodes 1；`hard_reset=true`；`init_states=true`；`control_mode=relative`；`num_steps=10`；`n_action_steps=50`；`chunk_size=50`；`max_parallel_tasks=1`；`use_async_envs=false`；`fps=20`；`observation 360×360` | 日志第 1–99 行 config dump |
| 任务 | `libero_spatial` task 0；英文任务串见 §2.2 | `outputs/spatial-task0/task_descriptions.json` |
| 权重完整性 | 两份 `model.safetensors` 实测 sha256 与 `logs/model-sha256.txt` 一致（见 §2.4） | 本次 `sha256sum` |
| 运行栈 | Python 3.12.16 / LeRobot 0.7.1（commit `83883ae`）/ torch 2.11.0+cu128 / transformers 5.22.2 / hf-libero 0.1.5 等 | `logs/pip-versions.txt`、`logs/conda-explicit.txt`、`git` |

### 1.1 建议的引用措辞

- **成功/链路**：「SmolVLA + LIBERO-Spatial task 0 的单回合 GPU 推理在 RTX 5070 上跑通：日志以 `End of eval` 结束，环境判定 success 1/1（sum_reward = max_reward = 1.0），评测计时 8.90 s，生成 H.264 360×360 / 20 fps / 80 帧视频且整段可解码。」
- **不夸大**：「该结果为 n=1、单 task、单 seed、单固定初态，只证明本地推理与仿真链路连通；LeRobot 给出的 95% Wilson 区间 [20.7%, 100.0%] 在 n=1 下不具备成功率估计意义，不能据此报告模型成功率。」
- **重跑条件（一句话）**：「当前机器重跑需要 `danger-full-access` 提权（沙箱内无 `/dev/nvidia*`），且在驱动未重启对齐前必须使用项目内 `driver-runtime-580.178.04` 的 `LD_LIBRARY_PATH` + `__EGL_VENDOR_LIBRARY_FILENAMES`，或重启使内核模块与用户态库对齐后再跑。」

---

## 2. 逐项核对

### 2.1 退出与完成标志（exit code / End of eval）

| 检查项 | 结果 |
| --- | --- |
| 日志末行 | `INFO 2026-11-20 20:05:13 bot_eval.py:942 End of eval`（`tail -c` 确认文件以 `End of eval\n` 结尾） |
| 异常栈 | `grep -c "Traceback"` = 0 |
| 日志时间窗 | 首行 `2026-11-20 20:05:00` → 末行 `2026-11-20 20:05:13`，13 秒窗口内完成建环境、加载策略、79 步 rollout、写 JSON/视频 |
| 产物落盘 | `eval_info.json`、`task_descriptions.json` 同一 mtime `20:05:13.412974559`；mp4 mtime `20:05:13.238961175`（体积完整 85 754 B） |
| 退出码 | **项目内没有任何记录退出码的文件**：`logs/` 无 wrapper 日志、`outputs/` 无 marker；`~/.bash_history` 最后更新为 10 月 31 日，未包含本次会话。`exit 0` 仅出现在 `reproduction_notes.md` §9 的表格里。 |

**结论**：`End of eval`、成功指标、产物完整可以独立复核；**「退出码 0」目前属于「执行会话的记录」，项目内没有二方可复核的证据**。建议主文档引用时写「日志以 `End of eval` 结束、产物完整、无异常栈」，或注明退出码来自执行会话记录。

**给重跑的建议**（把退出码留痕，成本极低）：

```bash
# 在原命令末尾追加（不要改原命令本身），把退出码写进日志：
... > logs/eval-spatial-task0.log 2>&1; ec=$?; echo "EXIT_CODE=$ec" | tee -a logs/eval-spatial-task0.log
```

### 2.2 任务描述、成功判定与 CI95

#### 文件内容

`outputs/spatial-task0/task_descriptions.json`：

```json
{ "libero_spatial_0": "pick up the black bowl between the plate and the ramekin and place it on the plate" }
```

`outputs/spatial-task0/eval_info.json`（结构与关键值）：

- `per_task[0]`：`task_group="libero_spatial"`、`task_id=0`、`metrics={sum_rewards:[1.0], max_rewards:[1.0], successes:[true], per_episode:[{episode_ix:0, sum_reward:1.0, max_reward:1.0, success:true, seed:1000}]}`、`n_episodes=1`、`n_success=1`、`pc_success=100.0`、`pc_success_ci95=[20.654931437723747, 100.0]`
- `per_group.libero_spatial` / `overall`：`avg_sum_reward=avg_max_reward=1.0`、`pc_success=100.0`、`n_episodes=1`、`n_success=1`、`eval_s=8.896803617477417`、`video_paths=["outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4"]`、`predicted_video_paths=[]`

#### 任务描述原文（英文）——有两处，注意区别

1. **实际喂给策略的指令**（也就是 `task_descriptions.json` 里的串，小写、无 “akita”）：

   > `pick up the black bowl between the plate and the ramekin and place it on the plate`

   来源链路（源码核对）：`libero_task_map["libero_spatial"][0]`（`hf-libero` 包 `libero/libero/benchmark/libero_suite_task_map.py`）→ `grab_language_from_filename()` 由**文件名**生成语言串（`libero/libero/benchmark/__init__.py`）→ `Task.language` → `LiberoEnv.task_description`（`lerobot/src/lerobot/sim/integrations/libero.py:216`）→ rollout 里 `observation["task"] = env.call("task_description")` → 策略的语言输入。`lerobot_eval.py:936` 把它写成 `task_descriptions.json`。

2. **LIBERO BDDL 文件里的 `:language`**（大写、含 “akita”）：

   > `Pick the akita black bowl between the plate and the ramekin and place it on the plate`

   来源：`.../site-packages/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl:3`。

**引用提示**：这两串**不一致**。LeRobot/hf-libero 评测实际使用的是「文件名派生」的第一串；若在论文/文档里引用 LIBERO 官方 BDDL 语言，需注明与本地产物中的任务串不同，否则会被误认为数据/代码不一致。

#### 成功判定来源

LIBERO 环境 `check_success()` → Gymnasium `info` 的 `is_success` → `rollout()` 汇总为 `successes` → `success_summary()` 得到 `n_success/pc_success/pc_success_ci95`。单回合只产生一个 0/1 样本。

#### 成功率与 95% CI 的准确含义（n=1）

- `pc_success=100.0` 只是**样本比例** `1/1`，不是对真实成功率的可靠估计。
- `pc_success_ci95` 由 LeRobot 的 `wilson_interval(1, 1)` 计算（源码：`lerobot/src/lerobot/utils/eval_stats.py:45`，`success_summary` 在 `:87`；评测日志也打印 “95% Wilson interval 20.7% to 100.0%”）。Wilson score interval，`z=z(0.95)≈1.959964`，无连续性校正。公式：

  ```text
  centre = (p + z²/(2n)) / (1 + z²/n)
  half   = z·sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
  CI = [max(0, centre-half), min(1, centre+half)]
  ```

  代入 `n=1, k=1 (p=1)` 得下界 `20.654931437723747%`、上界 `100%`——与 `eval_info.json` 完全一致（本次已用脚本复算）。
- **统计意义**：它只描述「若把该策略在同类回合上重复采样，二项比例 95% 的区间覆盖方法」；在 n=1 时区间极宽、上界被截到 100%。它**不**代表模型在 10 个 LIBERO-Spatial 任务上的成功率，**不**覆盖 seed/初始状态/环境随机性带来的差异，也**不**支持「成功率 100%」的说法。
- 本次运行的确定性强约束：`n_episodes=1`、`seed=1000`、`init_states=true`、`hard_reset=true`，且 `init_state_id = episode_index = 0` —— 即只测了**一个固定的初始状态**。

### 2.3 动作步数与视频规格（ffprobe/ffmpeg 已重新验一次）

#### 动作步数

- 日志进度条：`Running rollout with at most 280 steps: 79/280 ... running_success_rate=100.0%`，随后进入 metrics 打印 → 环境在**第 79 步**判定 success 后 `done`，rollout 提前结束（LIBERO-Spatial `_max_episode_steps=280`，见 `lerobot/src/lerobot/sim/integrations/libero.py:135`）。
- 视频 80 帧 = reset 后 1 帧 + 79 个动作步各 1 帧（源码：`rollout()` 在 `env.reset()` 后调用一次 `render_callback`，之后每个 `env.step()` 后再调一次，`lerobot/src/lerobot/scripts/lerobot_eval.py:235-236, 343-344`；写视频时取 `stacked_frames[: done_index + 1]`）。
- 因此「**79 个动作步**」与「**80 帧 / 4.0 s**」互相自洽。按 `n_action_steps=50` 推断本回合有 2 次策略前向（50 + 29 个动作），但该细节日志未直接记录。

#### 视频规格（本次用环境内 ffprobe 复测）

复测工具：`/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffprobe`（version 7.1.1）

| 字段 | 值 |
| --- | --- |
| 文件 | `outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4`（85 754 B；sha256 `128d27a018a624d11e12654174cc515efe3f7898d3d3083b67c90635f21f31d0`） |
| codec | `h264` / `avc1` / profile **High** / level 21 |
| 分辨率 | 360×360（coded 360×360） |
| pix_fmt | `yuv420p` |
| 帧率 | `r_frame_rate=20/1`、`avg_frame_rate=20/1` |
| 时长 | `4.000000 s`（`duration_ts=40960`, `time_base=1/10240`） |
| 帧数 | `nb_frames=80` |
| 码率 | 167 984 bit/s |
| 容器 | MP4（`isom/iso2/avc1/mp41`，`encoder=Lavf61.7.100`） |
| 整段解码 | `ffmpeg -v error -i <mp4> -f null -` → **exit 0**，无错误输出 |

与 `logs/video-probe.json`（原复现时保存的 ffprobe 结果：h264 / 360×360 / `20/1` / `4.000000` / 80 帧）一致；本核对另存了完整复测 JSON：`figures/repro/video_probe_recheck.json`。

#### 抽帧（本次新增，存 `figures/repro/`）

帧索引 `n` ⇔ 环境步 `n`（帧 0 = reset 观测，帧 79 = 第 79 步后观测）：

| 帧 | 环境步 | 时间 | 原始 PNG | sha256（前 12 位） | 画面 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.00 s | `frame_000.png` | `0a29c488120d` | reset 初始观测：桌面、盘子、黑碗、烤盅、饼干盒、机械臂在上方 |
| 20 | 20 | 1.00 s | `frame_020.png` | `91979fb6be38` | 25%：机械臂下降到黑碗上方 |
| 40 | 40 | 2.00 s | `frame_040.png` | `2d9fbaff77af` | 50%：夹爪夹住黑碗 |
| 60 | 60 | 3.00 s | `frame_060.png` | `71277d0b8820` | 75%：抬起并向盘子移动 |
| 79 | 79 | 3.95 s | `frame_079.png` | `f2596f5509cf` | 末帧：黑碗已被放到盘子上（视觉确认；成功以环境 `check_success` 为准） |

同一目录另有：

- `annotated_frame_000/020/040/060/079.png`：单帧 + 烧录文字（step 数 / 秒数 / 帧号）。
- `contact_sheet.png`（780×1570，2×3）：5 帧 + 1 个 “Evidence panel”，每格标注 `step N / 79 · t = X s` 与画面说明。
- `video_probe_recheck.json`：本次 ffprobe 复测结果原文。

抽帧命令（可复算）：

```bash
FF=/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffmpeg
VID=/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4
OUT=/home/lxy/桌面/embodied/figures/repro
mkdir -p "$OUT"
for n in 0 20 40 60 79; do
  "$FF" -v error -y -i "$VID" -vf "select=eq(n\,$n)" -vsync 0 -frames:v 1 "$OUT/frame_$(printf '%03d' "$n").png"
done
```

（contact sheet 用 conda 环境自带 Pillow 13.0.0 + DejaVuSans 字体生成；不依赖 GPU/网络。）

### 2.4 权重完整性

本次对 `hf-cache` 中两个 `model.safetensors` 的**实际 blob 文件**重算 sha256（不只看记录）：

| 权重 | 大小 | 本次实测 sha256 | 与 `logs/model-sha256.txt` | Hub 缓存路径 |
| --- | --- | --- | --- | --- |
| `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d` | 906 712 520 B（≈865 MiB） | `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8` | **一致** | `models--lerobot--smolvla_libero/snapshots/a24f1f…/model.safetensors` → `blobs/9a9f6413…` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@4cde9d208e19519c36917aa170fffc7e381aa0ad` | 2 029 990 624 B（≈1.89 GiB） | `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3` | **一致** | `models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/4cde9d2…/model.safetensors` → `blobs/b9bfd456…` |

合计 2 936 703 144 B（≈2.94 GB）。两个文件都 <3 GB，因此**已做完整重算**（不是只比大小）。

补充说明：

- HF 缓存 `blobs/<sha256>` 的文件名就是该文件的 LFS sha256 oid；本次重算值同时等于「记录值」「blob 文件名」，因此等价于与 Hub 侧 LFS 哈希一致。
- 快照目录名与 revision 一致：主权重 `a24f1f38…`、骨干 `4cde9d20…`；骨干缓存 `refs/main` = `4cde9d208e19519c36917aa170fffc7e381aa0ad`（与 `reproduction_notes.md` §2 记录的提交一致）。
- ⚠️ **主权重缓存没有 `refs/main`**（只有 `snapshots/<sha>` 与 `trees/<sha>.json`），见 §5.3；离线复跑必须显式给 revision。
- 其余小文件（`config.json`、`train_config.json`、processor 等）未与 Hub 哈希逐项核对；只按快照结构核对了完整性（主权重快照 9 个文件含 `.gitattributes`，骨干快照 12 个文件，与下载日志 “Fetching 9 files/12 files” 的数量吻合）。

### 2.5 运行栈版本（与 notes §9 逐项对照）

| 组件 | 实测 | 证据 |
| --- | --- | --- |
| Python | 3.12.16 | `envs/lerobot-smolvla-repro/bin/python -V` |
| FFmpeg/ffprobe | 7.1.1（conda-forge） | `ffmpeg -version`、`logs/conda-explicit.txt`（`ffmpeg-7.1.1-gpl_hbbdf940_911.conda`） |
| LeRobot | 0.7.1（pip） | `logs/pip-versions.txt` |
| LeRobot 源码 commit | `83883ae60a8e1682552b6853e4170c74d83a4525`；`git status --porcelain` 为空 | `git -C lerobot rev-parse HEAD` / `git status` |
| PyTorch | 2.11.0+cu128（`cuda_available=False` 于本沙箱） | `pip-versions.txt` + 本会话 import |
| torchvision | 0.26.0+cu128 | `pip-versions.txt` |
| TorchCodec | 0.12.0+cpu | `pip-versions.txt` |
| Transformers | 5.22.2 | `pip-versions.txt` |
| huggingface_hub | 2.4.0 | `pip-versions.txt` |
| Accelerate | 1.16.0 | `pip-versions.txt` |
| Gymnasium | 1.4.0 | `pip-versions.txt` |
| hf-libero | 0.1.5 | `pip-versions.txt` |
| robosuite / MuJoCo | 1.4.0 / 3.8.1 | `pip-versions.txt` |
| NumPy / SciPy / PyAV | 2.2.6 / 1.18.1 / 15.1.0 | `pip-versions.txt` |
| `pip check` | `No broken requirements found.` | `logs/pip-check.log` |
| conda 版本 | 25.5.1（生成 explicit 清单的版本） | `logs/conda-explicit.txt` 头部 |

与 `reproduction_notes.md` §9 的版本列表**逐项一致**，未发现差异。

### 2.6 时间线与耗时（能推的写出来，不能推的写“未记录”）

| 完成时刻（文件 mtime） | 步骤 | 可推断耗时 |
| --- | --- | --- |
| 19:42:57 | `git clone` 完成（`logs/clone.log`） | 起点未记录 |
| 19:43:37 | conda 环境创建完成（`logs/conda-create.log`） | clone 后 ≤40 s 窗口内完成；精确耗时未记录 |
| 19:45:45–19:49:10 | 系统 apt 顺带升级 NVIDIA 580.178.04→580.196.07，**并安装新内核 6.8.0-151**（`/var/log/apt/history.log`；`/var/log/dpkg.log` 19:46:23 起 upgrade） | 与复现无关的外部变更；`/var/run/reboot-required` 19:49 生成 |
| 19:50:04 | `pip install huggingface-hub` 完成 | 未记录 |
| 19:50:18 | torch/torchvision wheel 下载完成 | torch wheel 820.3 MB @14.5 MB/s，日志 **0:00:53** |
| 19:50:56 | 骨干首次下载失败（SSL `UNEXPECTED_EOF_WHILE_READING`） | — |
| 19:51:44 | torchcodec wheel 下载完成 | 日志 **0:00:07** |
| 19:51:53 | 主 checkpoint 下载完成（9 files） | 日志 **0:00:45** |
| 19:52:42 | 骨干重试下载完成（12 files，实际落盘） | 重试耗时未记录（≈1 m 46 s 窗口，含启动） |
| 19:54:35 | `logs/model-sha256.txt` 生成 | 未记录 |
| 19:55:04 / 19:55:05 | 检查 hf-libero wheel / 启动全量 LIBERO 素材下载 | 全量素材下载**未完成**（进度停滞，后于评测后终止） |
| 20:00:50 | `pip install -e '.[evaluation,smolvla,libero]'` 完成 | 距上一日志 5 m 45 s 窗口（含人工/命令间隔）；pip 本身总耗时未记录 |
| 20:01:32 | task0 素材子集下载完成（57 files） | 日志 **0:00:18** |
| 20:01:42 | pip check / pip freeze / conda list --explicit | 未记录 |
| 20:01:57 | 预检失败（驱动错配，CUDA `Error 804`） | — |
| 20:02:38 | `LD_DEBUG=libs` 加载器日志（证实加载系统 `libcuda.so.580.196.07`） | 未记录 |
| 20:04:26 | scoped `nvidia-smi` 成功识别 RTX 5070（`logs/nvidia-scoped.log`） | — |
| 20:04:28 | scoped CUDA 张量测试成功（`cuda_tensor_result 3`，`EGL renderer None`） | — |
| **20:05:00 → 20:05:13** | **评测执行**：日志窗口 **13.41 s**；其中 `eval_s=8.8968 s`（rollout 8.63 s + 输出处理） | 进程启动前的 import 耗时未记录；窗口内约 4.5 s 用于建环境/加载策略/收尾（反推，未单独记录） |
| 20:06:10 | 视频探测（`logs/video-probe.json`） | — |
| 20:07:54 | `reproduction_notes.md` 写入 | — |

**总耗时**：19:42:57（第一条日志）→ 20:06:10（最终产物）≈ **23 分 13 秒**；到 `reproduction_notes.md` 完成 ≈ **24 分 58 秒**。注意这是「文件 mtime 差」，不是统一计时；各命令启动时刻大多没有记录，因此**不能**把 23 分钟精确拆到每一步。

**明确“未记录”的耗时**：conda create、pip install（lerobot）、checkpoint 之外的骨干下载、全量 LIBERO 素材（未完成）、`sha256sum`、各项预检、进程启动/import。

---

## 3. 当前 GPU / 驱动现场状态

### 3.1 现场读数（本次只读检查）

| 项 | 实测值 |
| --- | --- |
| 已加载内核模块 | `/proc/driver/nvidia/version` → `580.178.04`；`/sys/module/nvidia/version` → `580.178.04` |
| 磁盘上的内核模块 | `modinfo -F version nvidia` → `580.196.07`；文件 `/lib/modules/6.8.0-149-generic/updates/dkms/nvidia.ko` |
| 系统用户态驱动 | `readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1` → `/usr/lib/x86_64-linux-gnu/libcuda.so.580.196.07` |
| dpkg 包状态 | `libnvidia-compute-580 / gl / cfg1 / extra / nvidia-utils-580` 均为 `580.196.07-0ubuntu0.22.04.1` |
| 待重启 | `/var/run/reboot-required` 存在（`*** System restart required ***`，19:49），同级 `reboot-required.pkgs` 也在 |
| 内核 | 运行 `6.8.0-149-generic`；apt 于 19:45–19:49 安装了 `6.8.0-151`（重启会切内核） |
| 设备节点 | **本会话（workspace-write 沙箱）内 `/dev/nvidia*` 不存在** |
| `nvidia-smi` | `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.` |
| PyTorch | `torch 2.11.0+cu128, cuda_available False` |
| 项目内旧驱动运行时 | `driver-runtime-580.178.04/usr/lib/x86_64-linux-gnu/libcuda.so.1 → libcuda.so.580.178.04`；含 `libEGL_nvidia.so.0`、`libnvidia-eglcore.so.580.178.04`、`libGLX_nvidia.so.0`；EGL vendor json：`driver-runtime-580.178.04/usr/share/glvnd/egl_vendor.d/10_nvidia.json` |
| 当日 scoped 证据 | `logs/nvidia-scoped.log`：`NVIDIA GeForce RTX 5070, 580.178.04, 12227 MiB`；`logs/preflight-scoped.log`：`2.11.0+cu128 … (12, 0) … cuda_tensor_result 3`、`EGL renderer None` |

### 3.2 现在重跑要满足什么条件

**（1）权限：必须用 `danger-full-access`（或等价具有 `/dev/nvidia*` 访问权的）会话执行。**
本会话是 `workspace-write` 沙箱，看不到 `/dev/nvidia*`，任何 GPU 调用都会失败；本会话且已禁用审批提示，**无法从会话内部自行提权**。需要由有提权权限的会话（lead）执行。

**（2）驱动版本错配：二选一**

- **方案 A（不重启，当日已验证可行）**：使用项目内 `driver-runtime-580.178.04` 的用户态库（从 apt 缓存解出的 5 个 amd64 包），在评测命令里加：

  ```bash
  LD_LIBRARY_PATH=/home/lxy/lerobot-smolvla-repro/driver-runtime-580.178.04/usr/lib/x86_64-linux-gnu
  __EGL_VENDOR_LIBRARY_FILENAMES=/home/lxy/lerobot-smolvla-repro/driver-runtime-580.178.04/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  ```

  当日该组合已成功：scoped `nvidia-smi` 识别 GPU、CUDA 张量测试返回 3、评测成功（见 §3.1 日志）。这也是 `reproduction_notes.md` §8 命令能跑通的原因。

- **方案 B（重启对齐）**：重启后加载的内核模块为磁盘上的 `580.196.07`，与用户态库 `580.196.07` 对齐；此时**去掉**上面两个环境变量，用系统驱动即可。复跑前先执行 `nvidia-smi` 和一次 CUDA 张量测试确认。
  - 额外注意：重启同时会把内核从 `6.8.0-149` 切到已安装的 `6.8.0-151`，请确认新内核的 DKMS 模块版本也是 `580.196.07`（`modinfo -F version nvidia`、`/proc/driver/nvidia/version` 一致，`nvidia-smi` 正常）再评测。`reproduction_notes.md` 只提到“驱动对齐”，未提到内核切换（见 §5.4）。

**（3）离线复用条件（当前资产已够复跑 task 0）**

- `HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache` + `HF_HUB_OFFLINE=1`：权重（2.94 GB）与 tokenizer 等已落盘。
- **必须保留** `--policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d`，或直接把 `--policy.path` 指向本地快照目录 `hf-cache/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d`——因为主权重缓存没有 `refs/main`（§5.3），只写模型名在离线模式下可能解析不到。
- `LIBERO_CONFIG_PATH=/home/lxy/lerobot-smolvla-repro/libero-config`；`site-packages/libero/libero/assets` 是指向项目 `libero-assets/` 的符号链接（若重建 conda 环境需重新建链）。
- 输出用新的 `--output_dir` 和新的日志路径，避免覆盖当日证据。

**（4）建议顺带做的留痕**：记录退出码（§2.1）、`nvidia-smi --query-gpu=… --format=csv`、`ffprobe -of json` 结果。

---

## 4. 复现手册（可执行清单，供主文档引用）

> 命令保留原始绝对路径；「一次性」= 需要联网且只需做一次；「离线」= 之后不需要网络。

### 4.0 路径与环境变量约定

```bash
PROJ=/home/lxy/lerobot-smolvla-repro
ENV=/home/lxy/miniconda3/envs/lerobot-smolvla-repro
export HF_HOME=$PROJ/hf-cache
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

- 宿主：Ubuntu 22.04.5 LTS、RTX 5070 12 GB、Python 3.12（conda env `lerobot-smolvla-repro`）。
- 依赖的公共 wheel 缓存：`/home/lxy/ml-basic-workspace/.wheelhouse`（本机存在，可只读复用；换机需改为联网安装 PyTorch cu128 构建）。

### 4.1 一次性下载与安装（需联网）

```bash
# 1) 代码：固定提交
mkdir -p /home/lxy/lerobot-smolvla-repro/logs
cd /home/lxy/lerobot-smolvla-repro
git ls-remote https://github.com/huggingface/lerobot.git HEAD
git ls-remote https://huggingface.co/lerobot/smolvla_libero refs/heads/main
git clone --no-checkout https://github.com/huggingface/lerobot.git lerobot > logs/clone.log 2>&1
git -C lerobot switch --detach 83883ae60a8e1682552b6853e4170c74d83a4525

# 2) conda 环境（Python 3.12 + FFmpeg 7.1.1）
/home/lxy/miniconda3/bin/conda create -y -n lerobot-smolvla-repro --override-channels -c conda-forge \
  python=3.12 pip ffmpeg=7.1.1 > logs/conda-create.log 2>&1

# 3) 预下载 PyTorch / TorchCodec wheel
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/ml-basic/bin/python -B -m pip download --no-deps --only-binary=:all: \
  --python-version 3.12 --index-url https://download.pytorch.org/whl/cu128 --dest wheels \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 > logs/download-torch.log 2>&1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip download --no-deps \
  --index-url https://download.pytorch.org/whl/cpu --dest wheels torchcodec==0.12.0 > logs/download-torchcodec.log 2>&1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip install \
  'huggingface-hub>=1.31.0,<3.0.0' > logs/install-hub.log 2>&1

# 4) Hugging Face 权重（≈2.94 GB）
export HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/hf download lerobot/smolvla_libero \
  --revision a24f1f380461b665158d52acd140152c79e8b36d --max-workers 2 > logs/download-checkpoint.log 2>&1
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  config.json generation_config.json model.safetensors preprocessor_config.json processor_config.json \
  tokenizer.json tokenizer_config.json added_tokens.json special_tokens_map.json chat_template.json \
  merges.txt vocab.json --max-workers 2 > logs/download-vlm-retry.log 2>&1

# 5) LIBERO 场景/纹理素材（全量 repo 在当时下载停滞；task0 子集另见 4.2）
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/hf download lerobot/libero-assets --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --local-dir libero-assets \
  --max-workers 4 > logs/download-libero-assets.log 2>&1

# 6) 素材路径接入 + 权重校验留痕
ln -s /home/lxy/lerobot-smolvla-repro/libero-assets \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/lib/python3.12/site-packages/libero/libero/assets
sha256sum hf-cache/hub/models--*/snapshots/*/model.safetensors > logs/model-sha256.txt

# 7) 安装 LeRobot（editable，三组 extras）
cd /home/lxy/lerobot-smolvla-repro/lerobot
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip install \
  --find-links /home/lxy/ml-basic-workspace/.wheelhouse ../wheels/*.whl \
  -e '.[evaluation,smolvla,libero]' > ../logs/install-lerobot.log 2>&1

# 8) 版本留痕
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip check > ../logs/pip-check.log 2>&1
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -m pip freeze > ../logs/pip-versions.txt
/home/lxy/miniconda3/bin/conda list -n lerobot-smolvla-repro --explicit > ../logs/conda-explicit.txt
```

若目标机处于「内核模块与用户态库错配」窗口（本次当日情况），额外做一次（一次性）：

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

### 4.2 可离线复用的资产（无需网络）

| 资产 | 路径 | 占用 | 说明 |
| --- | --- | --- | --- |
| HF 权重缓存 | `$PROJ/hf-cache` | 2.8 GB | 主权重 866 MB + 骨干 1.89 GB + 元数据；`HF_HUB_OFFLINE=1` 可用；⚠️ 主权重无 `refs/main`，需带 revision |
| torch/torchcodec wheels | `$PROJ/wheels` | 793 MB | 重建环境时 `--find-links` 复用 |
| 公共 wheel 缓存 | `/home/lxy/ml-basic-workspace/.wheelhouse` | — | 本机旧的通用依赖 wheel（只读复用；换机需联网装 torch cu128） |
| conda 环境 | `/home/lxy/miniconda3/envs/lerobot-smolvla-repro` | 9.6 GB | Python 3.12.16 + ffmpeg 7.1.1 + 全部包 |
| LeRobot 源码 | `$PROJ/lerobot` | 294 MB | commit `83883ae`，`git status` 干净 |
| LIBERO 素材 | `$PROJ/libero-assets` | 225 MB / 1139 files | **不是完整素材**（全量下载未完成）；task0 实测离线可跑，其它任务可能缺素材 |
| LIBERO 配置 | `$PROJ/libero-config/config.yaml` | — | 供 `LIBERO_CONFIG_PATH`，避免交互式询问 |
| 旧驱动运行时 | `$PROJ/driver-runtime-580.178.04` | — | 仅在驱动错配窗口需要；重启对齐后可弃用 |
| task0 素材子集 | `$PROJ/libero-assets` 内（`stable_scanned_objects/akita_black_bowl`、`plate`、`glazed_rim_porcelain_ramekin`、`stable_hope_objects/cookies`、`textures`） | — | 本次 57 文件补齐，日志记录 57/57 成功 |

task0 素材子集补齐命令（全量下载停滞时的加速手段，需联网）：

```bash
cd /home/lxy/lerobot-smolvla-repro
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 HF_HOME=/home/lxy/lerobot-smolvla-repro/hf-cache \
  HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -B -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download("lerobot/libero-assets", repo_type="dataset", revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6", local_dir="libero-assets", allow_patterns=["stable_scanned_objects/akita_black_bowl/*", "stable_scanned_objects/plate/*", "stable_scanned_objects/glazed_rim_porcelain_ramekin/*", "stable_hope_objects/cookies/*", "textures/*"], max_workers=6))' \
  > logs/download-task0-assets.log 2>&1
```

### 4.3 每次评测命令（需要 GPU；两种驱动条件见 §3.2）

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

- **重启对齐后**：去掉 `LD_LIBRARY_PATH=...` 与 `__EGL_VENDOR_LIBRARY_FILENAMES=...` 两项（其余不变）。
- **重跑时**：`--output_dir` 与日志换成新路径；建议追加退出码留痕（§2.1）。
- 实测产物：`outputs/spatial-task0/{eval_info.json, task_descriptions.json, videos/libero_spatial_0/eval_episode_0.mp4}`。

### 4.4 结果复核命令

```bash
PROJ=/home/lxy/lerobot-smolvla-repro
FF=/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffmpeg
FP=/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffprobe
VID=$PROJ/outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4

# 视频规格 + 整段解码
"$FP" -v error -show_format -show_streams -of json "$VID"
"$FF" -v error -i "$VID" -f null - ; echo "decode_exit=$?"

# 抽帧命令见 §2.3；指标与任务串：
cat $PROJ/outputs/spatial-task0/eval_info.json
cat $PROJ/outputs/spatial-task0/task_descriptions.json
# 权重校验（记录值 + 实时重算）
cat $PROJ/logs/model-sha256.txt
sha256sum $PROJ/hf-cache/hub/models--*/snapshots/*/model.safetensors
# 源码版本与干净度
git -C $PROJ/lerobot rev-parse HEAD && git -C $PROJ/lerobot status --porcelain
```

### 4.5 资产占用与耗时汇总

| 阶段 | 一次性/每次 | 耗时（有据可查的值） |
| --- | --- | --- |
| 代码 clone + checkout | 一次性 | 未记录（19:42:57 完成） |
| conda 环境创建 | 一次性 | 未记录（19:43:37 完成，clone 后 ≤40 s 窗口） |
| 权重下载（主 checkpoint 9 files） | 一次性 | 45 s（HF 进度条） |
| 权重下载（骨干 12 files） | 一次性 | 未记录（首试失败；重试窗口约 1 m 46 s） |
| torch wheel 下载（820.3 MB） | 一次性 | 53 s（14.5 MB/s） |
| torchcodec wheel（2.4 MB） | 一次性 | 7 s |
| LIBERO 全量素材 | 一次性 | **未完成**（停滞，后终止） |
| task0 素材子集（57 files） | 一次性 | 18 s |
| `pip install -e .[evaluation,smolvla,libero]` | 一次性 | 未记录（≤5 m 45 s 窗口） |
| 驱动错配窗口的 runtime 解包 | 一次性（当日） | 未记录 |
| **单回合评测** | 每次 | **`eval_s = 8.897 s`**；日志窗口 13.41 s；进程启动/import 未记录 |
| 视频探测 / 抽帧+拼图 | 核对用 | 探测 <1 s；抽帧+contact sheet <1 s（本次新增，不属于原复现） |

磁盘占用：`hf-cache` 2.8 GB、`wheels` 793 MB、`lerobot` 294 MB、`libero-assets` 225 MB；项目合计 4.9 GB；conda 环境 9.6 GB。

---

## 5. 与 `reproduction_notes.md` 的差异与补充

以下不是「复现失败」或「数据造假」，而是**引用前需要补正/注明**的点。

### 5.1 「退出状态 0」缺少独立记录（notes §9）

- 现状：项目内没有任何文件记录退出码；`~/.bash_history` 未更新到本次会话；日志只证明 `End of eval` + 产物完整 + 无 Traceback。
- 影响：主文档若写「exit code 0」，严格说来源是执行会话的记录，不是项目内可复核证据。
- 建议：引用时降级为「日志以 `End of eval` 结束且无异常栈」，或重跑时把 `$?` 写入日志（§2.1）。

### 5.2 任务指令串与 LIBERO 官方 BDDL `:language` 不一致（notes 未展开）

- `task_descriptions.json`（实际喂给策略）：`pick up the black bowl between the plate and the ramekin and place it on the plate`（由文件名派生）。
- BDDL 文件 `:language`：`Pick the akita black bowl between the plate and the ramekin and place it on the plate`。
- notes §9 只写了中文意译「把盘子和小烤盅之间的黑碗放到盘子上」，没有暴露英文串有两个版本。主文档引用英文任务串时需注明「LeRobot/hf-libero 使用文件名派生串」。

### 5.3 主权重缓存没有 `refs/main`（notes §3 只提示了骨干）

- `models--lerobot--smolvla_libero/` 下只有 `blobs/`、`snapshots/a24f1f…/`、`trees/a24f1f….json`，**没有 `refs/` 目录**；骨干缓存则 `refs/main = 4cde9d20…` 存在。
- 影响：离线复跑若省略 `--policy.pretrained_revision=a24f…`，按名字解析可能失败；必须带 revision 或直接给本地 snapshot 路径。
- notes §3 末尾只写了「回放时可先确认骨干缓存 refs/main 指向该提交」，没有提主权重这一点，属于遗漏补充。

### 5.4 驱动升级时间窗、以及「重启」不只换驱动（notes §7/§10）

- notes §7：「dpkg 日志记录 NVIDIA 软件包在约 19:49 更新到 580.196.07」。实际 `/var/log/dpkg.log` 显示升级从 **19:46:23** 开始、约 **19:49:10** 结束；`/var/run/reboot-required` 生成于 19:49。引用时建议写「19:46–19:49 之间升级完成」。
- 同一次 apt 还安装了新内核 `6.8.0-151`（当前运行 `6.8.0-149`）。重启会**同时切换内核**，复跑前需重新确认新内核下的 NVIDIA 模块版本与用户态一致；notes §10 只提「让已安装的驱动模块与用户态库对齐」。

### 5.5 notes 未记载的资产占用与可复算耗时（本核对补充）

- 项目合计 4.9 GB（`hf-cache` 2.8 GB、`wheels` 793 MB、`lerobot` 294 MB、`libero-assets` 225 MB），conda 环境 9.6 GB。
- 可复算耗时：checkpoint 下载 45 s、torch wheel 53 s、torchcodec 7 s、task0 素材 18 s、评测 8.897 s；其余「未记录」（§2.6）。
- 这些数字适合放进主文档的环境/成本段落。

### 5.6 已逐条核对一致的点（从略，供放心引用）

| notes 说法 | 核对结果 |
| --- | --- |
| 退出状态：日志末尾 `End of eval` | ✅ 末行确为 `End of eval`，224 行，无 Traceback |
| success 1/1，sum_reward = max_reward = 1.0 | ✅ `eval_info.json` 一致 |
| 评测计时约 8.90 s | ✅ `eval_s = 8.896803617477417`；batch 8.63 s/it |
| 视频 H.264 / 360×360 / 20 fps / 4 s / 80 帧 | ✅ ffprobe 复核逐项一致，额外确认 High profile、yuv420p、85 754 B |
| 79 个动作步 | ✅ rollout 进度 79/280 后 success 提前结束；80 帧 = 1 reset + 79 |
| 两份权重与 Hub LFS SHA-256 一致 | ✅ 本次重算 sha256 与记录、blob 名三方一致 |
| 源码提交 83883ae，`git status --porcelain` 为空 | ✅ 一致 |
| 版本清单（Python 3.12.16、FFmpeg 7.1.1、LeRobot 0.7.1、torch 2.11.0+cu128、transformers 5.22.2 等） | ✅ `pip-versions.txt` / `conda-explicit.txt` 逐项一致 |
| 相机映射 agentview→camera1、eye_in_hand→camera2 | ✅ checkpoint `train_config.json` 的 `features_map` + `rename_map` + 日志 config 一致 |
| control_mode=relative / use_delta=True | ✅ `train_config.json` 为 relative；`libero.py:392-395` 在 relative 下设 `use_delta=True` |
| 训练配置 fps=30，评测 env 默认 20 Hz | ✅ `train_config.json` `env.fps=30`；评测日志 `fps=20` |
| 驱动：内核模块 580.178.04 / libcuda 580.196.07 / reboot-required / modinfo 580.196.07 | ✅ 本次现场读数完全一致 |
| 沙箱无 /dev/nvidia*、nvidia-smi 失败 | ✅ 本会话复现该现象（`torch.cuda.is_available()=False`） |
| 全量 LIBERO 素材未下载完 | ✅ `libero-assets` 仅 225 MB / 1139 files，`logs/download-libero-assets.log` 无完成记录 |
| 首试/重试的 TLS 中断与恢复 | ✅ `download-checkpoint.log`（3 次重试后成功）、`download-vlm.log`（Error + SSL）一致 |

---

## 6. 证据边界（不要夸大的地方）

1. **n=1**：单回合成功只说明「这条链路能跑通」。LeRobot 的 Wilson 区间 `[20.7%, 100.0%]` 在 n=1 下没有成功率估计意义；不能写「成功率 100%」。
2. **单 task**：只测 `libero_spatial` 的 task 0（该 suite 共 10 个任务），未覆盖其它任务。
3. **单 seed / 单初态**：`seed=1000`、`init_states=true`、`hard_reset=true`、`episode_index=0` → 只测了一个固定初始状态。
4. **未训练/微调**：使用公开权重推理，未做任何训练或微调；也未做多 seed、完整 suite 评测。
5. **素材不完整**：LIBERO 全量素材未下载完；本次只是「现有已落盘素材恰好够 task0 离线跑」。
6. **驱动窗口是临时绕过**：当日评测是在「内核模块 580.178.04 + 用户态 580.196.07 错配、用项目内 580.178.04 用户态库」的条件下成功的；机器仍处于待重启状态，这不是干净环境。
7. **环境差异**：以上结论来自 2026-11-20 的当日宿主机（RTX 5070、Ubuntu 22.04.5）；换机器/重启后需按 §3.2 重新确认。
8. **退出码**：见 §5.1，不能作为「已复核」项引用。

---

## 7. 本次核对使用的只读命令（便于复算）

```bash
# 日志与产物
cat /home/lxy/lerobot-smolvla-repro/reproduction_notes.md
cat /home/lxy/lerobot-smolvla-repro/logs/eval-spatial-task0.log
cat /home/lxy/lerobot-smolvla-repro/logs/{pip-versions.txt,conda-explicit.txt,model-sha256.txt,video-probe.json,nvidia-scoped.log,preflight-scoped.log,pip-check.log}
cat /home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/{eval_info.json,task_descriptions.json}
stat -c '%y  %s  %n' /home/lxy/lerobot-smolvla-repro/logs/* | sort

# 权重（本次实测重算）
sha256sum /home/lxy/lerobot-smolvla-repro/hf-cache/hub/models--lerobot--smolvla_libero/blobs/9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8
sha256sum /home/lxy/lerobot-smolvla-repro/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/blobs/b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3

# 视频
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffprobe -v error -show_format -show_streams -of json \
  /home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffmpeg -v error -i \
  /home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4 -f null -

# GPU / 驱动现场
cat /proc/driver/nvidia/version
cat /sys/module/nvidia/version
modinfo -F version nvidia
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
cat /var/run/reboot-required; ls -l /dev/nvidia* 2>&1
/usr/bin/nvidia-smi 2>&1 | head -3
grep -n "580.196.07" /var/log/dpkg.log | head; tail -30 /var/log/apt/history.log

# 版本与源码
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python -V
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/ffmpeg -version | head -2
git -C /home/lxy/lerobot-smolvla-repro/lerobot rev-parse HEAD
git -C /home/lxy/lerobot-smolvla-repro/lerobot status --porcelain
```

---

## 附录 A：本次核对涉及的文件校验值

```
# 核心产物（sha256）
e52e567ee2c0f8b32dbdb3fea549afb365239c3acd531c840390d6599d324cf5  outputs/spatial-task0/eval_info.json
8544df2e1d6a8416128f03f2855af25d8f26cc5369637e944373c5efe32b05d1  outputs/spatial-task0/task_descriptions.json
128d27a018a624d11e12654174cc515efe3f7898d3d3083b67c90635f21f31d0  outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4
7bf88de709a74d22e3bf680ab0962415a8ced8471053e51748057448c2e94899  logs/eval-spatial-task0.log
8cb769c3e957124decdad552d1bea61713a6e2b8a26d7532dd3a0527f36f69f9  logs/model-sha256.txt
f922205761d3cdc356740bd1bc0096ad53c85b3abf6a1c65b5f1806e4752978f  logs/video-probe.json

# 权重（本次实测）
906712520 B  9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8  smolvla_libero/model.safetensors
2029990624 B b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3  SmolVLM2-500M/model.safetensors

# figures/repro（本次新增，绝对路径 /home/lxy/桌面/embodied/figures/repro/）
0a29c488120d06248ddc5ed92b02694a06a17f83f536ba7d1fc44801205e859a  frame_000.png
91979fb6be38a34a4168e6c573168b4e9660f3bb9291a65b4ebd9e8757a3c0b5  frame_020.png
2d9fbaff77af9a9c85466b673a44ea835f97dfd19c009b90cc1a4921cba1bef5  frame_040.png
71277d0b8820e2d904a65680aad8bb62132b70f067c515b63ee5639b9402df1a  frame_060.png
f2596f5509cfec87e57b406465ecfdf44d17510efc00c56c056fb95b48ca23b6  frame_079.png
491b0d3fa65a90fae53a4025e9d7287cefab1ff2f06055bdf3b6ea7c52e569fc  contact_sheet.png
```

## 附录 B：图与文件索引

| 文件 | 说明 |
| --- | --- |
| `figures/repro/frame_000.png` … `frame_079.png` | 首帧 / 25% / 50% / 75% / 末帧（原始 360×360） |
| `figures/repro/annotated_frame_000.png` … `annotated_frame_079.png` | 同上，烧录 step / 秒数 / 帧号文字 |
| `figures/repro/contact_sheet.png` | 2×3 contact sheet（5 帧 + Evidence panel） |
| `figures/repro/video_probe_recheck.json` | 本次 ffprobe 复测原文 |
| `/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4` | 原复现视频（只读引用） |
| `/home/lxy/lerobot-smolvla-repro/logs/eval-spatial-task0.log` | 原复现完整日志（只读引用） |
| `/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/eval_info.json` | 指标（只读引用） |
| `/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/task_descriptions.json` | 任务串（只读引用） |
| `/home/lxy/lerobot-smolvla-repro/logs/model-sha256.txt` | 权重 sha256 记录（只读引用） |
