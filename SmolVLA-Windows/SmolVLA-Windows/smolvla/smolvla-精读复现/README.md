# SmolVLA：论文精读 + 本地复现

> 论文：[SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)（arXiv:2506.01844v1，2025-06-02，24 页，HuggingFace 团队）
> 代码：[huggingface/lerobot](https://github.com/huggingface/lerobot) commit `83883ae60`（LeRobot 0.7.1）
> 本机复现：RTX 5070 12 GB · Ubuntu 22.04 · 单回合 GPU 推理 **success 1/1**

这份材料分两条线，可以分开看，也可以交叉看：

1. **精读线**：逐节拆论文的问题、方法、公式、每张表和每个消融，并指出不自洽/可质疑处（`notes/01`–`notes/04`）。
2. **复现线**：把论文公式映射到 LeRobot 代码，记录本机这次复现的每一步、证据、与论文设置的差异、以及怎么重跑（`notes/05`、`repro/`、`notes/_raw/repro-evidence.md`）。

---

## 一、复现状态（一句话版）

**已经跑通"论文模型 + 官方权重 + LIBERO-Spatial task 0 + 单回合 GPU 推理"这条链路**：`lerobot/smolvla_libero` 在 RTX 5070 上以 seed 1000 完成 79 步 rollout，`success=true`，产出 360×360 / 20 fps 的成功视频（黑碗被放到盘子上）。
**并且做了第二次独立复跑（2026-11-20 20:55）：退出码 0、success 1/1，产出视频与第一次逐字节相同（SHA-256 一致）**——同 seed 下这条链路是确定性的。
但这是 **n=1 的链路验证，不是成功率结论**（Wilson 95% 区间 20.7%–100%）；训练/微调、多回合成功率统计、异步推理栈都还没做。

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 环境（conda `lerobot-smolvla-repro`，Python 3.12，torch 2.11+cu128） | ✅ 可用 | `notes/05-代码对照与复现.md` §4 |
| 权重（`lerobot/smolvla_libero@a24f1f3` 906 MB + SmolVLM2-500M 2.03 GB） | ✅ sha256 已与 Hub LFS 核对 | `notes/_raw/repro-evidence.md` §2.4 |
| LIBERO 素材（`lerobot/libero-assets@0b3ea86`） | ✅ 够 task 0 用（全量未下完） | 同上 §4.5 |
| 单回合推理 | ✅ success 1/1，首次 8.90 s / 复跑 4.55 s，79 步 | `figures/repro/contact_sheet.png` |
| 复跑确定性验证 | ✅ 第二次独立运行：exit code **0**，success 1/1，视频 SHA-256 与首次完全一致 | `repro/rerun_check.json` |
| 执行协议探针 | ✅ `n_action_steps=50` 成功（79 步）/ `=1` 失败（280 步，n=1 单例） | `repro/protocol_probe.json`、`notes/05` §6.6 |
| 多回合/多任务成功率 | ⬜ 未做（论文每任务 10 trials；LeRobot 社区常用每任务 50 episodes） | — |
| 训练/微调复现 | ⬜ 未做（需 100k–200k steps） | — |
| 异步推理栈复现 | ⬜ 未做（需要真机或双进程模拟） | — |
| **现在能否立即重跑** | ✅ 能（本次已用 `danger-full-access` + 项目内驱动垫片成功复跑） | 见下 |

> ⚠️ **当前机器的两个坑**（复现前必读）：
> 1. **驱动版本错配**：已加载内核模块是 `580.178.04`，用户态 `libcuda.so.1` 已升级到 `580.196.07`（`/var/run/reboot-required` 存在，同一次 apt 还装了新内核 6.8.0-151）。上次运行是用项目内 `driver-runtime-580.178.04/` 里的旧库 + `LD_LIBRARY_PATH` 绕过的。
> 2. **沙箱看不到 GPU**：普通 bash 沙箱里没有 `/dev/nvidia*`（`torch.cuda.is_available()==False`），要跑 GPU 必须用 `danger-full-access` 提权。**本次复跑已验证：提权 + 项目内 `driver-runtime-580.178.04` 垫片可以在不重启的情况下正常识别 RTX 5070 并完成评测**（见 `repro/rerun_check.json`）。
>
> 最干净的做法仍然是**重启机器**，让内核模块和用户态库都变成 580.196.07，然后去掉 `LD_LIBRARY_PATH` 重跑。

---

## 二、文档地图

| 文件 | 内容 | 建议读法 |
| --- | --- | --- |
| [`notes/01-导读与背景.md`](notes/01-导读与背景.md) | 论文定位、5 个贡献、阅读路线、前置知识、8 个带读问题、术语表 | 先读，20 分钟 |
| [`notes/02-方法精读.md`](notes/02-方法精读.md) | §3 逐节：架构、layer skipping、64 视觉 token、flow matching 公式逐项、CA/SA 交错掩码、异步推理算法与 g 的数学 | 对照 PDF 第 3–8 页 + 图 1/2/3 + 算法 1 |
| [`notes/03-实验与消融精读.md`](notes/03-实验与消融精读.md) | §4 逐表：13 张表的读法、关键数字、每表支持什么结论、表间 baseline 是否自洽 | 对照 PDF 第 8–14 页 |
| [`notes/04-批判性阅读.md`](notes/04-批判性阅读.md) | 15 条可疑点/未回答问题，审稿人视角 | 第三遍读 |
| [`notes/05-代码对照与复现.md`](notes/05-代码对照与复现.md) | 论文公式 → `文件:行号` 对照；本次复现全过程、结果 vs 论文、重跑命令；官方 checkpoint 的实际训练配置 | 动手前必读 |
| [`notes/06-自测题.md`](notes/06-自测题.md) | 20 道题（含答案），检验是否真的读懂了 | 读完后自测 |
| [`notes/_raw/`](notes/_raw/) | 原始素材：论文实验数据数字化、代码对照长笔记、复现证据核对（631 行） | 需要细节时查 |
| [`figures/`](figures/) | 论文图 1–4、我自己画的 3 张教学图、复现视频抽帧 | 读 02 时对着看 |
| [`repro/verify_model_facts.py`](repro/verify_model_facts.py) | 用本地缓存权重核对参数量/视觉 token 数（只读） | 想自己验证结构时跑 |

**我自己画的 3 张教学图**（论文里没有对应图，是理解关键）：

| 图 | 解决什么问题 |
| --- | --- |
| [`figures/fig-dataflow.png`](figures/fig-dataflow.png) | 一次推理的张量形状数据流：什么 token 进 VLM、什么进 expert、KV 怎么复用 |
| [`figures/fig-attention-masks.png`](figures/fig-attention-masks.png) | 论文只是文字描述"交错 CA/SA + 因果掩码"，这里画出三种掩码的真实形状（227×227 / 50×227 / 50×177） |
| [`figures/fig-flow-matching.png`](figures/fig-flow-matching.png) | 论文公式的 $\tau$ 与代码里的 $t$ 互为镜像（这是读代码时第一个会踩的坑）和 $\tau\sim\mathrm{Beta}$ 的真实含义 |

---

## 三、两条阅读路线

**路线 A：30 分钟速览（只想搞清论文说了什么）**
1. 本文件 §4 的"关键数字速查表"
2. `notes/01` §0–§2（定位 + 三个贡献）
3. `figures/fig-dataflow.png` + `paper-fig1-architecture.png`
4. `notes/03` 的 Table 2 / Table 5 / Table 13 三张表的解读（主结果、预训练价值、动作执行策略）

**路线 B：3 小时精读（要吃透、要复现/改模型）**
1. `notes/01` 全文（20 min）
2. 打开 PDF，按 `notes/02` 逐节走 §3.1–§3.3，每读完一节回来看对应的掩码图/数据流图（90 min）
3. `notes/03` 对着 PDF 第 8–14 页逐表读，重点看"表间 baseline 是否自洽"（40 min）
4. `notes/04`（20 min）
5. `notes/05` + 跑一遍 `repro/verify_model_facts.py`（30 min）

---

## 四、关键数字速查表

| 维度 | 论文数字 | 本地实测/核对 |
| --- | --- | --- |
| 总参数量 | 450M（其中 action expert ≈100M） | `450,046,176`；expert `98,245,840` ✅ |
| VLM 骨干 | SmolVLM2-500M（SigLIP + SmolLM2，**32 层只用前 16 层**） | 实测 32 层；16 层文本参数量 157M |
| 视觉 token | 每帧 **64 个**（不做 tiling + pixel shuffle） | 实测：1024 patch → connector → 64 token ✅ |
| 语言 token | 最多 48 | `tokenizer_max_length=48` ✅ |
| 动作 chunk | n = 50；训练/评测时 10 步 Euler | `chunk_size=50, num_steps=10` ✅ |
| 动作维度 | 7 维（3 平移 + 3 旋转 + 1 夹爪），内部 pad 到 32 | LIBERO 配置：state 8 维 / action 7 维 ✅ |
| 预训练数据 | 481 个社区数据集 / 22.9K episodes / 10.6M frames | 论文 Table 1（caption 写 "~10M episodes" 是笔误） |
| 预训练规模 | 200k steps，global batch 256，4 GPU，≈30k GPU-hours | 论文 §4.3 |
| LIBERO 主结果 | 0.45B 模型 87.3%（4 个子集平均，**未用机器人数据预训练**） | 论文 Table 2；官方 release 的 checkpoint 是另一套训练配方（见 `notes/05` §5） |
| 真机 SO100 主结果 | 78.3%（多任务+预训练；ACT 48.3 / π₀ 61.7） | 论文 Table 3 |
| 异步推理 | 13.75 s → 9.70 s（≈30% 更快）；固定时间 9 → 19 次成功 | 论文 Figure 5 |
| 本地复现 | LIBERO-Spatial task 0 单回合 success 1/1 | 8.90 s，79 步，360×360/20fps，seed 1000 |

---

## 五、证据索引（想自己核查时用）

- **本仓库内自包含归档**：[`repro/artifacts/`](repro/artifacts/)（三次运行的日志、指标 JSON、任务描述、三段视频、原始复现记录）——在 Windows/网页上直接点开就能看
- 复现视频：`/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/videos/libero_spatial_0/eval_episode_0.mp4`
- 复跑视频（逐字节相同）：`/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0-rerun/videos/libero_spatial_0/eval_episode_0.mp4`
- 复跑证据 JSON（退出码/GPU/哈希）：`repro/rerun_check.json`
- 执行协议探针证据（n_action_steps 50 vs 1）：`repro/protocol_probe.json`、`figures/repro/probe-n1/`
- 抽帧 + 标注 + contact sheet：[`figures/repro/`](figures/repro/)
- 评测日志（含 policy 配置、running success、End of eval）：`/home/lxy/lerobot-smolvla-repro/logs/eval-spatial-task0.log`（首次）、`logs/eval-rerun.log`（复跑，末行 `EXIT_CODE=0`）；仓库内副本见 `repro/artifacts/`
- 指标 JSON：`/home/lxy/lerobot-smolvla-repro/outputs/spatial-task0/eval_info.json`
- 权重 sha256 核对：`notes/_raw/repro-evidence.md` §2.4
- 复现全过程原始记录：`/home/lxy/lerobot-smolvla-repro/reproduction_notes.md`；仓库内副本：[`repro/artifacts/reproduction_notes_原始记录.md`](repro/artifacts/reproduction_notes_原始记录.md)

> 📄 论文原文（PDF）因版权未入库，请从 [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) 下载；本仓库的页码引用都基于 arXiv v1 版（24 页）。
- 结构核对脚本输出：`repro/verify_model_facts.json`

## 六、重跑一句话

```bash
# 重启机器后（推荐），在 /home/lxy/lerobot-smolvla-repro 下：
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=$PWD/libero-config \
HF_HOME=$PWD/hf-cache HF_HUB_OFFLINE=1 \
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d \
  --policy.device=cuda --policy.num_steps=10 --policy.n_action_steps=50 \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
  --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false --seed=1000 \
  --output_dir=outputs/spatial-task0-rerun
```

完整环境变量（驱动错配时的 `LD_LIBRARY_PATH` / EGL 垫片）、耗时、资产清单见 [`notes/05-代码对照与复现.md`](notes/05-代码对照与复现.md) §6 和 [`notes/_raw/repro-evidence.md`](notes/_raw/repro-evidence.md) §4。

> 想跟论文数字对齐时注意：论文仿真评测是**每执行 1 个动作就重新预测**（`n_action_steps=1`，Table 13 的 80.3%），而这次复现用的是 checkpoint 自带的 `n_action_steps=50`（Table 13 里只有 51.8% 的配置）。两者不可直接比，详见 `notes/05` §5.2。
