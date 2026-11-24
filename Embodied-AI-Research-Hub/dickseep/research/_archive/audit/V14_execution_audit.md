---
title: "V14：OpenVLA-OFT 动作队列与实时执行机制代码审计"
date: "2026-11-01"
status: "静态审计与标准库独立验证完成；模型与仿真未运行"
repository: "https://github.com/moojink/openvla-oft"
branch_resolved: "main"
commit: "e4287e94541f459edc4feabc4e181f537cd569a8"
---

# V14 执行机制审计

## 1. 结论与证据边界

**源码事实 F1：LIBERO 配置 P=8、动作维度 7；get_action 返回完整的 8 个动作，队列才决定保留哪些动作。**

**源码事实 + 独立验证 F2：原实现使用有界 deque 接收完整预测块。E<P 时执行的是后 E 步，不是前 E 步；E=1 首次执行索引 7。** 因此只改 num_open_loop_steps 得到的成功率曲线不能解释为标准的“执行前 E 步后重规划”曲线。

**源码事实 F3：所审计 LIBERO 入口是同步的，队列为空后阻塞推理，再 env.step；没有动作执行与模型推理的流水线重叠。** 因而旧方案的 max(E·Δt,L) 公式和“L≤P·Δt 就能不间断闭环”不适用于此入口。

标记约定：**[F]** 为固定 SHA 源码事实；**[V]** 为已执行的独立逻辑验证；**[D]** 为在明确假设下的推导；**[H]** 为尚需实测的假设。没有下载权重，没有导入 PyTorch/NumPy/LIBERO，没有运行机器人或仿真。本报告不声称已验证 checkpoint 数值输出、成功率或 RTX 5070 延迟。

## 2. 固定版本与代码位置

审计时间：2026-11-01，Asia/Shanghai。GitHub API 将 main 解析为 **e4287e94541f459edc4feabc4e181f537cd569a8**；提交时间为 **2025-09-09T06:25:06Z**，提交说明为 `Update pyproject.toml: Pin diffusers version`。[提交记录](https://github.com/moojink/openvla-oft/commit/e4287e94541f459edc4feabc4e181f537cd569a8)。以下结论绑定这个 SHA，不绑定未来的 main。

工作区 `D:\Desktop\dickseep` 不是 Git 仓库（git status 返回 not a git repository）；没有发现本地官方 checkout。本次只将少量固定 SHA 源文件保存为 work/V14/source 下的审计快照，不是克隆或修改官方仓库。原始科研报告在审计前保存 SHA256 清单，以便结束时核对未变更。

| 源文件（永久链接） | 行号 | 审计作用 |
|---|---|---|
| [prismatic/vla/constants.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/vla/constants.py#L26-L87) | 26–30、48–77 | LIBERO 常量、平台检测与导出 |
| [experiments/robot/libero/run_libero_eval.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/run_libero_eval.py#L279-L359) | 100、302–350 | E 默认值、有界队列、同步 rollout |
| [experiments/robot/robot_utils.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/robot_utils.py#L99-L146) | 99–146 | get_action 封装与实际转发 |
| [experiments/robot/openvla_utils.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/openvla_utils.py#L715-L798) | 780–798 | 模型调用、完整块转 list |
| [prismatic/extern/hf/modeling_prismatic.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/extern/hf/modeling_prismatic.py#L912-L1058) | 923–942、1055–1058 | reshape、CPU NumPy、反归一化和返回值 |
| [prismatic/models/action_heads.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/models/action_heads.py#L84-L107) | 98–107 | L1 head 输出时间维 |
| [experiments/robot/libero/libero_utils.py](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/libero_utils.py#L18-L25) | 18–25、53 | 外部环境构造；视频 fps 并非控制周期 |

额外文件用于追踪三个指定入口的被调用函数，未扩大为环境复现。来源 URL、抓取时间及每文件 SHA256 保存在 `work/V14/source_manifest.json`，完整提交响应在 `work/V14/commit.json`。

## 3. P/E 与动作输出

| 变量 | 定义 | 固定版本行为 |
|---|---|---|
| P | 一次模型调用预测的时间步数 | LIBERO 的 NUM_ACTIONS_CHUNK=8 |
| D_a | 每步动作维度 | LIBERO ACTION_DIM=7 |
| E_cfg | num_open_loop_steps 配置值，本报告简写 E | 默认 8；作为 deque.maxlen |
| E_run | 某块实际 env.step 次数 | 1≤E≤P、无提前结束时为 E；done/异常/步数上限可使其更短 |
| Δt_sim | env.step 一次推进的仿真时间 | 来自实际 LIBERO/robosuite 环境配置，本次未加载；不能用墙钟 step 耗时或视频 fps 替代 |
| Δt_wall,j | 第 j 次 env.step 调用到返回的墙钟耗时 | 包含该环境实现的物理/渲染等成本，未实测 |
| L_k | 第 k 次 get_action 调用到完整动作列表可用的墙钟延迟 | 包含其内部图像处理、设备搬运、模型计算和输出转换；不包含调用前 prepare_observation |
| H_j | 每步除 get_action 和 env.step 以外的串行开销 | prepare_observation、图像留存、队列操作、process_action、tolist 等 |
| Δt_target | 拟部署执行器的目标墙钟控制周期 | 外部设计参数，不是本代码自动保证的周期 |

[F] 常量并非全平台统一为 8：ALOHA=25、BRIDGE=5。constants.py 根据 sys.argv 中的 libero/aloha/bridge 选择平台，无法判断则默认 LIBERO；标准 run_libero_eval.py 命令路径含 libero。包装入口需记录实际 ROBOT_PLATFORM，不能仅看权重名称推定。

[F] 默认 use_l1_regression=True、use_diffusion=False。对正常的 batch=1、LIBERO L1 推理链：

1. L1 head 按时间块 reshape 为 `[B,P,...]`，产生 `[B,P,7]`。
2. predict_action 内将其 reshape 为 `[8,7]`，转换为 CPU NumPy，反归一化不改变时间轴，实际返回 **tuple(actions, actions_hidden_states)**。该函数的 `-> np.ndarray` 注解与真实 tuple 返回不完全一致，不能据注解判断。
3. get_vla_action 丢弃 hidden states，按 i=0…len(action)-1 构造 **list[np.ndarray]**；长度 8，每个元素 shape `(7,)`。get_action 原样返回这个 list。它的 Union 类型注解更宽，但本分支的实际容器是 list；外层 list 自身没有 `.shape`，堆叠后的等价 shape 才是 `(8,7)`。
4. 记 `a_i` 为该块第 i 个预测时间槽，返回顺序是 `a_0,a_1,…,a_7`；没有时间翻转或按 E 截断。process_action 位于 popleft 后，只处理单步夹爪数值，不负责补回被丢弃的时间槽。

[D] 对与该代码匹配的 LIBERO checkpoint，源码接口预期完整 P=8 步输出。**这不是已加载 checkpoint 的实测结论**：没有下载或核对某个权重的 revision、远程自定义模型代码、归一化统计及 head 文件。任意第三方 checkpoint 不在此保证范围内。

## 4. deque 索引追踪与问题

[F] run_episode 在队列为空时才查询：`deque(maxlen=E)` → `extend(actions)` → 每次 `popleft()` → process_action → env.step。Python 有界 deque 从右添加元素时，溢出会从左淘汰；这不是“只添加前 E 个”。

[V] 以下是 P=8、空队列、一次 extend 后的真实标准库结果。索引从 0 开始；假设不提前结束。

| E | extend 后实际队列（左→右） | 第一次执行索引 | 该块完整执行顺序 | 被丢弃的前缀 |
|---|---|---|---|---|
| 1 | [7] | 7 | 7 | 0…6 |
| 2 | [6,7] | 6 | 6→7 | 0…5 |
| 4 | [4,5,6,7] | 4 | 4→5→6→7 | 0…3 |
| 8 | [0,1,2,3,4,5,6,7] | 0 | 0→1→2→3→4→5→6→7 | 无 |

[V] 记下一次预测块为 b，连续两块的实际执行分别为：E=1：a7→b7；E=2：a6→a7→b6→b7；E=4：a4→a5→a6→a7→b4→b5→b6→b7；E=8：a0…a7→b0…b7。每块内部顺序未反转，但首次执行的预测时间槽偏移为 P−E。

[D] E<P 时同时改变了“执行次数”和“预测起始时间槽”，构成 T1′ 的混杂因素。不能将其成功率变化直接归因于重规划频率、偏离训练配置或开环误差。官方 WARNING 只是建议执行完整块，不能证明后缀选择是设计意图，更不能证明所有 E<P 必然降分。

[F/D] 边界也应在实验包装层校验：E=0 的 deque 不保留动作，随后 popleft 会抛异常并被 episode 的异常处理捕获；E<0 在构造 deque 时就失败；E>P 不是执行 E 步，而是仅有 P 步。原片段未显式约束 1≤E≤P。默认 E=P 不受此前缀淘汰问题影响；本审计不据此否定论文默认配置结果。

## 5. 同步与异步时序

### 5.1 当前实现：同步阻塞

[F] 空队列时发生普通函数调用，没有 future/线程/后台推理提交。get_action 的 no_grad/inference_mode 只控制自动求导，不是异步执行器。模型输出 `.cpu().numpy()` 使 CPU 动作在返回前可用；CUDA 内部 kernel launch 的异步性不等价于 env.step 与推理重叠。

```text
墙钟时间 →
主线程   prepare(obs_k) | get_action_k，等待 L_k | 填队列 | step_1 | prepare | step_2 | … | step_E
模型                         [完整预测块]
环境     本入口未调用 step，未主动推进仿真           [推进]            [推进]       [推进]
下一轮   队列已空 → prepare(obs_{k+1}) → get_action_{k+1}，再次等待 → step_1 …
```

[F] 初始化的 num_steps_wait 是前若干次 dummy env.step 的数量（319–321 行），不是秒数，也不是为了掩盖推理延迟的异步等待。本轮审计公式排除这段预热与 reset、视频写出等 episode 外成本。

[F] 准备观测在每个有效动作迭代都执行，即使队列非空也执行；只有空队列迭代调用模型。此入口没有显式控制频率 sleep、deadline 或实时节拍器。OffScreenRenderEnv 来自外部依赖，本次没有审计其锁定版本内部；准确的物理推进语义和 Δt_sim 仍待检查，不能凭这一入口宣称环境有后台实时推进。

### 5.2 根据顺序推导当前墙钟指标

[D] 对一个完整执行 E 步的块 k，定义 `S_k=Σ_j Δt_wall,kj`、`H_k=Σ_j H_kj`。以该块开始的迭代至最后一个 env.step 返回为周期：

```text
T_block,k = L_k + S_k + H_k
T_action,kj = H_kj + Δt_wall,kj + 1{队列为空}·L_k
平均每动作墙钟成本 = T_block,k / E
f_exec,k = E / (L_k + S_k + H_k)
f_replan,k = 1 / (L_k + S_k + H_k)
推理阻塞占比 rho_infer,k = L_k / (L_k + S_k + H_k)
```

非查询动作迭代没有 L，但块边界动作周期额外包含整个 L，因此平均频率不能掩盖边界长间隔。若近似每步 `Δt_wall=s`、`H_j=h`：`T_block=L+E(s+h)`，`f_exec=1/(s+h+L/E)`，`rho_infer=L/[L+E(s+h)]`。

对不等长/提前结束的真实 episode，应报 `f_exec=总执行动作数/总rollout墙钟时间`，`rho_infer=ΣL/总rollout墙钟时间`，不能直接平均各块的频率。等待占比描述的是调用方没有送出新动作的时间份额，不代表 GPU 空闲，更不代表真机器人完全静止；真实执行器可能保持旧命令。

[D] 仿真累计推进量在固定步长下为 `N·Δt_sim`，real-time factor 为 `N·Δt_sim/T_wall`。Δt_wall 是计算耗时，Δt_sim 是物理时间推进量，两个量不能替换。若仿真仅由 step 推进，添加纯 sleep 只会拉长墙钟，不会模拟等待期间物体继续运动。

[D] 若仅要求**平均**执行频率达到 `1/Δt_target`，上述同步循环的必要比较为：

```text
L + E(s+h) ≤ E·Δt_target
即 L ≤ E[Δt_target − (s+h)]
```

这仅是平均速率预算，既不是固定间隔控制保证，也不是任务成功保证。边界动作还有 L，必须另查最坏单步间隔、p95/p99 和 deadline miss。若 s+h 已占满目标周期，串行增加 L 就无法维持该平均目标。同步推理等待不会因为 L<E·Δt_sim 自动消失。即使目标平均频率不达标，仍可低频闭环；“不满足约束”等于“不能闭环”的表述过强。

### 5.3 异步是待实现架构，不是当前事实

```text
目标墙钟节拍 →  | slot 0 | slot 1 | … | slot E−1 | 下一窗口
执行端          [a_0]     [a_1]          [a_E−1]  [替换/对齐后的 b]
推理端          取带时间戳的新观测 → [推理 b + 搬运] → 完整候选块进入另一缓冲区
切换点          校验 deadline、观测时间戳、动作有效槽；原子替换；超时则触发明确策略
```

[D] 若在当前执行窗口开始时提交下一次预测，且剩余可覆盖动作数为 R、实际每槽目标周期为 Δt_target，则局部“不饿死队列”条件是 `L_effective + J + M ≤ R·Δt_target`。L_effective 应在并发负载下包含预处理、通信、设备搬运及完整块可用时间；J 为抖动预算，M 为切换安全余量。R≤当前剩余的有效动作数；固定前缀执行 E 的窗口通常 R≤E，不能无条件换成 P。启动时仍需等待第一块，或明确定义启动策略。

[D] 在每 E 个节拍提交一次请求、并行执行与推理、无额外瓶颈等假设下，理想稳态块周期才可写成 `max(E·Δt_target,L_effective)`。这不是当前顺序循环的实测公式。还要满足执行端 H+step 不超节拍、后台推理长期处理能力匹配请求率、队列线程安全、无资源竞争导致漏期。

[D/H] 吞吐条件也不足以保证动作时序正确。若新观测产生后经过 d 个控制槽才开始执行对应块，应显式处理这些已过时的预测槽：例如跳过 d 槽或用延迟条件策略/对齐算法。简单跳过的方案至少需要 d<P；若还要连续执行 E 个有效槽，需要 d+E≤P。该条件只适用于这种对齐策略，不是所有异步方案的通用定理。直接接上旧观测的 a0 会引入观测陈旧问题；L≤P·Δt_target 不能证明闭环稳定、接触安全或成功率不降。独立双缓冲加线程也不等于实现 RTC。

## 6. 最小修复建议（未修改官方文件）

[D] 若实验定义是“每次预测完整 P 步、从当前观测对应的起始动作执行前 E 步”，最小改法是在入队时明确前缀截断，并加参数/输出断言：

```python
# 独立实验包装层的建议；没有应用到官方仓库
E = cfg.num_open_loop_steps
assert 1 <= E <= NUM_ACTIONS_CHUNK
# actions = get_action(...) 返回完整块后
assert len(actions) == NUM_ACTIONS_CHUNK
assert all(a.shape == (7,) for a in actions)  # 这里只针对 LIBERO
assert len(action_queue) == 0
# 原为 action_queue.extend(actions)
action_queue.extend(actions[:E])
```

保留 deque(maxlen=E) 即可；也可用无界 deque，但必须同时截断，否则无界队列会执行全部 P 步，E 不再生效。不要用 appendleft/reverse 修复，此举会改动作顺序。不要为扫描 E 改 NUM_ACTIONS_CHUNK；那会改预测接口。前缀修复解决索引语义，不解决同步阻塞；异步改造应作为独立实验因素。

[D] 推理延迟不要用任意丢弃后缀或前缀自动“补偿”，必须绑定观测/执行时间戳。建议每条动作记录 prediction_id、source_observation_step、predicted_slot、execution_step、墙钟时间戳，另记触发推理时 remaining_queue。

## 7. 最小独立验证：真实执行记录

[V] 使用已有 bundled Python 3.12.14，Windows 11；仅标准库 ast、collections、types、json 等，无安装、无 GPU、无模型/仿真依赖。确定性符号索引，无随机采样，seed 记为不适用。

验证脚本 `work/V14/validate_logic.py` 包含两层：

- 直接运行 collections.deque，核对 E=1/2/4/8 的 extend 与逐个 popleft，共 4 个索引用例。
- AST 只提取快照中的真实 run_episode 函数，使用假的 model/env/action 和观测处理函数，每配置走两块；分别测试原逻辑与**仅在内存中**替换为 actions[:E] 的逻辑，共 8 个用例。逐项 assert 执行动作索引、推理调用数和事件顺序，拒绝函数内部捕获的异常被当作成功。

结果：**全部通过**。原逻辑两块均执行后缀，内存修复两块均执行前缀；E=8 两者相同。事件始终是 infer_start→infer_end→step…step→infer_start，未出现交错推理与执行。此验证证明所提取控制流和 deque 语义，不证明真实模型耗时、CUDA 调度或物理行为。

复现命令（PowerShell，在 D:\Desktop\dickseep 执行）：

```powershell
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' work/V14/collect_source.py
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' work/V14/validate_logic.py
```

collect_source.py 的首次 main 解析仅用于发现 SHA；**本次证据以 source_manifest.json 中的 SHA 和逐文件 hash 为准**。未来复核应直接使用 manifest 的固定 raw URL，避免重新解析 main 得到新版。`logic_validation.json` 保存参数、环境、各用例全部事件；`logic_validation.log` 保存最终 stdout。

执行中曾有两个已处理失败：PowerShell Invoke-RestMethod 的网络认证/TLS 请求失败，改用已有 Python urllib 成功；AST 初稿漏放 GenerateConfig 类型注解 stub，发生 NameError，补齐测试命名空间后重跑通过。这些都不是模型/环境实验失败；最终日志记录修正后的验证，没有隐去真实模型运行未开展的状态。

## 8. 对 T1′ 方案的影响：只提出审阅意见，不改原报告

本节引用本地既有方案的表述仅用于指出需要 DSH 重新审查的位置，未修改那些文件。

| 既有位置/主张 | V14 结论 | 给 DSH 的修订建议 |
|---|---|---|
| experiment_plan.md §0.5 B4：改 E 后队列长度变化即生效 | [F/V] 长度正确但内容是后缀，闸门不足 | 增加“首次执行槽=0、连续执行槽=0…E−1、输出完整 P”的索引验收 |
| experiment_plan.md §1：直接扫描 E=1…8 | [D] 同时扫描了起始槽偏移 P−E | 区分 original_suffix 与 corrected_prefix 两条实现；E=P 为共同基准；先做逻辑闸门再评估是否需要成功率实验 |
| E=1 是精度上界 | [F/D] 原实现执行 a7；即使修复执行 a0，也没有精度上界定理 | 改成“每步重规划对照”，精度高低留给实验 |
| experiment_plan.md §0.2 达成步频 E/max(EΔt,L) | [F/D] 假定了不存在的重叠 | 当前同步入口用 E/[L+Σstep_wall+H]；异步扩展另列，禁止混报 |
| candidate_topics.md T1′：L>PΔt 则根本不能闭环；neighbor_audit §7：L<PΔt 就能不间断闭环 | [D] 混淆目标周期预算、异步吞吐与闭环成立 | 换成“给定执行架构和目标节拍的 deadline/平均速率检验”；闭环成功和稳定性另验 |
| 用 sleep 加 50/100ms 检验实时成功率损失 | [D/H] step 驱动仿真可能在等待时暂停世界；sleep 只改变墙钟 | 同步 baseline 用来度量墙钟成本；要模拟真实延迟，需让世界按时钟推进并明确 hold-last-action/空队列策略，不能靠 sleep 自动等价 |
| U=SR·(1−idle) 等价于单位墙钟成功任务数 | [D] 不成立，任务动作数与长度会随 E 改变 | 若使用该效用，明确它是自定义加权分；真实任务吞吐另报 成功episode数/总墙钟时间，并声明超时与reset成本范围 |
| 只改 E 不改前向，显存完全与 E 无关 | [D/H] 预测形状固定，但不同策略改变观测分布/历史；CPU队列长度也变化 | 仅说“无直接的预测张量形状变化”；是否峰值显存近似不变需测量，不做绝对保证 |

[D] 建议的最小重启顺序：先保留 E=P=8 的官方同步行为作 baseline；明确前缀实验包装层的索引合同；独立度量 L/step_wall/H 与边界间隔；最后才由 DSH 决定是否新增 paced asynchronous 或真实时间推进的延迟条件。这是修复测量有效性，未自行变更 T1′ 的科研目标，也没有开始大规模 E 扫描。

## 9. 未验证事项与交付状态

| 状态 | 项目 |
|---|---|
| 已执行 | 固定 commit，读取三个指定文件及必要调用链；保存少量源码快照及 hash |
| 已执行 | 4 个 deque 用例、8 个真实函数 AST + stub 用例，通过 |
| 已执行 | 只生成本报告与 work/V14 审计辅助文件；不改原始报告/官方代码 |
| 尚未执行 | 指定 checkpoint revision/head/config 的真实加载及 `(8,7)` 数值、dtype、NaN 检查 |
| 尚未执行 | RTX 5070 12GB 的准入、峰值显存、L p50/p95/p99、step_wall/H 分解与deadline测量 |
| 尚未执行 | 固定版本 LIBERO/robosuite 的 Δt_sim、control_freq、后台推进和 seed 行为核验 |
| 尚未执行 | 后缀/前缀选择对成功率、jerk、接触行为的影响；不得把代码缺陷推断成已测到性能下降 |
| 尚未执行 | 异步执行器、缓冲切换、观测陈旧补偿、超时策略与推理/执行资源竞争 |
| 尚未执行 | OpenFleX 目标控制周期、底层 hold 行为与通信延迟；平台不是本次审计前提 |

本报告的强结论是**动作索引与当前控制流语义**。关于实时频率的式子是逐项声明假设的推导；关于模型性能、设备可行性、闭环任务表现的结论仍需后续授权的真实实验。
