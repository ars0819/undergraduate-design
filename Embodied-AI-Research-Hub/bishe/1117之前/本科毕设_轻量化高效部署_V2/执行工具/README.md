# 执行工具说明

这不是已经跑通的GPU复现仓库。脚本已做Python语法与CLI检查，GPU、Hub下载和机器人任务路径尚未执行。

- `collect_evidence.py`：本次实际执行的公开源码获取；网络失败保留记录，不安装模型。
- `audit_local.py`：本次实际执行的本地原稿/培养方案/24篇共120份笔记读取与hash登记、配对顺序生成。不会修改原稿。
- `supplement.py`：补充网络和缓存解析；首次运行因Windows子进程编码错误在缓存解析处中断；后已用`python -X utf8`单独重跑三份Atom解析。保留失败说明，未重跑网络部分。
- `lock_hub.py`：待在Ubuntu运行；完整快照下载并hash检查。例：`python lock_hub.py lerobot/smolvla_libero ./checkpoints/smolvla_libero`。请同时冻结配置中VLM/tokenizer和数据repo，不用浮动main跑正式实验。
- `gpu_probe.py`：待在实际Ubuntu/A100环境运行，例：`python gpu_probe.py --out runs/probe/gpu.json`。仅测试驱动与小矩阵，不能代替VLA验收。
- `benchmark_smol.py`：待在已安装冻结LeRobot的Ubuntu运行。输入必须由官方processor在真实数据/环境观测上产生并保存为tensor字典，图像键、token与归一化须先核对。例：`python benchmark_smol.py --checkpoint ./checkpoints/smolvla_libero --batch ./prepared_batch.pt --out ./runs/eager01 --precision fp32`；优化只加`--compile`且用新out。分别独立进程运行3次。

benchmark只测GPU上已准备输入到新动作块，排除预处理/后处理/D2H/环境，并将这些排除项写入summary；完整端到端计时和模块profiler需在02所述官方eval入口加入日志后另跑。它每次显式新块预测，避免select_action缓存误计。单帧回放不是任务成功率；当前还没有生成prepared_batch.pt。

所有安装/评测命令见02。正式使用前核对每个CLI在固定代码中的`--help`；第一次运行如果报接口错误，保留trace并修正后记录diff，不隐瞒为“已验证脚本”。
