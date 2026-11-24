import pathlib,json,re,hashlib
R=pathlib.Path(__file__).parent
entries={}
for name in ['metadata_parsed.json','metadata_extra.json','metadata_related.json']:
 for x in json.loads((R/name).read_text(encoding='utf-8-sig'))['entries']: entries[x['arxiv_id']]=x
ids=['2212.06817','2307.15818','2406.09246','2502.19645','2506.01844','2410.24164','2504.16054','2303.04137','2304.13705','2506.07339','2607.24008','2611.01390','2611.05155','2607.14695','2611.01011','2611.03103','2610.34717','2502.19417','2503.20020','2510.03342','2506.09985','2601.21998','2602.15922','2606.00537','2605.25537','2611.02101','2204.01691','2207.05608','2209.07753','2512.10675','2610.20442',None,None,'2512.01031','2605.08168','2606.31132','2609.00065']
deep={4,5,10,11,12,13,14,15,16,17,18,20,21,22,23}
inst={1:'Robotics at Google、Everyday Robots、Google Research Brain',2:'Google DeepMind',3:'Stanford、UC Berkeley、Toyota Research Institute、Google DeepMind、Physical Intelligence、MIT 等论文署名机构',4:'Stanford University',5:'Hugging Face、Sorbonne University、valeo.ai、ENS Paris-Saclay',6:'Physical Intelligence',7:'Physical Intelligence',8:'Columbia University、Toyota Research Institute、MIT、Stanford（所取扩展版本）',9:'Stanford、UC Berkeley、Meta',10:'Physical Intelligence、UC Berkeley',11:'四川大学、电子科技大学、University of Alberta',12:'University of Nottingham Ningbo China',13:'INSAIT、Sofia University',14:'北京邮电大学计算机学院',15:'Leapmotor，Embodied Algorithm Group',16:'NUS、NTU、Stanford、NVIDIA',17:'NUS、0G Labs',18:'Physical Intelligence；完整个人跨机构映射未逐一核实',19:'Google DeepMind',20:'Google DeepMind',21:'FAIR at Meta、Mila／Polytechnique Montréal',22:'官方 Robbyant 项目；PDF 首页未列完整机构映射，不据姓名补齐',23:'NVIDIA',24:'北京大学、JD Explore Academy',25:'本次未可靠定位机构，留待作者页核实',26:'Texas Tech University',27:'Robotics at Google、Everyday Robots',28:'Robotics at Google',29:'Robotics at Google',30:'Google DeepMind',31:'CUHK MMLab、复旦、NovaXBot、清华、浙江大学、中山大学',34:'MIT、NVIDIA、清华、UC Berkeley、UCSD、Caltech',35:'École polytechnique；ETH Zürich Systems Group',36:'Carnegie Mellon University',37:'K-Dense Inc.'}
codes={1:'https://robotics-transformer1.github.io/',2:'https://robotics-transformer2.github.io/',3:'https://github.com/openvla/openvla',4:'https://openvla-oft.github.io/',5:'https://github.com/huggingface/lerobot',6:'https://github.com/Physical-Intelligence/openpi',7:'https://github.com/Physical-Intelligence/openpi',8:'https://diffusion-policy.cs.columbia.edu/',9:'https://tonyzhaozh.github.io/aloha/',10:'https://www.pi.website/research/real_time_chunking',11:'https://jianghaiscu.github.io/FutureRTC_proj/',13:'https://arp.insait.ai',14:'https://github.com/9yc/Reflex',16:'https://github.com/codestrongestx/VLALagBench',18:'https://www.pi.website/research/hirobot',21:'https://github.com/facebookresearch/vjepa2',22:'https://github.com/robbyant/lingbot-va',23:'https://dreamzero0.github.io/',27:'https://say-can.github.io/',28:'https://innermonologue.github.io/',29:'https://code-as-policies.github.io/',30:'https://veo-robotics.github.io/',34:'https://github.com/mit-han-lab/vlash',35:'https://github.com/TheAyos/async-vla-inference'}
notes={1:'扩展多任务策略与真实数据规模；不是实时调度论文。本文不提取其跨条件性能百分比。',2:'VLM 与机器人动作共同微调；语义迁移路线。大模型动作解码不构成电机实时保障。',3:'7B、融合视觉特征、动作 token 与参数高效微调；是 OFT 的基础模型，不能用 OFT 配方结果替代原方法。',6:'VLM 与 flow action expert；需要训练。RSS 2025 状态见 arXiv comment。作为模型路线而非本文本机基准。',7:'异构数据共训练、语义与动作层次；机器人能力与系统时序均非只由网络规模决定。',8:'条件动作扩散与滚动时域；所取 PDF 为原 RSS 2023 工作的扩展稿。未把扩展稿所有数字归于原会议版本。',9:'ACT：CVAE／Transformer 动作序列与 temporal ensemble，低成本 ALOHA；需模仿学习训练。',19:'初代 Gemini Robotics 与 ER 技术报告；提供真实多平台实例。未将其参数、硬件或完整权重开放状态推断为已公开。',24:'PACE，低速谷选择执行前缀；正文 §3–5 与 Table 1／4／5 核验。作者 comment 声称 NeurIPS 2026 接收，正式会议记录未独立核实。',25:'Soft RTC，需训练的 action-prior denoising；Kinetix 与初步真机。本文只使用原文限定指标，不推断安全等级。',26:'延迟×视角变化的模拟交互实验；一策略一任务套，未真实硬件复现。',27:'语言概率结合技能可供性，说明语言计划需要可执行性约束；完整训练硬件未核实。',28:'把执行／环境反馈文本回送 LLM，仿真与真实机器人；底层技能不等于由 LLM 实时控制电机。',29:'代码生成组织感知与运动 API；生成代码不自动拥有实时或安全保证。',30:'多视角动作条件 Veo 评测系统；用途以离线策略评测为主，不将其误用为在线低延迟证据。',31:'背景综述，非新系统实验；用其术语框架，不用综述引用替代原始方法证据。',34:'VLASH，未来本体状态条件与异步执行；本报告只核读部分正文，未复核摘要全部速度倍数。',35:'同模型／代码条件下比较 IT-RTC、TT-RTC、VLASH、A2C2；本文未完成其全部实验表格复核，不将排名泛化。',36:'顺序与并行 test-time compute 的 meta-MDP／强化学习元策略；底层策略冻结但元策略要训练。摘要定量结果未进一步用于本报告推荐。',37:'软件方法引用；不是机器人研究样本。Skill 文档实际版本见 §2.2。'}
venue={4:'RSS 2025（API comment）',10:'NeurIPS 2025（会议论文集与 API）',18:'ICML 2025（API comment）',6:'RSS 2025（API comment）',14:'预印本；作者 comment 声称 ICML2026 接收，未独立确认',24:'预印本；作者 comment 声称 NeurIPS2026 接收，未独立确认'}
out=[]; ledger=[]
for n,aid in enumerate(ids,1):
 if aid is None: continue
 e=entries[aid]; level='F' if n in deep else 'P'
 authors=e['authors']; authors=[a.get('name',str(a)) if isinstance(a,dict) else a for a in authors]
 auth='；'.join(authors if len(authors)<=16 else authors[:6])+('；等（完整作者序列见 sources 元数据）' if len(authors)>16 else '')
 first=e['published'][:10]; upd=e['updated'][:10]
 title=e['title'].replace('$','')
 out.append(f'**[{n}] {title}.** {auth}。机构：{inst.get(n,"未核实")}。首次 {first}；最近更新 {upd}；读取 {e["arxiv_id_versioned"]}；状态：{venue.get(n,"arXiv 预印本／技术报告，会议状态本次未独立核实")}；深度 **{level}**。')
 out.append(f'论文：[arXiv 原始记录](https://arxiv.org/abs/{aid})；[PDF](https://arxiv.org/pdf/{e["arxiv_id_versioned"]})。'+(f'代码／项目入口：[作者入口]({codes[n]})；具体开放状态按第 6–7 节限定。' if n in codes else '代码：本次未核实可运行 release，不构造仓库地址。'))
 out.append(notes.get(n,'方法、模型／平台、实验协议、基线、定量结果、训练依赖与局限见第 6 节对应小节；未确认的硬件和 release 保持未核实。'))
 ledger.append(dict(number=n,**e,evidence_depth=level,institution=inst.get(n),pdf_sha256=hashlib.sha256((R/(aid+'.pdf')).read_bytes()).hexdigest() if (R/(aid+'.pdf')).exists() else None))
special='''**[32] Serving DNNs like Clockwork: Performance Predictability from the Bottom Up.** Arpan Gujarati；Reza Karimi；Safya Alzayat；Wei Hao；Antoine Kaufmann；Ymir Vigfusson；Jonathan Mace。机构：Max Planck Institute for Software Systems、Emory University。OSDI 2020，2020-11；首次预印本／最近更新本次未核实；深度 **A**（官方论文页与摘要）。[USENIX 原文入口](https://www.usenix.org/conference/osdi20/presentation/gujarati)。可预测模型执行、资源隔离与请求接纳；不是机器人动作安全验证，不将服务 SLO 外推为机器人保证。

**[33] Paella: Low-latency Model Serving with Software-defined GPU Scheduling.** Kelvin K.W. Ng；Henri Maxime Demoulin；Vincent Liu。University of Pennsylvania、DBOS Inc.。SOSP 2023，2023-10-23—26；预印本最新修订时间未核实；深度 **A**（作者 PDF 身份／摘要与问题段落）。[作者 PDF](https://vincen.tl/files/ng23paella.pdf)；[DOI](https://doi.org/10.1145/3600006.3613163)；[公开代码](https://github.com/eniac/paella)。软件控制 GPU kernel 次序并减少路径开销；本报告仅借用机制，不引述其未完整核读的性能结果。'''
# Keep reference numbering ordered.
text='\n\n'.join(out)
pos=text.index('**[34]')
text=text[:pos]+special+'\n\n'+text[pos:]
text+='''

**[38] LeRobot 代码核验证据（不计入研究论文数量）。** Hugging Face，v0.4.4，commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`。[robot_client.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/robot_client.py)、[policy_server.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/policy_server.py)、[configs.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/configs.py)、[helpers.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/helpers.py)。实际读取源码，未运行机器人。`sources/lerobot_*` 为取回快照；`code_probe.py` 中固定 SHA 为真实下载版本，终端打印的 `sha=9e5...` 是另查 main 的变量，不能误当这些快照版本。

本报告由 AI 辅助检索、提取与综合；未冒充人工双评审、实际机器人实验或投稿批准。正文的本科方案是可审查研究设计，下一步应先验证策略基本能力与基线时序，再决定是否值得增加算法复杂度。
'''
p=R.parent/'具身智能系统架构与协同调度_深度研究报告.md'
s=p.read_text(encoding='utf-8'); s=s.split('<!-- REFERENCES_INSERT -->')[0]+'<!-- REFERENCES_INSERT -->\n\n'+text
p.write_text(s,encoding='utf-8')
(R/'included_evidence.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
print('research references',36,'focused full-text',len(deep),'Chinese chars',sum('\u4e00'<=c<='\u9fff' for c in s),'total chars',len(s))
