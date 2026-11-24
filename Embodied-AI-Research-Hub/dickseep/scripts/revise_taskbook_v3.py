# -*- coding: utf-8 -*-
"""V3: modify the FIRST-version task book in place (copy), following advisor feedback.

Base : D:\\Desktop\\09_本科毕业设计任务书_已填写_导师讨论稿.docx  (first version, untouched)
Out  : <OUTDIR>/本科毕业设计任务书_V3_高效推理部署_导师讨论稿.docx
"""
import os
import shutil

from docx import Document
from docx.oxml.ns import qn

BASE = r"D:\Desktop\09_本科毕业设计任务书_已填写_导师讨论稿.docx"
OUTDIR = r"D:\Desktop\dickseep\本科毕设任务书_第一版修订_V3"
OUT = os.path.join(OUTDIR, "本科毕业设计任务书_V3_高效推理部署_导师讨论稿.docx")

TITLE_CN = "面向多自由度机械臂操作的视觉语言动作模型高效推理与部署研究"
TITLE_EN = ("Efficient Inference and Deployment of Vision-Language-Action Models "
            "for Multi-Degree-of-Freedom Robotic Manipulation")

IND = ["2.3", "3.3", "6.2", "7.2", "8.2", "10.3", "11.2", "12.2"]


def ind_line(checked):
    return "支撑指标点：" + "；".join(("☑" if k in checked else "□") + k for k in IND)


def clear_cell(cell):
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p0 = cell.paragraphs[0]
    for r in p0.runs[1:]:
        r._element.getparent().remove(r._element)
    return p0


def _copy_font(dst_run, src_run):
    if src_run is None:
        return
    if src_run.font.name:
        dst_run.font.name = src_run.font.name
    if src_run.font.size:
        dst_run.font.size = src_run.font.size
    if src_run.font.bold is not None:
        dst_run.font.bold = src_run.font.bold
    src_rpr = src_run._element.rPr
    if src_rpr is not None:
        sf = src_rpr.find(qn('w:rFonts'))
        if sf is not None:
            rpr = dst_run._element.get_or_add_rPr()
            rf = rpr.find(qn('w:rFonts'))
            if rf is None:
                rf = rpr.makeelement(qn('w:rFonts'), {})
                rpr.append(rf)
            for a in ('w:eastAsia', 'w:ascii', 'w:hAnsi'):
                v = sf.get(qn(a))
                if v:
                    rf.set(qn(a), v)


def write_cell(cell, lines):
    p0 = clear_cell(cell)
    tpl = p0.runs[0] if p0.runs else None
    for i, line in enumerate(lines):
        p = p0 if i == 0 else cell.add_paragraph()
        if i == 0:
            if p.runs:
                p.runs[0].text = line
            else:
                _copy_font(p.add_run(line), tpl)
        else:
            p.paragraph_format.first_line_indent = p0.paragraph_format.first_line_indent
            p.paragraph_format.line_spacing = p0.paragraph_format.line_spacing
            _copy_font(p.add_run(line), tpl)


TASKS = [
    "主要任务及目标：（任务共4项；任务1为调研类，故任务数不少于4项。指标点按机械工程（机器人试验班）培养方案）",
    "任务1：调研视觉语言动作模型（VLA）及其高效推理技术的发展现状，重点分析代表性开源 VLA 的模型结构、"
    "动作生成机制、推理计算与部署需求；结合多自由度机械臂操作任务，比较 SmolVLA、OpenVLA-OFT 等模型在"
    "计算资源、推理延迟、显存占用与实验环境方面的适用条件，形成技术需求、约束分析与方案比选报告。",
    ind_line({"2.3", "10.3", "12.2"}),
    "任务2：针对多自由度机械臂操作任务，设计包含环境感知、模型推理、动作转换、控制执行与反馈记录的机器人"
    "实验系统，明确图像与状态输入、动作空间、控制接口、执行时序、软硬件部署方式及异常处理机制，完成系统"
    "总体方案、接口规范与性能测试方案设计。",
    ind_line({"3.3", "6.2", "7.2", "11.2"}),
    "任务3：选择一种具有公开代码与模型权重的 VLA 模型，搭建推理与机器人仿真环境，完成模型加载、观测预处理、"
    "动作生成与任务执行的基本流程复现，建立原始模型在目标计算平台上的推理延迟、显存占用、吞吐量与机器人任务"
    "表现基线，分析主要计算与部署瓶颈。",
    ind_line({"2.3", "8.2", "11.2", "12.2"}),
    "任务4：在原始推理基线基础上选择一种成熟的推理优化方法复现，保持模型权重、任务与主要实验条件一致，测量"
    "优化前后的推理延迟、显存占用、动作输出差异与机器人任务成功率，分析优化收益、额外开销、性能退化及适用"
    "条件，完成工程验证与部署方案评价。",
    ind_line({"3.3", "6.2", "7.2", "8.2", "10.3", "11.2", "12.2"}),
]

CONTENT = [
    "主要内容：（与上述四项任务一一对应；下列“支撑毕业要求指标点”按机械工程（机器人试验班）培养方案撰写）",
    "内容一（对应任务1）：VLA 模型结构与高效推理技术研究、机械臂操作任务需求分析及技术方案比较。",
    "支撑毕业要求指标点：",
    "2.3：在同一任务约束下比较 SmolVLA、OpenVLA-OFT 等模型的参数量、动作生成机制、推理延迟与显存占用，"
    "给出可替代方案与选型理由，形成“多方案比较—影响因素分析—有效结论”的完整论证链；",
    "10.3：主要参考文献以外文会议与期刊论文为主，完成英文文献检索、阅读与要点提炼，并在论文中提供英文"
    "摘要与英文图表题；",
    "12.2：跟踪 2025—2026 年 VLA 高效推理与量化的快速演进，形成对新兴技术的主动追踪与批判性判断。",
    "内容二（对应任务2）：机器人感知—推理—动作执行实验系统设计、接口约束、软硬件部署与工程验证方案。",
    "支撑毕业要求指标点：",
    "3.3：针对“机械臂操作实时性不足”这一工程需求，完成机器人控制系统与软硬件协同执行系统的方案设计，"
    "交付系统架构图、通信时序图、传感—计算—执行接口定义、动作时序与控制约束、超时与失效处理方案及方案"
    "比选理由；设计对象为可部署的机器人操作实验系统，而非仅绘制系统框图；",
    "6.2：在方案阶段纳入关节限位、速度与力矩保护、通信中断与队列耗尽后的保持或终止策略，并评估这些约束"
    "对操作安全性与作业连续性的影响；",
    "7.2：比较本地推理与远端推理在算力占用与能耗上的差异，给出面向可持续性的部署建议；",
    "11.2：在“本机 12 GB 显卡 / 实验室 A100 40 GB”两种部署条件下，结合显存、延迟与成本完成配置选择，"
    "形成成本—性能决策表。",
    "内容三（对应任务3）：开源 VLA 原始推理复现、仿真任务运行、GPU 性能测量及计算瓶颈分析。",
    "支撑毕业要求指标点：",
    "2.3：把端到端时间分解为观测预处理、视觉编码、语言主干、动作生成、后处理与下发等阶段，逐段比较不同"
    "配置的代价，识别关键环节并给出改进方向的比较与取舍；",
    "8.2：固定模型权重版本、随机种子与评测脚本，保存原始日志与视频，遵守开源许可证要求，保证结果可追溯、"
    "可复现；",
    "11.2：记录不同配置下的显存占用与推理开销，为“能否在 12 GB 单卡上完成实验”给出工程决策依据；",
    "12.2：面对框架版本迭代与接口不兼容问题，通过官方文档、源码阅读与实验复现完成自学与适配，形成可复现"
    "的版本与配置记录。",
    "内容四（对应任务4）：已有高效推理方法复现、同条件性能比较、动作误差分析、机器人任务评价及工程适用性研究。",
    "支撑毕业要求指标点：",
    "3.3：以优化前后的实测数据验证前述系统设计与部署方案的可行性与有效性，完成方案—实现—验证的闭环；",
    "6.2：评估优化引入的风险（动作质量下降、超时、异常输出），并给出对应的失效处理与安全约束；",
    "7.2：比较优化前后在算力占用与能耗上的变化，说明其可持续性含义；",
    "8.2：保持优化前后除优化手段外的模型、数据、任务与硬件一致，如实报告负结果；",
    "10.3：以论文正文、答辩汇报与英文摘要完成技术表达与交流；",
    "11.2：给出优化方法的收益—代价评价及是否值得在实际部署中采用的结论；",
    "12.2：对复现过程中方法的适用边界与失败模式进行反思与总结。",
]

REFS = [
    "主要参考文献：（不少于10篇，含近五年文献；以下条目均经 arXiv 元数据核对，预印本与正式发表状态如实标注）",
    "[1] Zhao T Z, Kumar V, Levine S, et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware[C]. "
    "Robotics: Science and Systems (RSS), 2023. arXiv:2304.13705.",
    "[2] Chi C, Xu Z, Feng S, et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion[J]. "
    "arXiv preprint arXiv:2303.04137, 2023.",
    "[3] Kim M J, Pertsch K, Karamcheti S, et al. OpenVLA: An Open-Source Vision-Language-Action Model[J]. "
    "arXiv preprint arXiv:2406.09246, 2024.",
    "[4] Kim M J, Finn C, Liang P. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success[C]. "
    "Robotics: Science and Systems (RSS), 2025. arXiv:2502.19645.",
    "[5] Shukor M, Aubakirova D, Capuano F, et al. SmolVLA: A Vision-Language-Action Model for Affordable and "
    "Efficient Robotics[J]. arXiv preprint arXiv:2506.01844, 2025.",
    "[6] Black K, Brown N, Driess D, et al. π0: A Vision-Language-Action Flow Model for General Robot Control[C]. "
    "Robotics: Science and Systems (RSS), 2025. arXiv:2410.24164.",
    "[7] Liu B, Zhu Y, Gao C, et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning[J]. "
    "arXiv preprint arXiv:2306.03310, 2023.",
    "[8] Lin J, Tang J, Tang H, et al. AWQ: Activation-aware Weight Quantization for LLM Compression and "
    "Acceleration[C]. MLSys, 2024（Best Paper Award）. arXiv:2306.00978.",
    "[9] Xiao G, Lin J, Seznec M, et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large "
    "Language Models[C]. ICML, 2023. arXiv:2211.10438.",
    "[10] Li M, Lin Y, Zhang Z, et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion "
    "Models[C]. ICLR, 2025（Spotlight）. arXiv:2411.05007.",
    "[11] Xu J, Jin Q, Chen M, et al. VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for "
    "Vision-Language-Action Models[J]. arXiv preprint arXiv:2609.25376, 2026.",
    "[12] Zhang J, Hsieh Y, Wan Z, et al. QuantVLA: Scale-Calibrated Post-Training Quantization for "
    "Vision-Language-Action Models[C]. CVPR, 2026. arXiv:2602.20309.",
    "[13] Xu B, Tan H, Jiang W, et al. VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware[J]. "
    "arXiv preprint arXiv:2610.34717, 2026.",
    "[14] Bukhari S U, Almgren M, Ali-Eldin A. LoopQ: Fair, Closed-Loop Evaluation of Quantized "
    "Vision-Language-Action Models[J]. arXiv preprint arXiv:2611.05093, 2026.",
    "注：文献 [8][9][10] 的方法面向语言模型与图像扩散模型，迁移到 VLA 需经兼容性与效果验证；本课题不预设"
    "其对本课题所选模型必然有效。",
]

SCHEDULE = [
    "进度安排：（参照学院 2027 届毕设工作安排，起点为秋季学期第15周，每2周为一个节点）",
    "第一阶段（2026.12.7-2026.12.18）完成核心文献阅读、课题需求分析、模型选择及技术路线确定。",
    "第二阶段（2026.12.21-2026.12.31）完成开题报告、感知—推理—执行系统方案与实验环境设计。",
    "第三阶段（2027.1.4-2027.1.15）搭建模型与仿真环境，完成权重、数据接口与原始推理的初步验收。",
    "第四阶段（2027.3.1-2027.3.12）完成原始模型性能基线与机器人仿真任务复现。",
    "第五阶段（2027.3.15-2027.3.26）确定优化方法，完成首轮复现实验，并接受中期检查。",
    "第六阶段（2027.3.29-2027.4.9）开展优化前后推理性能测量与对照实验。",
    "第七阶段（2027.4.12-2027.4.23）完成机器人任务评测、动作误差与失败分析。",
    "第八阶段（2027.4.26-2027.5.7）完成工程验证、能耗与成本分析，修改论文初稿。",
    "第九阶段（2027.5.10-2027.5.21）完成论文查重与内审，准备答辩材料（按学院实际通知执行）。",
    "第十阶段（2027.5.24-2027.6.4）完成论文定稿、答辩及答辩后修改与材料整理（按学院正式安排执行）。",
    "第十一阶段（2027.6.7-2027.6.18）完成后续修改、总结与资料归档。",
]

NOTE4 = ("说明四：本文件为 V3 修订稿，直接在第一版任务书基础上修改，未重建模板、未改动学生信息与指导教师"
         "信息、保留原四项任务结构与 11 个进度阶段日期。修订依据导师意见：不预设新 VLA 架构、不将 ACT 训练／"
         "SmolVLA 全参数微调／真机装配列为必做，主线调整为“开源 VLA 推理复现 → 系统设计 → 原始性能评测 → "
         "一种已有高效推理方法复现 → 对照实验与工程验证”。本科核心平台为机器人仿真环境，真机仅作可选附加验证。")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    shutil.copyfile(BASE, OUT)
    doc = Document(OUT)
    t = doc.tables[0]
    rows = t.rows

    write_cell(rows[3].cells[1], ["（中文）" + TITLE_CN])
    write_cell(rows[4].cells[1], ["（英文）" + TITLE_EN])
    write_cell(rows[9].cells[1], TASKS)
    write_cell(rows[10].cells[1], CONTENT)
    write_cell(rows[11].cells[1], REFS)
    write_cell(rows[12].cells[1], SCHEDULE)

    doc.add_paragraph(NOTE4)
    doc.save(OUT)
    print("OK", OUT)


if __name__ == "__main__":
    main()
