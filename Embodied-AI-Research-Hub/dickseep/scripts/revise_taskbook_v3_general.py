# -*- coding: utf-8 -*-
"""V3(泛化版): rewrite task-book body at framework level.

Base : D:\\Desktop\\09_本科毕业设计任务书_已填写_导师讨论稿.docx (untouched)
Out  : <OUTDIR>/本科毕业设计任务书_V3_高效推理部署_导师讨论稿.docx  (generalised body)
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


def _copy_font(dr, sr):
    if sr is None:
        return
    if sr.font.name:
        dr.font.name = sr.font.name
    if sr.font.size:
        dr.font.size = sr.font.size
    if sr.font.bold is not None:
        dr.font.bold = sr.font.bold
    srpr = sr._element.rPr
    if srpr is not None:
        sf = srpr.find(qn('w:rFonts'))
        if sf is not None:
            rpr = dr._element.get_or_add_rPr()
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
    "任务1：调研视觉语言动作模型及其高效推理技术的发展现状，分析代表性开源模型的结构特点、动作生成方式、"
    "推理计算特点与部署需求；结合多自由度机械臂操作任务，比较不同类型模型在计算资源、推理效率与实验条件"
    "方面的适用性，形成技术需求、约束分析与方案比选报告。",
    ind_line({"2.3", "10.3", "12.2"}),
    "任务2：针对多自由度机械臂操作任务，设计包含环境感知、模型推理、动作转换与控制执行的机器人实验系统，"
    "明确输入输出接口、动作空间、执行时序、软硬件部署方式与异常处理机制，完成系统总体方案、接口规范与"
    "测试验证方案设计。",
    ind_line({"3.3", "6.2", "7.2", "11.2"}),
    "任务3：选择一种具有公开代码与模型权重的视觉语言动作模型，搭建推理与仿真实验环境，完成模型加载、"
    "观测处理、动作生成与任务执行的基本流程复现，建立该模型在推理效率与任务表现方面的原始基线，并分析"
    "主要计算与部署瓶颈。",
    ind_line({"2.3", "8.2", "11.2", "12.2"}),
    "任务4：在原始基线基础上复现一类成熟的推理优化方法，保持模型、任务与实验条件一致，比较优化前后的推理"
    "效率、资源占用、动作输出差异与任务表现，分析优化收益、额外开销与适用条件，完成工程验证与部署方案评价。",
    ind_line({"3.3", "6.2", "7.2", "8.2", "10.3", "11.2", "12.2"}),
]

CONTENT = [
    "主要内容：（与上述四项任务一一对应；下列“支撑毕业要求指标点”按机械工程（机器人试验班）培养方案撰写）",
    "内容一（对应任务1）：视觉语言动作模型结构及其高效推理技术研究，机械臂操作任务需求分析与技术方案比较。",
    "支撑毕业要求指标点：",
    "2.3：在统一任务约束下比较不同类型开源模型的结构、动作生成方式与推理特点，给出可替代方案与选型理由，"
    "形成“多方案比较—影响因素分析—有效结论”的完整论证链；",
    "10.3：主要参考文献以外文会议与期刊论文为主，完成英文文献的检索、阅读与要点提炼，并在论文中提供"
    "英文摘要与英文图表题；",
    "12.2：跟踪该方向近年快速演进，形成对新兴技术的主动追踪与批判性判断。",
    "内容二（对应任务2）：机器人感知—推理—动作执行系统设计、接口约束、软硬件部署与工程验证方案。",
    "支撑毕业要求指标点：",
    "3.3：针对机械臂操作任务的实时性需求，完成机器人控制系统与软硬件协同执行系统的方案设计，交付系统"
    "架构、执行时序、传感—计算—执行接口定义、动作约束以及异常与失效处理方案，并给出方案比选与验证依据；"
    "设计对象是可用于实验验证的机器人操作实验系统，而非仅停留于方案框图；",
    "6.2：在设计阶段即纳入关节限位、速度与力矩保护、通信中断与执行中止等约束，评估其对操作安全性与"
    "作业连续性的影响；",
    "7.2：比较不同部署方式在算力占用与能耗上的差异，给出面向可持续性的部署建议；",
    "11.2：结合资源条件完成部署方案的技术经济比较，形成成本—性能工程决策依据。",
    "内容三（对应任务3）：代表性开源视觉语言动作模型的推理复现、任务运行、性能测量与瓶颈分析。",
    "支撑毕业要求指标点：",
    "2.3：将整体推理过程按功能环节拆分比较，识别主要耗时环节并给出改进方向的比较与取舍；",
    "8.2：固定模型版本、实验脚本与评测条件，保存原始记录，遵守开源许可要求，保证结果可追溯、可复现；",
    "11.2：记录不同配置下的资源占用与推理开销，为实验条件与部署方式的工程选择提供依据；",
    "12.2：面对框架演进与接口差异，通过官方文档、源码阅读与实验复现完成自学与适配，形成可复现的配置"
    "与版本记录。",
    "内容四（对应任务4）：已有高效推理方法复现、同条件性能比较、动作质量分析、任务表现评价与工程适用性研究。",
    "支撑毕业要求指标点：",
    "3.3：以优化前后的实测数据验证系统设计与部署方案的有效性，形成方案—实现—验证的闭环；",
    "6.2：评估优化可能引入的风险（动作质量下降、执行超时、异常输出），并给出对应的失效处理与安全约束；",
    "7.2：比较优化前后在资源占用与能耗方面的变化，说明其可持续性含义；",
    "8.2：保持优化手段以外的条件一致，客观报告结果，包括未取得改善的情形；",
    "10.3：以论文正文、答辩汇报与英文摘要完成技术表达与交流；",
    "11.2：给出优化方法的收益—代价评价及是否具备部署价值的结论；",
    "12.2：对方法的适用边界与失败模式进行反思与总结。",
]

REFS = [
    "主要参考文献：（不少于10篇，含近五年文献；以下条目均经 arXiv 元数据核对，预印本与正式发表状态如实标注）",
    "[1] Zhao T Z, Kumar V, Levine S, et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost "
    "Hardware[C]. Robotics: Science and Systems (RSS), 2023. arXiv:2304.13705.",
    "[2] Chi C, Xu Z, Feng S, et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion[J]. "
    "arXiv preprint arXiv:2303.04137, 2023.",
    "[3] Kim M J, Pertsch K, Karamcheti S, et al. OpenVLA: An Open-Source Vision-Language-Action Model[J]. "
    "arXiv preprint arXiv:2406.09246, 2024.",
    "[4] Kim M J, Finn C, Liang P. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success[C]. "
    "Robotics: Science and Systems (RSS), 2025. arXiv:2502.19645.",
    "[5] Shukor M, Aubakirova D, Capuano F, et al. SmolVLA: A Vision-Language-Action Model for Affordable and "
    "Efficient Robotics[J]. arXiv preprint arXiv:2506.01844, 2025.",
    "[6] Black K, Brown N, Driess D, et al. π0: A Vision-Language-Action Flow Model for General Robot "
    "Control[C]. Robotics: Science and Systems (RSS), 2025. arXiv:2410.24164.",
    "[7] Liu B, Zhu Y, Gao C, et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning[J]. "
    "arXiv preprint arXiv:2306.03310, 2023.",
    "[8] Lin J, Tang J, Tang H, et al. AWQ: Activation-aware Weight Quantization for LLM Compression and "
    "Acceleration[C]. MLSys, 2024. arXiv:2306.00978.",
    "[9] Xiao G, Lin J, Seznec M, et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for "
    "Large Language Models[C]. ICML, 2023. arXiv:2211.10438.",
    "[10] Li M, Lin Y, Zhang Z, et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion "
    "Models[C]. ICLR, 2025. arXiv:2411.05007.",
    "[11] Xu J, Jin Q, Chen M, et al. VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for "
    "Vision-Language-Action Models[J]. arXiv preprint arXiv:2609.25376, 2026.",
    "[12] Zhang J, Hsieh Y, Wan Z, et al. QuantVLA: Scale-Calibrated Post-Training Quantization for "
    "Vision-Language-Action Models[C]. CVPR, 2026. arXiv:2602.20309.",
    "[13] Xu B, Tan H, Jiang W, et al. VLA-HWBench: Where the Time Goes When VLA Models Run on Edge "
    "Hardware[J]. arXiv preprint arXiv:2610.34717, 2026.",
    "[14] Bukhari S U, Almgren M, Ali-Eldin A. LoopQ: Fair, Closed-Loop Evaluation of Quantized "
    "Vision-Language-Action Models[J]. arXiv preprint arXiv:2611.05093, 2026.",
    "注：文献 [8][9][10] 的方法面向语言模型与图像生成模型，迁移至视觉语言动作模型需经适用性验证；"
    "本课题不预设其必然有效。",
]

SCHEDULE = [
    "进度安排：（参照学院 2027 届毕设工作安排，起点为秋季学期第15周，每2周为一个节点）",
    "第一阶段（2026.12.7-2026.12.18）完成核心文献阅读与课题需求分析，确定研究路线与实验方案。",
    "第二阶段（2026.12.21-2026.12.31）完成开题报告，形成系统总体方案与实验环境设计。",
    "第三阶段（2027.1.4-2027.1.15）搭建模型与仿真实验环境，完成模型加载与初步推理验证。",
    "第四阶段（2027.3.1-2027.3.12）完成原始模型的推理基线与任务流程复现。",
    "第五阶段（2027.3.15-2027.3.26）确定优化方法并完成首轮复现实验，接受中期检查。",
    "第六阶段（2027.3.29-2027.4.9）开展优化前后的推理性能测量与对照实验。",
    "第七阶段（2027.4.12-2027.4.23）完成任务表现评测、动作质量与失败情形分析。",
    "第八阶段（2027.4.26-2027.5.7）完成工程验证与资源评价，修改论文初稿。",
    "第九阶段（2027.5.10-2027.5.21）完成论文查重与内审，准备答辩材料（按学院通知执行）。",
    "第十阶段（2027.5.24-2027.6.4）完成论文定稿、答辩及答辩后修改与材料整理（按学院安排执行）。",
    "第十一阶段（2027.6.7-2027.6.18）完成后续修改、工作总结与资料归档。",
]

NOTE4 = ("说明四：本文件为 V3 修订稿，在第一版任务书基础上修改，保留原有表格栏目、四项任务结构、"
         "毕业要求指标点对应关系与 11 个进度阶段日期。修订后的研究主线为：代表性开源视觉语言动作模型的"
         "推理复现 → 机器人操作实验系统设计 → 原始推理性能评测 → 一类已有高效推理方法的复现与验证。")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    shutil.copyfile(BASE, OUT)
    doc = Document(OUT)
    rows = doc.tables[0].rows
    write_cell(rows[3].cells[1], ["（中文）" + TITLE_CN])
    write_cell(rows[4].cells[1], ["（英文）" + TITLE_EN])
    write_cell(rows[9].cells[1], TASKS)
    write_cell(rows[10].cells[1], CONTENT)
    write_cell(rows[11].cells[1], REFS)
    write_cell(rows[12].cells[1], SCHEDULE)
    doc.add_paragraph(NOTE4)
    doc.save(OUT)
    print("OK", OUT)


if __name__ == '__main__':
    main()
