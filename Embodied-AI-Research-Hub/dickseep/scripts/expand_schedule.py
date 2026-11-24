# -*- coding: utf-8 -*-
"""Expand the 11-stage schedule with domain-specific activities (no specific method names).
Header line is preserved verbatim from the current file.
"""
import os

from docx import Document
from docx.oxml.ns import qn

OUTDIR = r"D:\Desktop\dickseep\本科毕设任务书_第一版修订_V3"
SRC = os.path.join(OUTDIR, "本科毕业设计任务书_V3_高效推理部署_导师讨论稿_修订.docx")
DST = os.path.join(OUTDIR, "本科毕业设计任务书_V3_高效推理部署_导师讨论稿.docx")

STAGES = [
    ("第一阶段（2026.12.7-2026.12.18）",
     "完成开题报告撰写，于12月18日前上传教务系统。围绕视觉语言动作模型的推理链路开展定向文献阅读，"
     "梳理模型结构、动作生成方式、观测预处理与推理部署的关键环节，结合机械臂操作任务明确功能需求与"
     "实时性约束，形成技术路线与实验方案初稿。"),
    ("第二阶段（2026.12.21-2026.12.31）",
     "配合学院开题检查并完善方案。完成感知—推理—执行实验系统的总体设计：输入输出接口、动作空间与"
     "归一化方式、执行时序与超时处理、软硬件部署分工，以及异常与失效处理机制；形成接口规范与测试"
     "验证方案。"),
    ("第三阶段（2027.1.4-2027.1.15）",
     "搭建模型与仿真实验环境。确定运行环境与依赖版本，记录模型权重与配置版本，打通观测数据接口与"
     "动作接口，并完成模型加载、单次前向推理与动作输出格式的初步验证。"),
    ("第四阶段（2027.3.1-2027.3.12）",
     "完成原始模型的推理基线与任务流程复现。测量并记录推理延迟、计算资源占用与吞吐等性能指标，"
     "跑通仿真环境中的任务执行流程，形成可追溯的原始性能基线。"),
    ("第五阶段（2027.3.15-2027.3.26）",
     "确定待复现的推理优化方向并完成首轮实验，接受中期检查。按模型计算、动作生成过程与部署配置等"
     "环节归类候选方法，评估其在本课题条件下的可行性，提交中期检查材料与不少于二分之一的论文框架。"),
    ("第六阶段（2027.3.29-2027.4.9）",
     "开展优化前后的推理性能对照实验。在模型、任务与实验条件一致的前提下，比较推理延迟、资源占用、"
     "吞吐与运行稳定性，并记录优化引入的额外开销与适用前提。"),
    ("第七阶段（2027.4.12-2027.4.23）",
     "完成任务表现评测与失败情形分析。统计任务成功率及其分布，比较优化前后的动作输出差异，按失败"
     "类型（如动作偏差、执行中止、超时）分类归因，评估对操作精度与执行稳定性的影响。"),
    ("第八阶段（2027.4.26-2027.5.7）",
     "完成工程验证与资源评价，修改论文初稿。结合部署条件评价优化方法的收益与代价，完成资源占用与"
     "能耗对比，形成系统设计与工程验证结论，并据此修改论文初稿。"),
    ("第九阶段（2027.5.10-2027.5.21）",
     "完成论文查重与内审，准备答辩材料（按学院通知执行）。补充实验记录与图表，核对毕业要求指标点的"
     "支撑材料。"),
    ("第十阶段（2027.5.24-2027.6.4）",
     "完成论文定稿、答辩及答辩后修改与材料整理（按学院安排执行）。"),
    ("第十一阶段（2027.6.7-2027.6.18）",
     "完成后续修改、工作总结与资料归档。"),
]


def set_text(p, text):
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
    else:
        p.add_run(text)


def main():
    doc = Document(SRC)
    cell = doc.tables[0].rows[12].cells[1]
    paras = cell.paragraphs

    header = None
    for p in paras:
        if p.text.strip().startswith("进度安排"):
            header = p.text
            break
    if header is None:
        raise RuntimeError("header paragraph not found")

    # remove all paragraphs except the header, then append expanded stages
    for p in paras:
        if p.text.strip().startswith("进度安排"):
            continue
        p._element.getparent().remove(p._element)

    hp = cell.paragraphs[0]
    tpl = hp.runs[0] if hp.runs else None
    for prefix, body in STAGES:
        p = cell.add_paragraph()
        p.paragraph_format.first_line_indent = hp.paragraph_format.first_line_indent
        p.paragraph_format.line_spacing = hp.paragraph_format.line_spacing
        run = p.add_run(prefix + body)
        if tpl is not None:
            if tpl.font.name:
                run.font.name = tpl.font.name
            if tpl.font.size:
                run.font.size = tpl.font.size
            srpr = tpl._element.rPr
            if srpr is not None:
                sf = srpr.find(qn('w:rFonts'))
                if sf is not None:
                    rpr = run._element.get_or_add_rPr()
                    rf = rpr.find(qn('w:rFonts'))
                    if rf is None:
                        rf = rpr.makeelement(qn('w:rFonts'), {})
                        rpr.append(rf)
                    for a in ('w:eastAsia', 'w:ascii', 'w:hAnsi'):
                        v = sf.get(qn(a))
                        if v:
                            rf.set(qn(a), v)

    try:
        doc.save(DST)
        print("SAVED", DST)
    except PermissionError as e:
        alt = DST.replace(".docx", "_v2.docx")
        doc.save(alt)
        print("LOCKED, saved to", alt, "|", str(e)[:50])


if __name__ == "__main__":
    main()
