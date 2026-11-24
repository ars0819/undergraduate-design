# -*- coding: utf-8 -*-
"""Replace ONLY stage-1 and stage-2 lines in the 进度安排 cell; keep header and
stages 3-11 byte-identical. Also rewrite the cross-college application form.
"""
import os
import shutil

from docx import Document
from docx.oxml.ns import qn

OUTDIR = r"D:\Desktop\dickseep\本科毕设任务书_第一版修订_V3"
V3 = os.path.join(OUTDIR, "本科毕业设计任务书_V3_高效推理部署_导师讨论稿.docx")
V3_ALT = os.path.join(OUTDIR, "本科毕业设计任务书_V3_高效推理部署_导师讨论稿_修订.docx")

FORM_BASE = r"D:\Desktop\10_校院外毕业设计申请表_已填写_导师讨论稿.docx"
FORM_OUT = os.path.join(OUTDIR, "10_校院外毕业设计申请表_V3_导师讨论稿.docx")

TITLE_CN = "面向多自由度机械臂操作的视觉语言动作模型高效推理与部署研究"
TITLE_EN = ("Efficient Inference and Deployment of Vision-Language-Action Models "
            "for Multi-Degree-of-Freedom Robotic Manipulation")

S1 = ("第一阶段（2026.12.7-2026.12.18）完成开题报告撰写，于12月18日前上传教务系统；"
      "前期完成的文献阅读与需求分析作为开题基础。")
S2 = "第二阶段（2026.12.21-2026.12.31）配合学院开题检查，完善系统总体方案与实验环境设计。"

MAIN_CONTENT = [
    "毕设主要内容：",
    "1. 视觉语言动作模型及其高效推理技术的发展现状分析，明确机械臂操作任务的需求与主要性能瓶颈；",
    "2. 机器人感知—推理—动作执行实验系统的总体设计，包括输入输出接口、动作空间、执行时序、软硬件"
    "部署方式与异常处理机制；",
    "3. 代表性开源视觉语言动作模型的推理流程复现，建立推理效率与任务表现的原始基线，分析主要计算与"
    "部署瓶颈；",
    "4. 一类成熟推理优化方法的复现，保持模型与实验条件一致，比较优化前后的推理效率、资源占用与"
    "动作质量；",
    "5. 任务表现评测与失败情形分析，结合工程约束评价优化方法的适用条件；",
    "6. 形成系统设计方案、实验记录与工程验证结论，为后续在轮式双臂平台上开展研究提供基础。",
    "上述工作以模型复现、系统设计与工程评测为主，不涉及新模型结构的自主设计。",
]


def set_para(p, text):
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
    else:
        p.add_run(text)


def patch_schedule():
    doc = Document(V3)
    cell = doc.tables[0].rows[12].cells[1]
    hit = 0
    for p in cell.paragraphs:
        t = p.text.strip()
        if t.startswith("第一阶段"):
            set_para(p, S1); hit += 1
        elif t.startswith("第二阶段"):
            set_para(p, S2); hit += 1
    if hit != 2:
        raise RuntimeError("expected 2 replacements, got %d" % hit)
    try:
        doc.save(V3)
        return V3
    except PermissionError:
        doc.save(V3_ALT)
        return V3_ALT


def clear_cell(cell):
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p0 = cell.paragraphs[0]
    for r in p0.runs[1:]:
        r._element.getparent().remove(r._element)
    return p0


def write_cell(cell, lines):
    p0 = clear_cell(cell)
    tpl = p0.runs[0] if p0.runs else None
    for i, line in enumerate(lines):
        p = p0 if i == 0 else cell.add_paragraph()
        if i == 0:
            set_para(p, line)
        else:
            p.paragraph_format.line_spacing = p0.paragraph_format.line_spacing
            run = p.add_run(line)
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


def rewrite_form():
    shutil.copyfile(FORM_BASE, FORM_OUT)
    doc = Document(FORM_OUT)
    rows = doc.tables[0].rows
    write_cell(rows[7].cells[1], [TITLE_CN])
    write_cell(rows[8].cells[1], [TITLE_EN])
    write_cell(rows[9].cells[1], [
        "1、工程实践类□  研究设计类☑（建议）  理论分析类□（三类中各选一项，待导师确认）",
        "2、软件□  硬件□  软硬结合☑（建议）  非软硬件□",
        "3、题目是否来源于科研项目：  是□      否☑",
    ])
    write_cell(rows[10].cells[1], MAIN_CONTENT)
    doc.save(FORM_OUT)
    return FORM_OUT


if __name__ == "__main__":
    print("TASK:", patch_schedule())
    print("FORM:", rewrite_form())
