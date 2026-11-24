# -*- coding: utf-8 -*-
"""Fill BUPT 2027 task-book and cross-college application form from templates.

Cell indices were measured from the actual templates (merged-cell aware):
  RWS (8 cols): 学院 value=c1, 专业 value=c6
                学生姓名=c1, 学号=c4, 班级=c7
                指导教师=c1, 所在单位=c4, 职称=c7
                签字 value=c2, 日期 value=c5
  APP (12 cols): 学生姓名=c1, 班级=c4, 学号=c8, 班内序号=c11
                专业=c1, 手机=c8, 邮箱=c11
                导师姓名=c1, 性别=c5, 所属单位=c10
                学历=c1, 职称=c5, 出生日期=c10
                联系电话=c1, 手机=c5, 邮箱=c10
Originals are never modified (copies are edited).
"""
import os
import shutil

from docx import Document
from docx.oxml.ns import qn

SRC_RWS = r"D:\Desktop\附件6：信通院2027届本科毕设常用特殊表格.docx"
SRC_APP = r"D:\Desktop\bishe\附件7.校（院）外毕业设计申请表---现代邮政学院(自动化学院).docx"
OUT_DIR = r"D:\Desktop\dickseep\research\thesis\2026-11-04_导师交流初稿"
OUT_RWS = os.path.join(OUT_DIR, "09_本科毕业设计任务书_已填写_导师讨论稿.docx")
OUT_APP = os.path.join(OUT_DIR, "10_校院外毕业设计申请表_已填写_导师讨论稿.docx")

TITLE_CN = "面向多自由度机械臂操作的视觉语言动作模型高效推理与执行优化研究"
TITLE_EN = ("Efficient Inference and Execution Optimization of Vision-Language-Action "
            "Models for Multi-Degree-of-Freedom Robotic Manipulation")

# 由学生本人确认提供的信息（2026-11-04）
STU_NAME = "刘鑫宇"
STU_ID = "2023212129"
STU_CLASS = "2023211704"
STU_SEQ = "无"
STU_PHONE = "18445450819"
STU_MAIL = "ars0819@163.com"
ADV_NAME = "王强"
ADV_TITLE = "教授"


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
    src_rpr = src_run._element.rPr
    if src_rpr is not None:
        src_fonts = src_rpr.find(qn('w:rFonts'))
        if src_fonts is not None:
            rpr = dst_run._element.get_or_add_rPr()
            rfonts = rpr.find(qn('w:rFonts'))
            if rfonts is None:
                rfonts = rpr.makeelement(qn('w:rFonts'), {})
                rpr.append(rfonts)
            for attr in ('w:eastAsia', 'w:ascii', 'w:hAnsi'):
                v = src_fonts.get(qn(attr))
                if v:
                    rfonts.set(qn(attr), v)


def write_cell(cell, lines):
    p0 = clear_cell(cell)
    tpl_run = p0.runs[0] if p0.runs else None
    for i, line in enumerate(lines):
        p = p0 if i == 0 else cell.add_paragraph()
        if i == 0:
            if p.runs:
                p.runs[0].text = line
            else:
                _copy_font(p.add_run(line), tpl_run)
        else:
            p.paragraph_format.first_line_indent = p0.paragraph_format.first_line_indent
            p.paragraph_format.line_spacing = p0.paragraph_format.line_spacing
            _copy_font(p.add_run(line), tpl_run)


def strip_after_first_table(doc):
    body = doc.element.body
    children = list(body)
    idx = next((i for i, el in enumerate(children) if el.tag == qn('w:tbl')), None)
    if idx is None:
        raise RuntimeError("no table")
    for el in children[idx + 1:]:
        if el.tag == qn('w:sectPr'):
            continue
        body.remove(el)


IND = ["2.3", "3.3", "6.2", "7.2", "8.2", "10.3", "11.2", "12.2"]


def ind_line(checked):
    return "支撑指标点：" + "；".join(
        ("☑" if k in checked else "□") + k for k in IND
    )


def fill_renwushu():
    shutil.copyfile(SRC_RWS, OUT_RWS)
    doc = Document(OUT_RWS)
    strip_after_first_table(doc)
    tbl = doc.tables[0]
    rows = tbl.rows

    # ---- 基本信息 ----
    r = rows[0].cells
    write_cell(r[1], ["现代邮政学院（自动化学院）"])
    write_cell(r[6], ["机械工程（机器人试验班）"])

    r = rows[1].cells
    write_cell(r[1], [STU_NAME])
    write_cell(r[4], [STU_ID])
    write_cell(r[7], [STU_CLASS])

    r = rows[2].cells
    write_cell(r[1], [ADV_NAME])
    write_cell(r[4], ["信息与通信工程学院"])
    write_cell(r[7], [ADV_TITLE])

    # ---- 题目 ----
    write_cell(rows[3].cells[1], ["（中文）" + TITLE_CN])
    write_cell(rows[4].cells[1], ["（英文）" + TITLE_EN])

    # ---- 类型 / 来源 ----
    write_cell(rows[5].cells[1],
               ["工程实践类□   研究设计类☑（建议，待导师确认）   理论分析类□  其他□"])
    write_cell(rows[6].cells[1],
               ["题目是否来源于科研项目       是□      否☑"])
    write_cell(rows[7].cells[1], ["科研项目名称：（不适用——题目不来源于科研项目）"])
    write_cell(rows[8].cells[1], ["科研项目负责人：（不适用——题目不来源于科研项目）"])

    # ---- 任务 ----
    task_block = [
        "主要任务及目标：（任务共4项；任务1为调研类，故任务数不少于4项。指标点已按机械工程"
        "（机器人试验班）培养方案替换，见文末说明）",
        "任务1：调研机器人学习与视觉语言动作模型（VLA）的原理、代表方法与研究现状，分析目标"
        "机械臂操作任务的功能需求与实时性约束，形成需求与约束清单。",
        ind_line({"2.3", "10.3", "12.2"}),
        "任务2：完成机器人感知、动作输出、控制接口与推理执行链路的总体方案设计，明确控制周期、"
        "动作空间、相机配置、超时与失效处理策略，并给出可测的性能指标定义。",
        ind_line({"3.3", "11.2", "6.2"}),
        "任务3：部署并评测代表性机器人学习策略（ACT 作为模仿学习基线，SmolVLA 作为主要 VLA），"
        "测量推理延迟、动作队列覆盖与成功率，定位性能瓶颈，并针对其中一个环节尝试一种有依据的"
        "优化方法。",
        ind_line({"2.3", "11.2", "12.2"}),
        "任务4：搭建仿真与（条件允许时的）SO-101 真机实验平台，开展对照测试、消融实验、失败"
        "分类统计与安全性评估，给出优化前后的性能对比与适用条件分析。",
        ind_line({"3.3", "6.2", "7.2", "8.2", "10.3"}),
    ]
    write_cell(rows[9].cells[1], task_block)

    # ---- 主要内容 ----
    content_block = [
        "主要内容：（与上述四项任务一一对应；下列“支撑毕业要求指标点”按机械工程（机器人试验班）"
        "培养方案撰写）",
        "内容一（对应任务1）：机器人学习与 VLA 高效推理的原理与现状调研，以及机械臂操作任务的"
        "需求分析。",
        "支撑毕业要求指标点：",
        "2.3：在综述中比较同步推理、异步推理与动作分块等至少三条技术路线的适用条件与代价，给出"
        "可替代方案与选型理由，形成“多方案比较—影响因素分析—有效结论”的完整论证链；",
        "10.3：主要参考文献以外文期刊与会议论文为主，完成英文文献的检索、阅读与要点提炼，并在"
        "论文中提供英文摘要与英文图表题；",
        "12.2：跟踪 2025—2026 年的快速演进（异步执行、自适应执行长度、边缘端延迟分解等），"
        "在综述中体现对新兴技术的主动追踪与批判性判断。",
        "内容二（对应任务2）：面向机械臂操作的感知—推理—执行系统总体方案设计。",
        "支撑毕业要求指标点：",
        "3.3：针对“机械臂操作实时性不足”这一工程需求，完成机器人控制系统与软硬件协同执行系统的"
        "方案设计，交付系统架构图、通信时序图、传感—计算—执行接口定义、超时与失效处理方案及"
        "方案比选理由；",
        "11.2：在“本机 12 GB 显卡 / 实验室 A100 40 GB”两种部署条件下，结合显存、延迟与成本"
        "做出配置选择与工程决策，形成成本—性能决策表；",
        "6.2：在方案阶段即纳入关节限位、速度与力矩保护、通信中断后的保持或终止策略，并评估超时"
        "策略对操作安全性与作业连续性的影响。",
        "内容三（对应任务3）：代表性策略的部署、性能测量、瓶颈定位及一项优化尝试。",
        "支撑毕业要求指标点：",
        "2.3：把总延迟分解为“曝光—读帧—请求—返回—动作下发—物理响应”六段，逐段比较不同执行"
        "模式的代价，识别关键环节并给出改进方向的比较与取舍；",
        "11.2：在相同算力预算下比较不同执行策略的“单位时间成功任务数”与推理请求率，说明工程"
        "取舍而非只报最优指标；",
        "12.2：面对框架版本快速迭代与接口不兼容问题，通过源码阅读、社区文档与实验复现完成自学"
        "与适配，形成可复现的版本与配置记录。",
        "内容四（对应任务4）：仿真与真机实验平台搭建、对照实验与工程验证。",
        "支撑毕业要求指标点：",
        "3.3：以实验平台与执行系统的整体设计与实测数据，验证前述方案设计的可实施性与有效性，"
        "完成方案—实现—验证的闭环；",
        "6.2：统计并复核失败模式（启动失败、抓取失败、掉落、放置失败、超时、保护停止），评估"
        "安全相关失败的比例与处置方式；",
        "7.2：比较本地部署与远端推理在算力占用与能耗上的差异，给出面向可持续性的部署建议；",
        "8.2：固定随机种子与 checkpoint，保留原始日志与视频，公开评测脚本，遵守学术诚信与"
        "开源许可要求；",
        "10.3：以论文正文、答辩汇报与英文摘要完成技术表达与交流。",
    ]
    write_cell(rows[10].cells[1], content_block)

    # ---- 参考文献 ----
    refs = [
        "主要参考文献：（不少于10篇，含近五年文献；以下条目均经 arXiv 元数据或本地原文核对，"
        "预印本性质如实标注）",
        "[1] Zhao T Z, Kumar V, Levine S, et al. Learning Fine-Grained Bimanual Manipulation "
        "with Low-Cost Hardware[C]. Robotics: Science and Systems (RSS), 2023. arXiv:2304.13705.",
        "[2] Chi C, Xu Z, Feng S, et al. Diffusion Policy: Visuomotor Policy Learning via "
        "Action Diffusion[J]. arXiv preprint arXiv:2303.04137, 2023.",
        "[3] Kim M J, Pertsch K, Karamcheti S, et al. OpenVLA: An Open-Source "
        "Vision-Language-Action Model[J]. arXiv preprint arXiv:2406.09246, 2024.",
        "[4] Kim M J, Finn C, Liang P. Fine-Tuning Vision-Language-Action Models: Optimizing "
        "Speed and Success[C]. Robotics: Science and Systems (RSS), 2025. arXiv:2502.19645.",
        "[5] Shukor M, Aubakirova D, Capuano F, et al. SmolVLA: A Vision-Language-Action Model "
        "for Affordable and Efficient Robotics[J]. arXiv preprint arXiv:2506.01844, 2025.",
        "[6] Black K, Brown N, Driess D, et al. π0: A Vision-Language-Action Flow Model for "
        "General Robot Control[C]. Robotics: Science and Systems (RSS), 2025. arXiv:2410.24164.",
        "[7] Physical Intelligence, Black K, Brown N, et al. π0.5: a Vision-Language-Action "
        "Model with Open-World Generalization[J]. arXiv preprint arXiv:2504.16054, 2025.",
        "[8] Black K, Galliker M Y, Levine S. Real-Time Execution of Action Chunking Flow "
        "Policies[C]. NeurIPS, 2025. arXiv:2506.07339.",
        "[9] Jiang H, Zou Y, Liang B, et al. FutureRTC: Real-Time Robot Execution with "
        "Anticipatory-Conditioned Action Chunking[J]. arXiv preprint arXiv:2607.24008, 2026.",
        "[10] Liang Y, Wang X, Wang K, et al. Adaptive Action Chunking at Inference-time for "
        "Vision-Language-Action Models[C]. CVPR, 2026. arXiv:2604.04161.",
        "[11] Xu R, Shan X, Dai S, et al. Knowing When to Stop: Adaptive Action Chunking via "
        "Internal Cross-Attention Dynamics in VLAs[J]. arXiv preprint arXiv:2609.00908, 2026.",
        "[12] Wen Q, Xu B, Amro H, et al. VLALagBench: Understanding When and Why VLA "
        "Inference Latency Matters[J]. arXiv preprint arXiv:2611.03103, 2026.",
        "[13] Xu B, Tan H, Jiang W, et al. VLA-HWBench: Where the Time Goes When VLA Models "
        "Run on Edge Hardware[J]. arXiv preprint arXiv:2610.34717, 2026.",
        "[14] Xu J, Jin Q, Chen M, et al. VLAQuantBench: Closed-Loop Evaluation of "
        "Post-Training Quantization for Vision-Language-Action Models[J]. arXiv preprint "
        "arXiv:2609.25376, 2026.",
        "[15] Li Z, Yang H, Chen Z, et al. From Inference Efficiency to Embodied Efficiency: "
        "Revisiting Efficiency Metrics for Vision-Language-Action Models[J]. arXiv preprint "
        "arXiv:2603.19131, 2026.",
        "[16] Cowling S. Does Validation Loss Predict Real-Robot Success? Evidence from Four "
        "Imitation-Learning Policies on a Low-Cost Arm[J]. arXiv preprint arXiv:2611.00283, 2026.",
        "[17] Liu B, Zhu Y, Gao C, et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong "
        "Robot Learning[J]. arXiv preprint arXiv:2306.03310, 2023.",
    ]
    write_cell(rows[11].cells[1], refs)

    # ---- 进度 ----
    schedule = [
        "进度安排：（参照学院 2027 届毕设工作安排，起点为秋季学期第15周，每2周为一个节点）",
        "第一阶段（2026.12.7-2026.12.18）精读核心文献，锁定 LeRobot 版本与依赖清单，明确题目"
        "范围与任务边界，完成需求与约束清单初稿。",
        "第二阶段（2026.12.21-2026.12.31）完成本科毕业设计（论文）开题报告；提交系统方案设计"
        "初稿（系统架构图、通信时序图、接口定义）。",
        "第三阶段（2027.1.4-2027.1.15）搭建软件环境，完成策略“可加载、可推理”两级准入验证；"
        "使用公开数据完成数据接口与动作格式检查。",
        "第四阶段（2027.3.1-2027.3.12）完成平台装配与标定（真机路线）；确定数据契约与日志"
        "字段；开展流程预实验并估算复位成本。",
        "第五阶段（2027.3.15-2027.3.26）完成演示采集与 ACT 基线训练，获得首组成功率数据；"
        "填写中期检查表并参加中期答辩，提交不少于二分之一的论文框架。",
        "第六阶段（2027.3.29-2027.4.9）完成 SmolVLA 微调；实施推理延迟六段分解测量，定位"
        "瓶颈环节，确定待验证的优化点。",
        "第七阶段（2027.4.12-2027.4.23）完成对照实验与消融实验第一轮；整理实验数据与图表，"
        "撰写论文核心章节。",
        "第八阶段（2027.4.26-2027.5.7）在独立测试布局上复核结论；完成失败分类统计与安全性"
        "评估；按内审意见修改论文。",
        "第九阶段（2027.5.10-2027.5.21）完成论文查重检测与格式检测并取得合格结果；补齐过程性"
        "材料。",
        "第十阶段（2027.5.24-2027.6.4）论文定稿；准备答辩汇报与演示材料。",
        "第十一阶段（2027.6.7-2027.6.18）参加答辩；按答辩意见修改并在答辩后三日内提交定稿；"
        "完成材料归档。",
    ]
    write_cell(rows[12].cells[1], schedule)

    # ---- 签字栏：保持空白，不动“指导教师签字”标签 ----
    write_cell(rows[13].cells[2], [""])
    write_cell(rows[13].cells[5], ["年   月   日"])

    note = ("说明一：本表“支撑指标点”的选项已由信通院（电子信息工程）模板原列的 "
            "3.1/3.3/4.1/4.3/5.1/6.2/10.1/10.2/10.3/12.1，替换为机械工程（机器人试验班）"
            "培养方案中“毕业设计”列为 H 支撑的指标点 2.3/3.3/6.2/7.2/8.2/10.3/11.2/12.2。"
            "此替换须经指导教师与学院确认；若学院要求沿用原模板选项，请以学院口径为准。")
    note2 = ("说明二：指标点 3.2（机械单元/零部件设计）未被培养方案列为毕业设计的 H 支撑项，"
             "故本表未勾选；是否需要在课题中补充相应工作，请学院确认。")
    note3 = ("说明三：本文件为导师讨论稿。学生姓名、学号、班级由本人提供（刘鑫宇 / 2023212129 / "
             "2023211704）；题目类型勾选仍为建议值，须经导师确认。题目来源已确认为“不来源于"
             "科研项目”。")
    doc.add_paragraph(note)
    doc.add_paragraph(note2)
    doc.add_paragraph(note3)
    doc.save(OUT_RWS)
    return OUT_RWS


def fill_application():
    shutil.copyfile(SRC_APP, OUT_APP)
    doc = Document(OUT_APP)
    rows = doc.tables[0].rows

    # 学生信息
    r = rows[1].cells
    write_cell(r[1], [STU_NAME])
    write_cell(r[4], [STU_CLASS])
    write_cell(r[8], [STU_ID])
    write_cell(r[11], [STU_SEQ])

    r = rows[2].cells
    write_cell(r[1], ["机械工程（机器人试验班）"])
    write_cell(r[8], [STU_PHONE])
    write_cell(r[11], [STU_MAIL])

    # 导师信息
    r = rows[4].cells
    write_cell(r[1], [ADV_NAME])
    write_cell(r[5], ["待填"])
    write_cell(r[10], ["北京邮电大学 信息与通信工程学院"])

    r = rows[5].cells
    write_cell(r[1], ["待填"])
    write_cell(r[5], [ADV_TITLE])
    write_cell(r[10], ["待填"])

    r = rows[6].cells
    write_cell(r[1], ["待填"])
    write_cell(r[5], ["待填"])
    write_cell(r[10], ["待填"])

    # 题目
    write_cell(rows[7].cells[1], [TITLE_CN])
    write_cell(rows[8].cells[1], [TITLE_EN])

    # 题目分类
    write_cell(rows[9].cells[1], [
        "1、工程实践类□  研究设计类☑（建议）  理论分析类□（三类中各选一项，待导师确认）",
        "2、软件□  硬件□  软硬结合☑（建议）  非软硬件□",
        "3、题目是否来源于科研项目：  是□      否☑",
    ])

    # 主要内容
    main_content = [
        "毕设主要内容：",
        "1. 机械臂操作与视觉语言动作模型（VLA）的研究现状分析，明确真实部署中的实时性能瓶颈；",
        "2. 机器人感知—推理—动作执行系统的总体方案设计：控制周期、动作空间、相机配置、通信"
        "时序、超时处理，以及机器人与服务器的职责划分；",
        "3. ACT（模仿学习基线）与 SmolVLA（主要 VLA）的部署与基线评测；",
        "4. 推理延迟、控制周期抖动、动作队列覆盖率与任务成功率的测量，以及瓶颈环节定位；",
        "5. 针对已定位的瓶颈，尝试一种有依据的执行调度或部署优化方法，并做消融实验；",
        "6. 在仿真环境中完成主实验；在条件允许时使用 SO-101 机械臂开展真机验证与安全性评估。",
        "上述工作以系统设计与工程实验为主，不以训练更大规模模型为目标。",
    ]
    write_cell(rows[10].cells[1], main_content)

    # 地点
    write_cell(rows[11].cells[1], [
        "做毕设地点：",
        "待确认。拟在北京邮电大学校内开展，主要依托信息与通信工程学院王强老师指导下的科研环境"
        "（具体房间、场地与可用设备以导师与学院的实际授权为准；如涉及校外单位，须另行说明）。",
    ])

    # 申请原因
    reason = [
        "申请原因：",
        "本人为现代邮政学院（自动化学院）机械工程（机器人试验班）2023 级本科生，已与信息与"
        "通信工程学院王强老师建立后续科研指导关系。本人未来研究方向涉及具身智能与机器人操作、"
        "视觉语言动作模型（VLA）的高效推理与部署，需要机器人学习、智能控制、模型部署与工程"
        "实验等方面的系统指导。",
        "本次拟开展的本科课题围绕机械臂操作任务，研究感知、推理与动作执行链路的系统设计与"
        "性能评测，涉及机器人控制系统方案、软硬件接口设计与对照实验验证，与本人本科专业的"
        "培养方向具有明确的衔接点。在王强老师指导下开展该课题，有利于使用相关的科研环境与"
        "研究条件，但具体可使用的设备、算力与场地均以实际授权为准。",
        "本人承诺严格遵守现代邮政学院（自动化学院）关于开题、中期检查、论文撰写、查重、答辩"
        "与材料归档的各项管理要求，按学院规定的时间节点完成各环节工作，并按时回本科所在学院"
        "参加中期检查与毕业答辩。",
        "",
        "申请人签字：____________________            年    月    日",
    ]
    write_cell(rows[12].cells[1], reason)

    # 各意见栏、签字栏保持空白，不代拟内容
    doc.save(OUT_APP)
    return OUT_APP


if __name__ == "__main__":
    print("OK", fill_renwushu())
    print("OK", fill_application())
