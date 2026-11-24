# -*- coding: utf-8 -*-
"""Build the 10-slide advisor report deck."""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

BASE = r"D:\Desktop\dickseep\research\thesis\2026-11-04_综合研究方案"
OUT = os.path.join(BASE, "导师交流汇报.pptx")

C_TITLE = RGBColor(0x1F, 0x3A, 0x5F)
C_BODY = RGBColor(0x22, 0x22, 0x22)
C_ACC = RGBColor(0xB0, 0x3A, 0x2E)


def add_title_slide(prs, title, sub):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = C_TITLE
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = sub
    r2.font.size = Pt(15); r2.font.color.rgb = C_BODY
    return s


def add_content(prs, title, bullets, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = C_TITLE

    body = s.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(12.1), Inches(6.3))
    tf = body.text_frame
    tf.word_wrap = True
    for i, (txt, lvl) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        p.space_after = Pt(5)
        r = p.add_run(); r.text = ('• ' if lvl == 0 else '– ') + txt
        r.font.size = Pt(13 if lvl == 0 else 11.5)
        r.font.color.rgb = C_BODY
    if note:
        nb = s.shapes.add_textbox(Inches(0.6), Inches(7.15), Inches(12.1), Inches(0.5))
        p = nb.text_frame.paragraphs[0]
        r = p.add_run(); r.text = note
        r.font.size = Pt(10); r.font.italic = True; r.font.color.rgb = C_ACC
    return s


def add_table(prs, title, header, rows, col_w, note=None, fs=9.5):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = C_TITLE

    nrow, ncol = len(rows) + 1, len(header)
    left, top = Inches(0.45), Inches(1.15)
    width = Inches(sum(col_w))
    shape = s.shapes.add_table(nrow, ncol, left, top, width, Inches(0.35 * nrow))
    table = shape.table
    for j, w in enumerate(col_w):
        table.columns[j].width = Inches(w)
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.text = h
        for pr in cell.text_frame.paragraphs:
            for r in pr.runs:
                r.font.size = Pt(fs); r.font.bold = True
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(val)
            for pr in cell.text_frame.paragraphs:
                for r in pr.runs:
                    r.font.size = Pt(fs)
    if note:
        nb = s.shapes.add_textbox(Inches(0.45), Inches(7.1), Inches(12.3), Inches(0.6))
        p = nb.text_frame.paragraphs[0]
        r = p.add_run(); r.text = note
        r.font.size = Pt(9.5); r.font.italic = True; r.font.color.rgb = C_ACC
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "面向多自由度机械臂操作的 VLA 高效推理与执行优化研究",
        "导师讨论稿｜刘鑫宇（机械工程·机器人试验班 2023 级）｜2026-11-04\n"
        "全部结论标注证据等级；未做的实验一律标“待做”，不以假设充当结果"
    )

    add_content(prs, "一、选题与核心目标", [
        ("题目：面向多自由度机械臂操作的视觉语言动作模型高效推理与执行优化研究（待导师确认）", 0),
        ("不做：新基础模型、新 VLA 架构、自建评测框架、以机械结构设计为主线", 0),
        ("要做：统一执行层 + 六段延迟测量 + 真机边界检验 + 系统设计文档（支撑 3.3）", 0),
        ("为什么现在必须收窄：2026-10-31 至 11-03 四天内出现 4 篇直接工作，执行层已是红海", 0),
        ("PTS：训练无关执行层，LIBERO 注入延迟下吞吐 +33%(π0.5)/+104%(SmolVLA)，超零延迟 1×", 1),
        ("ARP：可训练异步架构，Franka +18% 成功率 / −20% 时间，需重训", 1),
        ("Reflex：整块缓存+两级校正，延迟 1.4—11×；Leap-VLA：缓存年龄条件化 + 无梯度 RTC 改写", 1),
    ], note="证据：arXiv 2611.01390 / 2611.05155 / 2611.00524 / 2611.01011，均为 2026-10-31~11-03 预印本，摘要级核验")

    add_table(prs, "二、最新进展与关键数据（含条件）",
              ["论文（日期）", "机制", "关键结果（含条件）", "对本课题"],
              [["LoopQ（11-03）", "同 checkpoint/layer/engine 下比量化", "省显存 2—3×；H100 batch=1 无量化更快(慢1.06—1.58×)；标准LIBERO掩盖1.5—4.3分损失", "量化≠加速"],
               ["VLALagBench（11-02）", "五因子延迟敏感性", "100ms 使 71.5%→40.5%（另一任务 86.5%→74.0%）；标定 RMSE 5.7—7.8%", "延迟—成功基准"],
               ["VLA-HWBench（10-27）", "7 阶段分解，18 模型", "动作生成在 14/19 配置为最大阶段；参数只解释 ≤15% 方差", "归因方法"],
               ["SO-101 Bench（10-27）", "Isaac Lab 数字孪生", "仿真-真机排序 ρ=1.00，但仿真高估成功率；最佳策略≈61%", "支持仿真为主"],
               ["Yu & Qiu（06-11）", "SO-101 真机四策略", "“执行不稳定”是最主要失败源", "最接近的失败证据"],
               ["Cowling（10-30）", "离线指标 vs 真机，930 rollout", "启动失败占 DP 96%/SmolVLA 47%；跨策略排名反转 ρ=+0.90", "必须拆分启动失败"]],
              [1.7, 2.6, 5.6, 2.2],
              note="全部为 arXiv 自述结果，未见独立复现；机构未在元数据声明者不作推断")

    add_table(prs, "三、研究重叠矩阵：空白在哪一行",
              ["判据", "PTS", "ARP", "Reflex", "Leap-VLA", "RTC", "VLALag", "LoopQ"],
              [["训练无关执行层", "★", "—", "△", "—", "△", "○", "—"],
               ["异步/双频架构", "△", "★", "○", "★", "★", "△", "—"],
               ["块缓存/复用", "○", "○", "★", "○", "△", "—", "—"],
               ["量化/压缩", "—", "—", "—", "—", "—", "—", "★"],
               ["低成本 SO-101 真机", "—", "—", "—", "—", "—", "○", "—"],
               ["执行器物理约束 L_mech", "○", "—", "—", "—", "—", "—", "—"],
               ["闭环墙钟+队列层测量", "△", "○", "○", "○", "○", "△", "△"]],
              [3.3, 1.15, 1.15, 1.15, 1.4, 1.15, 1.35, 1.2],
              note="★ 正面对撞　△ 部分重叠　○ 可共存。最后两行几乎全空——是位置，也说明缺少对照基线，风险同源")

    add_content(prs, "四、本科阶段的具体研究问题（三条假设，任一结局可答辩）", [
        ("H1 衰减：同一执行层规则在高精度平台上的吞吐增益，在舵机臂上至少衰减一半", 0),
        ("检验：两平台增益比值 < 0.5 即成立", 1),
        ("H2 归因：衰减可由伺服响应延迟 L_mech 与路径可行域收紧解释（回归残差 < 20%）", 0),
        ("检验：残差过大即不成立，转为“无法归因”的负结论", 1),
        ("H3 无差别：在低成本臂上，PTS 类规则与“简单缩短执行长度”无显著差异", 0),
        ("检验：配对非劣性检验，预设非劣界 δ", 1),
        ("为什么这样设计：三种结局都产出可用结论——边界、判据、或“别过度工程”的工程建议", 0),
    ], note="本页为待验证假设，尚未做任何实验")

    add_content(prs, "五、技术架构与实验平台", [
        ("链路：相机采集 → 观测缓冲(统一时间戳) → 推理(本地12GB / 远端A100) → 动作队列(H步) → 总线舵机 → L_mech → 关节状态闭环", 0),
        ("端到端闭环延迟 = 关节到位 − 曝光时刻；GPU kernel 时间只作归因，不作闭环延迟", 0),
        ("SO-101 规格（官方 BOM）：STS3215 舵机，每臂 6 个；堵转 16.5 kg·cm@6V(7.4V版)；参考 ￥97.72/个", 0),
        ("选它的理由：总线舵机位置环 → L_mech 与回程间隙可测且不可忽略，正好是 H2 的自变量来源", 0),
        ("模型安排：ACT(非VLA基线，验证链路) → SmolVLA(主对象) → π0.5(可选第二架构)", 0),
        ("两条线并行：仿真跑统计量，真机做锚点验证；仿真结论须注明“在此配置下成立”", 0),
    ], note="SO-101 规格取自官方 BOM/README 文本副本；L_mech 可测性待实测确认")

    add_table(prs, "六、实验设计与指标",
              ["类别", "内容", "说明"],
              [["基线", "B0 官方同步／B1 官方异步／B2 固定 H=K／B3 PTS 类规则", "避免只战胜人为削弱的同步"],
               ["对照", "2 平台 × 2 规则 × 2 延迟分布；独立测试布局", "布局不参与调参"],
               ["消融", "去进度对齐／去限速／固定 H", "确认增益来源"],
               ["延迟指标", "推理延迟 p50/p95、端到端延迟、控制周期抖动、队列耗尽率", "报样本数，不报小样本 p99"],
               ["效果指标", "成功率（Wilson 95% CI）、完成时间（含失败）、启动失败率", "分母是全部开始的 trial"],
               ["统计", "配对风险差 + 预设非劣界 δ；多格比较用 Holm；排序用 AnyRank 类序贯法", "不以 p>0.05 写等效"],
               ["资源", "12GB 本机做测量，A100 做微调；SO-101 双臂 + 1 相机 + 台面", "配额与经费待确认"]],
              [1.6, 6.6, 3.9],
              note="指标定义与统计方案须在开题报告预先冻结，不得看到结果后放宽")

    add_content(prs, "七、预期成果与毕业要求对应", [
        ("工程成果：统一执行层（K/H 可配、执行槽序号、时间戳）+ 六段测量脚本 + 日志 schema", 0),
        ("研究结论：H1/H2/H3 的判定 + 适用边界表 + 失败分类统计", 0),
        ("设计文档：架构图、通信时序图、传感—计算—执行接口定义、失效处理方案（3.3 举证）", 0),
        ("3.3 依据：培养方案 p54 原文为“机械系统或生产制造工艺流程设计”，支撑课程含“机器人控制系统设计”；毕业设计对 3.3 标 H，对 3.2 无标注", 0),
        ("其他指标：2.3 方案比较｜6.2 失效处理｜7.2 本地vs远端能耗｜8.2 可复现｜10.3 英文｜11.2 成本决策｜12.2 技术追踪", 0),
        ("仍是待确认：控制系统方案+接口+验收能否兑现 3.3；是否需要补实物件；指标点选项集替换是否被接受", 0),
    ], note="不预设“必须做夹爪/CAD/有限元”；也不预设纯软件评测必然满足 3.3")

    add_content(prs, "八、风险与降级方案", [
        ("风险1 增益不衰减：H1 被否证 → 题目退化为“复现 PTS”；须导师事先认可这种交付", 0),
        ("风险2 真机吞吐不足：单 trial 含复位约 3—6 分钟，240 trial 即 12—24 小时纯操作；<20 trial/天只能判大效应", 0),
        ("风险3 仿真-真机不可比：SO-101 Bench 显示排序可迁移但仿真高估成功率 → 绝对成功率不可混报", 0),
        ("降级 L1：仿真 + 真机 20—30 trial 锚点", 0),
        ("降级 L2：纯仿真闭环 + 真机可行性记录（装配/标定/回放视频）", 0),
        ("降级 L3：纯仿真 + 公开数据软件验证", 0),
        ("红线：仿真与真机数据分列，不合并样本量；不把 GPU 计时当闭环延迟", 0),
    ], note="降级策略需导师批准后写入任务书，避免中期被动")

    add_content(prs, "九、需要导师决定的三个问题", [
        ("1) 贡献形态：是否接受“边界检验 + 系统实测”作为主要贡献？", 0),
        ("若要求新方法 → 本方案需重构，改走模型级优化（方向 B），工期与算力要求更高", 1),
        ("2) 3.3 口径：机器人控制系统方案 + 接口 + 失效处理 + 真机验收，是否足够？是否需要补实物件？", 0),
        ("3) 资源：A100 配额与排队规则；SO-101 采购经费；硬件延误时是否批准仿真主线降级", 0),
        ("另需确认：是否纳入 π0.5 作为第二架构（会显著增加工期）", 0),
        ("本周我可独立完成、不占用老师时间：环境准入（两策略可加载可推理）与需求约束表初稿", 0),
    ], note="题目与方案均为候选，未获任何审批；任务书与院外申请表已按流程填写但保留待确认项")

    prs.save(OUT)
    print("PPTX:", OUT, "slides:", len(prs.slides))


if __name__ == '__main__':
    main()
