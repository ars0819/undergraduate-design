# -*- coding: utf-8 -*-
"""Build DOCX / PPTX / appendix from the consolidated research plan."""
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn

BASE = r"D:\Desktop\dickseep\research\thesis\2026-11-04_综合研究方案"
MD = os.path.join(BASE, "本科毕设综合研究方案.md")
DOCX = os.path.join(BASE, "本科毕设综合研究方案.docx")
APP_MD = os.path.join(BASE, "文献证据附录.md")
BIB = r"D:\Desktop\dickseep\research\thesis\2026-11-04_导师交流初稿\references.bib"

ASCII_ARCH = """[系统架构（原 Mermaid 图，此处为等价文字版）]
  相机采集 ──┐
  关节/夹爪状态 ──┴─→ 观测缓冲（统一时间戳）
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
        本地推理(12GB)       远端推理(A100)
              └─────────┬─────────┘
                        ↓
                 动作队列（执行 H 步）
                        ↓
                  总线舵机下发
                        ↓
              物理响应 L_mech → 关节状态（闭环）
                        ↓
        日志（帧/请求/下发/响应）→ 测量与统计"""


def set_font(run, size=10.5, bold=False, name="宋体"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = rpr.makeelement(qn('w:rFonts'), {})
        rpr.append(rf)
    rf.set(qn('w:eastAsia'), name)
    rf.set(qn('w:ascii'), 'Times New Roman')
    rf.set(qn('w:hAnsi'), 'Times New Roman')


def add_rich(par, text, size=10.5, base_bold=False):
    """Handle **bold** and `code` inline markers."""
    tokens = re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', text)
    for tk in tokens:
        if not tk:
            continue
        if tk.startswith('**') and tk.endswith('**'):
            r = par.add_run(tk[2:-2]); set_font(r, size, True)
        elif tk.startswith('`') and tk.endswith('`'):
            r = par.add_run(tk[1:-1]); set_font(r, size - 0.5, base_bold, "Consolas")
        else:
            r = par.add_run(tk); set_font(r, size, base_bold)


def is_sep(row):
    return all(re.fullmatch(r':?-{2,}:?', c.strip()) or c.strip() == '' for c in row)


def split_row(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in s.split('|')]


def build_docx(md_path, out_path, title=None, body=10.5, h1=13, h2=11.5, margin=2.2, tblfs=8.5):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(margin)
    sec.top_margin = sec.bottom_margin = Cm(max(1.2, margin - 0.4))

    lines = open(md_path, encoding='utf-8').read().splitlines()
    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        ln = lines[i]
        s = ln.rstrip()

        if s.startswith('```'):
            if not in_code:
                in_code = True
                code_buf = []
                lang = s[3:].strip().lower()
            else:
                in_code = False
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(6)
                text = ASCII_ARCH if lang == 'mermaid' else '\n'.join(code_buf)
                for k, cl in enumerate(text.split('\n')):
                    r = p.add_run(('' if k == 0 else '\n') + cl)
                    set_font(r, 8.5, False, "Consolas")
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue

        if s.startswith('|') and i + 1 < len(lines) and lines[i + 1].strip().startswith('|'):
            header = split_row(s)
            j = i + 1
            rows = []
            while j < len(lines) and lines[j].strip().startswith('|'):
                rows.append(split_row(lines[j]))
                j += 1
            tb_rows = [r for r in rows if not is_sep(r)]
            if tb_rows and is_sep(rows[0]) is False:
                header, tb_rows = tb_rows[0], tb_rows[1:]
            ncol = max(len(header), max((len(r) for r in tb_rows), default=0))
            tbl = doc.add_table(rows=0, cols=ncol)
            tbl.style = 'Table Grid'
            tbl.autofit = True
            hr = tbl.add_row().cells
            for c in range(ncol):
                cell = hr[c]
                cell.paragraphs[0].text = ''
                add_rich(cell.paragraphs[0], header[c] if c < len(header) else '', tblfs, True)
            for r in tb_rows:
                rc = tbl.add_row().cells
                for c in range(ncol):
                    rc[c].paragraphs[0].text = ''
                    add_rich(rc[c].paragraphs[0], r[c] if c < len(r) else '', tblfs)
            doc.add_paragraph()
            i = j
            continue

        if s.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_rich(p, s[2:], h1 + 3, True); doc.add_paragraph()
        elif s.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            add_rich(p, s[3:], h1, True)
        elif s.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            add_rich(p, s[4:], h2, True)
        elif s.startswith('---'):
            pass
        elif s.startswith('- ') or s.startswith('* '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_after = Pt(1)
            add_rich(p, '· ' + s[2:], body)
        elif re.match(r'^\d+\.\s', s):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_after = Pt(1)
            add_rich(p, s, body)
        elif s.strip() == '':
            pass
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.first_line_indent = Cm(0)
            add_rich(p, s, body)
        i += 1

    doc.save(out_path)
    return out_path


def build_appendix():
    esc = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'π': 'pi'}
    txt = open(BIB, encoding='utf-8').read()
    body = '\n'.join(l for l in txt.splitlines() if not l.strip().startswith('%'))
    entries = re.findall(r'@misc\{([^,]+),(.*?)\n\}', body, re.S)
    lines = ['# 文献证据附录', '',
             '> 配套《本科毕设综合研究方案》。全部条目元数据经 arXiv Atom API 或本地 PDF 首页核验；',
             '> **无一篇在本轮读到全文**（OFT 与 SmolVLA 的关键章节在早前会话读过）。venue 仅转述论文自述。',
             '> 机构栏：🟦 已在 PDF 首页或元数据读到；🟨 据项目页/代码组织名/作者长期单位推断；🟥 元数据未给出。', '',
             '## 1. 参考文献总表（%d 条）' % len(entries), '',
             '| # | key | 标题 | 年份 | arXiv | 备注 |', '|---|---|---|---|---|---|']
    for k, (key, blk) in enumerate(entries, 1):
        def get(field):
            m = re.search(r'\b' + field + r'\s*=\s*\{(.*?)\}', blk, re.S)
            return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
        title = get('title').replace('{', '').replace('}', '')
        year = get('year')
        eprint = get('eprint')
        note = get('note')[:60]
        lines.append('| %d | %s | %s | %s | %s | %s |' % (k, key, title[:70], year, eprint, note))

    lines += ['', '## 2. 检索日志摘要', '',
              '| 轮次 | 通道 | 检索式/参数 | 命中 | 取回 | 备注 |',
              '|---|---|---|---|---|---|',
              '| R1 | arXiv Atom | 12 个 id_list 批量 | — | 24 | 2026-11-03，经 API 核验 |',
              '| R1 | arXiv Atom | `all:"Fin Ray" AND all:gripper` | 21 | 5 | 机械侧 |',
              '| R1 | arXiv Atom | `abs:"mobile manipulation" AND abs:wheeled AND abs:grasping` | 5 | 4 | 移动操作 |',
              '| R2 | arXiv Atom | `ti:"SO-101"` | 2 | 2 | SO-101 Bench、Yu & Qiu |',
              '| R2 | arXiv Atom | `ti:"VLA-Eval" OR ti:"AnyRank" OR ti:"SO-ARMs"` | 2 | 2 | vla-eval、AnyRank |',
              '| R3 | arXiv Atom | `abs:"vision-language-action" AND abs:latency`（按时间倒序） | 270 | 7 | 最新进展 |',
              '| R3 | arXiv Atom | `abs:"action chunking"`（按时间倒序） | 526 | 7 | 最新进展 |',
              '',
              '**通道限制**：进程级 `curl`/`Invoke-WebRequest` 全部 HTTP 000（沙箱阻断）；`web_fetch` 可访问 `export.arxiv.org`，',
              '但 `huggingface.co`、`api.github.com`、`raw.githubusercontent.com` 均报 "resolves to a non-public IP"（代理 fake-ip 模式所致）。',
              'SO-101 规格取自本机保留的官方 BOM/README 文本副本。',
              '',
              '## 3. 研究重叠审计（按条件逐项比对）', '',
              '| 判据 | PTS | ARP | Reflex | Leap-VLA | RTC | VLALag | LoopQ | SO-101 Bench | Yu&Qiu |',
              '|---|---|---|---|---|---|---|---|---|---|',
              '| 训练无关执行层 | 是 | 否 | 部分 | 否 | 部分 | 否 | 否 | 否 | 否 |',
              '| 异步/双频架构 | 部分 | 是 | 否 | 是 | 是 | 部分 | 否 | 否 | 否 |',
              '| 块缓存/复用 | 否 | 否 | 是 | 部分 | 部分 | 否 | 否 | 否 | 否 |',
              '| 量化/压缩 | 否 | 否 | 否 | 否 | 否 | 否 | 是 | 否 | 否 |',
              '| 低成本 SO-101 真机 | 否 | 否 | 否 | 否 | 否 | 部分 | 否 | 是 | 是 |',
              '| **执行器物理约束（L_mech/间隙）** | 部分 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 部分 |',
              '| **闭环墙钟+队列层测量** | 部分 | 否 | 否 | 否 | 部分 | 部分 | 部分 | 否 | 否 |',
              '',
              '> 结论：最后两行在任何一列都不是"完整覆盖"。但**空缺同时意味着缺少对照基线**——这是本课题的主要风险来源。',
              '',
              '## 4. 待核实清单', '',
              '1. PTS 原文（22 页）是否讨论舵机/执行器约束的实测 —— 决定方向 A 的位置是否仍存在。',
              '2. ARP 的 vision-action expert 能否直接接 SmolVLA —— 影响其是否可作为基线。',
              '3. LoopQ 的结论在 5070（Blackwell、12 GB）上的可复现性。',
              '4. LeRobot 当前 policy 列表、`K/H` 默认值、`E<K` 时取前缀还是后缀 —— 需装包读源码。',
              '5. 机械工程方向期刊文献（IEEE/ASME/中文核心）—— 本轮无通道。',
              ]
    open(APP_MD, 'w', encoding='utf-8').write('\n'.join(lines))
    return APP_MD, len(entries)


ONE_PAGE = """# 导师一页研究决策摘要

**学生**：刘鑫宇（机械工程（机器人试验班）2023 级，2023212129）｜**导师**：王强 教授
**日期**：2026-11-04｜**性质**：决策用一页摘要，非最终开题材料

## 推荐题目

**面向多自由度机械臂操作的视觉语言动作模型高效推理与执行优化研究**
（备选：面向低成本平台的 VLA 执行层部署适配与边界检验）

## 主要研究问题

在注入等价延迟的条件下，PTS 一类**训练无关执行层**在**舵机驱动的低成本臂**上带来的吞吐增益，相对其在高精度/仿真平台上的增益**衰减多少**？衰减能否被**伺服响应延迟 L_mech** 与**路径可行域收紧**解释？

三条假设覆盖三种结局：H1 衰减成立 → 给出低成本部署边界；H2 归因成立 → 给出可预测判据；H3 无差别 → 说明低端硬件上复杂调度不划算。**任一结局都可答辩。**

## 已有论文做到哪一步（关键事实）

- **PTS**（arXiv:2611.01390，2026-11-01）：训练无关执行层，LIBERO 注入延迟下吞吐 **+33%（π0.5）/ +104%（SmolVLA）**，成功率不降，甚至超过零延迟 1×；未在舵机臂验证。
- **ARP**（2611.05155，11-03）：可训练异步架构，Franka **+18% 成功率 / −20% 时间**；需重训。
- **LoopQ**（2611.05093，11-03）：18 万+ episode 证明量化稳定省显存 2—3×，但 **batch=1 下无配置更快**，标准 LIBERO 掩盖 1.5—4.3 分损失。
- **VLALagBench**（2611.03103）：100 ms 延迟使某任务成功率 **71.5%→40.5%**；静态基准延迟只影响完成时间。
- **SO-101 Bench**（2610.36420）：仿真-真机策略排序 **ρ=1.00**，但仿真高估成功率。
- **Yu & Qiu**（2606.08881v2）：SO-101 真机四策略，**"执行不稳定"为最主要失败源**。
- **Cowling**（2611.00283）：启动失败占 DP **96%**、SmolVLA **47%**，离线数据无法暴露。

## 本科阶段具体做什么

1. 建 SO-101 平台（STS3215 舵机，每臂 6 个，官方 BOM 参考 ￥97.72/个）+ 仿真线；
2. ACT 基线 → SmolVLA 微调；
3. **统一执行层**（`K/H` 可配、执行槽序号、时间戳）与**六段延迟测量**；
4. 接入 PTS 类规则，测 `L_mech` 与回程间隙；
5. 仿真 vs 真机对照，检验 H1/H2/H3；
6. 独立布局复核 + 失败分类 + 边界结论。

## 工程研究价值（不依赖创新成立）

可复用的执行层与测量代码、日志 schema、低成本平台部署判据表、系统设计与失效处理文档（用于 3.3 举证）。

## 最少需要的硬件与模型

SO-101 双臂 + 1 相机 + 台面；本机 5070 12 GB 做小模型与测量，A100 做微调。模型：SmolVLA（主）、ACT（基线）、π0.5（可选）。

## 可能的创新点与风险

**创新点（严格限定）**：执行层优化在高 `L_mech` 平台上的**衰减与归因**——边界结论，非新方法。
**最大风险**：若增益不衰减，题目退化为"复现 PTS"；若真机吞吐 <20 trial/天，只能判大效应。

## 需要导师决定的三件事

1. **贡献形态**：是否接受"边界检验 + 系统实测"作为主要贡献（若要求新方法，本方案需重构）。
2. **3.3 口径**：以"机器人控制系统方案 + 传感—计算—执行接口 + 失效处理 + 真机验收"兑现，是否足够？是否需要补实物件？
3. **资源**：A100 配额与排队规则；SO-101 采购经费；硬件延误时是否批准"仿真主线 + 真机可行性记录"的降级方案。
"""


def main():
    os.makedirs(BASE, exist_ok=True)
    p1 = build_docx(MD, DOCX)
    one_md = os.path.join(BASE, "导师一页研究决策摘要.md")
    open(one_md, 'w', encoding='utf-8').write(ONE_PAGE)
    one_docx = os.path.join(BASE, "导师一页研究决策摘要.docx")
    build_docx(one_md, one_docx, body=8.6, h1=10.5, h2=9.2, margin=1.45, tblfs=8.2)
    app, n = build_appendix()
    app_docx = os.path.join(BASE, "文献证据附录.docx")
    build_docx(app, app_docx)
    print("DOCX:", p1)
    print("ONE :", one_docx)
    print("APP :", app_docx, "refs:", n)


if __name__ == '__main__':
    main()
