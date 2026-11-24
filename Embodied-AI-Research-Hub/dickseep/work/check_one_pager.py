# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

p = Path(r"D:\Desktop\dickseep\research\v3\advisor_one_page.md")
t = p.read_text(encoding="utf-8")
lines = t.splitlines()

FENCE = "`" * 3
issues = []
if t.count(FENCE) % 2:
    issues.append("unbalanced fences")

block = []
for i, l in enumerate(lines + [""], 1):
    if l.lstrip().startswith("|"):
        block.append((i, l))
    else:
        if len(block) >= 2:
            counts = {}
            for ln, x in block:
                counts.setdefault(x.replace("\\|", "").count("|"), []).append(ln)
            if len(counts) > 1:
                issues.append("table@%d %s" % (block[0][0], {k: len(v) for k, v in counts.items()}))
        block = []

print("%d lines, %d B, table_issues=%s" % (len(lines), p.stat().st_size, issues or "none"))
print("sections:")
for l in lines:
    if l.startswith("## "):
        print("  " + l)

stale = [
    "12GB 准入分级与预算基准",
    "必须有一个 bf16 能进 12GB",
    "因果分解完",
    "已完成对重规划频率",
]
for s in stale:
    if s in t:
        print("WARN residual:", s)
