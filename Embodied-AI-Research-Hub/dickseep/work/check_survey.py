# -*- coding: utf-8 -*-
"""Sanity check for the survey report: tables, fences, section order, reference count."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

p = Path(r"D:\Desktop\dickseep\research\毕设选题.md")
t = p.read_text(encoding="utf-8")
lines = t.splitlines()

issues = []
FENCE = "`" * 3
if t.count(FENCE) % 2:
    issues.append("unbalanced code fences")

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
                issues.append("table@%d -> %s" % (block[0][0], {k: len(v) for k, v in counts.items()}))
        block = []

refs = re.findall(r"^\d+\.\s", t, re.M)
arx = re.findall(r"arXiv:(\d{4}\.\d{4,5})", t)

print("lines=%d  bytes=%d" % (len(lines), p.stat().st_size))
print("tables_issues=%s" % (issues or "none"))
print("numbered_refs=%d  distinct_arxiv_ids=%d" % (len(refs), len(set(arx))))
print("sections:")
for l in lines:
    if l.startswith("## "):
        print("  " + l)
print("missing? expected 46 arxiv ids, got %d" % len(set(arx)))
