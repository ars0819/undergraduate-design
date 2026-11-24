# -*- coding: utf-8 -*-
"""Light markdown sanity check: table column consistency, code-fence balance, heading order."""
import re
from pathlib import Path

ROOT = Path(r"D:\Desktop\dickseep\research\topics")

for name in ["literature_matrix.md", "candidate_topics.md", "experiment_plan.md", "advisor_discussion.md"]:
    text = (ROOT / name).read_text(encoding="utf-8")
    lines = text.splitlines()
    issues = []

    if text.count("```") % 2 != 0:
        issues.append(f"unbalanced code fences ({text.count('```')})")

    # table blocks
    block, start = [], 0
    for i, line in enumerate(lines + [""], 1):
        if line.lstrip().startswith("|"):
            if not block:
                start = i
            block.append((i, line))
        else:
            if len(block) >= 2:
                counts = {}
                for ln, l in block:
                    n = l.replace("\\|", "").count("|")
                    counts.setdefault(n, []).append(ln)
                if len(counts) > 1:
                    detail = "; ".join(f"{n} pipes @ lines {v[:4]}" for n, v in counts.items())
                    issues.append(f"table starting line {start}: inconsistent columns -> {detail}")
            block = []

    heads = re.findall(r"^(#{1,3})\s+(.+)$", text, re.M)
    print(f"== {name} ==")
    print(f"   lines={len(lines)} level1-3 headings={len(heads)}")
    print(f"   issues: {issues if issues else 'none'}")
