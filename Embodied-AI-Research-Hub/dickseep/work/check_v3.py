import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
root = Path(r"D:\Desktop\dickseep\research\v3")
for p in sorted(root.glob("*.md")):
    t = p.read_text(encoding="utf-8")
    lines = t.splitlines()
    issues = []
    if t.count("```") % 2: issues.append("unbalanced fences")
    block = []
    for i, l in enumerate(lines + [""], 1):
        if l.lstrip().startswith("|"): block.append((i, l))
        else:
            if len(block) >= 2:
                c = {}
                for ln, x in block:
                    n = x.replace("\\|", "").count("|"); c.setdefault(n, []).append(ln)
                if len(c) > 1: issues.append(f"table@{block[0][0]} cols={ {k:len(v) for k,v in c.items()} }")
            block = []
    print(f"{p.name:28s} {len(lines):4d} lines  {p.stat().st_size:6d} B  issues={issues if issues else 'none'}")
# cross-file: forbid stale formulas
bad = ["max(E·Δt", "K / L_chunk", "L ≤ P·Δt 就能", "E=1 是精度上界"]
for p in sorted(root.glob("*.md")):
    t = p.read_text(encoding="utf-8")
    for b in bad:
        if b in t:
            # allowed only inside the "superseded" section of decision_log
            print(f"  WARN {p.name}: contains '{b}' (check context)")
