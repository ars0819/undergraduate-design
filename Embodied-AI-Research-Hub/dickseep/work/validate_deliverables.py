# -*- coding: utf-8 -*-
"""Validate deliverables: bib well-formedness + citation-key coverage + file inventory."""
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\Desktop\dickseep\research\topics")
BIB = ROOT / "references.bib"
MDS = ["literature_matrix.md", "candidate_topics.md", "experiment_plan.md", "advisor_discussion.md"]

bib_text = BIB.read_text(encoding="utf-8")

# ---- 1. brace balance ----
depth = 0
line_no = 1
bad = []
for ch in bib_text:
    if ch == "\n":
        line_no += 1
    elif ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth < 0:
            bad.append(line_no)
            depth = 0
print(f"[bib] final brace depth = {depth} (expect 0); premature closings at lines: {bad[:5]}")

# ---- 2. entry parsing ----
entries = re.findall(r"@(\w+)\{([^,]+),", bib_text)
keys = [k.strip() for _, k in entries]
print(f"[bib] entries parsed = {len(entries)}; types = {sorted(set(t for t, _ in entries))}")
dupes = {k for k in keys if keys.count(k) > 1}
print(f"[bib] duplicate keys = {sorted(dupes) if dupes else 'none'}")

# per-entry required fields
problems = []
for m in re.finditer(r"@(\w+)\{([^,]+),((?:[^{}]|\{[^{}]*\})*)\}", bib_text, re.S):
    typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
    for field in ("author", "title", "year"):
        if not re.search(rf"\b{field}\s*=", body):
            problems.append(f"{key}: missing {field}")
print(f"[bib] field problems = {problems if problems else 'none'}")

# ---- 3. citation coverage ----
cited = set()
for name in MDS:
    text = (ROOT / name).read_text(encoding="utf-8")
    cited |= set(re.findall(r"@([A-Za-z0-9_]+)", text))
missing = sorted(c for c in cited if c not in set(keys))
print(f"[cite] distinct keys cited in md = {len(cited)}")
print(f"[cite] cited but MISSING in bib = {missing if missing else 'none'}")
uncited = sorted(k for k in keys if k not in cited)
print(f"[bib] entries never cited in md ({len(uncited)}) = {uncited}")

# ---- 4. file inventory ----
print("\n[files]")
total = 0
for p in sorted(ROOT.iterdir()):
    if p.is_file():
        size = p.stat().st_size
        total += size
        if p.suffix == ".md":
            n = len(p.read_text(encoding="utf-8").splitlines())
            print(f"  {p.name:28s} {size:7d} B   {n:5d} lines")
        else:
            print(f"  {p.name:28s} {size:7d} B")
print(f"  total {total} B")

sys.exit(0 if (not missing and not dupes and not problems and depth == 0) else 1)
