import re
from pathlib import Path
p=Path(r"D:\Desktop\dickseep\research\reviews\v11_v13_audit_2026-11-01.md")
t=p.read_text(encoding="utf-8")
print("fences:", t.count("```"), "| headings:", len(re.findall(r"^## ", t, re.M)))
lines=t.splitlines()
block=[]; issues=[]
for i,l in enumerate(lines+[""],1):
    if l.lstrip().startswith("|"): block.append((i,l))
    else:
        if len(block)>=2:
            counts={}
            for ln,x in block:
                n=x.replace("\\|","").count("|")
                counts.setdefault(n,[]).append(ln)
            if len(counts)>1: issues.append((block[0][0], {k:len(v) for k,v in counts.items()}))
        block=[]
print("table issues:", issues if issues else "none")
print("sections:", [l[:34] for l in lines if l.startswith("## ")])
