import sys, re
sys.path.insert(0, r"D:\Desktop\dickseep")
import pdf_text2 as P
p = r"D:\Desktop\dickseep\work\zip\4-其他-90.00元-资兴市极脉控模型店（个体工商户）-2026.05.22-发票.pdf"
pdf = P.PDF(p)
for pno, res, contents in pdf.pages():
    if not contents: continue
    m = re.search(rb"/Font\s*<<(.*?)>>", res, re.S)
    names = {}
    if m:
        for fm in re.finditer(rb"/([\w.\-]+)\s+(\d+)\s+\d+\s+R", m.group(1)):
            names[fm.group(1).decode()] = int(fm.group(2))
    print("fonts:", names)
    for k,v in names.items():
        b = pdf.body(v)
        print(k, b[:300])
    # raw text ops
    for t in re.finditer(rb"<([0-9A-Fa-f]+)>\s*Tj|\((?:[^()\\]|\\.)*\)\s*Tj", contents):
        pass
    print("--- hex strings, try gbk ---")
    seen=set()
    for t in re.finditer(rb"<([0-9A-Fa-f]{4,})>\s*Tj", contents):
        h = t.group(1).decode()
        if h in seen: continue
        seen.add(h)
        raw = bytes.fromhex(h)
        for enc in ("gbk","gb18030","big5"):
            try:
                s = raw.decode(enc)
                print(enc, "OK:", s)
                break
            except Exception as e:
                pass
        else:
            print("FAIL", h[:60])
