import sys, re
sys.path.insert(0, r"D:\Desktop\dickseep")
import pdf_text2 as P
p = r"D:\Desktop\dickseep\work\zip\4-其他-90.00元-资兴市极脉控模型店（个体工商户）-2026.05.22-发票.pdf"
pdf = P.PDF(p)
for n in (22, 24):
    cs = pdf.stream(n)
    print("=== obj", n, "len", len(cs) if cs else None)
    if cs: print(cs[:1200].decode("latin-1"))
