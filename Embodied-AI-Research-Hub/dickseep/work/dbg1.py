import sys, re
sys.path.insert(0, r"D:\Desktop\dickseep")
import pdf_text2
t = pdf_text2.extract(r"D:\Desktop\dickseep\out\材料费发票合并-11张-2074.35元.pdf")
chunks = t.split("----PAGE----")
c = chunks[0]
print("=== PAGE 1 RAW ===")
print(c)
print("=== digits only ===")
print(re.sub(r"[^0-9]", "", c))
