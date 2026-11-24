# -*- coding: utf-8 -*-
"""Verify the merged PDF: xref, stream lengths, page count, per-page invoice."""
import re
import sys

sys.path.insert(0, r"D:\Desktop\dickseep")
import pdf_text2

PATH = sys.argv[1]
EXPECTED = [
    ("26432000001155188101", "90.00"),
    ("26422000001843792021", "370.00"),
    ("26952000002150270851", "42.80"),
    ("26372000002442814951", "39.60"),
    ("26952000002149565131", "36.16"),
    ("26952000002142529411", "28.90"),
    ("26422000001844364376", "9.20"),
    ("26442000005814129826", "720.00"),
    ("26332000004444834906", "125.09"),
    ("26372000003682690861", "366.60"),
    ("26372000003681578386", "246.00"),
]

data = open(PATH, "rb").read()
print(f"size={len(data)} bytes  header={data[:8]}  %%EOF={data.rstrip().endswith(b'%%EOF')}")

# ---- xref ----
m = re.search(rb"startxref\s+(\d+)", data)
xref_pos = int(m.group(1))
assert data[xref_pos:xref_pos + 4] == b"xref", "startxref does not point at xref"
head = re.match(rb"xref\r?\n(\d+) (\d+)\r?\n", data[xref_pos:])
first, count = int(head.group(1)), int(head.group(2))
print(f"xref: startxref={xref_pos} first={first} count={count}")

bad = 0
for i in range(count):
    off = xref_pos + head.end() + i * 20
    e = data[off:off + 20]
    num = first + i
    if num == 0 or len(e) < 18 or e[17:18] != b"n":
        continue
    f_off = int(e[0:10])
    if not data[f_off:f_off + 40].lstrip().startswith(f"{num} 0 obj".encode()):
        print(f"  BAD xref offset obj {num}")
        bad += 1
print(f"xref offsets wrong: {bad}")

# ---- stream /Length consistency ----
stream_bad = 0
for pm in re.finditer(rb"(?<![0-9])(\d+) 0 obj", data):
    num = int(pm.group(1))
    start = pm.end()
    sm = re.compile(rb"\bstream\r?\n?").search(data, start)
    if not sm:
        continue
    dict_part = data[start:sm.start()]
    lm = re.search(rb"/Length\s+(\d+)", dict_part)
    if not lm:
        continue
    length = int(lm.group(1))
    body = data[sm.end():sm.end() + length]
    after = data[sm.end() + length:sm.end() + length + 16]
    if not after.lstrip().startswith(b"endstream"):
        print(f"  BAD /Length obj {num}: declared {length}, next={after!r}")
        stream_bad += 1
    if not body:
        print(f"  empty stream obj {num}")
print(f"stream /Length mismatches: {stream_bad}")

# ---- pages ----
pages = pdf_text2.PDF(PATH).pages()
print(f"page objects: {len(pages)}")

for pno, res, contents in pages:
    body = pdf_text2.PDF(PATH).body(pno)
    has_media = b"/MediaBox" in body
    if not has_media:
        print(f"  page obj {pno} missing explicit /MediaBox")

text = pdf_text2.extract(PATH)
chunks = text.split("----PAGE----")
print(f"text chunks: {len(chunks)}")

ok = stream_bad == 0 and bad == 0 and len(pages) == 11
for i, (no, amt) in enumerate(EXPECTED):
    digits = re.sub(r"[^0-9.]", "", chunks[i] if i < len(chunks) else "")
    # strip the decimal points when hunting the invoice number
    hits_no = no in digits
    hits_amt = amt in digits or amt.replace(".", "") in digits.replace(".", "")
    good = hits_no and hits_amt
    ok &= good
    print(f"  page {i+1:>2}: {no} / {amt:>7} -> "
          f"{'OK' if good else 'PROBLEM'} (no={hits_no} amt={hits_amt})")

print("PDF MERGE OK" if ok else "PROBLEM")
