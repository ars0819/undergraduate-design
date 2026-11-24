# -*- coding: utf-8 -*-
"""Merge single-page invoice PDFs into one document (pure Python, no deps).

Strategy: copy every object of every source PDF with a fresh object number,
remap indirect references in the dictionary part only (stream bytes are copied
byte-for-byte), collect the page objects, then build a new /Pages tree and
write a classic xref table.
"""
import re
import sys

REF_RE = re.compile(rb"(\d+)\s+(\d+)\s+R")
OBJ_RE = re.compile(rb"(?<![0-9])(\d+)\s+(\d+)\s+obj\b")


def parse_objects(data):
    """Return list of (num, gen, dict_part, stream_part) preserving byte layout."""
    objs = []
    pos = 0
    while True:
        m = OBJ_RE.search(data, pos)
        if not m:
            break
        num, gen = int(m.group(1)), int(m.group(2))
        start = m.end()
        sm = re.compile(rb"\bstream\r?\n?").search(data, start)
        em = data.find(b"endobj", start)
        if sm and (em == -1 or sm.start() < em):
            dict_part = data[start:sm.start()]
            sstart = sm.end()
            lm = re.search(rb"/Length\s+(\d+)(?!\s+\d+\s+R)", dict_part)
            if lm:
                send = sstart + int(lm.group(1))
            else:
                send = data.find(b"endstream", sstart)
            # the copied slice must carry its own "endstream" terminator
            es = data.find(b"endstream", send if send != -1 else sstart)
            if es == -1:
                es = data.find(b"endstream", sstart)
            if es == -1:
                raise ValueError(f"obj {num}: stream without endstream")
            stream_part = data[sm.start():es + len(b"endstream")]
            eo = data.find(b"endobj", es + len(b"endstream"))
            pos = (eo + 6) if eo != -1 else len(data)
            objs.append((num, gen, dict_part, stream_part))
        else:
            end = em if em != -1 else len(data)
            objs.append((num, gen, data[start:end], b""))
            pos = (end + 6) if em != -1 else len(data)
    return objs


def find_page_objects(objs):
    """Object numbers whose dict is /Type /Page (not /Pages)."""
    pages = []
    for num, gen, d, s in objs:
        if re.search(rb"/Type\s*/Page(?![s\w])", d):
            pages.append(num)
    return pages


def remap(dict_part, mapping):
    def sub(m):
        old = int(m.group(1))
        new = mapping.get(old)
        return f"{new} 0 R".encode() if new is not None else m.group(0)
    return REF_RE.sub(sub, dict_part)


def merge(paths, out_path):
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    next_num = 1
    page_refs = []

    # new /Pages object gets the lowest number we allocate later; reserve ids first
    reserved_pages_num = 0  # filled after all sources are walked
    # We must know ids before writing, so do a two-pass: collect all sources first.
    sources = []
    for p in paths:
        data = open(p, "rb").read()
        sources.append(parse_objects(data))

    # assign numbers
    plan = []  # list of (source_page_index, objs, mapping)
    for objs in sources:
        mapping = {}
        for num, gen, d, s in objs:
            mapping[num] = next_num
            next_num += 1
        plan.append((objs, mapping))
    pages_obj_num = next_num
    next_num += 1
    catalog_num = next_num
    next_num += 1

    for objs, mapping in plan:
        for num, gen, d, s in objs:
            newnum = mapping[num]
            offsets[newnum] = len(out)
            if re.search(rb"/Type\s*/Page(?![s\w])", d):
                d = re.sub(rb"/Parent\s+\d+\s+\d+\s+R",
                           f"/Parent {pages_obj_num} 0 R".encode(), d)
                page_refs.append(newnum)
            body = remap(d, mapping)
            out += f"{newnum} 0 obj".encode() + body
            if s:
                out += s
            out += b"\nendobj\n"

    kids = b" ".join(f"{n} 0 R".encode() for n in page_refs)
    offsets[pages_obj_num] = len(out)
    out += (f"{pages_obj_num} 0 obj\n<</Type/Pages/Count {len(page_refs)}"
            f"/Kids[{kids.decode()}]/MediaBox[0 0 595.28 396.85]>>\nendobj\n").encode()

    offsets[catalog_num] = len(out)
    out += (f"{catalog_num} 0 obj\n<</Type/Catalog/Pages {pages_obj_num} 0 R>>"
            f"\nendobj\n").encode()

    size = next_num
    xref_pos = len(out)
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for n in range(1, size):
        off = offsets.get(n)
        out += b"0000000000 00000 f \n" if off is None else f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<</Size {size}/Root {catalog_num} 0 R>>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()

    open(out_path, "wb").write(bytes(out))
    return len(page_refs), size, xref_pos


if __name__ == "__main__":
    out_path = sys.argv[1]
    paths = sys.argv[2:]
    n, size, xp = merge(paths, out_path)
    print(f"merged {len(paths)} files -> {out_path}")
    print(f"pages={n} objects={size} xref_at={xp}")
