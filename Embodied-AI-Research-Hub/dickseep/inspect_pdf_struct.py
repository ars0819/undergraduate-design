# -*- coding: utf-8 -*-
"""Inspect PDF internals: header, xref kind, object streams, page tree."""
import re
import sys


def inspect(path):
    data = open(path, "rb").read()
    print("=" * 90)
    print("FILE:", path.split("\\")[-1], "size", len(data))
    print("  header:", data[:9])
    print("  has /ObjStm:", b"/ObjStm" in data, "| has /XRef:", b"/XRef" in data,
          "| has startxref:", b"startxref" in data)
    n_obj = len(re.findall(rb"\d+\s+\d+\s+obj\b", data))
    print("  literal ? 0 obj count:", n_obj)
    m = re.search(rb"/Type\s*/Catalog", data)
    print("  Catalog:", bool(m))
    for key in (rb"/Type\s*/Pages", rb"/Type\s*/Page\b", rb"/Encrypt", rb"/Linearized",
                rb"/Type\s*/XRef", rb"/Type\s*/ObjStm"):
        print(f"  {key.decode():22s}", bool(re.search(key, data)))
    # trailer
    tm = re.search(rb"trailer\s*<<(.*?)>>", data, re.S)
    print("  trailer:", tm.group(1)[:200] if tm else None)
    # page objects
    for pm in re.finditer(rb"(\d+)\s+\d+\s+obj(.*?)endobj", data, re.S):
        body = pm.group(2)
        if re.search(rb"/Type\s*/Page\b", body):
            keys = re.findall(rb"/(\w+)", body[:800])
            print(f"  page obj {pm.group(1).decode()}: keys={[k.decode() for k in keys][:14]}")
            print(f"    body head: {body[:260]!r}")


for p in sys.argv[1:]:
    inspect(p)
