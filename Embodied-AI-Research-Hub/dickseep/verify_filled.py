# -*- coding: utf-8 -*-
"""Verify the merged filled form against the two invoice batches."""
import sys
from openpyxl import load_workbook

path = sys.argv[1]
wb = load_workbook(path, data_only=False)
ws = wb["Sheet1"]

TOTAL, COUNT = 2074.35, 11

print("sheets:", wb.sheetnames, "| dims:", ws.dimensions,
      "| merged:", len(ws.merged_cells.ranges))

expected = {
    "A1": "报销清单",
    "B2": "自动化",
    "B5": "材料费",
    "C5": COUNT,
    "D5": TOTAL,
    "B6": "图书资料",
    "B7": "设备费",
    "B9": "刘鑫宇",
    "C9": "2023212129",
    "D9": "18445450819",
    "E9": TOTAL,
    "A21": "合计",
    "B21": TOTAL,
    "A23": "项目负责人签字",
    "A25": "指导老师签字",
    "D25": "基地主任签字",
}
ok = True
for ref, want in expected.items():
    got = ws[ref].value
    good = got == want
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {ref}: {got!r}")

print("D2:", repr(ws["D2"].value))
print("number formats: D5=%s E9=%s C9=%s D9=%s" % (
    ws["D5"].number_format, ws["E9"].number_format,
    ws["C9"].number_format, ws["D9"].number_format))

print("--- must stay blank ---")
blank_ok = True
for ref in ["C6", "D6", "C7", "D7", "E5", "B10", "C10", "D10", "E10",
            "B11", "C11", "D11", "E11", "B13", "B16", "B17", "B19", "E19", "B20"]:
    if ws[ref].value is not None:
        blank_ok = False
        print(f"  BAD {ref} = {ws[ref].value!r}")
print(f"  blank check: {'OK' if blank_ok else 'FAILED'}")

print("arithmetic: 9+2=%s  1461.75+612.60=%s  consistent=%s" % (
    COUNT, TOTAL, ws["C5"].value == COUNT and
    ws["D5"].value == ws["E9"].value == ws["B21"].value == TOTAL))

print("styles on filled cells:")
for ref in ("C5", "D5", "B9", "C9", "D9", "E9", "B21"):
    c = ws[ref]
    b = c.border
    bd = "".join(x[0] for x in ["left", "right", "top", "bottom"]
                 if getattr(b, x) and getattr(b, x).style)
    print(f"  {ref}: border={bd} align=({c.alignment.horizontal},{c.alignment.vertical}) "
          f"font=({c.font.name},{c.font.sz})")

print("ALL OK" if (ok and blank_ok) else "PROBLEM")
