# -*- coding: utf-8 -*-
"""Dump styling / layout details of the template sheet."""
import sys
from openpyxl import load_workbook

path = sys.argv[1]
wb = load_workbook(path, data_only=False)
ws = wb["Sheet1"]

print("### column widths")
for k, d in ws.column_dimensions.items():
    print(f"  {k}: width={d.width} hidden={d.hidden} bestFit={d.bestFit}")

print("### row heights")
for i in range(1, ws.max_row + 1):
    d = ws.row_dimensions.get(i)
    print(f"  row {i}: height={d.height if d else None}")

print("### page setup")
ps = ws.page_setup
print(f"  orientation={ps.orientation} paper={ps.paperSize} fitToWidth={ps.fitToWidth} fitToHeight={ps.fitToHeight} scale={ps.scale}")
print(f"  margins: left={ws.page_margins.left} right={ws.page_margins.right} top={ws.page_margins.top} bottom={ws.page_margins.bottom}")
print(f"  print_area={ws.print_area}")
print(f"  sheet_view.showGridLines={ws.sheet_view.showGridLines}")

print("### cell styles")
for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=9):
    for c in row:
        f = c.font
        b = c.border
        al = c.alignment
        bd = "".join(x[0] for x in ["left", "right", "top", "bottom"] if getattr(b, x) and getattr(b, x).style)
        print(
            f"  {c.coordinate}: val={str(c.value)[:18]!r} font=({f.name},{f.sz},b={f.bold}) "
            f"align=(h={al.horizontal},v={al.vertical},wrap={al.wrap_text}) "
            f"border={bd} fill={c.fill.fgColor.rgb if c.fill and c.fill.fill_type else None} "
            f"fmt={c.number_format}"
        )
