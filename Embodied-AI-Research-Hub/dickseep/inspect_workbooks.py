# -*- coding: utf-8 -*-
"""Dump structure of the template and the reimbursement-detail workbooks."""
import sys
from openpyxl import load_workbook

def dump(path, label):
    print("=" * 100)
    print(f"### {label}: {path}")
    wb = load_workbook(path, data_only=False)
    print(f"sheets: {wb.sheetnames}")
    for ws in wb.worksheets:
        print("-" * 100)
        print(f"[sheet] {ws.title!r} dims={ws.dimensions} max_row={ws.max_row} max_col={ws.max_column}")
        print(f"  merged: {[str(r) for r in ws.merged_cells.ranges]}")
        print(f"  freeze: {ws.freeze_panes}")
        print("  --- cells ---")
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
            for c in row:
                if c.value is not None and str(c.value).strip() != "":
                    print(f"  {c.coordinate}: {c.value!r}")

if __name__ == "__main__":
    dump(sys.argv[1], sys.argv[2])
