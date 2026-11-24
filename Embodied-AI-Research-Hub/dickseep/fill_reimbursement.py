# -*- coding: utf-8 -*-
"""Fill the BUPT reimbursement form with all material-fee invoices (two batches)."""
import shutil
from openpyxl import load_workbook

SRC = r"D:\Desktop\1、2020-自动化大创-报销清单（一式两份）.xlsx"
OUT = r"D:\Desktop\dickseep\out\1、2020-自动化大创-报销清单（一式两份）-已填写.xlsx"

# ---- batches, each verified against its invoice PDFs (价税合计 is authoritative) ----
# batch A: 2026-05-27, 9 invoices (详见第一批 PDF)
BATCH_A_COUNT, BATCH_A_TOTAL = 9, 1461.75
# batch B: 2026-07-28, 2 invoices from 潍坊朗途: 366.60 + 246.00
BATCH_B_COUNT, BATCH_B_TOTAL = 2, 612.60

TOTAL = round(BATCH_A_TOTAL + BATCH_B_TOTAL, 2)
COUNT = BATCH_A_COUNT + BATCH_B_COUNT

STUDENT = "刘鑫宇"
STU_ID = "2023212129"
PHONE = "18445450819"
DATE_TEXT = "2026年7月28日"  # latest invoice date

assert TOTAL == 2074.35 and COUNT == 11, (TOTAL, COUNT)

shutil.copyfile(SRC, OUT)
wb = load_workbook(OUT, data_only=False)
ws = wb["Sheet1"]

# header date
ws["D2"] = f"日期：{DATE_TEXT}"

# 材料费 row: invoice count + amount
ws["C5"] = COUNT
ws["D5"] = TOTAL
ws["D5"].number_format = "0.00"

# 报销至学生 (row 9)
ws["B9"] = STUDENT
ws["C9"] = STU_ID
ws["D9"] = PHONE
ws["E9"] = TOTAL
ws["E9"].number_format = "0.00"
for ref in ("C9", "D9"):
    ws[ref].number_format = "@"

# 合计
ws["B21"] = TOTAL
ws["B21"].number_format = "0.00"

wb.save(OUT)
print(f"saved: {OUT}")
print(f"count={COUNT} total={TOTAL} (A={BATCH_A_TOTAL} + B={BATCH_B_TOTAL})")
