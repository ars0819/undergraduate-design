# -*- coding: utf-8 -*-
"""Build the ordered invoice list, then merge into one PDF."""
import os
import sys

sys.path.insert(0, r"D:\Desktop\dickseep")
from merge_pdfs import merge

ZIP1 = r"D:\Desktop\dickseep\work\zip"
ZIP2 = r"D:\Desktop\dickseep\work\zip2"
OUT = r"D:\Desktop\dickseep\out\材料费发票合并-11张-2074.35元.pdf"

# (date, amount, seller, invoice_no, filename) -- order: date asc, amount desc
INVOICES = [
    ("2026-05-22", "90.00", "资兴市极脉控模型店", "26432000001155188101", ZIP1),
    ("2026-05-25", "370.00", "武汉酷点机器人科技有限公司", "26422000001843792021", ZIP1),
    ("2026-05-25", "42.80", "深圳市天航世邦科技有限公司", "26952000002150270851", ZIP1),
    ("2026-05-25", "39.60", "烟台塔克电子科技有限公司", "26372000002442814951", ZIP1),
    ("2026-05-25", "36.16", "深圳市优维客科技有限公司", "26952000002149565131", ZIP1),
    ("2026-05-25", "28.90", "深圳市华燕品冠科技有限公司", "26952000002142529411", ZIP1),
    ("2026-05-25", "9.20", "武汉酷点机器人科技有限公司", "26422000001844364376", ZIP1),
    ("2026-05-26", "720.00", "东莞快马模型开发有限公司", "26442000005814129826", ZIP1),
    ("2026-05-27", "125.09", "杭州炟成科技有限公司", "26332000004444834906", ZIP1),
    ("2026-07-28", "366.60", "潍坊朗途科技有限公司", "26372000003682690861", ZIP2),
    ("2026-07-28", "246.00", "潍坊朗途科技有限公司", "26372000003681578386", ZIP2),
]


def resolve(folder, amount, invoice_no):
    for name in os.listdir(folder):
        if not name.lower().endswith(".pdf"):
            continue
        if amount in name and name.startswith(tuple("0123456789")):
            return os.path.join(folder, name)
    raise FileNotFoundError((folder, amount))


def main():
    paths, total = [], 0.0
    print(f"{'#':>3} {'日期':<11} {'金额':>8}  {'发票号码':<22} 文件")
    for i, (date, amount, seller, no, folder) in enumerate(INVOICES, 1):
        p = resolve(folder, amount, no)
        paths.append(p)
        total += float(amount)
        print(f"{i:>3} {date:<11} {amount:>8}  {no:<22} {os.path.basename(p)}")
    print(f"\n合计 {len(paths)} 张 / {total:.2f} 元")
    assert len(paths) == 11, len(paths)
    assert abs(total - 2074.35) < 1e-9, total
    n, size, xp = merge(paths, OUT)
    print(f"merged -> {OUT}\npages={n} objects={size} xref_at={xp}")


if __name__ == "__main__":
    main()
