# -*- coding: utf-8 -*-
"""Validate relative links, image references, archive hashes and key files.

Usage:  python tools/verify_hub.py
Exit:   0 = passed (warnings allowed), 1 = errors
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

HUB = Path(__file__).resolve().parents[1]
REPORT = HUB / "90_Archive" / "Migration_Reports" / "verify_report.txt"

KEY_FILES = [
    "AGENTS.md",
    "README.md",
    "00_Home/README.md",
    "00_Home/使用手册.md",
    "00_Home/Dashboard.md",
    "00_Home/Current_Status.md",
    "00_Home/Next_Actions.md",
    "00_Home/Decisions.md",
    "01_Roadmap/Current_Roadmap.md",
    "01_Roadmap/Skills_Map.md",
    "01_Roadmap/Foundation_Plan.md",
    "01_Roadmap/VLA_Inference_Track.md",
    "01_Roadmap/Long_Term_Research.md",
    "01_Roadmap/Original_V3_Comparison.md",
    "90_Archive/Original_Documents/SOURCE_MANIFEST.md",
    "90_Archive/Original_Documents/SHA256SUMS.txt",
    "90_Archive/Migration_Reports/迁移与完整性报告.md",
]

SOURCE_EXPECT = {
    Path(r"D:\Desktop\机器人学习与具身智能科研路线_V3.0_简化执行版.docx"):
        "8c1e4c095422fcd83f55c1e76af805cf9ff8ce7170460543367d9d5e20154ee3",
    Path(r"D:\Desktop\926150.docx"):
        "05c1e7311cbc24a274dbb2fae12bf405b5e30ffae9145bb6f0ffc6286536d1e0",
    Path(r"D:\Desktop\机器人学习与具身智能科研路线_表格版.xlsx"):
        "00c5460cacb220e317b75683fd03326234da0e72220bde8206946f790db82dec",
}

LINK_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)
SUMS_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_link_targets(path: Path):
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK_RE.finditer(line):
            raw = m.group(1).strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1].strip()
            yield lineno, raw


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in KEY_FILES:
        if not (HUB / rel).is_file():
            errors.append(f"missing key file: {rel}")

    md_files = sorted(HUB.rglob("*.md"))
    if not md_files:
        errors.append("no Markdown files found")

    link_count = 0
    image_refs = 0
    python_blocks = 0
    for md in md_files:
        lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
        fence_lines = [i for i, line in enumerate(lines, 1) if line.startswith("```")]
        if len(fence_lines) % 2:
            errors.append(f"odd code-fence count: {md.relative_to(HUB).as_posix()} ({len(fence_lines)})")
        in_fence = False
        lang = ""
        buf: list[str] = []
        for i, line in enumerate(lines, 1):
            if line.startswith("```"):
                if not in_fence:
                    in_fence = True
                    lang = line[3:].strip()
                    buf = []
                else:
                    if lang.lower() == "python":
                        python_blocks += 1
                        try:
                            ast.parse("\n".join(buf))
                        except SyntaxError as e:
                            errors.append(
                                f"python syntax error: {md.relative_to(HUB).as_posix()}:{i} block: {e.msg}"
                            )
                    in_fence = False
                    lang = ""
                    buf = []
                continue
            if in_fence:
                buf.append(line)
        for lineno, raw in iter_link_targets(md):
            if not raw or raw.startswith("#"):
                continue
            parsed = urlparse(raw)
            if parsed.scheme in {"http", "https", "mailto", "tel", "data", "file"}:
                continue
            if re.match(r"^[A-Za-z]:[\\/]", raw):
                # External absolute Windows path; allowed in text, not a portable link.
                warnings.append(f"{md.relative_to(HUB).as_posix()}:{lineno}: external absolute path link: {raw}")
                continue
            target = unquote(parsed.path).replace("\\", "/")
            if not target:
                continue
            link_count += 1
            candidate = (md.parent / target).resolve()
            if not candidate.exists():
                errors.append(f"broken link: {md.relative_to(HUB).as_posix()}:{lineno} -> {raw}")
            if target.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")):
                image_refs += 1

    # Archive hash manifest (paths are relative to the manifest's directory).
    sums = HUB / "90_Archive" / "Original_Documents" / "SHA256SUMS.txt"
    hash_lines = 0
    if not sums.is_file():
        errors.append("missing SHA256SUMS.txt")
    else:
        for line in sums.read_text(encoding="utf-8").splitlines():
            m = SUMS_RE.match(line.strip())
            if not m:
                if line.strip():
                    warnings.append(f"unparsed hash line: {line[:80]}")
                continue
            expected, rel = m.group(1), m.group(2)
            p = HUB / rel
            if not p.is_file():
                errors.append(f"hash manifest missing file: {rel}")
                continue
            actual = sha256(p)
            if actual != expected:
                errors.append(f"hash mismatch: {rel}")
            hash_lines += 1

    # Originals are optional (they live outside the hub) but should match if present.
    for src, expected in SOURCE_EXPECT.items():
        if src.exists():
            actual = sha256(src)
            if actual != expected:
                errors.append(f"source changed: {src}")
        else:
            warnings.append(f"source not present now (archive still valid): {src}")

    # Source counts / coverage checks.
    excel_extract = HUB / "90_Archive" / "Old_Roadmaps" / "Excel_七工作表结构化提取.md"
    if excel_extract.is_file():
        text = excel_extract.read_text(encoding="utf-8")
        for name in ["总览", "32周主线", "前14天", "课程资源", "数学与MLS", "框架与CS课程", "验收与自测"]:
            if f"## 工作表：{name}" not in text:
                errors.append(f"Excel extract missing sheet: {name}")
    else:
        errors.append("missing Excel extract")

    v3_extract = HUB / "90_Archive" / "Old_Roadmaps" / "V3.0_完整机械提取.md"
    if v3_extract.is_file():
        text = v3_extract.read_text(encoding="utf-8")
        # 32 weekly rows are present as "| 01 |" ... "| 32 |" in the comparison, not extract;
        # check the original 32-week table is present at all.
        if "32周整体科研路线" not in text:
            errors.append("V3 extract missing 32-week section")
        if len(set(re.findall(r"<!-- (P\d{4})", text))) != 413:
            errors.append("V3 extract does not contain 413 distinct paragraph markers")
        if len(re.findall(r"<!-- TABLE \d+ -->", text)) != 17:
            errors.append("V3 extract does not contain 17 tables")
        if len(re.findall(r"^#### D\d+", text, re.M)) != 14:
            errors.append("V3 extract does not contain 14 day headings")
    else:
        errors.append("missing V3 extract")

    notes_extract = HUB / "90_Archive" / "Original_Documents" / "926150_原始笔记逐段提取.md"
    if notes_extract.is_file():
        text = notes_extract.read_text(encoding="utf-8")
        if len(set(re.findall(r"<!-- (P\d{4})", text))) != 473:
            errors.append("notes extract does not contain 473 distinct paragraph markers")
        if len(re.findall(r"!\[", text)) != 5:
            errors.append("notes extract does not contain 5 image references")
    else:
        errors.append("missing notes extract")

    comparison = HUB / "01_Roadmap" / "Original_V3_Comparison.md"
    if comparison.is_file():
        text = comparison.read_text(encoding="utf-8")
        if len(re.findall(r"^\| \d{2} \|", text, re.M)) != 32:
            errors.append("comparison does not contain 32 weekly rows")
    else:
        errors.append("missing V3 comparison")

    status = HUB / "00_Home" / "Current_Status.md"
    if status.is_file():
        st = status.read_text(encoding="utf-8")
        for word in ["未开始", "学习中", "已接触但待复习", "已通过独立验收", "暂缓"]:
            if word not in st:
                errors.append(f"Current_Status missing status word: {word}")

    image_dir = HUB / "99_Assets" / "Images"
    image_files = [p for p in image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}] if image_dir.is_dir() else []
    if len(image_files) != 5:
        errors.append(f"expected 5 learning images, found {len(image_files)}")

    large = [p for p in HUB.rglob("*") if p.is_file() and p.stat().st_size > 20 * 1024 * 1024]
    for p in large:
        warnings.append(f"large file in hub (>20 MiB): {p.relative_to(HUB)}")

    # Cheap secret-pattern check.
    secret_patterns = ["BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "AKIA"]
    for md in md_files:
        text = md.read_text(encoding="utf-8", errors="replace")
        for pat in secret_patterns:
            if pat in text:
                errors.append(f"possible secret pattern {pat!r} in {md.relative_to(HUB)}")

    lines = [
        "Embodied-AI-Research-Hub verification",
        f"time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"hub: {HUB}",
        f"markdown files: {len(md_files)}",
        f"relative links checked: {link_count}",
        f"image references: {image_refs}",
        f"python code blocks checked: {python_blocks}",
        f"hash entries checked: {hash_lines}",
        f"errors: {len(errors)}",
        f"warnings: {len(warnings)}",
        "",
    ]
    if errors:
        lines += ["ERRORS"] + [f"- {e}" for e in errors] + [""]
    if warnings:
        lines += ["WARNINGS"] + [f"- {w}" for w in warnings] + [""]
    if not errors:
        lines += ["RESULT: PASS", ""]
    report = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
