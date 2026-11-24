param(
  [string]$MetaJson = "D:\Desktop\dickseep\work\arxiv_meta.json",
  [string]$OutBib   = "D:\Desktop\dickseep\research\topics\references.bib"
)

$meta = @{}
foreach ($r in @(Get-Content $MetaJson -Raw | ConvertFrom-Json)) {
  if ($r.status -eq "OK") { $meta[$r.id] = $r }
}

# key | arxiv id | note (venue only when confirmed on the arXiv abstract page comments field)
$rows = @(
  @("openvla2024",            "2406.09246", "arXiv preprint"),
  @("openvlaoft2025",         "2502.19645", "Accepted to Robotics: Science and Systems (RSS) 2025"),
  @("pi0flow2024",            "2410.24164", "Published in RSS 2025"),
  @("pi05generalization2025", "2504.16054", "arXiv preprint"),
  @("fast2025",               "2501.09747", "arXiv preprint"),
  @("act2023",                "2304.13705", "RSS 2023"),
  @("diffusionpolicy2023",    "2303.04137", "RSS 2023; extended journal version"),
  @("octo2024",               "2405.12213", "arXiv preprint"),
  @("tinyvla2024",            "2409.12514", "arXiv preprint"),
  @("smolvla2025",            "2506.01844", "arXiv preprint"),
  @("rt1_2022",               "2212.06817", "arXiv preprint"),
  @("rt2_2023",               "2307.15818", "arXiv preprint"),
  @("rdt1b2024",              "2410.07864", "arXiv preprint"),
  @("cogact2024",             "2411.19650", "arXiv preprint"),
  @("spatialvla2025",         "2501.15830", "arXiv preprint"),
  @("roboflamingo2023",       "2311.01378", "arXiv preprint"),
  @("consistencypolicy2024",  "2405.07503", "arXiv preprint"),
  @("onestepdp2024",          "2410.21257", "arXiv preprint"),
  @("vlacache2025",           "2502.02175", "Accepted to NeurIPS 2025"),
  @("bitvla2025",             "2506.07530", "arXiv preprint (work in progress)"),
  @("kvefficientvla2025",     "2509.21354", "arXiv preprint"),
  @("spvla2025",              "2506.12723", "arXiv preprint"),
  @("sqapvla2025",            "2509.09090", "arXiv preprint"),
  @("qvla2026",               "2602.03782", "ICLR 2026"),
  @("quantvla2026",           "2602.20309", "CVPR 2026"),
  @("vlapruner2025",          "2511.16449", "arXiv preprint"),
  @("specvla2025",            "2507.22424", "Accepted by EMNLP 2025 (main conference)"),
  @("specvla_codesign2026",   "2608.15636", "arXiv preprint"),
  @("rtc2025",                "2506.07339", "Published in NeurIPS 2025"),
  @("pdvla2025",              "2503.02310", "Accepted by IROS 2025"),
  @("efficiencymetrics2026",  "2603.19131", "arXiv preprint"),
  @("robodual2024",           "2410.08001", "arXiv preprint"),
  @("dualvla2025",            "2511.22134", "arXiv preprint"),
  @("fastinslow2025",         "2506.01953", "arXiv preprint"),
  @("grootn1_2025",           "2503.14734", "arXiv preprint"),
  @("hirobot2025",            "2502.19417", "ICML 2025"),
  @("vlasurvey2024",          "2405.14093", "arXiv preprint"),
  @("efficientsurvey2025",    "2510.24795", "Accepted by IEEE TPAMI"),
  @("oxe2023",                "2310.08864", "arXiv preprint"),
  @("libero2023",             "2306.03310", "arXiv preprint"),
  @("simplerenv2024",         "2405.05941", "arXiv preprint"),
  @("droid2024",              "2403.12945", "arXiv preprint"),
  @("calvin2021",             "2112.03227", "IEEE Robotics and Automation Letters (RA-L)")
)

function BibEscape([string]$s) {
  $s = $s -replace '\\', '\\textbackslash{}'
  $s = $s -replace '([&%$#_])', '\$1'
  return $s
}

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("% ============================================================================")
[void]$sb.AppendLine("% references.bib")
[void]$sb.AppendLine("% 生成方式：由脚本从 arXiv 摘要页 citation_* 元数据自动生成，非手工抄录。")
[void]$sb.AppendLine("% 元数据核实日期：2026-11-01（Asia/Shanghai）")
[void]$sb.AppendLine("% 每条目字段来源：https://arxiv.org/abs/<eprint> 页面 meta 标签")
[void]$sb.AppendLine("% 注意：venue 仅在 arXiv 摘要页 comments 字段明确写出时才填写；")
[void]$sb.AppendLine("%       未确认的录用信息一律写 arXiv preprint，避免虚构出处。")
[void]$sb.AppendLine("% ============================================================================")
[void]$sb.AppendLine("")

$missing = @()
foreach ($row in $rows) {
  $key = $row[0]; $id = $row[1]; $note = $row[2]
  if (-not $meta.ContainsKey($id)) { $missing += "$key ($id)"; continue }
  $m = $meta[$id]

  $auths = @($m.authors | Where-Object { $_ -ne "" })
  # 去掉 arXiv 页面里把机构名当作作者的错误切分（如 "NVIDIA, :"）
  $auths = @($auths | Where-Object { $_ -ne ":" -and $_ -notmatch '^[\s:]*$' })
  $use = @()
  $i = 0
  foreach ($a in $auths) { if ($i -lt 6) { $use += ($a -replace '\s+', ' ').Trim() }; $i++ }
  $authorField = ($use -join " and ")
  if ($auths.Count -gt 6) { $authorField += " and others" }
  $authorField = BibEscape $authorField

  $title = BibEscape (($m.title -replace '\s+', ' ').Trim())
  $year = ""
  if ($m.date -match '^(\d{4})') { $year = $matches[1] }

  $type = "misc"
  [void]$sb.AppendLine("@$type{$key,")
  [void]$sb.AppendLine("  author        = {$authorField},")
  [void]$sb.AppendLine("  title         = {{$title}},")
  [void]$sb.AppendLine("  year          = {$year},")
  [void]$sb.AppendLine("  eprint        = {$id},")
  [void]$sb.AppendLine("  archivePrefix = {arXiv},")
  [void]$sb.AppendLine("  primaryClass  = {cs.RO},")
  [void]$sb.AppendLine("  doi           = {10.48550/arXiv.$id},")
  [void]$sb.AppendLine("  url           = {https://arxiv.org/abs/$id},")
  [void]$sb.AppendLine("  note          = {$note},")
  [void]$sb.AppendLine("}")
  [void]$sb.AppendLine("")
}

# 非 arXiv 条目：平台与工具链（手工核实）
[void]$sb.AppendLine("% ---- 非 arXiv 条目（平台/工具链，仓库存在性经 GitHub API 核实）----")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("@misc{openflex_repo2026,")
[void]$sb.AppendLine("  author       = {{OpenFleX-Wheeled-Humanoid}},")
[void]$sb.AppendLine("  title        = {{OpenFleX}: Whole-Body Robot System Documentation (hardware control, {VR} teleoperation, {ros2\_control})},")
[void]$sb.AppendLine("  year         = {2026},")
[void]$sb.AppendLine("  howpublished = {\url{https://github.com/OpenFleX-Wheeled-Humanoid/OpenFleX}},")
[void]$sb.AppendLine("  note         = {Accessed 2026-11-01; repository last push 2026-10-22; license declared NOASSERTION. Technical-companion listing: \url{https://www.aiman.world/robot/openflex-lunshiyidongshuangbi}},")
[void]$sb.AppendLine("}")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("@misc{openpi2026,")
[void]$sb.AppendLine("  author       = {{Physical Intelligence}},")
[void]$sb.AppendLine("  title        = {{openpi}: Open-source repository for $\pi_0$, $\pi_{0.5}$ and related robot policies},")
[void]$sb.AppendLine("  year         = {2026},")
[void]$sb.AppendLine("  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},")
[void]$sb.AppendLine("  note         = {GitHub API checked 2026-11-01: exists, last push 2026-08-24},")
[void]$sb.AppendLine("}")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("@misc{lerobot2026,")
[void]$sb.AppendLine("  author       = {{Hugging Face}},")
[void]$sb.AppendLine("  title        = {{LeRobot}: State-of-the-art machine learning for real-world robotics},")
[void]$sb.AppendLine("  year         = {2026},")
[void]$sb.AppendLine("  howpublished = {\url{https://github.com/huggingface/lerobot}},")
[void]$sb.AppendLine("  note         = {GitHub API checked 2026-11-01: exists; hosts SmolVLA and $\pi_0$/$\pi_{0.5}$ checkpoints},")
[void]$sb.AppendLine("}")
[void]$sb.AppendLine("")

New-Item -ItemType Directory -Force -Path (Split-Path $OutBib) | Out-Null
$sb.ToString() | Set-Content -Path $OutBib -Encoding UTF8
Write-Output ("written: {0}  entries={1}" -f $OutBib, $rows.Count)
if ($missing.Count -gt 0) { Write-Output ("MISSING META: " + ($missing -join "; ")) }
