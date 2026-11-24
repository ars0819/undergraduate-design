param(
  [string]$Url,
  [string]$OutTxt,
  [string]$Keywords = "chunk,open-loop,execution length,replan,latency,throughput,kernel",
  [int]$MaxSnippets = 25
)

$ua = "Mozilla/5.0 (research-audit)"
try {
  $resp = Invoke-WebRequest -Uri $Url -TimeoutSec 90 -UseBasicParsing -Headers @{ "User-Agent" = $ua } -ErrorAction Stop
  $html = $resp.Content
} catch {
  Write-Output "FETCH_FAIL $Url :: $($_.Exception.Message)"
  exit 1
}

Write-Output ("BYTES {0}" -f $html.Length)

# --- headings before stripping ---
$hits = [regex]::Matches($html, '(?is)<h([123])[^>]*>(.*?)</h\1>')
$heads = @()
foreach ($m in $hits) {
  $t = ($m.Groups[2].Value -replace '(?s)<[^>]+>', ' ')
  $t = [System.Net.WebUtility]::HtmlDecode($t)
  $t = ($t -replace '\s+', ' ').Trim()
  if ($t.Length -gt 0 -and $t.Length -lt 160) { $heads += ("h" + $m.Groups[1].Value + "  " + $t) }
}
Write-Output ("---- HEADINGS ({0}) ----" -f $heads.Count)
$heads | Select-Object -First 80 | ForEach-Object { Write-Output $_ }

# --- plain text ---
$t = $html
$t = $t -replace '(?s)<script.*?</script>', ' '
$t = $t -replace '(?s)<style.*?</style>', ' '
$t = $t -replace '(?s)<math.*?</math>', ' [MATH] '
$t = $t -replace '(?s)<table.*?</table>', "`n[TABLE]`n"
$t = $t -replace '<[^>]+>', "`n"
$t = [System.Net.WebUtility]::HtmlDecode($t)
$lines = @($t -split "`n" | ForEach-Object { ($_ -replace '\s+', ' ').Trim() } | Where-Object { $_.Length -gt 1 })
if ($OutTxt) { $lines | Set-Content -Path $OutTxt -Encoding UTF8 }
Write-Output ("---- TEXT LINES {0} -> {1} ----" -f $lines.Count, $OutTxt)

# --- keyword hits ---
$kws = @($Keywords -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
foreach ($k in $kws) {
  $m = @($lines | Select-String -Pattern ([regex]::Escape($k)) -SimpleMatch)
  Write-Output ("==== KW '{0}' -> {1} hits" -f $k, $m.Count)
  $n = 0
  foreach ($x in $m) {
    if ($n -ge $MaxSnippets) { Write-Output "   ... (truncated)"; break }
    $s = $x.Line
    if ($s.Length -gt 300) { $s = $s.Substring(0, 300) + "..." }
    Write-Output ("   " + $s)
    $n++
  }
}
