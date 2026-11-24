param(
  [string]$IdList = "",
  [string]$OutJson = "D:\Desktop\dickseep\work\arxiv_abs.json",
  [switch]$Digest
)

$Ids = @($IdList -split "[,\s]+" | Where-Object { $_ -ne "" })
$store = @{}
if (Test-Path $OutJson) {
  try {
    $arr = @(Get-Content $OutJson -Raw | ConvertFrom-Json)
    foreach ($r in $arr) { $store[$r.id] = $r }
  } catch { }
}

$ua = "Mozilla/5.0 (research-literature-audit)"
$kw = '(?i)(%|×|x faster|\bHz\b|\bms\b|\bGB\b|\bMB\b|latency|speedup|throughput|memory|param|billion|state-of-art|state of the art|success rate|success)'

foreach ($id in $Ids) {
  if ($store.ContainsKey($id) -and $Digest) { }
  if (-not $store.ContainsKey($id)) {
    try {
      $resp = Invoke-WebRequest -Uri "https://arxiv.org/abs/$id" -TimeoutSec 45 -UseBasicParsing -Headers @{ "User-Agent" = $ua } -ErrorAction Stop
      $html = $resp.Content
    } catch {
      Write-Output "FAIL $id :: $($_.Exception.Message)"
      continue
    }
    $m = [regex]::Match($html, '(?s)<blockquote class="abstract[^"]*">(.*?)</blockquote>')
    $abs = ""
    if ($m.Success) {
      $abs = $m.Groups[1].Value -replace '(?s)<span class="descriptor">.*?</span>', ''
      $abs = $abs -replace '<[^>]+>', ' '
      $abs = [System.Net.WebUtility]::HtmlDecode($abs)
      $abs = ($abs -replace '\s+', ' ').Trim()
    }
    if ([string]::IsNullOrWhiteSpace($abs)) { Write-Output "WARN $id :: no abstract found"; continue }
    $store[$id] = [pscustomobject]@{ id = $id; abstract = $abs }
    Start-Sleep -Milliseconds 700
  }
  if ($Digest) {
    $abs = $store[$id].abstract
    $sents = $abs -split '(?<=[\.\?])\s+'
    $hit = @($sents | Where-Object { $_ -match $kw -and $_.Length -gt 25 })
    if ($hit.Count -eq 0) { $hit = @($sents | Select-Object -First 2) }
    $line = ($hit -join " ")
    if ($line.Length -gt 1100) { $line = $line.Substring(0, 1100) + " ..." }
    Write-Output ("=== {0} ===" -f $id)
    Write-Output $line
    Write-Output ""
  } else {
    Write-Output ("stored {0} ({1} chars)" -f $id, $store[$id].abstract.Length)
  }
}

@($store.Values) | ConvertTo-Json -Depth 4 | Set-Content -Path $OutJson -Encoding UTF8
Write-Output ("--- total records: {0}" -f $store.Count)
