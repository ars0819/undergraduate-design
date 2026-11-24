param(
  [string]$IdList = "",
  [string]$OutJson = "D:\Desktop\dickseep\work\arxiv_meta.json",
  [switch]$Append
)

$Ids = @($IdList -split "[,\s]+" | Where-Object { $_ -ne "" })

$results = @()
if ($Append -and (Test-Path $OutJson)) {
  try {
    $raw = Get-Content $OutJson -Raw
    $raw = $raw.TrimStart([char]0xFEFF)
    $results = @($raw | ConvertFrom-Json)
  } catch { $results = @() }
}
$known = @{}
foreach ($r in $results) { $known[$r.id] = $true }

$ua = "Mozilla/5.0 (research-literature-audit)"

foreach ($id in $Ids) {
  if ($known.ContainsKey($id)) { Write-Output "SKIP(already) $id"; continue }
  $url = "https://arxiv.org/abs/$id"
  try {
    $resp = Invoke-WebRequest -Uri $url -TimeoutSec 45 -UseBasicParsing -Headers @{ "User-Agent" = $ua } -ErrorAction Stop
    $html = $resp.Content
  } catch {
    Write-Output "FAIL $id :: $($_.Exception.Message)"
    $results += [pscustomobject]@{ id = $id; status = "FETCH_FAIL"; error = $_.Exception.Message }
    continue
  }

  function Get-Metas([string]$h, [string]$name) {
    $ms = [regex]::Matches($h, '<meta\s+name="' + [regex]::Escape($name) + '"\s+content="([^"]*)"')
    return @($ms | ForEach-Object { $_.Groups[1].Value })
  }

  $title  = (Get-Metas $html "citation_title") | Select-Object -First 1
  $date   = (Get-Metas $html "citation_date") | Select-Object -First 1
  $auths  = (Get-Metas $html "citation_author")
  $absid  = (Get-Metas $html "citation_arxiv_id") | Select-Object -First 1
  $doi    = (Get-Metas $html "citation_doi") | Select-Object -First 1

  # submission history / comments block
  $cmt = ""
  $m = [regex]::Match($html, '(?s)<td class="tablecell comments[^"]*">(.*?)</td>')
  if ($m.Success) { $cmt = ($m.Groups[1].Value -replace '<[^>]+>', ' ' -replace '\s+', ' ').Trim() }
  $jref = ""
  $mj = [regex]::Match($html, '(?s)<td class="tablecell jref">(.*?)</td>')
  if ($mj.Success) { $jref = ($mj.Groups[1].Value -replace '<[^>]+>', ' ' -replace '\s+', ' ').Trim() }
  $subj = ""
  $ms2 = [regex]::Match($html, '(?s)<td class="tablecell subjects">(.*?)</td>')
  if ($ms2.Success) { $subj = ($ms2.Groups[1].Value -replace '<[^>]+>', ' ' -replace '\s+', ' ').Trim() }

  if ([string]::IsNullOrWhiteSpace($title)) {
    Write-Output "WARN $id :: no citation_title (maybe withdrawn/redirect)"
    $results += [pscustomobject]@{ id = $id; status = "NO_META" }
    continue
  }

  $obj = [pscustomobject]@{
    id        = $id
    status    = "OK"
    arxiv_id  = $absid
    title     = ($title -replace '\s+', ' ').Trim()
    date      = $date
    doi       = $doi
    authors   = @($auths)
    n_authors = @($auths).Count
    comments  = $cmt
    jref      = $jref
    subjects  = $subj
    url       = $url
  }
  $results += $obj
  $a2 = (@($auths) | Select-Object -First 3) -join ", "
  Write-Output ("OK  {0} | {1} | {2} auth | {3} | {4}" -f $id, $date, @($auths).Count, $a2, $obj.title)
  if ($cmt) { Write-Output ("     comments: " + $cmt) }
  Start-Sleep -Milliseconds 2000
}

$results | ConvertTo-Json -Depth 4 | Set-Content -Path $OutJson -Encoding UTF8
Write-Output ("--- saved {0} records -> {1}" -f $results.Count, $OutJson)
