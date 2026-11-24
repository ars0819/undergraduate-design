param(
  [Parameter(Mandatory=$true)][string]$Source,
  [Parameter(Mandatory=$true)][string]$Target
)
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Source, $false, $true)
    if ($null -eq $doc) { throw "open returned null" }
    $doc.ExportAsFixedFormat($Target, 17)   # 17 = wdExportFormatPDF
    Write-Output ("PAGES=" + $doc.ComputeStatistics(2))
    $doc.Close(0)
    Write-Output "SAVED"
}
finally {
    try { $word.Quit(0) } catch {}
}
