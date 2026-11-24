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
    if ($null -eq $doc) { throw "Open returned null for $Source" }
    $pages = $doc.ComputeStatistics(2)
    Write-Output "PAGES=$pages"
    $doc.SaveAs2($Target, 16)
    Write-Output "SAVED=$Target"
    $doc.Close(0)
}
finally {
    try { $word.Quit(0) } catch {}
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
