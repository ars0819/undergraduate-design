param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$Pages,
    [int]$Scale = 2.0
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$asTaskOp = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]
$asTaskAct = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction'
})[0]

function AwaitOp($op, $t) {
    $net = $asTaskOp.MakeGenericMethod($t).Invoke($null, @($op))
    $net.Wait(-1) | Out-Null
    $net.Result
}
function AwaitAct($act) {
    $net = $asTaskAct.Invoke($null, @($act))
    $net.Wait(-1) | Out-Null
}

[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStream, Windows.Storage, ContentType = WindowsRuntime] | Out-Null

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$bytes = [System.IO.File]::ReadAllBytes($Source)
$ms = New-Object System.IO.MemoryStream(, $bytes)
$ras = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($ms)
$pdf = AwaitOp ([Windows.Data.Pdf.PdfDocument]::LoadFromStreamAsync($ras)) ([Windows.Data.Pdf.PdfDocument])
Write-Output ("PAGECOUNT=" + $pdf.PageCount)

$pageList = @($Pages -split '[,; ]+' | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })
foreach ($p in $pageList) {
    $idx = $p - 1
    if ($idx -lt 0 -or $idx -ge $pdf.PageCount) { Write-Output "skip $p"; continue }
    $page = $pdf.GetPage($idx)
    $opts = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $opts.DestinationWidth = [uint32]([math]::Round($page.Size.Width * $Scale))
    $opts.DestinationHeight = [uint32]([math]::Round($page.Size.Height * $Scale))
    $out = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    AwaitAct ($page.RenderToStreamAsync($out, $opts))
    $size = [uint32]$out.Size
    $out.Seek(0)
    $reader = New-Object Windows.Storage.Streams.DataReader($out.GetInputStreamAt(0))
    AwaitOp ($reader.LoadAsync($size)) ([uint32]) | Out-Null
    $buf = New-Object byte[] $size
    $reader.ReadBytes($buf)
    $path = Join-Path $OutDir ("page_{0:d3}.png" -f $p)
    [System.IO.File]::WriteAllBytes($path, $buf)
    Write-Output ("wrote " + $path + "  " + $size + " bytes")
    $reader.Dispose(); $out.Dispose(); $page.Dispose()
}
$ras.Dispose(); $ms.Dispose()
