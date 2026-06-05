$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'FLIMageProfileManager.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing $exe" }
$p = Start-Process -FilePath $exe -PassThru
Start-Sleep -Seconds 2
if ($p.HasExited) { throw "EXE exited early with code $($p.ExitCode)" }
$p.CloseMainWindow() | Out-Null
Start-Sleep -Milliseconds 500
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
Write-Host 'EXE smoke test OK'
