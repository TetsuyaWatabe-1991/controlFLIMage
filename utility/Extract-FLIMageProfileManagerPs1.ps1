# One-time helper: extract PowerShell body from legacy .bat into .ps1
$ErrorActionPreference = 'Stop'
$utilityDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat = Join-Path $utilityDir 'FLIMageProfileManager.bat'
$out = Join-Path $utilityDir 'FLIMageProfileManager.ps1'
$skip = 14

$lines = [System.IO.File]::ReadAllLines($bat)
if ($lines.Length -le $skip) {
    throw 'BAT header too short.'
}
$code = $lines[$skip..($lines.Length - 1)]

$bootstrap = @'
# FLIMage Profile Manager
# Source for FLIMageProfileManager.ps1 / .exe (build with Build-FLIMageProfileManager.ps1)
if ([string]::IsNullOrWhiteSpace($env:FLIM_PROFILE_DIR)) {
    $scriptRoot = $PSScriptRoot
    if (-not $scriptRoot -and $MyInvocation.MyCommand.Path) {
        $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    if ($scriptRoot) {
        $env:FLIM_PROFILE_DIR = $scriptRoot.TrimEnd('\')
    }
}

'@

$enc = New-Object System.Text.UTF8Encoding $true
[System.IO.File]::WriteAllText($out, $bootstrap + ($code -join [Environment]::NewLine), $enc)
Write-Host "Wrote $out ($((Get-Content -LiteralPath $out).Count) lines)"
