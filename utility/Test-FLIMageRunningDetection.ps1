# Quick check: Profile Manager exe must not count as FLIMage running
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FLIMageProfileManager.ps1') -ErrorAction SilentlyContinue 2>$null

# Dot-sourcing would run the GUI; call only the function by parsing is heavy.
# Inline the fixed logic for a standalone test:
function Test-FLIMageRunning {
    $selfPid = $PID
    $null -ne (Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Id -ne $selfPid -and ($_.ProcessName -eq 'FLIMage' -or $_.ProcessName -eq 'FLIMage2')
    })
}

$matches = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'FLIMage*' } | Select-Object -ExpandProperty ProcessName -Unique
Write-Host "Processes matching FLIMage* (old pattern): $($matches -join ', ')"
Write-Host "Test-FLIMageRunning (new): $(Test-FLIMageRunning)"
if (Test-FLIMageRunning) {
    throw 'Expected false when only Profile Manager / no FLIMage app is open'
}
Write-Host 'OK'
