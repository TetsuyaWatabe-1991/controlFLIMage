# Run automated bead FWHM tests (no user interaction required).

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

Write-Host "=== Running bead FWHM test suite ===" -ForegroundColor Cyan

$env:BEAD_FWHM_DATA_DIR = Join-Path $here "self_test_output"
New-Item -ItemType Directory -Force -Path $env:BEAD_FWHM_DATA_DIR | Out-Null

$pythonCandidates = @(
    $env:PYTHON,
    "C:\Users\yasudalab\Documents\Tetsuya_GIT\deepd3\Scripts\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
) | Where-Object { $_ -and (Test-Path $_) }

$python = $null
foreach ($cand in $pythonCandidates) {
    $ok = & $cand -c "import numpy, matplotlib, scipy, PyQt5, tifffile" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $python = $cand
        break
    }
}
if (-not $python) {
    Write-Error "No Python with numpy/matplotlib/scipy/PyQt5/tifffile was found."
    exit 1
}
Write-Host "Using Python: $python"

& $python (Join-Path $here "test_bead_fwhm.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nRunning GUI script self-test (BEAD_FWHM_SELF_TEST=1)..." -ForegroundColor Cyan
$env:BEAD_FWHM_SELF_TEST = "1"
& $python (Join-Path $here "bead_fwhm_gui.py")
$code = $LASTEXITCODE
Remove-Item Env:BEAD_FWHM_SELF_TEST -ErrorAction SilentlyContinue
if ($code -ne 0) { exit $code }

Write-Host "`nAll bead FWHM tests passed." -ForegroundColor Green
exit 0
