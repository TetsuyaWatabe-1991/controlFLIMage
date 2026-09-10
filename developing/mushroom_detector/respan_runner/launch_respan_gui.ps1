# Launch RESPAN GUI using the lab conda environments.
$ErrorActionPreference = "Stop"

$CondaRoot = Join-Path $env:LOCALAPPDATA "miniconda3"
$PythonGpu = Join-Path $CondaRoot "envs\respan_gpu\python.exe"
$RunnerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Workspace = (Resolve-Path (Join-Path $RunnerDir "..\..\..\..")).Path

$RespanRoot = $env:RESPAN_ROOT
if (-not $RespanRoot) {
    $ThirdParty = Join-Path $Workspace "third_party\RESPAN"
    $Legacy = Join-Path $Workspace "ongoing\RESPAN"
    if (Test-Path $ThirdParty) { $RespanRoot = $ThirdParty }
    elseif (Test-Path $Legacy) { $RespanRoot = $Legacy }
    else { throw "RESPAN clone not found. Set RESPAN_ROOT or clone under third_party\RESPAN." }
}

$GuiScript = Join-Path $RespanRoot "RESPAN\Scripts\RESPAN_GUI_DIST.py"

if (-not (Test-Path $PythonGpu)) {
    throw "respan_gpu not found at $PythonGpu. Run setup first."
}
if (-not (Test-Path $GuiScript)) {
    throw "GUI script not found at $GuiScript"
}

$CudaHome = Join-Path $CondaRoot "envs\respan_gpu\Library"
if (Test-Path $CudaHome) {
    $env:CUDA_PATH = $CudaHome
    $env:PATH = "$(Join-Path $CudaHome 'bin');$env:PATH"
}

Set-Location $RespanRoot
& $PythonGpu $GuiScript
