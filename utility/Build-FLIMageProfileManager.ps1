# Build FLIMageProfileManager.exe from FLIMageProfileManager.ps1 (PS2EXE)
$ErrorActionPreference = 'Stop'

$utilityDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$inputPs1 = Join-Path $utilityDir 'FLIMageProfileManager.ps1'
$outputExe = Join-Path $utilityDir 'FLIMageProfileManager.exe'

if (-not (Test-Path -LiteralPath $inputPs1)) {
    throw "Missing $inputPs1. Run Extract-FLIMageProfileManagerPs1.ps1 once if migrating from legacy .bat."
}

if (-not (Get-Module -ListAvailable -Name ps2exe)) {
    Write-Host 'Installing PS2EXE module (CurrentUser)...'
    Install-Module -Name ps2exe -Scope CurrentUser -Force -AllowClobber
}
Import-Module ps2exe -Force

Write-Host "Building $outputExe ..."
Invoke-ps2exe `
    -inputFile $inputPs1 `
    -outputFile $outputExe `
    -noConsole `
    -STA `
    -title 'FLIMage Profile Manager' `
    -description 'FLIMage profile backup and restore utility' `
    -company 'FLIMage' `
    -product 'FLIMage Profile Manager' `
    -version '1.0.0.0'

if (-not (Test-Path -LiteralPath $outputExe)) {
    throw "Build failed: $outputExe was not created."
}

$sizeMb = [math]::Round((Get-Item -LiteralPath $outputExe).Length / 1MB, 2)
Write-Host "Done: $outputExe ($sizeMb MB)"
