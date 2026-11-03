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

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:ProfilesBase = $null
$script:InitFilesPath = $null
$script:InitSubfolderExclude = @("Settings", "WindowsInfo", "Uncaging")
$script:ManagerConfigPath = $null
$script:LastMonthlyAutoBackup = $null
$script:PreserveUncagingCalibOnRestore = $true
$script:PreRestoreBackupMaxCount = 30
$script:ProfileHistoryMaxCount = 10
$script:ProfileHistoryFolderName = '_history'
$script:UncagingCalibSetupKeys = @('State.Uncaging.CalibV', 'State.Uncaging.Calib_beta')
# Never overwritten by Copy shared keys. Each file keeps its own value.
$script:SharedKeyCopyProtectedKeys = @('State.Acq.version')
# Setup keys that FLIMage updates per session (open/grab/stage) — omit from save_changelog.jsonl (see FLIMage2 ScanParameters / FLIMage_IO).
$script:ChangelogIgnoreSetupKeys = @(
    'State.Acq.triggerTime'
    'State.Acq.currentPositionPiezo_V'
    'State.Motor.motorPosition'
    'State.Motor.positionID'
    'State.Uncaging.currentPosition'
    'State.Ephys.currentPulseN'
    'State.Files.fileCounter'
    'State.Files.fileName'
    'State.Files.initFileName'
)
# Whole files that only reflect window layout / roaming UI state (not worth logging on every save).
$script:ChangelogIgnoreRelativePathPatterns = @(
    'Init_Files/WindowsInfo/*'
    'AppData_Roaming/*'
    'AppData_Local/*'
)

function Get-DefaultInitFilesPath {
    $docs = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
    Join-Path $docs "FLIMage\Init_Files"
}

function Get-DefaultProfilesBase {
    $docs = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
    Join-Path $docs "FLIMage\Profiles"
}

function Get-ManagerConfigPath {
    if ($script:ManagerConfigPath) { return $script:ManagerConfigPath }
    $dir = $env:FLIM_PROFILE_DIR
    if ([string]::IsNullOrWhiteSpace($dir)) {
        $dir = Get-DefaultProfilesBase
    }
    $script:ManagerConfigPath = Join-Path $dir "FLIMageProfileManager.config.json"
    return $script:ManagerConfigPath
}

function Load-ManagerConfig {
    $cfg = @{
        initFilesPath          = $null
        profilesBase           = $null
        lastMonthlyAutoBackup         = $null
        preserveUncagingCalibOnRestore = $null
    }
    $path = Get-ManagerConfigPath
    if (Test-Path -LiteralPath $path) {
        try {
            $loaded = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($loaded.initFilesPath) { $cfg.initFilesPath = [string]$loaded.initFilesPath }
            if ($loaded.profilesBase) { $cfg.profilesBase = [string]$loaded.profilesBase }
            if ($loaded.lastMonthlyAutoBackup) { $cfg.lastMonthlyAutoBackup = [string]$loaded.lastMonthlyAutoBackup }
            if ($null -ne $loaded.PSObject.Properties['preserveUncagingCalibOnRestore']) {
                $cfg.preserveUncagingCalibOnRestore = [bool]$loaded.preserveUncagingCalibOnRestore
            }
        } catch {
        }
    }
    return $cfg
}

function Save-ManagerConfig {
    $obj = [ordered]@{
        initFilesPath         = $script:InitFilesPath
        profilesBase          = $script:ProfilesBase
        lastMonthlyAutoBackup          = $script:LastMonthlyAutoBackup
        preserveUncagingCalibOnRestore = $script:PreserveUncagingCalibOnRestore
        saved_at                       = (Get-Date).ToUniversalTime().ToString("o")
    }
    $path = Get-ManagerConfigPath
    $parent = Split-Path -Parent $path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $utf8Bom = New-Object System.Text.UTF8Encoding $true
    [System.IO.File]::WriteAllText($path, ($obj | ConvertTo-Json), $utf8Bom)
}

function Get-InitFilesPath {
    if (-not [string]::IsNullOrWhiteSpace($script:InitFilesPath)) {
        return $script:InitFilesPath
    }
    Get-DefaultInitFilesPath
}

function Get-RoamingFlimagePath { Join-Path $env:APPDATA "FLIMage" }
function Get-LocalFlimagePath { Join-Path $env:LOCALAPPDATA "FLIMage" }

function Test-InitFilesPathValid {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }

    foreach ($name in @("FLIM_deviceFile_V2.txt", "FLIM_deviceFile_V1.txt", "FLIM_deviceFile.txt", "FLIM_init.txt")) {
        if (Test-Path -LiteralPath (Join-Path $Path $name)) { return $true }
    }
    if (Get-ChildItem -LiteralPath $Path -Filter "Default-*.txt" -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
        return $true
    }
    if (Test-Path -LiteralPath (Join-Path $Path "Settings") -PathType Container) { return $true }
    return $false
}

function Read-InitFolderPathFromFlimageSetup {
    param([string]$SearchDir)
    if (-not (Test-Path -LiteralPath $SearchDir -PathType Container)) { return $null }

    $filePaths = New-Object System.Collections.Generic.List[string]
    Get-ChildItem -LiteralPath $SearchDir -Filter "Default-*.txt" -File -ErrorAction SilentlyContinue | ForEach-Object {
        $filePaths.Add($_.FullName)
    }
    foreach ($name in @("FLIM_deviceFile_V2.txt", "FLIM_init.txt")) {
        $p = Join-Path $SearchDir $name
        if (Test-Path -LiteralPath $p) { $filePaths.Add($p) }
    }

    foreach ($filePath in $filePaths) {
        foreach ($line in [System.IO.File]::ReadLines($filePath)) {
            if ($line -match 'State\.Files\.initFolderPath\s*=\s*(.+?)\s*;?\s*$') {
                $value = $matches[1].Trim().Trim('"')
                if (-not [string]::IsNullOrWhiteSpace($value) -and (Test-Path -LiteralPath $value -PathType Container)) {
                    return $value
                }
            }
        }
    }
    return $null
}

function Resolve-InitFilesCandidate {
    param([string]$PreferredPath)

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) { $candidates.Add($PreferredPath) }

    $fromIni = Read-InitFolderPathFromFlimageSetup -SearchDir $PreferredPath
    if ($fromIni) { $candidates.Add($fromIni) }

    $candidates.Add((Get-DefaultInitFilesPath))

    $seen = @{}
    foreach ($c in $candidates) {
        if ([string]::IsNullOrWhiteSpace($c)) { continue }
        $norm = [System.IO.Path]::GetFullPath($c)
        if ($seen.ContainsKey($norm)) { continue }
        $seen[$norm] = $true
        if (Test-InitFilesPathValid $norm) { return $norm }
    }
    return $null
}

function Prompt-InitFilesPath {
    param([string]$TriedPath)

    $msg = "FLIMage Init_Files folder was not found or does not look valid."
    if (-not [string]::IsNullOrWhiteSpace($TriedPath)) {
        $msg += "`n`nChecked: $TriedPath"
    }
    $msg += "`n`nPlease select the FLIMage\Init_Files folder (e.g. ...\Documents\FLIMage\Init_Files)."
    $msg += "`n`nCancel to exit this tool."

    [System.Windows.Forms.MessageBox]::Show(
        $msg,
        "Init_Files not found",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    ) | Out-Null

    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = "Select FLIMage Init_Files folder"
    $guess = if ($TriedPath -and (Test-Path -LiteralPath $TriedPath)) { $TriedPath } else { Get-DefaultInitFilesPath }
    $parent = Split-Path -Parent $guess
    if ($parent -and (Test-Path -LiteralPath $parent)) { $dlg.SelectedPath = $parent }
    elseif (Test-Path -LiteralPath $guess) { $dlg.SelectedPath = $guess }

    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        return $null
    }
    return $dlg.SelectedPath
}

function Initialize-InitFilesPathAtStartup {
    $cfg = Load-ManagerConfig
    $script:ProfilesBase = if ($cfg.profilesBase) { $cfg.profilesBase } else { Get-DefaultProfilesBase }
    $script:LastMonthlyAutoBackup = $cfg.lastMonthlyAutoBackup
    if ($null -ne $cfg.preserveUncagingCalibOnRestore) {
        $script:PreserveUncagingCalibOnRestore = [bool]$cfg.preserveUncagingCalibOnRestore
    }
    $preferred = if ($cfg.initFilesPath) { $cfg.initFilesPath } else { Get-DefaultInitFilesPath }
    $resolved = Resolve-InitFilesCandidate -PreferredPath $preferred

    while (-not $resolved) {
        $picked = Prompt-InitFilesPath -TriedPath $preferred
        if (-not $picked) {
            exit 0
        }
        $preferred = $picked
        $resolved = Resolve-InitFilesCandidate -PreferredPath $preferred
        if (-not $resolved) {
            [System.Windows.Forms.MessageBox]::Show(
                "The selected folder does not look like a valid FLIMage Init_Files folder.`n`nExpected e.g. FLIM_deviceFile_V2.txt or Default-*.txt inside.",
                "Invalid Init_Files folder",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning
            ) | Out-Null
        }
    }

    $script:InitFilesPath = $resolved
    Save-ManagerConfig
}

function Get-ProfileRoot {
    param([string]$ProfileName)
    Join-Path $script:ProfilesBase $ProfileName
}

function Get-AutoBackupBase {
    $parent = Split-Path -Parent $script:ProfilesBase
    if ([string]::IsNullOrWhiteSpace($parent)) {
        $docs = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
        return (Join-Path (Join-Path $docs "FLIMage") "AutoBackup")
    }
    return (Join-Path $parent "AutoBackup")
}

function Get-AutoBackupRoot {
    param([string]$ProfileName)
    Join-Path (Get-AutoBackupBase) $ProfileName
}

function Resolve-ProfileStorageRoot {
    param([string]$ProfileName)
    if ([string]::IsNullOrWhiteSpace($ProfileName)) { return $null }

    $userRoot = Get-ProfileRoot $ProfileName
    if (Test-Path -LiteralPath $userRoot) {
        return @{ Root = $userRoot; Kind = 'user' }
    }

    $autoRoot = Get-AutoBackupRoot $ProfileName
    if (Test-Path -LiteralPath $autoRoot) {
        return @{ Root = $autoRoot; Kind = 'auto_backup' }
    }
    return $null
}

function Test-ProfilesFolderEmptyOrMissing {
    $base = $script:ProfilesBase
    if (-not (Test-Path -LiteralPath $base)) { return $true }
    $dirs = @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)
    return ($dirs.Count -eq 0)
}

function Test-AutoBackupBaseExists {
    Test-Path -LiteralPath (Get-AutoBackupBase) -PathType Container
}

function Test-ShouldRunProfilesEmptyAutoBackup {
    if (-not (Test-ProfilesFolderEmptyOrMissing)) { return $false }
    return -not (Test-AutoBackupBaseExists)
}

function Get-CurrentYearMonthKey {
    Get-Date -Format 'yyyy-MM'
}

function Test-HasMonthlyAutoBackupThisMonth {
    $base = Get-AutoBackupBase
    if (-not (Test-Path -LiteralPath $base)) { return $false }
    $prefix = "backup_$(Get-Date -Format 'yyyyMM')"
    return (@(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "$prefix*" })).Count -gt 0
}

function Test-ShouldRunMonthlyAutoBackup {
    if ($script:LastMonthlyAutoBackup -eq (Get-CurrentYearMonthKey)) { return $false }
    if (Test-HasMonthlyAutoBackupThisMonth) { return $false }
    return $true
}

function Set-MonthlyAutoBackupDone {
    $script:LastMonthlyAutoBackup = Get-CurrentYearMonthKey
    Save-ManagerConfig
}

function Get-NewAutoBackupProfileName {
    param([string]$NameStem = $null)

    $base = Get-AutoBackupBase
    if ([string]::IsNullOrWhiteSpace($NameStem)) {
        $NameStem = "backup_$(Get-Date -Format 'yyyyMMdd')"
    }
    if (-not (Test-Path -LiteralPath $base)) { return $NameStem }
    if (-not (Test-Path -LiteralPath (Join-Path $base $NameStem))) { return $NameStem }
    $n = 2
    while ($true) {
        $candidate = "${NameStem}_$n"
        if (-not (Test-Path -LiteralPath (Join-Path $base $candidate))) {
            return $candidate
        }
        $n++
    }
}

function Get-InitFilesBackupComponents {
    @{
        initRoot        = $true
        settings        = $true
        windowsInfo     = $true
        uncaging        = $true
        initOther       = $true
        appDataRoaming  = $false
        appDataLocal    = $false
    }
}

function Invoke-AutoBackupIfNeeded {
    if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) { return $null }
    if (Test-FLIMageRunning) { return $null }

    $autoBase = Get-AutoBackupBase
    New-Item -ItemType Directory -Path $autoBase -Force | Out-Null
    $components = Get-InitFilesBackupComponents

    if (Test-ShouldRunProfilesEmptyAutoBackup) {
        $name = Get-NewAutoBackupProfileName
        $root = Join-Path $autoBase $name
        Save-Profile -ProfileName $name -Components $components -StorageRoot $root -ProfileKind 'auto_backup'
        Set-MonthlyAutoBackupDone
        return $name
    }

    if (Test-ShouldRunMonthlyAutoBackup) {
        $stem = "backup_$(Get-Date -Format 'yyyyMM')"
        $name = Get-NewAutoBackupProfileName -NameStem $stem
        $root = Join-Path $autoBase $name
        Save-Profile -ProfileName $name -Components $components -StorageRoot $root -ProfileKind 'auto_backup'
        Set-MonthlyAutoBackupDone
        return $name
    }

    return $null
}

function Test-FLIMageRunning {
    # Match FLIMage app only (AssemblyName FLIMage). Do not use FLIMage* — it matches this tool's exe (FLIMageProfileManager).
    $selfPid = $PID
    $null -ne (Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Id -ne $selfPid -and ($_.ProcessName -eq 'FLIMage' -or $_.ProcessName -eq 'FLIMage2')
    })
}

function Set-CopiedItemTimestamps {
    param(
        [System.IO.FileSystemInfo]$SourceItem,
        [string]$DestinationPath
    )
    if ([string]::IsNullOrWhiteSpace($DestinationPath)) { return }
    if (-not (Test-Path -LiteralPath $DestinationPath)) { return }
    try {
        $created = $SourceItem.CreationTime
        $written = $SourceItem.LastWriteTime
        if ($SourceItem.PSIsContainer) {
            [System.IO.Directory]::SetCreationTime($DestinationPath, $created)
            [System.IO.Directory]::SetLastWriteTime($DestinationPath, $written)
        } else {
            [System.IO.File]::SetCreationTime($DestinationPath, $created)
            [System.IO.File]::SetLastWriteTime($DestinationPath, $written)
        }
    } catch {
        # Best effort. A read-only or locked item should not fail save or restore.
    }
}

function Sync-CopiedTreeTimestamps {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source)) { return }
    if (-not (Test-Path -LiteralPath $Destination)) { return }
    $srcRoot = Get-Item -LiteralPath $Source -Force
    $children = @(Get-ChildItem -LiteralPath $Source -Recurse -Force -ErrorAction SilentlyContinue |
        Sort-Object { $_.FullName.Length } -Descending)
    foreach ($item in $children) {
        $relative = $item.FullName.Substring($srcRoot.FullName.Length).TrimStart('\', '/')
        if ([string]::IsNullOrWhiteSpace($relative)) { continue }
        Set-CopiedItemTimestamps -SourceItem $item -DestinationPath (Join-Path $Destination $relative)
    }
    Set-CopiedItemTimestamps -SourceItem $srcRoot -DestinationPath $Destination
}

function Copy-Tree {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source)) { return }
    $parent = Split-Path -Parent $Destination
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    Sync-CopiedTreeTimestamps -Source $Source -Destination $Destination
}

function Copy-InitFilesRootOnly {
    param([string]$SourceDir, [string]$DestDir)
    if (-not (Test-Path -LiteralPath $SourceDir)) { return }
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    Get-ChildItem -LiteralPath $SourceDir -File -ErrorAction SilentlyContinue | ForEach-Object {
        $destPath = Join-Path $DestDir $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $destPath -Force
        Set-CopiedItemTimestamps -SourceItem $_ -DestinationPath $destPath
    }
    Set-CopiedItemTimestamps -SourceItem (Get-Item -LiteralPath $SourceDir -Force) -DestinationPath $DestDir
}

function Test-IsUncagingCalibSetupKey {
    param([string]$Key)
    $script:UncagingCalibSetupKeys -contains $Key
}

function Test-IsSharedKeyCopyProtectedKey {
    param([string]$Key)
    $script:SharedKeyCopyProtectedKeys -contains $Key
}

function Get-SetupLineValuePart {
    param([string]$Line)
    $eq = $Line.IndexOf('=')
    if ($eq -lt 0) { return $null }
    return $Line.Substring($eq + 1).Trim().TrimEnd(';').Trim()
}

function Test-UncagingCalibLineValid {
    param(
        [string]$Line,
        [switch]$IsBeta
    )
    if ([string]::IsNullOrWhiteSpace($Line)) { return $false }
    $valuePart = Get-SetupLineValuePart -Line $Line
    if ([string]::IsNullOrWhiteSpace($valuePart)) { return $false }

    $compact = $valuePart -replace '\s', '' -replace '[\{\}\[\]";]', ''
    if ([string]::IsNullOrWhiteSpace($compact)) { return $false }

    $parts = @($compact -split ',' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($parts.Count -lt 2) { return $false }

    foreach ($p in $parts) {
        if ($p -match '^(?i)nan$') { return $false }
        $d = 0.0
        if (-not [double]::TryParse($p, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$d)) {
            return $false
        }
        if ([double]::IsNaN($d) -or [double]::IsInfinity($d)) { return $false }
        if ($IsBeta -and $d -eq 0) { return $false }
    }
    return $true
}

function Get-DefaultUncagingCalibSetupLine {
    param([string]$Key)
    if ($Key -eq 'State.Uncaging.CalibV') { return 'State.Uncaging.CalibV = [0, 0];' }
    if ($Key -eq 'State.Uncaging.Calib_beta') { return 'State.Uncaging.Calib_beta = [1, 1];' }
    return $null
}

function Resolve-UncagingCalibLineForRestore {
    param(
        [string]$Key,
        [string]$ProfileLine,
        [hashtable]$LiveMap,
        [bool]$PreserveUncagingCalib
    )
    $isBeta = ($Key -eq 'State.Uncaging.Calib_beta')
    $defaultLine = Get-DefaultUncagingCalibSetupLine -Key $Key
    $profileValid = Test-UncagingCalibLineValid -Line $ProfileLine -IsBeta:$isBeta

    $liveLine = $null
    $liveValid = $false
    if ($LiveMap.ContainsKey($Key)) {
        $liveLine = $LiveMap[$Key]
        $liveValid = Test-UncagingCalibLineValid -Line $liveLine -IsBeta:$isBeta
    }

    if ($PreserveUncagingCalib) {
        if ($liveValid) { return $liveLine }
        if ($profileValid) { return $ProfileLine }
        return $defaultLine
    }

    if ($profileValid) { return $ProfileLine }
    return $defaultLine
}

function Write-DefaultSetupFromProfileRestore {
    param(
        [string]$ProfileFilePath,
        [string]$LiveFilePath,
        [bool]$PreserveUncagingCalib
    )
    if (-not (Test-Path -LiteralPath $ProfileFilePath)) { return }

    $liveMap = @{}
    if (Test-Path -LiteralPath $LiveFilePath) {
        $liveMap = Read-DefaultSetupKeyMap -FilePath $LiveFilePath
    }

    $newLines = New-Object System.Collections.Generic.List[string]
    foreach ($line in [System.IO.File]::ReadLines($ProfileFilePath)) {
        $key = Get-SetupLineKey -Line $line
        if ($key -eq 'State.Uncaging.CalibV' -or $key -eq 'State.Uncaging.Calib_beta') {
            $newLines.Add((Resolve-UncagingCalibLineForRestore -Key $key -ProfileLine $line -LiveMap $liveMap -PreserveUncagingCalib $PreserveUncagingCalib))
        } else {
            $newLines.Add($line)
        }
    }

    $parent = Split-Path -Parent $LiveFilePath
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($LiveFilePath, $newLines, $utf8NoBom)
}

function Copy-InitFilesRootForRestore {
    param(
        [string]$SourceDir,
        [string]$DestDir,
        [bool]$PreserveUncagingCalib
    )
    if (-not (Test-Path -LiteralPath $SourceDir)) { return }
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

    Get-ChildItem -LiteralPath $SourceDir -File -ErrorAction SilentlyContinue | ForEach-Object {
        $destPath = Join-Path $DestDir $_.Name
        if ($_.Name -like 'Default-*.txt') {
            Write-DefaultSetupFromProfileRestore -ProfileFilePath $_.FullName -LiveFilePath $destPath -PreserveUncagingCalib $PreserveUncagingCalib
        } else {
            Copy-Item -LiteralPath $_.FullName -Destination $destPath -Force
        }
        Set-CopiedItemTimestamps -SourceItem $_ -DestinationPath $destPath
    }
    Set-CopiedItemTimestamps -SourceItem (Get-Item -LiteralPath $SourceDir -Force) -DestinationPath $DestDir
}

function Copy-InitOtherSubfolders {
    param([string]$SourceDir, [string]$DestDir)
    if (-not (Test-Path -LiteralPath $SourceDir)) { return }
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    Get-ChildItem -LiteralPath $SourceDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $script:InitSubfolderExclude -notcontains $_.Name } |
        ForEach-Object {
            Copy-Tree $_.FullName (Join-Path $DestDir $_.Name)
        }
}

function Get-SetupLineKey {
    param([string]$Line)
    $trimmed = $Line.Trim()
    if ($trimmed.Length -eq 0) { return $null }
    if ($trimmed.StartsWith('#')) { return $null }
    $eq = $trimmed.IndexOf('=')
    if ($eq -lt 1) { return $null }
    return $trimmed.Substring(0, $eq).Trim()
}

function Read-DefaultSetupKeyMap {
    param([string]$FilePath)
    $map = @{}
    foreach ($line in [System.IO.File]::ReadLines($FilePath)) {
        $key = Get-SetupLineKey -Line $line
        if ($key) {
            $map[$key] = $line
        }
    }
    return $map
}

function Get-MasterDefaultSetupFile {
    param([string]$InitFilesPath)
    $files = @(Get-ChildItem -LiteralPath $InitFilesPath -Filter "Default-*.txt" -File -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return $null }
    return $files | Sort-Object LastWriteTimeUtc, FullName -Descending | Select-Object -First 1
}

function Merge-DefaultSetupFromMaster {
    param(
        [string]$MasterPath,
        [string]$TargetPath
    )
    $masterMap = Read-DefaultSetupKeyMap -FilePath $MasterPath
    $targetMap = Read-DefaultSetupKeyMap -FilePath $TargetPath
    if ($masterMap.Count -eq 0 -or $targetMap.Count -eq 0) {
        return 0
    }

    $keysUpdated = 0
    $newLines = New-Object System.Collections.Generic.List[string]
    foreach ($line in [System.IO.File]::ReadLines($TargetPath)) {
        $key = Get-SetupLineKey -Line $line
        if ($key -and $targetMap.ContainsKey($key) -and $masterMap.ContainsKey($key)) {
            if (Test-IsSharedKeyCopyProtectedKey -Key $key) {
                $newLines.Add($line)
                continue
            }
            if ($script:PreserveUncagingCalibOnRestore -and (Test-IsUncagingCalibSetupKey -Key $key)) {
                $newLines.Add($line)
                continue
            }
            $masterLine = $masterMap[$key]
            if ($masterLine -ne $line) {
                $keysUpdated++
            }
            $newLines.Add($masterLine)
        } else {
            $newLines.Add($line)
        }
    }

    if ($keysUpdated -gt 0) {
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllLines($TargetPath, $newLines, $utf8NoBom)
    }
    return $keysUpdated
}

function Sync-DefaultSettingsAcrossVersions {
    param([string]$InitFilesPath)

    $masterFile = Get-MasterDefaultSetupFile -InitFilesPath $InitFilesPath
    if (-not $masterFile) {
        throw "No Default-*.txt file found for sync."
    }

    $targets = @(Get-ChildItem -LiteralPath $InitFilesPath -Filter "Default-*.txt" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -ne $masterFile.FullName })

    $filesUpdated = 0
    $keysUpdated = 0
    foreach ($target in $targets) {
        $n = Merge-DefaultSetupFromMaster -MasterPath $masterFile.FullName -TargetPath $target.FullName
        if ($n -gt 0) {
            $filesUpdated++
            $keysUpdated += $n
        }
    }

    return [pscustomobject]@{
        MasterName      = $masterFile.Name
        MasterSavedAt   = $masterFile.LastWriteTime
        TargetNames     = @($targets | ForEach-Object { $_.Name })
        FilesUpdated    = $filesUpdated
        KeysUpdated     = $keysUpdated
    }
}

function Confirm-SyncDefaultToOtherVersions {
    param(
        [string]$MasterName,
        [datetime]$MasterSavedAt,
        [string[]]$TargetNames
    )
    $savedLabel = $MasterSavedAt.ToString('g')
    $targetList = ($TargetNames | ForEach-Object { "  - $_" }) -join "`n"
    $message = @"
Sync Default settings to other FLIMage versions in this profile?

Master (most recently saved Default file in Init_Files):
  $MasterName
  Last saved: $savedLabel

Will update shared keys only in the profile copy of:
$targetList

Your live Init_Files folder will NOT be modified.
Keys that exist only in a target file are kept unchanged.
Keys that exist only in the master file are not copied into older files.
"@
    $result = [System.Windows.Forms.MessageBox]::Show(
        $message,
        "Sync Default across versions",
        [System.Windows.Forms.MessageBoxButtons]::YesNoCancel,
        [System.Windows.Forms.MessageBoxIcon]::Question,
        [System.Windows.Forms.MessageBoxDefaultButton]::Button2
    )
    return $result
}

# Save no longer calls this. Shared-key copy is explicit via Advanced -> Copy shared keys.
function Invoke-DefaultSyncPromptOnSave {
    param([string]$InitFilesPath)

    $defaultFiles = @(Get-ChildItem -LiteralPath $InitFilesPath -Filter "Default-*.txt" -File -ErrorAction SilentlyContinue)
    if ($defaultFiles.Count -lt 2) {
        return @{ Action = 'Skip'; Summary = $null }
    }

    $masterFile = Get-MasterDefaultSetupFile -InitFilesPath $InitFilesPath
    if (-not $masterFile) {
        return @{ Action = 'Skip'; Summary = $null }
    }

    $targetNames = @($defaultFiles | Where-Object { $_.FullName -ne $masterFile.FullName } | ForEach-Object { $_.Name })
    if ($targetNames.Count -eq 0) {
        return @{ Action = 'Skip'; Summary = $null }
    }

    $dialogResult = Confirm-SyncDefaultToOtherVersions -MasterName $masterFile.Name -MasterSavedAt $masterFile.LastWriteTime -TargetNames $targetNames
    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::Cancel) {
        return @{ Action = 'Cancel'; Summary = $null }
    }
    if ($dialogResult -eq [System.Windows.Forms.DialogResult]::No) {
        return @{ Action = 'No'; Summary = $null }
    }

    return @{ Action = 'Yes'; Summary = $null }
}

function Get-ComponentDefinitions {
    @(
        @{ Key = "initRoot";        Label = "Init_Files (root files: Default-*, device, Setting-*)" }
        @{ Key = "settings";        Label = "Init_Files\Settings (panel UI prefs)" }
        @{ Key = "windowsInfo";     Label = "Init_Files\WindowsInfo (window layout)" }
        @{ Key = "uncaging";        Label = "Init_Files\Uncaging (.unc / .dig)" }
        @{ Key = "initOther";       Label = "Init_Files\other folders (Shading, ImageSequence, ...)" }
        @{ Key = "appDataRoaming";  Label = "%APPDATA%\FLIMage (UiState .ui.json)" }
        @{ Key = "appDataLocal";    Label = "%LOCALAPPDATA%\FLIMage (fallback Settings)" }
    )
}

function Get-ProfileSaveChangelogPath {
    param([string]$ProfileRoot)
    Join-Path $ProfileRoot 'save_changelog.jsonl'
}

function Get-RelativePathForChangelog {
    param(
        [string]$FullPath,
        [string]$BasePath
    )
    $base = $BasePath.TrimEnd('\', '/')
    if (-not $FullPath.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $FullPath
    }
    $rel = $FullPath.Substring($base.Length).TrimStart('\', '/')
    return ($rel -replace '\\', '/')
}

function Get-ScopedFileMap {
    param(
        [string]$Root,
        [string]$RelPrefix,
        [ValidateSet('files', 'tree')]
        [string]$Mode = 'tree'
    )
    $map = @{}
    if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root)) {
        return $map
    }
    if ($Mode -eq 'files') {
        foreach ($item in Get-ChildItem -LiteralPath $Root -File -ErrorAction SilentlyContinue) {
            $rel = if ($RelPrefix) { "$RelPrefix/$($item.Name)" } else { $item.Name }
            $map[($rel -replace '\\', '/')] = $item.FullName
        }
        return $map
    }
    foreach ($item in Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue) {
        $rel = Get-RelativePathForChangelog -FullPath $item.FullName -BasePath $Root
        if ($RelPrefix) { $rel = "$RelPrefix/$rel" }
        $map[($rel -replace '\\', '/')] = $item.FullName
    }
    return $map
}

function Merge-ScopedFileMaps {
    param(
        [hashtable]$OldMap,
        [hashtable]$NewMap
    )
    $merged = @{}
    foreach ($key in @($OldMap.Keys + $NewMap.Keys | Select-Object -Unique)) {
        $merged[$key] = @{
            OldPath = $OldMap[$key]
            NewPath = $NewMap[$key]
        }
    }
    return $merged
}

function Add-ToScopedFileMap {
    param(
        [hashtable]$Target,
        [hashtable]$Source
    )
    foreach ($key in $Source.Keys) {
        $Target[$key] = $Source[$key]
    }
}

function Get-LiveScopedRelativeFileMap {
    param([hashtable]$Components)

    $liveInit = Get-InitFilesPath
    $liveMap = @{}

    if ($Components.initRoot) {
        Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root $liveInit -RelPrefix 'Init_Files' -Mode 'files')
    }
    if ($Components.settings) {
        Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root (Join-Path $liveInit 'Settings') -RelPrefix 'Init_Files/Settings')
    }
    if ($Components.windowsInfo) {
        Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root (Join-Path $liveInit 'WindowsInfo') -RelPrefix 'Init_Files/WindowsInfo')
    }
    if ($Components.uncaging) {
        Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root (Join-Path $liveInit 'Uncaging') -RelPrefix 'Init_Files/Uncaging')
    }
    if ($Components.initOther) {
        if (Test-Path -LiteralPath $liveInit) {
            foreach ($dir in Get-ChildItem -LiteralPath $liveInit -Directory -ErrorAction SilentlyContinue) {
                if ($script:InitSubfolderExclude -contains $dir.Name) { continue }
                $prefix = "Init_Files/$($dir.Name)"
                Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root $dir.FullName -RelPrefix $prefix)
            }
        }
    }
    if ($Components.appDataRoaming) {
        Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root (Get-RoamingFlimagePath) -RelPrefix 'AppData_Roaming')
    }
    if ($Components.appDataLocal) {
        Add-ToScopedFileMap -Target $liveMap -Source (Get-ScopedFileMap -Root (Get-LocalFlimagePath) -RelPrefix 'AppData_Local')
    }

    return $liveMap
}

function Get-ProfileSaveScopedFileMaps {
    param(
        [string]$ProfileRoot,
        [hashtable]$Components
    )
    $archInit = Join-Path $ProfileRoot 'Init_Files'
    $oldCombined = @{}

    if ($Components.initRoot) {
        Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root $archInit -RelPrefix 'Init_Files' -Mode 'files')
    }
    if ($Components.settings) {
        Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root (Join-Path $archInit 'Settings') -RelPrefix 'Init_Files/Settings')
    }
    if ($Components.windowsInfo) {
        Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root (Join-Path $archInit 'WindowsInfo') -RelPrefix 'Init_Files/WindowsInfo')
    }
    if ($Components.uncaging) {
        Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root (Join-Path $archInit 'Uncaging') -RelPrefix 'Init_Files/Uncaging')
    }
    if ($Components.initOther) {
        if (Test-Path -LiteralPath $archInit) {
            foreach ($dir in Get-ChildItem -LiteralPath $archInit -Directory -ErrorAction SilentlyContinue) {
                if ($script:InitSubfolderExclude -contains $dir.Name) { continue }
                $prefix = "Init_Files/$($dir.Name)"
                Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root $dir.FullName -RelPrefix $prefix)
            }
        }
    }
    if ($Components.appDataRoaming) {
        Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root (Join-Path $ProfileRoot 'AppData_Roaming') -RelPrefix 'AppData_Roaming')
    }
    if ($Components.appDataLocal) {
        Add-ToScopedFileMap -Target $oldCombined -Source (Get-ScopedFileMap -Root (Join-Path $ProfileRoot 'AppData_Local') -RelPrefix 'AppData_Local')
    }

    return (Merge-ScopedFileMaps -OldMap $oldCombined -NewMap (Get-LiveScopedRelativeFileMap -Components $Components))
}

function Get-FileContentHashHex {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Format-ChangelogValue {
    param([string]$Value)
    if ($null -eq $Value) { return '' }
    $text = [string]$Value
    if ($text.Length -le 120) { return $text }
    return $text.Substring(0, 117) + '...'
}

function Test-ChangelogIgnoreSetupKey {
    param([string]$Key)
    if ([string]::IsNullOrWhiteSpace($Key)) { return $false }
    foreach ($ignored in $script:ChangelogIgnoreSetupKeys) {
        if ($Key.Equals($ignored, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Test-ChangelogIgnoreRelativePath {
    param([string]$RelativePath)
    $rel = ($RelativePath -replace '\\', '/').TrimStart('/')
    foreach ($pattern in $script:ChangelogIgnoreRelativePathPatterns) {
        $regex = '^' + ([regex]::Escape($pattern).Replace('\*', '.*')) + '$'
        if ($rel -match $regex) { return $true }
    }
    return $false
}

function Get-SetupKeyValueMapFromFile {
    param([string]$FilePath)
    $result = @{}
    $lineMap = Read-DefaultSetupKeyMap -FilePath $FilePath
    foreach ($key in $lineMap.Keys) {
        if (Test-ChangelogIgnoreSetupKey -Key $key) { continue }
        $result[$key] = (Get-SetupLineValuePart -Line $lineMap[$key])
    }
    return $result
}

function Get-SetupKeyChangesFromValueMaps {
    param(
        [hashtable]$OldValues,
        [hashtable]$NewValues
    )
    $keyChanges = @()
    foreach ($key in @($OldValues.Keys + $NewValues.Keys | Select-Object -Unique)) {
        $hasOld = $OldValues.ContainsKey($key)
        $hasNew = $NewValues.ContainsKey($key)
        if ($hasOld -and $hasNew) {
            if ([string]$OldValues[$key] -eq [string]$NewValues[$key]) { continue }
            $keyChanges += [ordered]@{
                key       = $key
                change    = 'modified'
                old_value = (Format-ChangelogValue -Value $OldValues[$key])
                new_value = (Format-ChangelogValue -Value $NewValues[$key])
            }
        } elseif ($hasOld) {
            $keyChanges += [ordered]@{
                key       = $key
                change    = 'removed'
                old_value = (Format-ChangelogValue -Value $OldValues[$key])
            }
        } else {
            $keyChanges += [ordered]@{
                key       = $key
                change    = 'added'
                new_value = (Format-ChangelogValue -Value $NewValues[$key])
            }
        }
    }
    return $keyChanges
}

function Get-SetupFileKeyChanges {
    param(
        [string]$OldPath,
        [string]$NewPath
    )
    return @(Get-SetupKeyChangesFromValueMaps `
            -OldValues (Get-SetupKeyValueMapFromFile -FilePath $OldPath) `
            -NewValues (Get-SetupKeyValueMapFromFile -FilePath $NewPath))
}

function Get-FileBaselineEntry {
    param([string]$FilePath)
    return @{
        hash = (Get-FileContentHashHex -Path $FilePath)
    }
}

function Build-ProfileSaveBaselineSnapshot {
    param([hashtable]$Components)

    $liveMap = Get-LiveScopedRelativeFileMap -Components $Components
    $files = @{}
    foreach ($rel in $liveMap.Keys) {
        if (Test-ChangelogIgnoreRelativePath -RelativePath $rel) { continue }
        $path = $liveMap[$rel]
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $files[$rel] = Get-FileBaselineEntry -FilePath $path
    }

    return [ordered]@{
        saved_at = (Get-Date).ToUniversalTime().ToString('o')
        scope    = @($Components.Keys | Where-Object { $Components[$_] } | Sort-Object)
        files    = $files
    }
}

function ConvertFrom-ManifestBaselineFiles {
    param($FilesObject)
    $files = @{}
    if (-not $FilesObject) { return $files }
    foreach ($prop in $FilesObject.PSObject.Properties) {
        $rel = [string]$prop.Name
        $val = $prop.Value
        if ($val.hash) {
            $files[$rel] = @{ hash = [string]$val.hash }
        }
    }
    return $files
}

function Get-ManifestChangelogBaselineFiles {
    param([string]$ProfileRoot)
    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { return $null }
    try {
        $json = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $json.changelog_baseline) { return $null }
        return (ConvertFrom-ManifestBaselineFiles -FilesObject $json.changelog_baseline.files)
    } catch {
        return $null
    }
}

function Compare-ProfileSaveAgainstBaseline {
    param(
        [string]$ProfileRoot,
        [hashtable]$BaselineFiles,
        [hashtable]$Components
    )
    $profileMap = @{}
    foreach ($pair in (Get-ProfileSaveScopedFileMaps -ProfileRoot $ProfileRoot -Components $Components).GetEnumerator()) {
        if ($pair.Value.OldPath) {
            $profileMap[$pair.Key] = $pair.Value.OldPath
        }
    }
    $liveMap = Get-LiveScopedRelativeFileMap -Components $Components
    $changes = @()
    $added = 0
    $removed = 0
    $modified = 0
    $keysChanged = 0

    foreach ($rel in @($BaselineFiles.Keys + $liveMap.Keys | Select-Object -Unique | Sort-Object)) {
        if (Test-ChangelogIgnoreRelativePath -RelativePath $rel) { continue }

        $hasOld = $BaselineFiles.ContainsKey($rel)
        $hasNew = $liveMap.ContainsKey($rel) -and (Test-Path -LiteralPath $liveMap[$rel])

        if (-not $hasOld -and $hasNew) {
            $added++
            $changes += [ordered]@{
                path     = $rel
                kind     = 'added'
                new_hash = (Get-FileContentHashHex -Path $liveMap[$rel])
            }
            continue
        }
        if ($hasOld -and -not $hasNew) {
            $removed++
            $changes += [ordered]@{
                path     = $rel
                kind     = 'removed'
                old_hash = $BaselineFiles[$rel].hash
            }
            continue
        }
        if (-not $hasOld -and -not $hasNew) { continue }

        $oldHash = $BaselineFiles[$rel].hash
        $newHash = Get-FileContentHashHex -Path $liveMap[$rel]
        if ($oldHash -eq $newHash) { continue }

        $modified++
        $entry = [ordered]@{
            path     = $rel
            kind     = 'modified'
            old_hash = $oldHash
            new_hash = $newHash
        }
        $oldPath = if ($profileMap.ContainsKey($rel)) { $profileMap[$rel] } else { $null }
        if ($oldPath -and (Test-Path -LiteralPath $oldPath) -and (Test-IsSetupKeyDiffFile -RelativePath $rel)) {
            $keyChanges = @(Get-SetupFileKeyChanges -OldPath $oldPath -NewPath $liveMap[$rel])
            if ($keyChanges.Count -gt 0) {
                $entry.keys = $keyChanges
                $keysChanged += $keyChanges.Count
            }
        }
        $changes += $entry
    }

    return @{
        IsEmpty       = ($changes.Count -eq 0)
        Changes       = $changes
        FilesAdded    = $added
        FilesRemoved  = $removed
        FilesModified = $modified
        KeysChanged   = $keysChanged
    }
}

function Compare-ProfileSaveForChangelog {
    param(
        [string]$ProfileRoot,
        [hashtable]$Components
    )
    $baselineFiles = Get-ManifestChangelogBaselineFiles -ProfileRoot $ProfileRoot
    if ($baselineFiles) {
        return (Compare-ProfileSaveAgainstBaseline -ProfileRoot $ProfileRoot -BaselineFiles $baselineFiles -Components $Components)
    }
    return (Compare-ProfileSaveAgainstLive -ProfileRoot $ProfileRoot -Components $Components)
}

function Test-IsSetupKeyDiffFile {
    param([string]$RelativePath)
    return ($RelativePath -match '\.txt$')
}

function Compare-ProfileSaveAgainstLive {
    param(
        [string]$ProfileRoot,
        [hashtable]$Components
    )
    $fileMap = Get-ProfileSaveScopedFileMaps -ProfileRoot $ProfileRoot -Components $Components
    $changes = @()
    $added = 0
    $removed = 0
    $modified = 0
    $keysChanged = 0

    foreach ($rel in @($fileMap.Keys | Sort-Object)) {
        if (Test-ChangelogIgnoreRelativePath -RelativePath $rel) { continue }

        $pair = $fileMap[$rel]
        $oldPath = $pair.OldPath
        $newPath = $pair.NewPath
        $hasOld = -not [string]::IsNullOrWhiteSpace($oldPath) -and (Test-Path -LiteralPath $oldPath)
        $hasNew = -not [string]::IsNullOrWhiteSpace($newPath) -and (Test-Path -LiteralPath $newPath)

        if (-not $hasOld -and $hasNew) {
            $added++
            $changes += [ordered]@{
                path  = $rel
                kind  = 'added'
                new_hash = (Get-FileContentHashHex -Path $newPath)
            }
            continue
        }
        if ($hasOld -and -not $hasNew) {
            $removed++
            $changes += [ordered]@{
                path     = $rel
                kind     = 'removed'
                old_hash = (Get-FileContentHashHex -Path $oldPath)
            }
            continue
        }
        if (-not $hasOld -and -not $hasNew) { continue }

        $oldHash = Get-FileContentHashHex -Path $oldPath
        $newHash = Get-FileContentHashHex -Path $newPath
        if ($oldHash -eq $newHash) { continue }

        $keyChanges = @()
        if (Test-IsSetupKeyDiffFile -RelativePath $rel) {
            $keyChanges = @(Get-SetupFileKeyChanges -OldPath $oldPath -NewPath $newPath)
            if ($keyChanges.Count -eq 0) { continue }
        }

        $modified++
        $entry = [ordered]@{
            path     = $rel
            kind     = 'modified'
            old_hash = $oldHash
            new_hash = $newHash
        }
        if ($keyChanges.Count -gt 0) {
            $entry.keys = $keyChanges
            $keysChanged += $keyChanges.Count
        }
        $changes += $entry
    }

    return @{
        IsEmpty     = ($changes.Count -eq 0)
        Changes     = $changes
        FilesAdded  = $added
        FilesRemoved = $removed
        FilesModified = $modified
        KeysChanged = $keysChanged
    }
}

function Append-ProfileSaveChangelog {
    param(
        [string]$ProfileRoot,
        [hashtable]$Components,
        $Diff
    )
    $scope = @($Components.Keys | Where-Object { $Components[$_] } | Sort-Object)
    $event = [ordered]@{
        saved_at     = (Get-Date).ToUniversalTime().ToString('o')
        windows_user = $env:USERNAME
        compared_to  = 'last_live_baseline'
        scope        = $scope
        summary      = [ordered]@{
            files_added     = $Diff.FilesAdded
            files_removed   = $Diff.FilesRemoved
            files_modified  = $Diff.FilesModified
            keys_changed    = $Diff.KeysChanged
        }
        changes = $Diff.Changes
    }
    $line = $event | ConvertTo-Json -Depth 8 -Compress
    $path = Get-ProfileSaveChangelogPath -ProfileRoot $ProfileRoot
    Add-Content -LiteralPath $path -Value $line -Encoding UTF8
}

function Save-Profile {
    param(
        [string]$ProfileName,
        [hashtable]$Components,
        [string]$StorageRoot = $null,
        [string]$ProfileKind = 'user',
        [hashtable]$ManifestExtra = $null
    )

    if ($Components.Values -notcontains $true) {
        throw "Select at least one component to save."
    }

    $root = if ($StorageRoot) { $StorageRoot } else { Get-ProfileRoot $ProfileName }
    New-Item -ItemType Directory -Path $root -Force | Out-Null

    $manifestPath = Join-Path $root 'manifest.json'
    $existingRestoredAt = $null
    $existingPasswordSalt = $null
    $existingPasswordHash = $null
    $existingPasswordIterations = $null
    if (Test-Path -LiteralPath $manifestPath) {
        try {
            $existingManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if (Test-ManifestHasProperty -Json $existingManifest -PropertyName 'restored_at') {
                $existingRestoredAt = [string]$existingManifest.restored_at
            }
            if (Test-ManifestHasProperty -Json $existingManifest -PropertyName 'password_hash') {
                $existingPasswordSalt = [string]$existingManifest.password_salt
                $existingPasswordHash = [string]$existingManifest.password_hash
                $existingPasswordIterations = 100000
                if ($existingManifest.password_iterations) {
                    $existingPasswordIterations = [int]$existingManifest.password_iterations
                }
            }
        } catch { }
    }
    $writeChangelog = (-not $StorageRoot) -and ($ProfileKind -eq 'user') -and (Test-Path -LiteralPath $manifestPath)
    if ($writeChangelog) {
        $diff = Compare-ProfileSaveForChangelog -ProfileRoot $root -Components $Components
        if ($diff.IsEmpty) {
            return @{ NoChanges = $true }
        }
        Append-ProfileSaveChangelog -ProfileRoot $root -Components $Components -Diff $diff
        Invoke-ProfileHistoryBackupBeforeSave -ProfileRoot $root -ProfileName $ProfileName
    }

    $liveInit = Get-InitFilesPath
    $archInit = Join-Path $root "Init_Files"

    if ($Components.initRoot) {
        Copy-InitFilesRootOnly $liveInit $archInit
    }
    if ($Components.settings) {
        Copy-Tree (Join-Path $liveInit "Settings") (Join-Path $archInit "Settings")
    }
    if ($Components.windowsInfo) {
        Copy-Tree (Join-Path $liveInit "WindowsInfo") (Join-Path $archInit "WindowsInfo")
    }
    if ($Components.uncaging) {
        Copy-Tree (Join-Path $liveInit "Uncaging") (Join-Path $archInit "Uncaging")
    }
    if ($Components.initOther) {
        Copy-InitOtherSubfolders $liveInit $archInit
    }
    if ($Components.appDataRoaming) {
        Copy-Tree (Get-RoamingFlimagePath) (Join-Path $root "AppData_Roaming")
    }
    if ($Components.appDataLocal) {
        Copy-Tree (Get-LocalFlimagePath) (Join-Path $root "AppData_Local")
    }
    if ($Components.initRoot -and (Test-Path -LiteralPath $liveInit) -and (Test-Path -LiteralPath $archInit)) {
        Set-CopiedItemTimestamps -SourceItem (Get-Item -LiteralPath $liveInit -Force) -DestinationPath $archInit
    }

    $defaultFiles = @()
    if (Test-Path -LiteralPath $archInit) {
        $defaultFiles = Get-ChildItem -LiteralPath $archInit -Filter "Default-*.txt" -File -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Name }
    }

    $nowUtc = (Get-Date).ToUniversalTime().ToString('o')
    $manifest = [ordered]@{
        saved_at      = $nowUtc
        modified_at   = $nowUtc
        windows_user  = $env:USERNAME
        profile_kind  = $ProfileKind
        components    = $Components
        default_files = $defaultFiles
    }
    if ($existingRestoredAt) {
        $manifest.restored_at = $existingRestoredAt
    }
    if ($writeChangelog) {
        $manifest.changelog_baseline = (Build-ProfileSaveBaselineSnapshot -Components $Components)
    }
    if ($ManifestExtra) {
        foreach ($key in $ManifestExtra.Keys) {
            $manifest[$key] = $ManifestExtra[$key]
        }
    }
    # A normal save must keep an existing overwrite password. Restore never reads it.
    if ($existingPasswordHash -and -not $manifest.Contains('password_hash')) {
        $manifest.password_salt = $existingPasswordSalt
        $manifest.password_hash = $existingPasswordHash
        $manifest.password_iterations = $existingPasswordIterations
    }
    Write-ProfileManifestJson -ProfileRoot $root -ManifestObject $manifest
    return @{ NoChanges = $false }
}

function Get-PreRestoreBackupBase {
    $parent = Split-Path -Parent $script:ProfilesBase
    if ([string]::IsNullOrWhiteSpace($parent)) {
        $docs = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
        return (Join-Path (Join-Path $docs "FLIMage") "PreRestoreBackup")
    }
    return (Join-Path $parent "PreRestoreBackup")
}

function Get-PreRestoreSlotSavedAtUtc {
    param([string]$SlotRoot)

    $manifestPath = Join-Path $SlotRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { return $null }
    try {
        $json = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $json.saved_at) { return $null }
        $parsed = [datetime]::MinValue
        if ([datetime]::TryParse(
                [string]$json.saved_at,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [System.Globalization.DateTimeStyles]::RoundtripKind,
                [ref]$parsed)) {
            if ($parsed.Kind -eq [System.DateTimeKind]::Utc) { return $parsed }
            return $parsed.ToUniversalTime()
        }
    } catch { }
    return $null
}

function Get-PreRestoreSlotIndexToWrite {
    $base = Get-PreRestoreBackupBase
    $max = $script:PreRestoreBackupMaxCount
    New-Item -ItemType Directory -Path $base -Force | Out-Null

    $occupied = New-Object System.Collections.Generic.List[object]
    for ($i = 0; $i -lt $max; $i++) {
        $slotRoot = Join-Path $base ("slot_{0:D2}" -f $i)
        if (-not (Test-Path -LiteralPath $slotRoot)) {
            return $i
        }
        $savedAtUtc = Get-PreRestoreSlotSavedAtUtc -SlotRoot $slotRoot
        if (-not $savedAtUtc) {
            $savedAtUtc = (Get-Item -LiteralPath $slotRoot).LastWriteTimeUtc
        }
        $occupied.Add([pscustomobject]@{
                Index      = $i
                SavedAtUtc = $savedAtUtc
            })
    }

    if ($occupied.Count -eq 0) { return 0 }
    return ($occupied | Sort-Object SavedAtUtc, Index | Select-Object -First 1).Index
}

function Invoke-PreRestoreBackupBeforeRestore {
    param(
        [hashtable]$Components,
        [string]$TargetProfileName
    )

    $base = Get-PreRestoreBackupBase
    $slot = Get-PreRestoreSlotIndexToWrite
    $slotRoot = Join-Path $base ("slot_{0:D2}" -f $slot)
    if (Test-Path -LiteralPath $slotRoot) {
        Remove-Item -LiteralPath $slotRoot -Recurse -Force
    }

    Save-Profile -ProfileName 'prerestore' -Components $Components -StorageRoot $slotRoot -ProfileKind 'pre_restore' -ManifestExtra @{
        slot_index             = $slot
        restore_target_profile = $TargetProfileName
    }

    return $slotRoot
}

function Get-ProfileHistoryBase {
    param([string]$ProfileRoot)
    Join-Path $ProfileRoot $script:ProfileHistoryFolderName
}

function Copy-ProfileDirectorySnapshot {
    param(
        [string]$SourceRoot,
        [string]$DestRoot
    )
    if (Test-Path -LiteralPath $DestRoot) {
        Remove-Item -LiteralPath $DestRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $DestRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $SourceRoot)) { return }
    Get-ChildItem -LiteralPath $SourceRoot -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne $script:ProfileHistoryFolderName } |
        ForEach-Object {
            Copy-Tree -Source $_.FullName -Destination (Join-Path $DestRoot $_.Name)
        }
}

function Get-ProfileHistorySlotIndexToWrite {
    param([string]$ProfileRoot)
    $base = Get-ProfileHistoryBase -ProfileRoot $ProfileRoot
    $max = $script:ProfileHistoryMaxCount
    New-Item -ItemType Directory -Path $base -Force | Out-Null

    $occupied = New-Object System.Collections.Generic.List[object]
    for ($i = 0; $i -lt $max; $i++) {
        $slotRoot = Join-Path $base ("slot_{0:D2}" -f $i)
        if (-not (Test-Path -LiteralPath $slotRoot)) {
            return $i
        }
        $savedAtUtc = Get-PreRestoreSlotSavedAtUtc -SlotRoot $slotRoot
        if (-not $savedAtUtc) {
            $savedAtUtc = (Get-Item -LiteralPath $slotRoot).LastWriteTimeUtc
        }
        $occupied.Add([pscustomobject]@{
                Index      = $i
                SavedAtUtc = $savedAtUtc
            })
    }

    if ($occupied.Count -eq 0) { return 0 }
    return ($occupied | Sort-Object SavedAtUtc, Index | Select-Object -First 1).Index
}

function Invoke-ProfileHistoryBackupBeforeSave {
    param(
        [string]$ProfileRoot,
        [string]$ProfileName
    )
    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { return }

    $base = Get-ProfileHistoryBase -ProfileRoot $ProfileRoot
    $slot = Get-ProfileHistorySlotIndexToWrite -ProfileRoot $ProfileRoot
    $slotRoot = Join-Path $base ("slot_{0:D2}" -f $slot)
    Copy-ProfileDirectorySnapshot -SourceRoot $ProfileRoot -DestRoot $slotRoot

    $nowUtc = (Get-Date).ToUniversalTime().ToString('o')
    $slotManifestPath = Join-Path $slotRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $slotManifestPath)) { return }
    try {
        $json = Get-Content -LiteralPath $slotManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $json | Add-Member -NotePropertyName 'profile_kind' -NotePropertyValue 'profile_history' -Force
        $json | Add-Member -NotePropertyName 'history_slot' -NotePropertyValue $slot -Force
        $json | Add-Member -NotePropertyName 'history_source_profile' -NotePropertyValue $ProfileName -Force
        $json | Add-Member -NotePropertyName 'history_captured_at' -NotePropertyValue $nowUtc -Force
        $json | Add-Member -NotePropertyName 'saved_at' -NotePropertyValue $nowUtc -Force
        Write-ProfileManifestJson -ProfileRoot $slotRoot -ManifestObject $json
    } catch { }
}

function Get-ProfileHistoryListEntries {
    param([string]$ProfileName)
    $entries = @()
    $profileRoot = Get-ProfileRoot $ProfileName
    if (-not (Test-Path -LiteralPath $profileRoot)) { return $entries }
    $base = Get-ProfileHistoryBase -ProfileRoot $profileRoot
    if (-not (Test-Path -LiteralPath $base)) { return $entries }
    for ($i = 0; $i -lt $script:ProfileHistoryMaxCount; $i++) {
        $slotRoot = Join-Path $base ("slot_{0:D2}" -f $i)
        if (-not (Test-Path -LiteralPath $slotRoot)) { continue }
        $meta = Get-ProfileManifestInfo -ProfileRoot $slotRoot
        $label = "slot_{0:D2}" -f $i
        if ($meta.ModifiedAtDisplay) {
            $label = "$label ($($meta.ModifiedAtDisplay))"
        }
        $entries += New-ProfileListEntry -Name $label -Kind 'profile_history' -StorageRoot $slotRoot `
            -SavedAt $meta.SavedAtDisplay -ModifiedAt $meta.ModifiedAtDisplay -RestoredAt $meta.RestoredAtDisplay `
            -ModifiedAtSort $meta.ModifiedAtSort -RestoredAtSort $meta.RestoredAtSort -HasRestoredAt $meta.HasRestoredAt `
            -RestoreTarget $ProfileName -HasSavePassword $meta.HasSavePassword
    }
    return @($entries | Sort-Object @{ Expression = 'ModifiedAtSort'; Descending = $true }, Name)
}

function Test-IsValidProfileName {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return $false }
    if ($Name.Trim() -ne $Name) { return $false }
    foreach ($c in [System.IO.Path]::GetInvalidFileNameChars()) {
        if ($Name.IndexOf($c) -ge 0) { return $false }
    }
    return $true
}

function Show-RenameProfileDialog {
    param([string]$CurrentName)

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = 'Rename profile'
    $dlg.ClientSize = New-Object System.Drawing.Size(380, 118)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Location = New-Object System.Drawing.Point(12, 14)
    $lbl.Size = New-Object System.Drawing.Size(356, 20)
    $lbl.Text = 'New profile name:'
    $dlg.Controls.Add($lbl)

    $txt = New-Object System.Windows.Forms.TextBox
    $txt.Location = New-Object System.Drawing.Point(12, 38)
    $txt.Size = New-Object System.Drawing.Size(356, 22)
    $txt.Text = $CurrentName
    $dlg.Controls.Add($txt)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(204, 72)
    $btnOk.Size = New-Object System.Drawing.Size(78, 28)
    $btnOk.Text = 'OK'
    $dlg.AcceptButton = $btnOk
    $dlg.Controls.Add($btnOk)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Location = New-Object System.Drawing.Point(290, 72)
    $btnCancel.Size = New-Object System.Drawing.Size(78, 28)
    $btnCancel.Text = 'Cancel'
    $dlg.CancelButton = $btnCancel
    $dlg.Controls.Add($btnCancel)

    $btnOk.Add_Click({
        $name = $txt.Text.Trim()
        if (-not (Test-IsValidProfileName $name)) {
            [System.Windows.Forms.MessageBox]::Show(
                "Enter a valid profile name (no invalid path characters).",
                "Rename profile",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning
            ) | Out-Null
            return
        }
        $dlg.Tag = $name
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $dlg.Close()
    })
    $btnCancel.Add_Click({
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
        $dlg.Close()
    })

    if ($dlg.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK) {
        return [string]$dlg.Tag
    }
    return $null
}

function Invoke-RenameProfile {
    param(
        [string]$OldName,
        [string]$NewName
    )

    if (Test-FLIMageRunning) {
        throw "Close FLIMage before renaming a profile."
    }

    $old = $OldName.Trim()
    $new = $NewName.Trim()
    if ($old -eq $new) { return $old }
    if (-not (Test-IsValidProfileName $new)) {
        throw "Invalid profile name: $new"
    }
    if (Test-ProfileExists $new) {
        throw "Profile already exists: $new"
    }

    $storage = Resolve-ProfileStorageRoot $old
    if (-not $storage) {
        throw "Profile not found: $old"
    }

    $destRoot = if ($storage.Kind -eq 'auto_backup') {
        Get-AutoBackupRoot $new
    } else {
        Get-ProfileRoot $new
    }
    if (Test-Path -LiteralPath $destRoot) {
        throw "Destination already exists: $destRoot"
    }

    $destParent = Split-Path -Parent $destRoot
    if ($destParent -and -not (Test-Path -LiteralPath $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }

    Move-Item -LiteralPath $storage.Root -Destination $destRoot
    return $new
}

function Restore-Profile {
    param(
        [string]$ProfileName,
        [string]$StorageRoot = $null,
        [hashtable]$Components
    )

    if ($Components.Values -notcontains $true) {
        throw "Select at least one component to restore."
    }

    if ($StorageRoot) {
        if (-not (Test-Path -LiteralPath $StorageRoot)) {
            throw "Backup folder not found: $StorageRoot"
        }
        $root = $StorageRoot
    } else {
        $storage = Resolve-ProfileStorageRoot $ProfileName
        if (-not $storage) {
            throw "Profile not found: $ProfileName (checked Profiles and AutoBackup folders)."
        }
        $root = $storage.Root
    }

    $flimParent = Split-Path (Get-InitFilesPath) -Parent
    if (-not (Test-Path -LiteralPath $flimParent)) {
        New-Item -ItemType Directory -Path $flimParent -Force | Out-Null
    }

    $liveInit = Get-InitFilesPath
    $archInit = Join-Path $root "Init_Files"
    New-Item -ItemType Directory -Path $liveInit -Force | Out-Null

    if ($Components.initRoot) {
        Copy-InitFilesRootForRestore -SourceDir $archInit -DestDir $liveInit -PreserveUncagingCalib $script:PreserveUncagingCalibOnRestore
    }
    if ($Components.settings) {
        Copy-Tree (Join-Path $archInit "Settings") (Join-Path $liveInit "Settings")
    }
    if ($Components.windowsInfo) {
        Copy-Tree (Join-Path $archInit "WindowsInfo") (Join-Path $liveInit "WindowsInfo")
    }
    if ($Components.uncaging) {
        Copy-Tree (Join-Path $archInit "Uncaging") (Join-Path $liveInit "Uncaging")
    }
    if ($Components.initOther) {
        Copy-InitOtherSubfolders $archInit $liveInit
    }
    if ($Components.appDataRoaming) {
        Copy-Tree (Join-Path $root "AppData_Roaming") (Get-RoamingFlimagePath)
    }
    if ($Components.appDataLocal) {
        Copy-Tree (Join-Path $root "AppData_Local") (Get-LocalFlimagePath)
    }
    if ($Components.initRoot -and (Test-Path -LiteralPath $archInit) -and (Test-Path -LiteralPath $liveInit)) {
        Set-CopiedItemTimestamps -SourceItem (Get-Item -LiteralPath $archInit -Force) -DestinationPath $liveInit
    }
}

function Get-LatestActivityUtcFromSortFields {
    param(
        [datetime]$ModifiedAtSort,
        [datetime]$RestoredAtSort
    )
    $latest = [datetime]::MinValue
    foreach ($candidate in @($ModifiedAtSort, $RestoredAtSort)) {
        if (-not $candidate) { continue }
        $utc = $candidate.ToUniversalTime()
        if ($utc -gt $latest) { $latest = $utc }
    }
    return $latest
}

function Get-ProfileLatestActivityUtc {
    param([string]$ProfileRoot)
    $meta = Get-ProfileManifestInfo -ProfileRoot $ProfileRoot
    return Get-LatestActivityUtcFromSortFields -ModifiedAtSort $meta.ModifiedAtSort -RestoredAtSort $meta.RestoredAtSort
}

function Get-ProfileNames {
    $base = $script:ProfilesBase
    if (-not (Test-Path -LiteralPath $base)) { return @() }
    Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            [pscustomobject]@{
                Name                = $_.Name
                LatestActivityUtc = Get-ProfileLatestActivityUtc -ProfileRoot $_.FullName
            }
        } |
        Sort-Object @{ Expression = 'LatestActivityUtc'; Descending = $true }, Name |
        ForEach-Object { $_.Name }
}

function Get-ManifestUtcDateTime {
    param([string]$IsoString)
    if ([string]::IsNullOrWhiteSpace($IsoString)) { return $null }
    # PowerShell 5.1 cannot resolve TryParse when [ref] target is $null.
    $parsed = [datetime]::MinValue
    if ([datetime]::TryParse(
            $IsoString,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::RoundtripKind,
            [ref]$parsed)) {
        if ($parsed.Kind -eq [System.DateTimeKind]::Utc) { return $parsed }
        return $parsed.ToUniversalTime()
    }
    return $null
}

function Format-ManifestDateDisplay {
    param([datetime]$DateTimeValue)
    if (-not $DateTimeValue) { return '' }
    return $DateTimeValue.ToLocalTime().ToString('yyyy-MM-dd HH:mm')
}

function Test-ManifestHasProperty {
    param(
        $Json,
        [string]$PropertyName
    )
    if (-not $Json) { return $false }
    $prop = $Json.PSObject.Properties[$PropertyName]
    if (-not $prop) { return $false }
    return -not [string]::IsNullOrWhiteSpace([string]$prop.Value)
}

function Get-ProfileManifestInfo {
    param([string]$ProfileRoot)

    $info = @{
        SavedAt              = $null
        SavedAtDisplay       = ''
        ModifiedAt           = $null
        ModifiedAtDisplay    = ''
        ModifiedAtSort       = $null
        RestoredAt           = $null
        RestoredAtDisplay    = ''
        RestoredAtSort       = $null
        HasRestoredAt        = $false
        ProfileKind          = 'user'
        RestoreTargetProfile = ''
        HasSavePassword      = $false
    }
    $item = Get-Item -LiteralPath $ProfileRoot
    $info.ModifiedAt = $item.LastWriteTime
    $info.ModifiedAtDisplay = $item.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
    $info.ModifiedAtSort = $item.LastWriteTime
    $info.RestoredAt = $info.ModifiedAt
    $info.RestoredAtDisplay = $info.ModifiedAtDisplay
    $info.RestoredAtSort = $info.ModifiedAtSort

    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    if (Test-Path -LiteralPath $manifestPath) {
        try {
            $json = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($json.profile_kind) {
                $info.ProfileKind = [string]$json.profile_kind
            }
            if ($json.restore_target_profile) {
                $info.RestoreTargetProfile = [string]$json.restore_target_profile
            }
            if (Test-ManifestHasProperty -Json $json -PropertyName 'password_hash') {
                $info.HasSavePassword = $true
            }

            $modifiedSource = $null
            if (Test-ManifestHasProperty -Json $json -PropertyName 'modified_at') {
                $modifiedSource = [string]$json.modified_at
            } elseif (Test-ManifestHasProperty -Json $json -PropertyName 'saved_at') {
                $modifiedSource = [string]$json.saved_at
            }
            if ($modifiedSource) {
                $modifiedDt = Get-ManifestUtcDateTime -IsoString $modifiedSource
                if ($modifiedDt) {
                    $info.ModifiedAt = $modifiedDt.ToLocalTime()
                    $info.ModifiedAtSort = $modifiedDt
                }
                $info.ModifiedAtDisplay = Format-ManifestDateDisplay -DateTimeValue $info.ModifiedAt
                if ([string]::IsNullOrWhiteSpace($info.ModifiedAtDisplay)) {
                    $info.ModifiedAtDisplay = $modifiedSource
                }
            }

            if (Test-ManifestHasProperty -Json $json -PropertyName 'saved_at') {
                $savedDt = Get-ManifestUtcDateTime -IsoString ([string]$json.saved_at)
                if ($savedDt) {
                    $info.SavedAt = $savedDt.ToLocalTime()
                    $info.SavedAtDisplay = Format-ManifestDateDisplay -DateTimeValue $info.SavedAt
                } else {
                    $info.SavedAtDisplay = [string]$json.saved_at
                }
            }

            if (Test-ManifestHasProperty -Json $json -PropertyName 'restored_at') {
                $info.HasRestoredAt = $true
                $restoredSource = [string]$json.restored_at
                $restoredDt = Get-ManifestUtcDateTime -IsoString $restoredSource
                if ($restoredDt) {
                    $info.RestoredAt = $restoredDt.ToLocalTime()
                    $info.RestoredAtSort = $restoredDt
                    $info.RestoredAtDisplay = Format-ManifestDateDisplay -DateTimeValue $info.RestoredAt
                } else {
                    $info.RestoredAtDisplay = $restoredSource
                    $info.RestoredAtSort = $info.ModifiedAtSort
                }
            }
        } catch { }
    }

    if ([string]::IsNullOrWhiteSpace($info.SavedAtDisplay)) {
        $info.SavedAt = $info.ModifiedAt
        $info.SavedAtDisplay = $info.ModifiedAtDisplay
    }
    if (-not $info.HasRestoredAt) {
        $info.RestoredAt = $info.ModifiedAt
        $info.RestoredAtDisplay = $info.ModifiedAtDisplay
        $info.RestoredAtSort = $info.ModifiedAtSort
    } elseif (-not $info.RestoredAtSort) {
        $info.RestoredAtSort = $info.ModifiedAtSort
    } elseif ([string]::IsNullOrWhiteSpace($info.RestoredAtDisplay) -and $info.RestoredAt) {
        $info.RestoredAtDisplay = Format-ManifestDateDisplay -DateTimeValue $info.RestoredAt
    }
    if (-not $info.ModifiedAtSort) {
        $info.ModifiedAtSort = $info.ModifiedAt
    }
    return $info
}

function Write-ProfileManifestJson {
    param(
        [string]$ProfileRoot,
        $ManifestObject
    )
    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    $ManifestObject | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}

function Update-ProfileManifestRestoredAt {
    param([string]$ProfileRoot)
    if ([string]::IsNullOrWhiteSpace($ProfileRoot)) { return }
    if (-not (Test-Path -LiteralPath $ProfileRoot)) { return }
    $nowUtc = (Get-Date).ToUniversalTime().ToString('o')
    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    try {
        if (Test-Path -LiteralPath $manifestPath) {
            $json = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $json | Add-Member -NotePropertyName 'restored_at' -NotePropertyValue $nowUtc -Force
            Write-ProfileManifestJson -ProfileRoot $ProfileRoot -ManifestObject $json
        } else {
            $manifest = [ordered]@{
                restored_at  = $nowUtc
                profile_kind = 'user'
            }
            Write-ProfileManifestJson -ProfileRoot $ProfileRoot -ManifestObject $manifest
        }
    } catch {
        throw "Failed to update restored_at in manifest: $($_.Exception.Message)"
    }
}

function New-ProfileListEntry {
    param(
        [string]$Name,
        [string]$Kind,
        [string]$StorageRoot,
        [string]$SavedAt,
        [string]$ModifiedAt,
        [string]$RestoredAt,
        [datetime]$ModifiedAtSort,
        [datetime]$RestoredAtSort,
        [bool]$HasRestoredAt = $false,
        [string]$RestoreTarget = '',
        [bool]$HasSavePassword = $false
    )
    return [pscustomobject]@{
        Name            = $Name
        Kind            = $Kind
        StorageRoot     = $StorageRoot
        SavedAt         = $SavedAt
        ModifiedAt      = $ModifiedAt
        RestoredAt      = $RestoredAt
        ModifiedAtSort  = $ModifiedAtSort
        RestoredAtSort  = $RestoredAtSort
        HasRestoredAt   = $HasRestoredAt
        RestoreTarget   = $RestoreTarget
        HasSavePassword = $HasSavePassword
    }
}

function Get-ProfileListEntries {
    $entries = @()
    $basePath = $script:ProfilesBase
    if (-not (Test-Path -LiteralPath $basePath)) { return $entries }
    foreach ($dir in Get-ChildItem -LiteralPath $basePath -Directory -ErrorAction SilentlyContinue) {
        $meta = Get-ProfileManifestInfo -ProfileRoot $dir.FullName
        $kind = if ($meta.ProfileKind -and $meta.ProfileKind -ne 'auto_backup' -and $meta.ProfileKind -ne 'pre_restore') {
            $meta.ProfileKind
        } else {
            'user'
        }
        $entries += New-ProfileListEntry -Name $dir.Name -Kind $kind -StorageRoot $dir.FullName `
            -SavedAt $meta.SavedAtDisplay -ModifiedAt $meta.ModifiedAtDisplay -RestoredAt $meta.RestoredAtDisplay `
            -ModifiedAtSort $meta.ModifiedAtSort -RestoredAtSort $meta.RestoredAtSort -HasRestoredAt $meta.HasRestoredAt `
            -RestoreTarget $meta.RestoreTargetProfile -HasSavePassword $meta.HasSavePassword
    }
    return @($entries | Sort-Object @{ Expression = 'RestoredAtSort'; Descending = $true }, Name)
}

function Get-MonthlyBackupListEntries {
    $entries = @()
    $basePath = Get-AutoBackupBase
    if (-not (Test-Path -LiteralPath $basePath)) { return $entries }
    foreach ($dir in Get-ChildItem -LiteralPath $basePath -Directory -ErrorAction SilentlyContinue) {
        $meta = Get-ProfileManifestInfo -ProfileRoot $dir.FullName
        $kind = if ($meta.ProfileKind) { $meta.ProfileKind } else { 'auto_backup' }
        $entries += New-ProfileListEntry -Name $dir.Name -Kind $kind -StorageRoot $dir.FullName `
            -SavedAt $meta.SavedAtDisplay -ModifiedAt $meta.ModifiedAtDisplay -RestoredAt $meta.RestoredAtDisplay `
            -ModifiedAtSort $meta.ModifiedAtSort -RestoredAtSort $meta.RestoredAtSort -HasRestoredAt $meta.HasRestoredAt `
            -RestoreTarget $meta.RestoreTargetProfile -HasSavePassword $meta.HasSavePassword
    }
    return @($entries | Sort-Object @{ Expression = 'ModifiedAtSort'; Descending = $true }, Name)
}

function Get-PreRestoreBackupListEntries {
    $entries = @()
    $basePath = Get-PreRestoreBackupBase
    if (-not (Test-Path -LiteralPath $basePath)) { return $entries }
    for ($i = 0; $i -lt $script:PreRestoreBackupMaxCount; $i++) {
        $slotRoot = Join-Path $basePath ("slot_{0:D2}" -f $i)
        if (-not (Test-Path -LiteralPath $slotRoot)) { continue }
        $meta = Get-ProfileManifestInfo -ProfileRoot $slotRoot
        $kind = if ($meta.ProfileKind) { $meta.ProfileKind } else { 'pre_restore' }
        $entries += New-ProfileListEntry -Name ("slot_{0:D2}" -f $i) -Kind $kind -StorageRoot $slotRoot `
            -SavedAt $meta.SavedAtDisplay -ModifiedAt $meta.ModifiedAtDisplay -RestoredAt $meta.RestoredAtDisplay `
            -ModifiedAtSort $meta.ModifiedAtSort -RestoredAtSort $meta.RestoredAtSort -HasRestoredAt $meta.HasRestoredAt `
            -RestoreTarget $meta.RestoreTargetProfile -HasSavePassword $meta.HasSavePassword
    }
    return @($entries)
}

function Get-ProfileListDisplayName {
    param($Entry)
    $name = [string]$Entry.Name
    if ($Entry.Kind -eq 'auto_backup') {
        $name = "$($Entry.Name) [AutoBackup]"
    } elseif ($Entry.Kind -eq 'pre_restore' -and $Entry.RestoreTarget) {
        $name = "$($Entry.Name) (before: $($Entry.RestoreTarget))"
    }
    if ($Entry.HasSavePassword) {
        $name = "$name [password]"
    }
    return $name
}

function Test-ProfileListEntryMatchesFilter {
    param(
        $Entry,
        [string]$Filter
    )
    if ([string]::IsNullOrWhiteSpace($Filter)) { return $true }
    # Use StringComparison.OrdinalIgnoreCase — IndexOf(string, int) would treat CompareOptions (1) as startIndex.
    $cmp = [System.StringComparison]::OrdinalIgnoreCase
    $displayName = Get-ProfileListDisplayName -Entry $Entry
    if ($displayName.IndexOf($Filter, $cmp) -ge 0) { return $true }
    if ($Entry.Name.IndexOf($Filter, $cmp) -ge 0) { return $true }
    if ($Entry.ModifiedAt -and $Entry.ModifiedAt.IndexOf($Filter, $cmp) -ge 0) { return $true }
    if ($Entry.RestoredAt -and $Entry.RestoredAt.IndexOf($Filter, $cmp) -ge 0) { return $true }
    if ($Entry.RestoreTarget -and $Entry.RestoreTarget.IndexOf($Filter, $cmp) -ge 0) { return $true }
    return $false
}

function Sort-ProfileListEntries {
    param(
        [array]$Entries,
        [int]$SortColumn,
        [bool]$Ascending
    )
    if ($Entries.Count -le 1) { return $Entries }
    if ($SortColumn -eq 1) {
        if ($Ascending) {
            return @($Entries | Sort-Object ModifiedAtSort, Name)
        }
        return @($Entries | Sort-Object @{ Expression = 'ModifiedAtSort'; Descending = $true }, Name)
    }
    if ($SortColumn -eq 2) {
        if ($Ascending) {
            return @($Entries | Sort-Object RestoredAtSort, Name)
        }
        return @($Entries | Sort-Object @{ Expression = 'RestoredAtSort'; Descending = $true }, Name)
    }
    if ($Ascending) {
        return @($Entries | Sort-Object Name)
    }
    return @($Entries | Sort-Object Name -Descending)
}

function Get-ArchivedProfilesBase {
    $parent = Split-Path -Parent $script:ProfilesBase
    if ([string]::IsNullOrWhiteSpace($parent)) {
        $docs = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
        return (Join-Path (Join-Path $docs "FLIMage") "ArchivedProfiles")
    }
    return (Join-Path $parent "ArchivedProfiles")
}

function Get-ProfileRestoreFolderForKind {
    param([string]$Kind)
    if ($Kind -eq 'auto_backup') {
        return (Get-AutoBackupBase)
    }
    return $script:ProfilesBase
}

function Get-ArchiveLocationHelpText {
    $folder = Get-ArchivedProfilesBase
    $profiles = $script:ProfilesBase
    return @(
        'Archive zips the selected profile and removes it from this list.'
        ''
        'The ZIP is saved in:'
        $folder
        ''
        'This is not the Windows Recycle Bin.'
        'To use the profile again, unzip that file and move the folder back into:'
        $profiles
    ) -join "`n"
}

function Show-ArchiveLocationDialog {
    param(
        [string]$Title,
        [string]$Message,
        [string]$FolderPath
    )
    if ([string]::IsNullOrWhiteSpace($FolderPath)) {
        $FolderPath = Get-ArchivedProfilesBase
    }
    $script:ArchiveHelpFolder = $FolderPath

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = $Title
    $dlg.ClientSize = New-Object System.Drawing.Size(520, 230)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $txt = New-Object System.Windows.Forms.TextBox
    $txt.Location = New-Object System.Drawing.Point(12, 12)
    $txt.Size = New-Object System.Drawing.Size(496, 160)
    $txt.Multiline = $true
    $txt.ReadOnly = $true
    $txt.ScrollBars = [System.Windows.Forms.ScrollBars]::Vertical
    $txt.Text = $Message
    $dlg.Controls.Add($txt)

    $btnOpen = New-Object System.Windows.Forms.Button
    $btnOpen.Location = New-Object System.Drawing.Point(12, 186)
    $btnOpen.Size = New-Object System.Drawing.Size(120, 28)
    $btnOpen.Text = 'Open folder'
    $dlg.Controls.Add($btnOpen)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(412, 186)
    $btnOk.Size = New-Object System.Drawing.Size(96, 28)
    $btnOk.Text = 'OK'
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $dlg.AcceptButton = $btnOk
    $dlg.CancelButton = $btnOk
    $dlg.Controls.Add($btnOk)

    $btnOpen.Add_Click({
        $folder = $script:ArchiveHelpFolder
        if ([string]::IsNullOrWhiteSpace($folder)) { return }
        if (-not (Test-Path -LiteralPath $folder)) {
            New-Item -ItemType Directory -Path $folder -Force | Out-Null
        }
        Start-Process -FilePath explorer.exe -ArgumentList $folder
    })

    [void]$dlg.ShowDialog($form)
    $dlg.Dispose()
}

function Confirm-ArchiveProfile {
    param(
        [string]$ProfileName,
        [string]$SourceRoot,
        [string]$Kind,
        [string]$ZipPath
    )
    $archiveBase = Get-ArchivedProfilesBase
    $restoreFolder = Get-ProfileRestoreFolderForKind -Kind $Kind
    $kindLine = ''
    if ($Kind -eq 'auto_backup') {
        $kindLine = "This profile is currently an auto-backup.`n"
    }
    $message = @(
        "Archive profile '$ProfileName'?"
        ''
        'It will be zipped and removed from the profile list.'
        'ZIP folder:'
        $archiveBase
        ''
        'File:'
        $ZipPath
        ''
        'The original folder will be removed:'
        $SourceRoot
        $kindLine
        'This is not the Windows Recycle Bin.'
        'To use it again, unzip the file and move the folder back into:'
        $restoreFolder
    ) -join "`n"

    $result = [System.Windows.Forms.MessageBox]::Show(
        $message,
        "Confirm archive",
        [System.Windows.Forms.MessageBoxButtons]::YesNo,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    )
    return ($result -eq [System.Windows.Forms.DialogResult]::Yes)
}

function Invoke-ArchiveProfile {
    param([string]$ProfileName)

    if (Test-FLIMageRunning) {
        throw "Close FLIMage before archiving a profile."
    }

    $storage = Resolve-ProfileStorageRoot $ProfileName
    if (-not $storage) {
        throw "Profile not found: $ProfileName"
    }

    $archiveBase = Get-ArchivedProfilesBase
    New-Item -ItemType Directory -Path $archiveBase -Force | Out-Null

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $zipFileName = "${ProfileName}_${timestamp}.zip"
    $zipPath = Join-Path $archiveBase $zipFileName

    if (Test-Path -LiteralPath $zipPath) {
        throw "Archive file already exists: $zipPath"
    }

    if (-not (Confirm-ArchiveProfile -ProfileName $ProfileName -SourceRoot $storage.Root -Kind $storage.Kind -ZipPath $zipPath)) {
        return $false
    }

    try {
        Compress-Archive -LiteralPath $storage.Root -DestinationPath $zipPath -CompressionLevel Optimal
        Remove-Item -LiteralPath $storage.Root -Recurse -Force
    } catch {
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
        }
        throw
    }

    $done = "Profile '$ProfileName' was removed from the list.`n`nZIP file:`n$zipPath`n`nFolder:`n$archiveBase"
    Show-ArchiveLocationDialog -Title 'Archive complete' -FolderPath $archiveBase -Message $done
    return $true
}

function Invoke-RestoreFromBrowseBackupEntry {
    param(
        $Entry,
        [hashtable]$Components,
        [string]$Scope
    )

    if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) {
        throw "Init_Files folder is missing or invalid. Set Init_Files folder first."
    }

    $displayName = Get-ProfileListDisplayName -Entry $Entry
    $sourceDescription = switch ($Entry.Kind) {
        'auto_backup'      { "Monthly auto-backup: $displayName" }
        'pre_restore'      { "Pre-restore snapshot: $displayName" }
        'profile_history'  { "Profile history: $displayName" }
        default            { $displayName }
    }
    $targetForPreBackup = if ($Entry.RestoreTarget) { $Entry.RestoreTarget } else { $Entry.Name }

    if (-not (Confirm-OverwriteRestore -ProfileName $displayName -Scope $Scope -SourceDescription $sourceDescription)) {
        return $false
    }

    $preRestorePath = Invoke-PreRestoreBackupBeforeRestore -Components $Components -TargetProfileName $targetForPreBackup
    Restore-Profile -StorageRoot $Entry.StorageRoot -Components $Components
    $targetStorage = Resolve-ProfileStorageRoot $targetForPreBackup
    if ($targetStorage) {
        Update-ProfileManifestRestoredAt -ProfileRoot $targetStorage.Root
    }
    Update-Status ("Restored from backup: {0}. Scope: {1}. Pre-restore backup saved." -f $displayName, $Scope)
    [System.Windows.Forms.MessageBox]::Show(
        "Settings restored from:`n$displayName`n`nScope: $Scope`n`nPre-restore backup saved to:`n$preRestorePath`n`nYou can start FLIMage now.",
        "Restore complete",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
    return $true
}

function Test-ProfileExists {
    param([string]$ProfileName)
    $null -ne (Resolve-ProfileStorageRoot $ProfileName)
}

function Confirm-OverwriteRestore {
    param(
        [string]$ProfileName,
        [string]$Scope,
        [string]$SourceDescription = $null
    )
    $preRestoreBase = Get-PreRestoreBackupBase
    $sourceLine = if ($SourceDescription) {
        "Source: $SourceDescription`n`n"
    } else {
        ''
    }
    $message = @"
This will overwrite your live FLIMage settings (Init_Files, etc.) with profile '$ProfileName'.

${sourceLine}Scope: $Scope

Your current settings will be saved automatically before restore (ring buffer of $($script:PreRestoreBackupMaxCount) snapshots under):
$preRestoreBase

Start FLIMage after restoring.
"@
    $result = [System.Windows.Forms.MessageBox]::Show(
        $message,
        "Confirm overwrite",
        [System.Windows.Forms.MessageBoxButtons]::OKCancel,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    )
    return ($result -eq [System.Windows.Forms.DialogResult]::OK)
}

function New-SavePasswordRecord {
    param([Parameter(Mandatory = $true)][string]$Password)
    $salt = New-Object byte[] 16
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($salt)
    } finally {
        $rng.Dispose()
    }
    $iterations = 100000
    $derive = New-Object System.Security.Cryptography.Rfc2898DeriveBytes($Password, $salt, $iterations)
    try {
        $hash = $derive.GetBytes(32)
    } finally {
        $derive.Dispose()
    }
    return [pscustomobject]@{
        Salt       = [Convert]::ToBase64String($salt)
        Hash       = [Convert]::ToBase64String($hash)
        Iterations = $iterations
    }
}

function Test-SavePassword {
    param(
        [string]$Password,
        [string]$SaltB64,
        [string]$HashB64,
        [int]$Iterations
    )
    if ([string]::IsNullOrEmpty($Password) -or [string]::IsNullOrWhiteSpace($SaltB64) -or [string]::IsNullOrWhiteSpace($HashB64)) {
        return $false
    }
    if ($Iterations -lt 1) { return $false }
    try {
        $salt = [Convert]::FromBase64String($SaltB64)
        $expected = [Convert]::FromBase64String($HashB64)
        $derive = New-Object System.Security.Cryptography.Rfc2898DeriveBytes($Password, $salt, $Iterations)
        try {
            $actual = $derive.GetBytes($expected.Length)
        } finally {
            $derive.Dispose()
        }
    } catch {
        return $false
    }
    if ($actual.Length -ne $expected.Length) { return $false }
    $diff = 0
    for ($i = 0; $i -lt $actual.Length; $i++) {
        $diff = $diff -bor ($actual[$i] -bxor $expected[$i])
    }
    return ($diff -eq 0)
}

function Get-ProfilePasswordRecord {
    param([string]$ProfileRoot)
    if ([string]::IsNullOrWhiteSpace($ProfileRoot)) { return $null }
    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { return $null }
    try {
        $json = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not (Test-ManifestHasProperty -Json $json -PropertyName 'password_hash')) { return $null }
        $iterations = 100000
        if ($json.password_iterations) { $iterations = [int]$json.password_iterations }
        return [pscustomobject]@{
            Salt       = [string]$json.password_salt
            Hash       = [string]$json.password_hash
            Iterations = $iterations
        }
    } catch {
        return $null
    }
}

function Set-ProfilePasswordOnManifest {
    param(
        [string]$ProfileRoot,
        [string]$Salt,
        [string]$Hash,
        [int]$Iterations = 100000,
        [switch]$Clear
    )
    $manifestPath = Join-Path $ProfileRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Profile manifest was not found."
    }
    $json = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($propName in @('password_salt', 'password_hash', 'password_iterations')) {
        if ($json.PSObject.Properties[$propName]) {
            $json.PSObject.Properties.Remove($propName)
        }
    }
    if (-not $Clear) {
        $json | Add-Member -NotePropertyName 'password_salt' -NotePropertyValue $Salt -Force
        $json | Add-Member -NotePropertyName 'password_hash' -NotePropertyValue $Hash -Force
        $json | Add-Member -NotePropertyName 'password_iterations' -NotePropertyValue $Iterations -Force
    }
    Write-ProfileManifestJson -ProfileRoot $ProfileRoot -ManifestObject $json
}

function Confirm-OverwriteProfileSave {
    param(
        [string]$ProfileName,
        $PasswordRecord = $null
    )
    $needsPassword = ($null -ne $PasswordRecord -and -not [string]::IsNullOrWhiteSpace([string]$PasswordRecord.Hash))
    if (-not $needsPassword) {
        $message = "Profile '$ProfileName' already exists.`nOverwrite it with the current settings?"
        $result = [System.Windows.Forms.MessageBox]::Show(
            $message,
            "Confirm overwrite save",
            [System.Windows.Forms.MessageBoxButtons]::OKCancel,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        )
        return ($result -eq [System.Windows.Forms.DialogResult]::OK)
    }

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = 'Confirm overwrite save'
    $dlg.ClientSize = New-Object System.Drawing.Size(440, 188)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Location = New-Object System.Drawing.Point(12, 12)
    $lbl.Size = New-Object System.Drawing.Size(416, 72)
    $lbl.Text = "Profile '$ProfileName' already exists and has a save password.`nEnter that password to overwrite it.`nRestore does not require the password."
    $dlg.Controls.Add($lbl)

    $txt = New-Object System.Windows.Forms.TextBox
    $txt.Location = New-Object System.Drawing.Point(12, 92)
    $txt.Size = New-Object System.Drawing.Size(416, 22)
    $txt.UseSystemPasswordChar = $true
    $dlg.Controls.Add($txt)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(250, 140)
    $btnOk.Size = New-Object System.Drawing.Size(84, 28)
    $btnOk.Text = 'OK'
    $dlg.AcceptButton = $btnOk
    $dlg.Controls.Add($btnOk)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Location = New-Object System.Drawing.Point(344, 140)
    $btnCancel.Size = New-Object System.Drawing.Size(84, 28)
    $btnCancel.Text = 'Cancel'
    $dlg.CancelButton = $btnCancel
    $dlg.Controls.Add($btnCancel)

    $btnOk.Add_Click({
        if (-not (Test-SavePassword -Password $txt.Text -SaltB64 $PasswordRecord.Salt -HashB64 $PasswordRecord.Hash -Iterations ([int]$PasswordRecord.Iterations))) {
            [System.Windows.Forms.MessageBox]::Show(
                'Password is incorrect.',
                'Confirm overwrite save',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning
            ) | Out-Null
            return
        }
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $dlg.Close()
    })
    $btnCancel.Add_Click({
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
        $dlg.Close()
    })

    $accepted = ($dlg.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK)
    $dlg.Dispose()
    return $accepted
}

function Show-SaveWithPasswordDialog {
    param(
        [string]$ProfileName,
        [bool]$ProfileExists,
        $PasswordRecord
    )
    $hasPassword = ($null -ne $PasswordRecord -and -not [string]::IsNullOrWhiteSpace([string]$PasswordRecord.Hash))

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = 'Save with password'
    $dlg.ClientSize = New-Object System.Drawing.Size(460, 292)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $intro = "Save current settings into '$ProfileName'.`nFuture overwrites of this profile will require the password.`nRestore does not ask for it."
    if ($ProfileExists) {
        $intro = "Overwrite '$ProfileName' with the current settings.`nFuture overwrites will require the password.`nRestore does not ask for it."
    }
    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Location = New-Object System.Drawing.Point(12, 10)
    $lbl.Size = New-Object System.Drawing.Size(436, 58)
    $lbl.Text = $intro
    $dlg.Controls.Add($lbl)

    $y = 74
    $txtCurrent = $null
    if ($hasPassword) {
        $lblCurrent = New-Object System.Windows.Forms.Label
        $lblCurrent.Location = New-Object System.Drawing.Point(12, $y)
        $lblCurrent.Size = New-Object System.Drawing.Size(436, 18)
        $lblCurrent.Text = 'Current password:'
        $dlg.Controls.Add($lblCurrent)
        $y += 20
        $txtCurrent = New-Object System.Windows.Forms.TextBox
        $txtCurrent.Location = New-Object System.Drawing.Point(12, $y)
        $txtCurrent.Size = New-Object System.Drawing.Size(436, 22)
        $txtCurrent.UseSystemPasswordChar = $true
        $dlg.Controls.Add($txtCurrent)
        $y += 30
    }

    $chkRemove = New-Object System.Windows.Forms.CheckBox
    $chkRemove.Location = New-Object System.Drawing.Point(12, $y)
    $chkRemove.Size = New-Object System.Drawing.Size(436, 22)
    $chkRemove.Text = 'Remove password (overwrite will no longer require it)'
    $chkRemove.Visible = $hasPassword
    $dlg.Controls.Add($chkRemove)
    if ($hasPassword) { $y += 26 }

    $lblNew = New-Object System.Windows.Forms.Label
    $lblNew.Location = New-Object System.Drawing.Point(12, $y)
    $lblNew.Size = New-Object System.Drawing.Size(436, 18)
    $lblNew.Text = 'New password:'
    $dlg.Controls.Add($lblNew)
    $y += 20
    $txtNew = New-Object System.Windows.Forms.TextBox
    $txtNew.Location = New-Object System.Drawing.Point(12, $y)
    $txtNew.Size = New-Object System.Drawing.Size(436, 22)
    $txtNew.UseSystemPasswordChar = $true
    $dlg.Controls.Add($txtNew)
    $y += 28
    $lblConfirm = New-Object System.Windows.Forms.Label
    $lblConfirm.Location = New-Object System.Drawing.Point(12, $y)
    $lblConfirm.Size = New-Object System.Drawing.Size(200, 18)
    $lblConfirm.Text = 'Confirm password:'
    $dlg.Controls.Add($lblConfirm)
    $y += 20
    $txtConfirm = New-Object System.Windows.Forms.TextBox
    $txtConfirm.Location = New-Object System.Drawing.Point(12, $y)
    $txtConfirm.Size = New-Object System.Drawing.Size(436, 22)
    $txtConfirm.UseSystemPasswordChar = $true
    $dlg.Controls.Add($txtConfirm)
    $y += 36

    $dlg.ClientSize = New-Object System.Drawing.Size(460, ($y + 40))

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(270, $y)
    $btnOk.Size = New-Object System.Drawing.Size(84, 28)
    $btnOk.Text = 'Save'
    $dlg.AcceptButton = $btnOk
    $dlg.Controls.Add($btnOk)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Location = New-Object System.Drawing.Point(364, $y)
    $btnCancel.Size = New-Object System.Drawing.Size(84, 28)
    $btnCancel.Text = 'Cancel'
    $dlg.CancelButton = $btnCancel
    $dlg.Controls.Add($btnCancel)

    $script:PwDlg = @{
        Dialog         = $dlg
        HasPassword    = $hasPassword
        PasswordRecord = $PasswordRecord
        TxtCurrent     = $txtCurrent
        ChkRemove      = $chkRemove
        TxtNew         = $txtNew
        TxtConfirm     = $txtConfirm
        LblNew         = $lblNew
        LblConfirm     = $lblConfirm
    }

    $chkRemove.Add_CheckedChanged({
        $state = $script:PwDlg
        $enabled = -not $state.ChkRemove.Checked
        $state.TxtNew.Enabled = $enabled
        $state.TxtConfirm.Enabled = $enabled
        $state.LblNew.Enabled = $enabled
        $state.LblConfirm.Enabled = $enabled
    })

    $btnOk.Add_Click({
        try {
            $state = $script:PwDlg
            $dialog = $state.Dialog
            if ($state.HasPassword) {
                $current = ''
                if ($state.TxtCurrent) { $current = $state.TxtCurrent.Text }
                $record = $state.PasswordRecord
                if (-not (Test-SavePassword -Password $current -SaltB64 $record.Salt -HashB64 $record.Hash -Iterations ([int]$record.Iterations))) {
                    [System.Windows.Forms.MessageBox]::Show(
                        'Current password is incorrect.',
                        'Save with password',
                        [System.Windows.Forms.MessageBoxButtons]::OK,
                        [System.Windows.Forms.MessageBoxIcon]::Warning
                    ) | Out-Null
                    return
                }
            }
            if ($state.ChkRemove.Checked) {
                $dialog.Tag = [pscustomobject]@{ Clear = $true; Salt = $null; Hash = $null; Iterations = 0 }
                $dialog.DialogResult = [System.Windows.Forms.DialogResult]::OK
                $dialog.Close()
                return
            }
            if ([string]::IsNullOrEmpty($state.TxtNew.Text)) {
                [System.Windows.Forms.MessageBox]::Show(
                    'Enter a password.',
                    'Save with password',
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Warning
                ) | Out-Null
                return
            }
            if ($state.TxtNew.Text -cne $state.TxtConfirm.Text) {
                [System.Windows.Forms.MessageBox]::Show(
                    'Password and confirmation do not match.',
                    'Save with password',
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Warning
                ) | Out-Null
                return
            }
            $rec = New-SavePasswordRecord -Password $state.TxtNew.Text
            $dialog.Tag = [pscustomobject]@{
                Clear      = $false
                Salt       = $rec.Salt
                Hash       = $rec.Hash
                Iterations = $rec.Iterations
            }
            $dialog.DialogResult = [System.Windows.Forms.DialogResult]::OK
            $dialog.Close()
        } catch {
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                'Save with password',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })
    $btnCancel.Add_Click({
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
        $dlg.Close()
    })

    $accepted = ($dlg.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK)
    $result = $dlg.Tag
    $dlg.Dispose()
    if (-not $accepted) { return $null }
    return $result
}

function Get-SharedKeyCopyCandidates {
    param(
        [string]$InitFilesPath,
        [ValidateSet('Default', 'Device')]
        [string]$Kind
    )
    if ([string]::IsNullOrWhiteSpace($InitFilesPath) -or -not (Test-Path -LiteralPath $InitFilesPath)) {
        return @()
    }
    $pattern = if ($Kind -eq 'Default') { 'Default-*.txt' } else { 'FLIM_deviceFile_V*.txt' }
    return @(Get-ChildItem -LiteralPath $InitFilesPath -Filter $pattern -File -ErrorAction SilentlyContinue | Sort-Object Name)
}

function Test-SharedKeyCopySameFamily {
    param(
        [string]$SourceName,
        [string]$TargetName
    )
    $sourceIsDefault = $SourceName -like 'Default-*.txt'
    $targetIsDefault = $TargetName -like 'Default-*.txt'
    $sourceIsDevice = $SourceName -like 'FLIM_deviceFile_V*.txt'
    $targetIsDevice = $TargetName -like 'FLIM_deviceFile_V*.txt'
    if ($sourceIsDefault -and $targetIsDefault) { return $true }
    if ($sourceIsDevice -and $targetIsDevice) { return $true }
    return $false
}

function Update-CopySharedKeyDialogLists {
    param(
        [ValidateSet('Default', 'Device')]
        [string]$Kind
    )
    $ui = $script:CopyKeyUi
    $ui.Kind = $Kind
    $files = @(Get-SharedKeyCopyCandidates -InitFilesPath $ui.InitFilesPath -Kind $Kind)
    $ui.ByName = @{}
    $ui.Source.Items.Clear()
    $ui.Target.Items.Clear()
    foreach ($file in $files) {
        $ui.ByName[$file.Name] = $file.FullName
        [void]$ui.Source.Items.Add($file.Name)
        [void]$ui.Target.Items.Add($file.Name)
    }

    $noun = if ($Kind -eq 'Default') { 'Default file' } else { 'FLIM_deviceFile_V* file' }
    if ($files.Count -lt 2) {
        $ui.Hint.Text = "Need at least two $noun files in Init_Files.`nDefault files and device files are never mixed."
        $ui.CopyButton.Enabled = $false
        return
    }

    $ui.CopyButton.Enabled = $true
    $master = $files | Sort-Object LastWriteTimeUtc, Name -Descending | Select-Object -First 1
    $ui.Source.SelectedItem = $master.Name
    $targetIndex = 0
    if ($ui.Target.Items.Count -gt 1 -and [string]$ui.Target.Items[0] -eq [string]$ui.Source.SelectedItem) {
        $targetIndex = 1
    }
    $ui.Target.SelectedIndex = $targetIndex

    $extra = "`nState.Acq.version on the target is not overwritten."
    if ($Kind -eq 'Default' -and $script:PreserveUncagingCalibOnRestore) {
        $extra += "`nUncaging calibration keys on the target are kept."
    }
    $ui.Hint.Text = "Copy shared keys from one $noun onto another in the live Init_Files folder.`nDefault files and device files are never mixed.`nKeys only in the target stay. Keys only in the source are not added.$extra"
}

function Show-CopySharedKeysDialog {
    param([string]$InitFilesPath)

    $defaultCount = @(Get-SharedKeyCopyCandidates -InitFilesPath $InitFilesPath -Kind 'Default').Count
    $deviceCount = @(Get-SharedKeyCopyCandidates -InitFilesPath $InitFilesPath -Kind 'Device').Count
    if ($defaultCount -lt 2 -and $deviceCount -lt 2) {
        [System.Windows.Forms.MessageBox]::Show(
            "Need at least two Default-*.txt files, or two FLIM_deviceFile_V*.txt files, in Init_Files.`n`n$InitFilesPath",
            'Copy shared keys',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return $null
    }

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = 'Copy shared keys'
    $dlg.ClientSize = New-Object System.Drawing.Size(560, 310)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Location = New-Object System.Drawing.Point(12, 10)
    $lbl.Size = New-Object System.Drawing.Size(536, 96)
    $dlg.Controls.Add($lbl)

    $radDefault = New-Object System.Windows.Forms.RadioButton
    $radDefault.Location = New-Object System.Drawing.Point(12, 112)
    $radDefault.Size = New-Object System.Drawing.Size(160, 22)
    $radDefault.Text = 'Default files'
    $dlg.Controls.Add($radDefault)

    $radDevice = New-Object System.Windows.Forms.RadioButton
    $radDevice.Location = New-Object System.Drawing.Point(180, 112)
    $radDevice.Size = New-Object System.Drawing.Size(360, 22)
    $radDevice.Text = 'Device files (FLIM_deviceFile_V*)'
    $dlg.Controls.Add($radDevice)

    $lblSource = New-Object System.Windows.Forms.Label
    $lblSource.Location = New-Object System.Drawing.Point(12, 146)
    $lblSource.Size = New-Object System.Drawing.Size(80, 20)
    $lblSource.Text = 'Source:'
    $dlg.Controls.Add($lblSource)

    $cmbSource = New-Object System.Windows.Forms.ComboBox
    $cmbSource.Location = New-Object System.Drawing.Point(100, 144)
    $cmbSource.Size = New-Object System.Drawing.Size(444, 22)
    $cmbSource.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
    $dlg.Controls.Add($cmbSource)

    $lblTarget = New-Object System.Windows.Forms.Label
    $lblTarget.Location = New-Object System.Drawing.Point(12, 178)
    $lblTarget.Size = New-Object System.Drawing.Size(80, 20)
    $lblTarget.Text = 'Target:'
    $dlg.Controls.Add($lblTarget)

    $cmbTarget = New-Object System.Windows.Forms.ComboBox
    $cmbTarget.Location = New-Object System.Drawing.Point(100, 176)
    $cmbTarget.Size = New-Object System.Drawing.Size(444, 22)
    $cmbTarget.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
    $dlg.Controls.Add($cmbTarget)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(340, 266)
    $btnOk.Size = New-Object System.Drawing.Size(100, 28)
    $btnOk.Text = 'Copy'
    $dlg.AcceptButton = $btnOk
    $dlg.Controls.Add($btnOk)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Location = New-Object System.Drawing.Point(448, 266)
    $btnCancel.Size = New-Object System.Drawing.Size(96, 28)
    $btnCancel.Text = 'Cancel'
    $dlg.CancelButton = $btnCancel
    $dlg.Controls.Add($btnCancel)

    $script:CopyKeyUi = @{
        Dialog      = $dlg
        InitFilesPath = $InitFilesPath
        Hint        = $lbl
        Source      = $cmbSource
        Target      = $cmbTarget
        CopyButton  = $btnOk
        ByName      = @{}
        Kind        = 'Default'
    }

    $radDefault.Tag = 'Default'
    $radDevice.Tag = 'Device'
    $radDefault.Add_CheckedChanged({
        param($sender, $eventArgs)
        if ($script:CopyKeySync) { return }
        if (-not $sender.Checked) { return }
        Update-CopySharedKeyDialogLists -Kind ([string]$sender.Tag)
    })
    $radDevice.Add_CheckedChanged({
        param($sender, $eventArgs)
        if ($script:CopyKeySync) { return }
        if (-not $sender.Checked) { return }
        Update-CopySharedKeyDialogLists -Kind ([string]$sender.Tag)
    })

    $btnOk.Add_Click({
        try {
            $ui = $script:CopyKeyUi
            $src = [string]$ui.Source.SelectedItem
            $dstName = [string]$ui.Target.SelectedItem
            if ([string]::IsNullOrWhiteSpace($src) -or [string]::IsNullOrWhiteSpace($dstName)) {
                [System.Windows.Forms.MessageBox]::Show(
                    'Choose a source file and a target file.',
                    'Copy shared keys',
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Information
                ) | Out-Null
                return
            }
            if ($src -eq $dstName) {
                [System.Windows.Forms.MessageBox]::Show(
                    'Choose a different target file.',
                    'Copy shared keys',
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Information
                ) | Out-Null
                return
            }
            if (-not (Test-SharedKeyCopySameFamily -SourceName $src -TargetName $dstName)) {
                [System.Windows.Forms.MessageBox]::Show(
                    'Default files and device files cannot be mixed.',
                    'Copy shared keys',
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Warning
                ) | Out-Null
                return
            }
            $ui.Dialog.Tag = [pscustomobject]@{
                Kind       = $ui.Kind
                SourceName = $src
                TargetName = $dstName
                SourcePath = $ui.ByName[$src]
                TargetPath = $ui.ByName[$dstName]
            }
            $ui.Dialog.DialogResult = [System.Windows.Forms.DialogResult]::OK
            $ui.Dialog.Close()
        } catch {
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                'Copy shared keys',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })
    $btnCancel.Add_Click({
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
        $dlg.Close()
    })

    $startKind = if ($defaultCount -ge 2) { 'Default' } else { 'Device' }
    $script:CopyKeySync = $true
    $radDefault.Checked = ($startKind -eq 'Default')
    $radDevice.Checked = ($startKind -eq 'Device')
    $script:CopyKeySync = $false
    Update-CopySharedKeyDialogLists -Kind $startKind

    $accepted = ($dlg.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK)
    $picked = $dlg.Tag
    $dlg.Dispose()
    if (-not $accepted) { return $null }
    return $picked
}

Initialize-InitFilesPathAtStartup

# --- GUI ---

$script:ProfileViewMode = 'profiles'
$script:ProfileSortColumn = 2
$script:ProfileSortAscending = $false
$script:HistoryProfileName = $null
$script:ProfileEntryByTag = @{}
$script:AdvancedExpanded = $false

$form = New-Object System.Windows.Forms.Form
$form.Text = 'FLIMage Profile Manager'
$form.ClientSize = New-Object System.Drawing.Size(660, 548)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
$form.MaximizeBox = $false

$lblInit = New-Object System.Windows.Forms.Label
$lblInit.Location = New-Object System.Drawing.Point(12, 12)
$lblInit.Size = New-Object System.Drawing.Size(110, 20)
$lblInit.Text = 'Init_Files folder:'
$form.Controls.Add($lblInit)

$txtInit = New-Object System.Windows.Forms.TextBox
$txtInit.Location = New-Object System.Drawing.Point(124, 10)
$txtInit.Size = New-Object System.Drawing.Size(410, 22)
$txtInit.Text = $script:InitFilesPath
$form.Controls.Add($txtInit)

$btnBrowseInit = New-Object System.Windows.Forms.Button
$btnBrowseInit.Location = New-Object System.Drawing.Point(544, 8)
$btnBrowseInit.Size = New-Object System.Drawing.Size(104, 25)
$btnBrowseInit.Text = 'Browse'
$form.Controls.Add($btnBrowseInit)

$lblBase = New-Object System.Windows.Forms.Label
$lblBase.Location = New-Object System.Drawing.Point(12, 42)
$lblBase.Size = New-Object System.Drawing.Size(110, 20)
$lblBase.Text = 'Profiles folder:'
$form.Controls.Add($lblBase)

$txtBase = New-Object System.Windows.Forms.TextBox
$txtBase.Location = New-Object System.Drawing.Point(124, 40)
$txtBase.Size = New-Object System.Drawing.Size(410, 22)
$txtBase.Text = $script:ProfilesBase
$form.Controls.Add($txtBase)

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Location = New-Object System.Drawing.Point(544, 38)
$btnBrowse.Size = New-Object System.Drawing.Size(104, 25)
$btnBrowse.Text = 'Browse'
$form.Controls.Add($btnBrowse)

$lblFilter = New-Object System.Windows.Forms.Label
$lblFilter.Location = New-Object System.Drawing.Point(12, 74)
$lblFilter.AutoSize = $true
$lblFilter.Text = 'Filter:'
$form.Controls.Add($lblFilter)

$txtFilter = New-Object System.Windows.Forms.TextBox
$txtFilter.Location = New-Object System.Drawing.Point(56, 70)
$txtFilter.Size = New-Object System.Drawing.Size(478, 22)
$form.Controls.Add($txtFilter)

$btnRefresh = New-Object System.Windows.Forms.Button
$btnRefresh.Location = New-Object System.Drawing.Point(544, 68)
$btnRefresh.Size = New-Object System.Drawing.Size(104, 25)
$btnRefresh.Text = 'Refresh'
$form.Controls.Add($btnRefresh)

$lblViewMode = New-Object System.Windows.Forms.Label
$lblViewMode.Location = New-Object System.Drawing.Point(12, 98)
$lblViewMode.Size = New-Object System.Drawing.Size(360, 18)
$lblViewMode.Text = 'Profiles'
$form.Controls.Add($lblViewMode)

$btnBackToProfiles = New-Object System.Windows.Forms.Button
$btnBackToProfiles.Location = New-Object System.Drawing.Point(430, 94)
$btnBackToProfiles.Size = New-Object System.Drawing.Size(218, 24)
$btnBackToProfiles.Text = 'Back to profiles'
$btnBackToProfiles.Visible = $false
$form.Controls.Add($btnBackToProfiles)

$list = New-Object System.Windows.Forms.ListView
$list.Location = New-Object System.Drawing.Point(12, 122)
$list.Size = New-Object System.Drawing.Size(636, 220)
$list.View = [System.Windows.Forms.View]::Details
$list.FullRowSelect = $true
$list.HideSelection = $false
$list.MultiSelect = $false
$list.GridLines = $true
[void]$list.Columns.Add('Profile', 300)
[void]$list.Columns.Add('Modified', 160)
[void]$list.Columns.Add('Restored', 160)
$form.Controls.Add($list)

$lblCount = New-Object System.Windows.Forms.Label
$lblCount.Location = New-Object System.Drawing.Point(12, 346)
$lblCount.Size = New-Object System.Drawing.Size(636, 18)
$form.Controls.Add($lblCount)

$lblSelection = New-Object System.Windows.Forms.Label
$lblSelection.Location = New-Object System.Drawing.Point(12, 364)
$lblSelection.Size = New-Object System.Drawing.Size(636, 36)
$lblSelection.Font = New-Object System.Drawing.Font($lblSelection.Font.FontFamily, ($lblSelection.Font.Size * 2), $lblSelection.Font.Style)
$lblSelection.AutoEllipsis = $true
$lblSelection.Text = 'No row selected.'
$form.Controls.Add($lblSelection)

$btnSave = New-Object System.Windows.Forms.Button
$btnSave.Location = New-Object System.Drawing.Point(12, 408)
$btnSave.Size = New-Object System.Drawing.Size(200, 34)
$btnSave.Text = 'Save'
$form.Controls.Add($btnSave)

$btnRestore = New-Object System.Windows.Forms.Button
$btnRestore.Location = New-Object System.Drawing.Point(220, 408)
$btnRestore.Size = New-Object System.Drawing.Size(200, 34)
$btnRestore.Text = 'Restore'
$form.Controls.Add($btnRestore)

$btnNewProfile = New-Object System.Windows.Forms.Button
$btnNewProfile.Location = New-Object System.Drawing.Point(428, 408)
$btnNewProfile.Size = New-Object System.Drawing.Size(220, 34)
$btnNewProfile.Text = 'New profile'
$form.Controls.Add($btnNewProfile)

$lblFlimageWarning = New-Object System.Windows.Forms.Label
$lblFlimageWarning.Location = New-Object System.Drawing.Point(12, 448)
$lblFlimageWarning.Size = New-Object System.Drawing.Size(636, 18)
$lblFlimageWarning.Text = 'Close FLIMage before Save or Restore.'
$lblFlimageWarning.ForeColor = [System.Drawing.Color]::FromArgb(192, 0, 0)
$lblFlimageWarning.Visible = $false
$form.Controls.Add($lblFlimageWarning)

$btnAdvancedToggle = New-Object System.Windows.Forms.Button
$btnAdvancedToggle.Location = New-Object System.Drawing.Point(12, 472)
$btnAdvancedToggle.Size = New-Object System.Drawing.Size(140, 28)
$btnAdvancedToggle.Text = 'Advanced >>'
$form.Controls.Add($btnAdvancedToggle)

$lblStatus = New-Object System.Windows.Forms.Label
$lblStatus.Location = New-Object System.Drawing.Point(160, 468)
$lblStatus.Size = New-Object System.Drawing.Size(488, 40)
$lblStatus.Text = 'Ready. Scope: all.'
$form.Controls.Add($lblStatus)

$grpAdvanced = New-Object System.Windows.Forms.GroupBox
$grpAdvanced.Location = New-Object System.Drawing.Point(12, 512)
$grpAdvanced.Size = New-Object System.Drawing.Size(636, 360)
$grpAdvanced.Text = 'Advanced'
$grpAdvanced.Visible = $false
$form.Controls.Add($grpAdvanced)

$chkAll = New-Object System.Windows.Forms.CheckBox
$chkAll.Location = New-Object System.Drawing.Point(14, 22)
$chkAll.Size = New-Object System.Drawing.Size(200, 22)
$chkAll.Text = 'All'
$chkAll.Checked = $true
$grpAdvanced.Controls.Add($chkAll)

$script:ComponentCheckboxes = @{}
$y = 46
foreach ($def in Get-ComponentDefinitions) {
    $chk = New-Object System.Windows.Forms.CheckBox
    $chk.Location = New-Object System.Drawing.Point(28, $y)
    $chk.Size = New-Object System.Drawing.Size(590, 22)
    $chk.Text = $def.Label
    $chk.Checked = $true
    $chk.Tag = $def.Key
    $grpAdvanced.Controls.Add($chk)
    $script:ComponentCheckboxes[$def.Key] = $chk
    $y += 24
}

$chkPreserveUncagingCalib = New-Object System.Windows.Forms.CheckBox
$chkPreserveUncagingCalib.Location = New-Object System.Drawing.Point(14, $y)
$chkPreserveUncagingCalib.AutoSize = $true
$chkPreserveUncagingCalib.MaximumSize = New-Object System.Drawing.Size(600, 0)
$chkPreserveUncagingCalib.Text = 'Restore profile settings except uncaging calibration (recommended)'
$chkPreserveUncagingCalib.Checked = $script:PreserveUncagingCalibOnRestore
$grpAdvanced.Controls.Add($chkPreserveUncagingCalib)
$y += 32

$btnArchive = New-Object System.Windows.Forms.Button
$btnArchive.Location = New-Object System.Drawing.Point(14, $y)
$btnArchive.Size = New-Object System.Drawing.Size(100, 28)
$btnArchive.Text = 'Archive'
$grpAdvanced.Controls.Add($btnArchive)

$script:UiToolTip = New-Object System.Windows.Forms.ToolTip
$script:UiToolTip.AutoPopDelay = 30000
$script:UiToolTip.InitialDelay = 300
$script:UiToolTip.ReshowDelay = 100
$script:UiToolTip.ShowAlways = $true

$btnRename = New-Object System.Windows.Forms.Button
$btnRename.Location = New-Object System.Drawing.Point(122, $y)
$btnRename.Size = New-Object System.Drawing.Size(100, 28)
$btnRename.Text = 'Rename'
$grpAdvanced.Controls.Add($btnRename)

$btnMonthlyRestore = New-Object System.Windows.Forms.Button
$btnMonthlyRestore.Location = New-Object System.Drawing.Point(230, $y)
$btnMonthlyRestore.Size = New-Object System.Drawing.Size(190, 28)
$btnMonthlyRestore.Text = 'Monthly backups'
$grpAdvanced.Controls.Add($btnMonthlyRestore)

$btnPreRestoreBrowse = New-Object System.Windows.Forms.Button
$btnPreRestoreBrowse.Location = New-Object System.Drawing.Point(428, $y)
$btnPreRestoreBrowse.Size = New-Object System.Drawing.Size(190, 28)
$btnPreRestoreBrowse.Text = 'Recent 30 backups'
$grpAdvanced.Controls.Add($btnPreRestoreBrowse)
$y += 36

$btnProfileHistory = New-Object System.Windows.Forms.Button
$btnProfileHistory.Location = New-Object System.Drawing.Point(14, $y)
$btnProfileHistory.Size = New-Object System.Drawing.Size(208, 28)
$btnProfileHistory.Text = 'Profile history (10)'
$grpAdvanced.Controls.Add($btnProfileHistory)

$btnCopyKeys = New-Object System.Windows.Forms.Button
$btnCopyKeys.Location = New-Object System.Drawing.Point(230, $y)
$btnCopyKeys.Size = New-Object System.Drawing.Size(388, 28)
$btnCopyKeys.Text = 'Copy shared keys...'
$grpAdvanced.Controls.Add($btnCopyKeys)
$y += 36

$btnSaveWithPassword = New-Object System.Windows.Forms.Button
$btnSaveWithPassword.Location = New-Object System.Drawing.Point(14, $y)
$btnSaveWithPassword.Size = New-Object System.Drawing.Size(208, 28)
$btnSaveWithPassword.Text = 'Save with password...'
$grpAdvanced.Controls.Add($btnSaveWithPassword)

$lblPasswordHint = New-Object System.Windows.Forms.Label
$lblPasswordHint.Location = New-Object System.Drawing.Point(230, ($y + 6))
$lblPasswordHint.Size = New-Object System.Drawing.Size(390, 20)
$lblPasswordHint.Text = 'Blocks overwrite only. Restore stays open.'
$grpAdvanced.Controls.Add($lblPasswordHint)

$grpAdvanced.Height = $y + 48

$script:SyncingCheckboxes = $false

function Set-AllComponentCheckboxes {
    param([bool]$Checked)
    $script:SyncingCheckboxes = $true
    foreach ($chk in $script:ComponentCheckboxes.Values) {
        $chk.Checked = $Checked
    }
    $chkAll.Checked = $Checked
    $script:SyncingCheckboxes = $false
}

function Update-AllCheckboxFromComponents {
    if ($script:SyncingCheckboxes) { return }
    $script:SyncingCheckboxes = $true
    $allOn = ($script:ComponentCheckboxes.Values | Where-Object { -not $_.Checked }).Count -eq 0
    $chkAll.Checked = $allOn
    $script:SyncingCheckboxes = $false
}

function Get-SelectedComponents {
    $result = @{}
    foreach ($key in $script:ComponentCheckboxes.Keys) {
        $result[$key] = $script:ComponentCheckboxes[$key].Checked
    }
    return $result
}

function Get-SelectedComponentSummary {
    $parts = @()
    foreach ($def in Get-ComponentDefinitions) {
        if ($script:ComponentCheckboxes[$def.Key].Checked) {
            $parts += $def.Key
        }
    }
    if ($parts.Count -eq 0) { return '(none)' }
    return ($parts -join ', ')
}

function Update-InitFilesPathFromUi {
    $path = $txtInit.Text.Trim()
    if (-not [string]::IsNullOrWhiteSpace($path)) {
        $script:InitFilesPath = [System.IO.Path]::GetFullPath($path)
        $txtInit.Text = $script:InitFilesPath
    }
}

function Update-ProfilesBaseFromUi {
    $path = $txtBase.Text.Trim()
    if (-not [string]::IsNullOrWhiteSpace($path)) {
        $script:ProfilesBase = $path
    }
}

function Sync-ManagerConfigFromUi {
    Update-InitFilesPathFromUi
    Update-ProfilesBaseFromUi
    if ($chkPreserveUncagingCalib) {
        $script:PreserveUncagingCalibOnRestore = $chkPreserveUncagingCalib.Checked
    }
    Save-ManagerConfig
}

function Update-Status {
    param([string]$Message)
    $lblStatus.Text = $Message
    $form.Refresh()
}

function Update-FlimageRunningWarning {
    $running = Test-FLIMageRunning
    if ($script:LastFlimageRunning -eq $running) { return }
    $script:LastFlimageRunning = $running
    $lblFlimageWarning.Visible = $running
}

function Confirm-NotRunning {
    if (-not (Test-FLIMageRunning)) { return $true }
    [System.Windows.Forms.MessageBox]::Show(
        'FLIMage appears to be running. Close it before Save or Restore.',
        'Warning',
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    ) | Out-Null
    return $false
}

function Get-MainProfileEntries {
    switch ($script:ProfileViewMode) {
        'monthly' { return @(Get-MonthlyBackupListEntries) }
        'prerestore' { return @(Get-PreRestoreBackupListEntries) }
        'profilehistory' { return @(Get-ProfileHistoryListEntries -ProfileName $script:HistoryProfileName) }
        default { return @(Get-ProfileListEntries) }
    }
}

function Get-SelectedProfileListEntry {
    if ($list.SelectedItems.Count -eq 0) { return $null }
    $tag = [string]$list.SelectedItems[0].Tag
    if ($script:ProfileEntryByTag.ContainsKey($tag)) {
        return $script:ProfileEntryByTag[$tag]
    }
    return $null
}

function Update-ProfileColumnHeaders {
    $profileHeader = 'Profile'
    $modifiedHeader = 'Modified'
    $restoredHeader = 'Restored'
    if ($script:ProfileSortColumn -eq 0) {
        $profileHeader += $(if ($script:ProfileSortAscending) { ' ^' } else { ' v' })
    } elseif ($script:ProfileSortColumn -eq 1) {
        $modifiedHeader += $(if ($script:ProfileSortAscending) { ' ^' } else { ' v' })
    } elseif ($script:ProfileSortColumn -eq 2) {
        $restoredHeader += $(if ($script:ProfileSortAscending) { ' ^' } else { ' v' })
    }
    $list.Columns[0].Text = $profileHeader
    $list.Columns[1].Text = $modifiedHeader
    $list.Columns[2].Text = $restoredHeader
}

function Update-ProfileSelectionLabel {
    $entry = Get-SelectedProfileListEntry
    if (-not $entry) {
        $lblSelection.Text = 'No row selected.'
        return
    }
    $suffix = ''
    if ($entry.HasSavePassword -and $script:ProfileViewMode -eq 'profiles') {
        $suffix = '  |  overwrite requires password'
    }
    $lblSelection.Text = ('Selected: {0}{1}' -f (Get-ProfileListDisplayName -Entry $entry), $suffix)
}

function Update-ProfileActionButtons {
    $isProfiles = ($script:ProfileViewMode -eq 'profiles')
    $hasSelection = ($list.SelectedItems.Count -gt 0)
    $btnSave.Enabled = $isProfiles
    $btnArchive.Enabled = $isProfiles -and $hasSelection
    $btnRename.Enabled = $isProfiles -and $hasSelection
    $btnProfileHistory.Enabled = $isProfiles -and $hasSelection
    $btnBackToProfiles.Visible = -not $isProfiles
    if ($isProfiles) {
        $lblViewMode.Text = 'Profiles'
        $btnRestore.Text = 'Restore'
    }
}

function Select-MainProfileByTag {
    param([string]$Tag)
    if ([string]::IsNullOrWhiteSpace($Tag)) { return }
    foreach ($item in $list.Items) {
        if ([string]$item.Tag -eq $Tag) {
            $item.Selected = $true
            $item.Focused = $true
            $item.EnsureVisible()
            break
        }
    }
}

function Select-MainProfileByName {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return }
    foreach ($entry in (Get-MainProfileEntries)) {
        if ($entry.Name -eq $Name) {
            Select-MainProfileByTag -Tag $entry.StorageRoot
            break
        }
    }
}

function Update-MainProfileList {
    $keepTag = $null
    if ($list.SelectedItems.Count -gt 0) {
        $keepTag = [string]$list.SelectedItems[0].Tag
    }

    $list.BeginUpdate()
    $list.Items.Clear()
    $script:ProfileEntryByTag = @{}
    $filter = $txtFilter.Text.Trim()
    $allEntries = @(Get-MainProfileEntries)
    $filtered = @($allEntries | Where-Object { Test-ProfileListEntryMatchesFilter -Entry $_ -Filter $filter })
    $sorted = @(Sort-ProfileListEntries -Entries $filtered -SortColumn $script:ProfileSortColumn -Ascending $script:ProfileSortAscending)
    foreach ($entry in $sorted) {
        $tag = [string]$entry.StorageRoot
        $script:ProfileEntryByTag[$tag] = $entry
        $item = New-Object System.Windows.Forms.ListViewItem((Get-ProfileListDisplayName -Entry $entry))
        [void]$item.SubItems.Add([string]$entry.ModifiedAt)
        [void]$item.SubItems.Add([string]$entry.RestoredAt)
        $item.Tag = $tag
        [void]$list.Items.Add($item)
    }
    $list.EndUpdate()
    Update-ProfileColumnHeaders
    $countLabel = switch ($script:ProfileViewMode) {
        'monthly' { 'monthly backup(s)' }
        'prerestore' { 'pre-restore snapshot(s)' }
        'profilehistory' { 'history snapshot(s)' }
        default { 'profile(s)' }
    }
    $lblCount.Text = if ($filter) {
        ('{0} shown / {1} total' -f $sorted.Count, $allEntries.Count)
    } else {
        ('{0} {1}' -f $allEntries.Count, $countLabel)
    }
    if ($keepTag -and $script:ProfileEntryByTag.ContainsKey($keepTag)) {
        Select-MainProfileByTag -Tag $keepTag
    } elseif ($list.Items.Count -gt 0) {
        $list.Items[0].Selected = $true
        $list.Items[0].Focused = $true
        $list.EnsureVisible(0)
    }
    Update-ProfileActionButtons
    Update-ProfileSelectionLabel
}

function Set-ProfileViewMode {
    param(
        [string]$Mode,
        [string]$HistoryName = $null
    )
    $script:ProfileViewMode = $Mode
    if ($Mode -eq 'profiles') {
        $script:HistoryProfileName = $null
        $script:ProfileSortColumn = 2
        $script:ProfileSortAscending = $false
        $lblViewMode.Text = 'Profiles'
        $btnRestore.Text = 'Restore'
    } elseif ($Mode -eq 'monthly') {
        $script:ProfileSortColumn = 1
        $script:ProfileSortAscending = $false
        $lblViewMode.Text = 'Monthly auto-backups (restore only)'
        $btnRestore.Text = 'Restore backup'
    } elseif ($Mode -eq 'profilehistory') {
        $script:HistoryProfileName = $HistoryName
        $script:ProfileSortColumn = 1
        $script:ProfileSortAscending = $false
        $lblViewMode.Text = "Profile history: $HistoryName (restore only)"
        $btnRestore.Text = 'Restore backup'
    } else {
        $script:ProfileSortColumn = 1
        $script:ProfileSortAscending = $false
        $lblViewMode.Text = 'Recent pre-restore backups (restore only)'
        $btnRestore.Text = 'Restore backup'
    }
    Update-MainProfileList
}

function Set-AdvancedPanelVisible {
    param([bool]$Visible)
    $script:AdvancedExpanded = $Visible
    $grpAdvanced.Visible = $Visible
    $wanted = $grpAdvanced.Bottom + 16
    if ($Visible) {
        $btnAdvancedToggle.Text = 'Advanced <<'
        $screen = [System.Windows.Forms.Screen]::FromControl($form).WorkingArea
        $chrome = $form.Height - $form.ClientSize.Height
        $maxClientH = [Math]::Max(560, $screen.Height - $chrome - 24)
        $form.ClientSize = New-Object System.Drawing.Size(660, ([Math]::Min($wanted, $maxClientH)))
        if ($maxClientH -lt $wanted) {
            $form.AutoScroll = $true
            $form.AutoScrollMinSize = New-Object System.Drawing.Size(0, $wanted)
        } else {
            $form.AutoScroll = $false
            $form.AutoScrollMinSize = New-Object System.Drawing.Size(0, 0)
        }
    } else {
        $btnAdvancedToggle.Text = 'Advanced >>'
        $form.AutoScroll = $false
        $form.AutoScrollMinSize = New-Object System.Drawing.Size(0, 0)
        $form.AutoScrollPosition = New-Object System.Drawing.Point(0, 0)
        $form.ClientSize = New-Object System.Drawing.Size(660, 548)
    }
}

function Update-ArchiveLocationHint {
    if (-not $script:UiToolTip) { return }
    $text = Get-ArchiveLocationHelpText
    $script:UiToolTip.SetToolTip($btnArchive, $text)
}

function Refresh-ProfileList {
    Sync-ManagerConfigFromUi
    if (-not (Test-Path -LiteralPath $script:ProfilesBase)) {
        New-Item -ItemType Directory -Path $script:ProfilesBase -Force | Out-Null
    }
    Update-MainProfileList
    Update-ArchiveLocationHint
    $initOk = if (Test-InitFilesPathValid (Get-InitFilesPath)) { 'OK' } else { 'INVALID' }
    Update-Status ("Init_Files ({0}). Scope: {1}." -f $initOk, (Get-SelectedComponentSummary))
    Update-FlimageRunningWarning
}

function Invoke-SaveCurrentSettings {
    param(
        [Parameter(Mandatory = $true)][string]$ProfileName,
        [switch]$SkipOverwriteConfirm,
        $PasswordAction = $null
    )
    if (-not (Confirm-NotRunning)) { return }
    if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) {
        throw 'Init_Files folder is missing or invalid. Set Init_Files folder first.'
    }
    if (-not (Test-IsValidProfileName $ProfileName)) {
        throw 'Enter a valid profile name (no invalid path characters).'
    }
    Sync-ManagerConfigFromUi
    $existed = Test-ProfileExists $ProfileName
    if ($existed -and -not $SkipOverwriteConfirm) {
        $record = Get-ProfilePasswordRecord -ProfileRoot (Get-ProfileRoot $ProfileName)
        if (-not (Confirm-OverwriteProfileSave -ProfileName $ProfileName -PasswordRecord $record)) { return }
    }

    $components = Get-SelectedComponents
    $saveResult = Save-Profile -ProfileName $ProfileName -Components $components
    $passwordNote = ''
    if ($PasswordAction) {
        if ($PasswordAction.Clear) {
            Set-ProfilePasswordOnManifest -ProfileRoot (Get-ProfileRoot $ProfileName) -Clear
            $passwordNote = "`n`nOverwrite password removed."
        } else {
            Set-ProfilePasswordOnManifest -ProfileRoot (Get-ProfileRoot $ProfileName) `
                -Salt $PasswordAction.Salt -Hash $PasswordAction.Hash -Iterations ([int]$PasswordAction.Iterations)
            $passwordNote = "`n`nOverwrite password is set. Restore does not require it."
        }
    }

    $noSettingChanges = ($saveResult -and $saveResult.NoChanges)
    if ($noSettingChanges -and -not $PasswordAction) {
        [System.Windows.Forms.MessageBox]::Show(
            "No changes were detected in the selected scope.`nThe profile was not modified and no changelog entry was added.",
            'No changes',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return
    }

    if ($script:ProfileViewMode -ne 'profiles') {
        $script:ProfileViewMode = 'profiles'
        $script:HistoryProfileName = $null
        $script:ProfileSortColumn = 2
        $script:ProfileSortAscending = $false
    }
    Refresh-ProfileList
    Select-MainProfileByName -Name $ProfileName

    $scope = Get-SelectedComponentSummary
    if ($noSettingChanges) {
        $saveMsg = "Settings were unchanged.`n`nProfile: $ProfileName$passwordNote"
        $saveTitle = 'Password updated'
    } elseif (-not $existed) {
        $saveMsg = "Saved as a new profile.`n`nProfile: $ProfileName`nScope: $scope$passwordNote"
        $saveTitle = 'New profile saved'
    } else {
        $saveMsg = "Profile overwritten successfully.`n`nProfile: $ProfileName`nScope: $scope"
        $changelogPath = Get-ProfileSaveChangelogPath -ProfileRoot (Get-ProfileRoot $ProfileName)
        if (Test-Path -LiteralPath $changelogPath) {
            $saveMsg += "`n`nChangelog appended to:`n$changelogPath"
        }
        $saveMsg += $passwordNote
        $saveTitle = 'Profile saved'
    }
    [System.Windows.Forms.MessageBox]::Show(
        $saveMsg,
        $saveTitle,
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
}

function Show-ProfileNameDialog {
    param(
        [string]$Title = 'New profile',
        [string]$Prompt = 'Profile name:'
    )

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = $Title
    $dlg.ClientSize = New-Object System.Drawing.Size(380, 118)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Location = New-Object System.Drawing.Point(12, 14)
    $lbl.Size = New-Object System.Drawing.Size(356, 20)
    $lbl.Text = $Prompt
    $dlg.Controls.Add($lbl)

    $txt = New-Object System.Windows.Forms.TextBox
    $txt.Location = New-Object System.Drawing.Point(12, 38)
    $txt.Size = New-Object System.Drawing.Size(356, 22)
    $dlg.Controls.Add($txt)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(204, 72)
    $btnOk.Size = New-Object System.Drawing.Size(78, 28)
    $btnOk.Text = 'OK'
    $dlg.AcceptButton = $btnOk
    $dlg.Controls.Add($btnOk)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Location = New-Object System.Drawing.Point(290, 72)
    $btnCancel.Size = New-Object System.Drawing.Size(78, 28)
    $btnCancel.Text = 'Cancel'
    $dlg.CancelButton = $btnCancel
    $dlg.Controls.Add($btnCancel)

    $btnOk.Add_Click({
        try {
            $name = $txt.Text.Trim()
            if (-not (Test-IsValidProfileName $name)) {
                [System.Windows.Forms.MessageBox]::Show(
                    'Enter a valid profile name (no invalid path characters).',
                    $dlg.Text,
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Warning
                ) | Out-Null
                return
            }
            $dlg.Tag = $name
            $dlg.DialogResult = [System.Windows.Forms.DialogResult]::OK
            $dlg.Close()
        } catch {
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                $dlg.Text,
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })
    $btnCancel.Add_Click({
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
        $dlg.Close()
    })

    if ($dlg.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK) {
        return [string]$dlg.Tag
    }
    return $null
}

function Get-SaveWithPasswordTargetName {
    if ($script:ProfileViewMode -eq 'profiles') {
        $entry = Get-SelectedProfileListEntry
        if ($entry) { return [string]$entry.Name }
    }
    return Show-ProfileNameDialog -Title 'Save with password' -Prompt 'Profile name to protect:'
}

$chkAll.Add_CheckedChanged({
    if ($script:SyncingCheckboxes) { return }
    Set-AllComponentCheckboxes $chkAll.Checked
})
foreach ($chk in $script:ComponentCheckboxes.Values) {
    $chk.Add_CheckedChanged({ Update-AllCheckboxFromComponents })
}

$txtFilter.Add_TextChanged({ Update-MainProfileList })
$list.Add_SelectedIndexChanged({
    Update-ProfileActionButtons
    Update-ProfileSelectionLabel
})
$list.Add_ColumnClick({
    param($sender, $e)
    if ($e.Column -gt 2) { return }
    if ($e.Column -eq $script:ProfileSortColumn) {
        $script:ProfileSortAscending = -not $script:ProfileSortAscending
    } else {
        $script:ProfileSortColumn = $e.Column
        $script:ProfileSortAscending = $true
    }
    Update-MainProfileList
})

$btnAdvancedToggle.Add_Click({
    Set-AdvancedPanelVisible -Visible (-not $script:AdvancedExpanded)
})

$btnBackToProfiles.Add_Click({
    Set-ProfileViewMode -Mode 'profiles'
})

$btnBrowseInit.Add_Click({
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = 'Select FLIMage Init_Files folder'
    Update-InitFilesPathFromUi
    if (Test-Path -LiteralPath (Get-InitFilesPath)) {
        $dlg.SelectedPath = Get-InitFilesPath
    }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtInit.Text = $dlg.SelectedPath
        if (-not (Test-InitFilesPathValid $dlg.SelectedPath)) {
            [System.Windows.Forms.MessageBox]::Show(
                'This folder does not look like a valid Init_Files folder yet.',
                'Warning',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning
            ) | Out-Null
        }
        $script:InitFilesPath = [System.IO.Path]::GetFullPath($dlg.SelectedPath)
        Save-ManagerConfig
        Refresh-ProfileList
    }
})

$btnBrowse.Add_Click({
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    Update-ProfilesBaseFromUi
    $dlg.SelectedPath = $script:ProfilesBase
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtBase.Text = $dlg.SelectedPath
        Save-ManagerConfig
        Refresh-ProfileList
    }
})

$btnRefresh.Add_Click({ Refresh-ProfileList })

$btnNewProfile.Add_Click({
    try {
        $name = Show-ProfileNameDialog -Title 'New profile' -Prompt 'New profile name:'
        if ([string]::IsNullOrWhiteSpace($name)) { return }
        Invoke-SaveCurrentSettings -ProfileName $name
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'New profile',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnSave.Add_Click({
    try {
        if ($script:ProfileViewMode -ne 'profiles') {
            throw 'Switch back to the profile list before overwriting a profile.'
        }
        $entry = Get-SelectedProfileListEntry
        if (-not $entry) {
            throw 'Select a profile in the list.'
        }
        Invoke-SaveCurrentSettings -ProfileName ([string]$entry.Name)
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'Save',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnRestore.Add_Click({
    if (-not (Confirm-NotRunning)) { return }
    try {
        if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) {
            throw 'Init_Files folder is missing or invalid. Set Init_Files folder first.'
        }
        Sync-ManagerConfigFromUi
        $entry = Get-SelectedProfileListEntry
        if (-not $entry) {
            throw 'Select a profile or backup row.'
        }
        if ($script:ProfileViewMode -eq 'profiles') {
            $name = [string]$entry.Name
            if (-not (Test-ProfileExists $name)) {
                throw "Profile not found: $name"
            }
            $scope = Get-SelectedComponentSummary
            if (-not (Confirm-OverwriteRestore -ProfileName $name -Scope $scope)) { return }
            $components = Get-SelectedComponents
            $preRestorePath = Invoke-PreRestoreBackupBeforeRestore -Components $components -TargetProfileName $name
            Restore-Profile -ProfileName $name -Components $components
            $restoredStorage = Resolve-ProfileStorageRoot $name
            if ($restoredStorage) {
                Update-ProfileManifestRestoredAt -ProfileRoot $restoredStorage.Root
            }
            Refresh-ProfileList
            Select-MainProfileByName -Name $name
            Update-Status ("Restored: {0}. Scope: {1}. Pre-restore backup saved. Start FLIMage." -f $name, $scope)
            [System.Windows.Forms.MessageBox]::Show(
                "Profile '$name' has been restored.`n`nScope: $scope`n`nPre-restore backup saved to:`n$preRestorePath`n`nYou can start FLIMage now.",
                'Restore complete',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
        } else {
            $components = Get-SelectedComponents
            $scope = Get-SelectedComponentSummary
            if (Invoke-RestoreFromBrowseBackupEntry -Entry $entry -Components $components -Scope $scope) {
                Refresh-ProfileList
            }
        }
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'Error',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnArchive.Add_Click({
    $entry = Get-SelectedProfileListEntry
    if (-not $entry -or $script:ProfileViewMode -ne 'profiles') {
        [System.Windows.Forms.MessageBox]::Show(
            'Select a profile to archive.',
            'Archive',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return
    }
    try {
        Sync-ManagerConfigFromUi
        if (Invoke-ArchiveProfile -ProfileName ([string]$entry.Name)) {
            Refresh-ProfileList
        }
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'Archive failed',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnRename.Add_Click({
    $entry = Get-SelectedProfileListEntry
    if (-not $entry -or $script:ProfileViewMode -ne 'profiles') {
        [System.Windows.Forms.MessageBox]::Show(
            'Select a profile to rename.',
            'Rename',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return
    }
    $name = [string]$entry.Name
    $newName = Show-RenameProfileDialog -CurrentName $name
    if (-not $newName) { return }
    try {
        Sync-ManagerConfigFromUi
        $renamed = Invoke-RenameProfile -OldName $name -NewName $newName
        Refresh-ProfileList
        Select-MainProfileByName -Name $renamed
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'Rename failed',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnMonthlyRestore.Add_Click({
    Set-ProfileViewMode -Mode 'monthly'
})

$btnPreRestoreBrowse.Add_Click({
    Set-ProfileViewMode -Mode 'prerestore'
})

$btnProfileHistory.Add_Click({
    $entry = Get-SelectedProfileListEntry
    if (-not $entry -or $script:ProfileViewMode -ne 'profiles') {
        [System.Windows.Forms.MessageBox]::Show(
            'Select a profile first, then open its history.',
            'Profile history',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        return
    }
    Set-ProfileViewMode -Mode 'profilehistory' -HistoryName ([string]$entry.Name)
})

$btnCopyKeys.Add_Click({
    if (-not (Confirm-NotRunning)) { return }
    try {
        if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) {
            throw 'Init_Files folder is missing or invalid. Set Init_Files folder first.'
        }
        Sync-ManagerConfigFromUi
        $picked = Show-CopySharedKeysDialog -InitFilesPath (Get-InitFilesPath)
        if (-not $picked) { return }
        if (-not (Test-SharedKeyCopySameFamily -SourceName $picked.SourceName -TargetName $picked.TargetName)) {
            throw 'Default files and device files cannot be mixed.'
        }
        $keysUpdated = Merge-DefaultSetupFromMaster -MasterPath $picked.SourcePath -TargetPath $picked.TargetPath
        $family = if ($picked.Kind -eq 'Device') { 'device files' } else { 'Default files' }
        [System.Windows.Forms.MessageBox]::Show(
            "Shared keys copied ($family).`n`nSource: $($picked.SourceName)`nTarget: $($picked.TargetName)`nKeys updated: $keysUpdated`n`nLive Init_Files was modified.",
            'Copy shared keys',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'Copy shared keys',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnSaveWithPassword.Add_Click({
    try {
        $name = Get-SaveWithPasswordTargetName
        if ([string]::IsNullOrWhiteSpace($name)) { return }
        $exists = Test-ProfileExists $name
        $record = $null
        if ($exists) {
            $record = Get-ProfilePasswordRecord -ProfileRoot (Get-ProfileRoot $name)
        }
        $action = Show-SaveWithPasswordDialog -ProfileName $name -ProfileExists $exists -PasswordRecord $record
        if (-not $action) { return }
        Invoke-SaveCurrentSettings -ProfileName $name -SkipOverwriteConfirm -PasswordAction $action
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            'Save with password',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$script:LastFlimageRunning = $null
$script:FlimageWatchTimer = New-Object System.Windows.Forms.Timer
$script:FlimageWatchTimer.Interval = 1500
$script:FlimageWatchTimer.Add_Tick({ Update-FlimageRunningWarning })
$form.Add_FormClosed({
    if ($script:FlimageWatchTimer) {
        $script:FlimageWatchTimer.Stop()
        $script:FlimageWatchTimer.Dispose()
        $script:FlimageWatchTimer = $null
    }
    if ($script:UiToolTip) {
        $script:UiToolTip.Dispose()
        $script:UiToolTip = $null
    }
})

[void](Invoke-AutoBackupIfNeeded)
Refresh-ProfileList
$script:FlimageWatchTimer.Start()
[void]$form.ShowDialog()
