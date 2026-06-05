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
}

function Copy-InitFilesRootOnly {
    param([string]$SourceDir, [string]$DestDir)
    if (-not (Test-Path -LiteralPath $SourceDir)) { return }
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    Get-ChildItem -LiteralPath $SourceDir -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $DestDir $_.Name) -Force
    }
}

function Test-IsUncagingCalibSetupKey {
    param([string]$Key)
    $script:UncagingCalibSetupKeys -contains $Key
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
    }
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
    if (Test-Path -LiteralPath $manifestPath) {
        try {
            $existingManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if (Test-ManifestHasProperty -Json $existingManifest -PropertyName 'restored_at') {
                $existingRestoredAt = [string]$existingManifest.restored_at
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
            -RestoreTarget $ProfileName
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
        [string]$RestoreTarget = ''
    )
    return [pscustomobject]@{
        Name           = $Name
        Kind           = $Kind
        StorageRoot    = $StorageRoot
        SavedAt        = $SavedAt
        ModifiedAt     = $ModifiedAt
        RestoredAt     = $RestoredAt
        ModifiedAtSort = $ModifiedAtSort
        RestoredAtSort = $RestoredAtSort
        HasRestoredAt  = $HasRestoredAt
        RestoreTarget  = $RestoreTarget
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
            -RestoreTarget $meta.RestoreTargetProfile
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
            -RestoreTarget $meta.RestoreTargetProfile
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
            -RestoreTarget $meta.RestoreTargetProfile
    }
    return @($entries)
}

function Get-ProfileListDisplayName {
    param($Entry)
    if ($Entry.Kind -eq 'auto_backup') { return "$($Entry.Name) [AutoBackup]" }
    if ($Entry.Kind -eq 'profile_history') { return $Entry.Name }
    if ($Entry.Kind -eq 'pre_restore') {
        if ($Entry.RestoreTarget) {
            return "$($Entry.Name) (before: $($Entry.RestoreTarget))"
        }
        return $Entry.Name
    }
    return $Entry.Name
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

function Confirm-ArchiveProfile {
    param(
        [string]$ProfileName,
        [string]$SourceRoot,
        [string]$Kind,
        [string]$ZipPath
    )
    $archiveBase = Get-ArchivedProfilesBase
    $restoreFolder = Get-ProfileRestoreFolderForKind -Kind $Kind
    $kindNote = if ($Kind -eq 'auto_backup') {
        "This profile is stored as an auto-backup."
    } else {
        "This is a user profile under your Profiles folder."
    }

    $message = @"
You are about to archive profile '$ProfileName'.

$kindNote

The profile folder will be compressed (ZIP) and saved as:
$ZipPath

The original folder will be removed:
$SourceRoot

To restore later:
1. Extract (unzip) the archive file above.
2. Move the extracted folder (name: $ProfileName) into:
$restoreFolder

Continue with archive?
"@

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

    [System.Windows.Forms.MessageBox]::Show(
        "Profile '$ProfileName' was archived to:`n$zipPath",
        "Archive complete",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
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

function Show-ProfileBrowseDialog {
    param(
        [string]$InitialFilter = '',
        [string]$InitialSelection = ''
    )

    $browseState = @{
        Archived           = $false
        Renamed            = $false
        RenamedTo          = $null
        RestoredFromBackup = $false
    }
    # Hashtable so event handlers (column sort, etc.) share state across scriptblock scopes.
    $browseUiState = @{
        ViewMode           = 'profiles'
        SortColumn         = 2
        Ascending          = $false
        HistoryProfileName = $null
    }
    $entryByTag = @{}

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = 'Browse profiles'
    $dlg.ClientSize = New-Object System.Drawing.Size(520, 430)
    $dlg.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dlg.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ShowInTaskbar = $false

    $lblFilter = New-Object System.Windows.Forms.Label
    $lblFilter.Location = New-Object System.Drawing.Point(12, 14)
    $lblFilter.AutoSize = $true
    $lblFilter.Text = 'Filter:'
    $dlg.Controls.Add($lblFilter)

    $txtFilter = New-Object System.Windows.Forms.TextBox
    $txtFilter.Location = New-Object System.Drawing.Point(56, 11)
    $txtFilter.Size = New-Object System.Drawing.Size(452, 22)
    $txtFilter.Text = $InitialFilter
    $dlg.Controls.Add($txtFilter)

    $list = New-Object System.Windows.Forms.ListView
    $list.Location = New-Object System.Drawing.Point(12, 42)
    $list.Size = New-Object System.Drawing.Size(496, 296)
    $list.View = [System.Windows.Forms.View]::Details
    $list.FullRowSelect = $true
    $list.HideSelection = $false
    $list.MultiSelect = $false
    $list.GridLines = $true
    [void]$list.Columns.Add('Profile', 220)
    [void]$list.Columns.Add('Modified', 132)
    [void]$list.Columns.Add('Restored', 132)
    $dlg.Controls.Add($list)

    $lblCount = New-Object System.Windows.Forms.Label
    $lblCount.Location = New-Object System.Drawing.Point(12, 344)
    $lblCount.Size = New-Object System.Drawing.Size(496, 18)
    $dlg.Controls.Add($lblCount)

    $btnArchive = New-Object System.Windows.Forms.Button
    $btnArchive.Location = New-Object System.Drawing.Point(12, 368)
    $btnArchive.Size = New-Object System.Drawing.Size(72, 28)
    $btnArchive.Text = 'Archive'
    $dlg.Controls.Add($btnArchive)

    $btnRename = New-Object System.Windows.Forms.Button
    $btnRename.Location = New-Object System.Drawing.Point(90, 368)
    $btnRename.Size = New-Object System.Drawing.Size(72, 28)
    $btnRename.Text = 'Rename'
    $dlg.Controls.Add($btnRename)

    $btnMonthlyRestore = New-Object System.Windows.Forms.Button
    $btnMonthlyRestore.Location = New-Object System.Drawing.Point(168, 368)
    $btnMonthlyRestore.Size = New-Object System.Drawing.Size(168, 28)
    $btnMonthlyRestore.Text = 'Restore from monthly backup'
    $dlg.Controls.Add($btnMonthlyRestore)

    $btnPreRestoreBrowse = New-Object System.Windows.Forms.Button
    $btnPreRestoreBrowse.Location = New-Object System.Drawing.Point(12, 400)
    $btnPreRestoreBrowse.Size = New-Object System.Drawing.Size(150, 28)
    $btnPreRestoreBrowse.Text = 'Recent 30 backups'
    $dlg.Controls.Add($btnPreRestoreBrowse)

    $btnProfileHistory = New-Object System.Windows.Forms.Button
    $btnProfileHistory.Location = New-Object System.Drawing.Point(168, 400)
    $btnProfileHistory.Size = New-Object System.Drawing.Size(168, 28)
    $btnProfileHistory.Text = 'Profile history (10)'
    $btnProfileHistory.Enabled = $false
    $dlg.Controls.Add($btnProfileHistory)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Location = New-Object System.Drawing.Point(342, 400)
    $btnOk.Size = New-Object System.Drawing.Size(82, 28)
    $btnOk.Text = 'OK'
    $dlg.Controls.Add($btnOk)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Location = New-Object System.Drawing.Point(430, 400)
    $btnCancel.Size = New-Object System.Drawing.Size(82, 28)
    $btnCancel.Text = 'Cancel'
    $dlg.CancelButton = $btnCancel
    $dlg.Controls.Add($btnCancel)

    $getEntriesForViewMode = {
        switch ($browseUiState.ViewMode) {
            'monthly'         { return @(Get-MonthlyBackupListEntries) }
            'prerestore'      { return @(Get-PreRestoreBackupListEntries) }
            'profilehistory'  { return @(Get-ProfileHistoryListEntries -ProfileName $browseUiState.HistoryProfileName) }
            default           { return @(Get-ProfileListEntries) }
        }
    }

    $updateProfileHistoryButtonState = {
        $btnProfileHistory.Enabled = (
            $browseUiState.ViewMode -eq 'profiles' -and
            $list.SelectedItems.Count -gt 0
        )
    }

    $applyBrowseViewMode = {
        $isProfiles = ($browseUiState.ViewMode -eq 'profiles')
        $btnArchive.Visible = $isProfiles
        $btnRename.Visible = $isProfiles
        $btnMonthlyRestore.Visible = $isProfiles
        $btnPreRestoreBrowse.Visible = $isProfiles
        $btnProfileHistory.Visible = $isProfiles
        if ($isProfiles) {
            $dlg.Text = 'Browse profiles'
            $btnOk.Text = 'OK'
            $btnCancel.Text = 'Cancel'
            $dlg.CancelButton = $btnCancel
            & $updateProfileHistoryButtonState
        } elseif ($browseUiState.ViewMode -eq 'monthly') {
            $dlg.Text = 'Monthly auto-backups (restore only)'
            $btnOk.Text = 'Restore'
            $btnCancel.Text = 'Back'
            $dlg.CancelButton = $null
            $browseUiState.SortColumn = 1
            $browseUiState.Ascending = $false
        } elseif ($browseUiState.ViewMode -eq 'profilehistory') {
            $histName = $browseUiState.HistoryProfileName
            $dlg.Text = "Profile history: $histName (restore only)"
            $btnOk.Text = 'Restore'
            $btnCancel.Text = 'Back'
            $dlg.CancelButton = $null
            $browseUiState.SortColumn = 1
            $browseUiState.Ascending = $false
        } else {
            $dlg.Text = 'Recent pre-restore backups (restore only)'
            $btnOk.Text = 'Restore'
            $btnCancel.Text = 'Back'
            $dlg.CancelButton = $null
            $browseUiState.SortColumn = 1
            $browseUiState.Ascending = $false
        }
        & $repopulateList
    }

    $acceptSelection = {
        if ($list.SelectedItems.Count -eq 0) { return }
        if ($browseUiState.ViewMode -eq 'profiles') {
            $dlg.DialogResult = [System.Windows.Forms.DialogResult]::OK
            $dlg.Close()
            return
        }
        if (-not (Confirm-NotRunning)) { return }
        $tag = [string]$list.SelectedItems[0].Tag
        if (-not $entryByTag.ContainsKey($tag)) { return }
        $entry = $entryByTag[$tag]
        try {
            Sync-ManagerConfigFromUi
            $components = Get-SelectedComponents
            $scope = Get-SelectedComponentSummary
            if (Invoke-RestoreFromBrowseBackupEntry -Entry $entry -Components $components -Scope $scope) {
                $browseState.RestoredFromBackup = $true
                $dlg.DialogResult = [System.Windows.Forms.DialogResult]::OK
                $dlg.Close()
            }
        } catch {
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                "Restore failed",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    }

    $updateBrowseColumnHeaders = {
        $profileHeader = 'Profile'
        $modifiedHeader = 'Modified'
        $restoredHeader = 'Restored'
        if ($browseUiState.SortColumn -eq 0) {
            $profileHeader += if ($browseUiState.Ascending) { ' ^' } else { ' v' }
        } elseif ($browseUiState.SortColumn -eq 1) {
            $modifiedHeader += if ($browseUiState.Ascending) { ' ^' } else { ' v' }
        } elseif ($browseUiState.SortColumn -eq 2) {
            $restoredHeader += if ($browseUiState.Ascending) { ' ^' } else { ' v' }
        }
        $list.Columns[0].Text = $profileHeader
        $list.Columns[1].Text = $modifiedHeader
        $list.Columns[2].Text = $restoredHeader
    }

    $repopulateList = {
        $keepTag = $null
        if ($list.SelectedItems.Count -gt 0) {
            $keepTag = [string]$list.SelectedItems[0].Tag
        }

        $list.BeginUpdate()
        $list.Items.Clear()
        $entryByTag.Clear()
        $filter = $txtFilter.Text.Trim()
        $allEntries = @(& $getEntriesForViewMode)
        $filtered = @($allEntries | Where-Object {
                Test-ProfileListEntryMatchesFilter -Entry $_ -Filter $filter
            })
        $sorted = Sort-ProfileListEntries -Entries $filtered -SortColumn $browseUiState.SortColumn -Ascending $browseUiState.Ascending
        foreach ($entry in $sorted) {
            $tag = $entry.StorageRoot
            $entryByTag[$tag] = $entry
            $item = New-Object System.Windows.Forms.ListViewItem((Get-ProfileListDisplayName -Entry $entry))
            [void]$item.SubItems.Add($entry.ModifiedAt)
            [void]$item.SubItems.Add($entry.RestoredAt)
            $item.Tag = $tag
            [void]$list.Items.Add($item)
        }
        $list.EndUpdate()
        & $updateBrowseColumnHeaders
        $countLabel = switch ($browseUiState.ViewMode) {
            'monthly'        { 'monthly backup(s)' }
            'prerestore'     { 'pre-restore snapshot(s)' }
            'profilehistory' { 'history snapshot(s)' }
            default          { 'profile(s)' }
        }
        $lblCount.Text = if ($filter) {
            "{0} shown / {1} total" -f $sorted.Count, $allEntries.Count
        } else {
            "{0} $countLabel" -f $allEntries.Count
        }
        if ($keepTag -and $entryByTag.ContainsKey($keepTag)) {
            & $selectByTag $keepTag
        } elseif ($list.Items.Count -gt 0) {
            $list.Items[0].Selected = $true
            $list.Items[0].Focused = $true
            $list.EnsureVisible(0)
        }
    }

    $selectByTag = {
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

    $selectByName = {
        param([string]$Name)
        if ([string]::IsNullOrWhiteSpace($Name)) { return }
        foreach ($entry in (& $getEntriesForViewMode)) {
            if ($entry.Name -eq $Name) {
                & $selectByTag $entry.StorageRoot
                break
            }
        }
    }

    $returnToProfilesView = {
        if ($browseUiState.ViewMode -eq 'profiles') { return }
        $browseUiState.ViewMode = 'profiles'
        $browseUiState.HistoryProfileName = $null
        $browseUiState.SortColumn = 2
        $browseUiState.Ascending = $false
        & $applyBrowseViewMode
    }

    $txtFilter.Add_TextChanged({ & $repopulateList })

    $list.Add_ColumnClick({
        param($sender, $e)
        if ($e.Column -gt 2) { return }
        if ($e.Column -eq $browseUiState.SortColumn) {
            $browseUiState.Ascending = -not $browseUiState.Ascending
        } else {
            $browseUiState.SortColumn = $e.Column
            $browseUiState.Ascending = $true
        }
        & $repopulateList
    })

    $list.Add_DoubleClick({ & $acceptSelection })
    $btnOk.Add_Click({ & $acceptSelection })

    $btnMonthlyRestore.Add_Click({
        $browseUiState.ViewMode = 'monthly'
        & $applyBrowseViewMode
    })

    $btnPreRestoreBrowse.Add_Click({
        $browseUiState.ViewMode = 'prerestore'
        & $applyBrowseViewMode
    })

    $btnProfileHistory.Add_Click({
        if ($list.SelectedItems.Count -eq 0) {
            [System.Windows.Forms.MessageBox]::Show(
                'Select a profile first, then open its history.',
                'Profile history',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
            return
        }
        $tag = [string]$list.SelectedItems[0].Tag
        if (-not $entryByTag.ContainsKey($tag)) { return }
        $browseUiState.HistoryProfileName = $entryByTag[$tag].Name
        $browseUiState.ViewMode = 'profilehistory'
        & $applyBrowseViewMode
    })

    $list.Add_SelectedIndexChanged({ & $updateProfileHistoryButtonState })

    $btnArchive.Add_Click({
        if ($list.SelectedItems.Count -eq 0) {
            [System.Windows.Forms.MessageBox]::Show(
                "Select a profile to archive.",
                "Archive",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
            return
        }
        $tag = [string]$list.SelectedItems[0].Tag
        if (-not $entryByTag.ContainsKey($tag)) { return }
        $name = $entryByTag[$tag].Name
        try {
            Sync-ManagerConfigFromUi
            if (Invoke-ArchiveProfile -ProfileName $name) {
                $browseState.Archived = $true
                & $repopulateList
            }
        } catch {
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                "Archive failed",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })
    $btnRename.Add_Click({
        if ($list.SelectedItems.Count -eq 0) {
            [System.Windows.Forms.MessageBox]::Show(
                "Select a profile to rename.",
                "Rename",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
            return
        }
        $tag = [string]$list.SelectedItems[0].Tag
        if (-not $entryByTag.ContainsKey($tag)) { return }
        $name = $entryByTag[$tag].Name
        $newName = Show-RenameProfileDialog -CurrentName $name
        if (-not $newName) { return }
        try {
            Sync-ManagerConfigFromUi
            $renamed = Invoke-RenameProfile -OldName $name -NewName $newName
            $browseState.Renamed = $true
            $browseState.RenamedTo = $renamed
            & $repopulateList
            & $selectByName $renamed
        } catch {
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                "Rename failed",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })
    $btnCancel.Add_Click({
        if ($browseUiState.ViewMode -ne 'profiles') {
            & $returnToProfilesView
            return
        }
        $dlg.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
        $dlg.Close()
    })

    $txtFilter.Add_KeyDown({
        param($sender, $e)
        if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Down -and $list.Items.Count -gt 0) {
            $list.Focus()
            $e.Handled = $true
        }
        if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape -and $browseUiState.ViewMode -ne 'profiles') {
            & $returnToProfilesView
            $e.Handled = $true
        }
    })

    $list.Add_KeyDown({
        param($sender, $e)
        if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Return) {
            & $acceptSelection
            $e.Handled = $true
        }
        if ($e.KeyCode -eq [System.Windows.Forms.Keys]::Escape -and $browseUiState.ViewMode -ne 'profiles') {
            & $returnToProfilesView
            $e.Handled = $true
        }
    })

    & $applyBrowseViewMode
    if (-not [string]::IsNullOrWhiteSpace($InitialSelection)) {
        & $selectByName $InitialSelection
    }

    $dlgResult = $dlg.ShowDialog($form)
    if ($browseState.Archived) {
        $script:BrowseDialogArchivedProfiles = $true
    }
    if ($browseState.Renamed) {
        $script:BrowseDialogRenamedProfile = $browseState.RenamedTo
    }
    if ($browseState.RestoredFromBackup) {
        $script:BrowseDialogRestoredFromBackup = $true
    }
    if ($dlgResult -eq [System.Windows.Forms.DialogResult]::OK -and $browseUiState.ViewMode -eq 'profiles') {
        if ($list.SelectedItems.Count -gt 0 -and $entryByTag.ContainsKey([string]$list.SelectedItems[0].Tag)) {
            return $entryByTag[[string]$list.SelectedItems[0].Tag].Name
        }
    }
    return $null
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

function Confirm-OverwriteProfileSave {
    param([string]$ProfileName)
    $message = "Profile '$ProfileName' already exists.`nOverwrite it with the current settings?"
    $result = [System.Windows.Forms.MessageBox]::Show(
        $message,
        "Confirm overwrite save",
        [System.Windows.Forms.MessageBoxButtons]::OKCancel,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    )
    return ($result -eq [System.Windows.Forms.DialogResult]::OK)
}

Initialize-InitFilesPathAtStartup

# --- GUI ---

$form = New-Object System.Windows.Forms.Form
$form.Text = "FLIMage Profile Manager"
$form.Size = New-Object System.Drawing.Size(520, 592)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
$form.MaximizeBox = $false

$lblInit = New-Object System.Windows.Forms.Label
$lblInit.Location = New-Object System.Drawing.Point(12, 12)
$lblInit.Size = New-Object System.Drawing.Size(100, 20)
$lblInit.Text = "Init_Files folder:"
$form.Controls.Add($lblInit)

$txtInit = New-Object System.Windows.Forms.TextBox
$txtInit.Location = New-Object System.Drawing.Point(115, 10)
$txtInit.Size = New-Object System.Drawing.Size(280, 22)
$txtInit.Text = $script:InitFilesPath
$form.Controls.Add($txtInit)

$btnBrowseInit = New-Object System.Windows.Forms.Button
$btnBrowseInit.Location = New-Object System.Drawing.Point(400, 8)
$btnBrowseInit.Size = New-Object System.Drawing.Size(95, 25)
$btnBrowseInit.Text = "Browse"
$form.Controls.Add($btnBrowseInit)

$lblBase = New-Object System.Windows.Forms.Label
$lblBase.Location = New-Object System.Drawing.Point(12, 42)
$lblBase.Size = New-Object System.Drawing.Size(100, 20)
$lblBase.Text = "Profiles folder:"
$form.Controls.Add($lblBase)

$txtBase = New-Object System.Windows.Forms.TextBox
$txtBase.Location = New-Object System.Drawing.Point(115, 40)
$txtBase.Size = New-Object System.Drawing.Size(280, 22)
$txtBase.Text = $script:ProfilesBase
$form.Controls.Add($txtBase)

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Location = New-Object System.Drawing.Point(400, 38)
$btnBrowse.Size = New-Object System.Drawing.Size(95, 25)
$btnBrowse.Text = "Browse"
$form.Controls.Add($btnBrowse)

$grpScope = New-Object System.Windows.Forms.GroupBox
$grpScope.Location = New-Object System.Drawing.Point(12, 72)
$grpScope.Size = New-Object System.Drawing.Size(488, 258)
$grpScope.Text = "Save / Restore scope"
$form.Controls.Add($grpScope)

$chkAll = New-Object System.Windows.Forms.CheckBox
$chkAll.Location = New-Object System.Drawing.Point(14, 22)
$chkAll.Size = New-Object System.Drawing.Size(200, 22)
$chkAll.Text = "All"
$chkAll.Checked = $true
$grpScope.Controls.Add($chkAll)

$script:ComponentCheckboxes = @{}
$y = 48
foreach ($def in Get-ComponentDefinitions) {
    $chk = New-Object System.Windows.Forms.CheckBox
    $chk.Location = New-Object System.Drawing.Point(28, $y)
    $chk.Size = New-Object System.Drawing.Size(450, 22)
    $chk.Text = $def.Label
    $chk.Checked = $true
    $chk.Tag = $def.Key
    $grpScope.Controls.Add($chk)
    $script:ComponentCheckboxes[$def.Key] = $chk
    $y += 26
}

$grpAdvanced = New-Object System.Windows.Forms.GroupBox
$grpAdvanced.Location = New-Object System.Drawing.Point(12, 332)
$grpAdvanced.Size = New-Object System.Drawing.Size(488, 58)
$grpAdvanced.Text = "Advanced"
$form.Controls.Add($grpAdvanced)

$chkPreserveUncagingCalib = New-Object System.Windows.Forms.CheckBox
$chkPreserveUncagingCalib.Location = New-Object System.Drawing.Point(14, 20)
$chkPreserveUncagingCalib.AutoSize = $true
$chkPreserveUncagingCalib.MaximumSize = New-Object System.Drawing.Size(460, 0)
$chkPreserveUncagingCalib.Text = "Restore profile settings except uncaging calibration (recommended)"
$chkPreserveUncagingCalib.Checked = $script:PreserveUncagingCalibOnRestore
$grpAdvanced.Controls.Add($chkPreserveUncagingCalib)

$lblProf = New-Object System.Windows.Forms.Label
$lblProf.Location = New-Object System.Drawing.Point(12, 392)
$lblProf.Size = New-Object System.Drawing.Size(100, 20)
$lblProf.Text = "Profile:"
$form.Controls.Add($lblProf)

$combo = New-Object System.Windows.Forms.ComboBox
$combo.Location = New-Object System.Drawing.Point(115, 390)
$combo.Size = New-Object System.Drawing.Size(148, 22)
$combo.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDown
$combo.AutoCompleteMode = [System.Windows.Forms.AutoCompleteMode]::SuggestAppend
$combo.AutoCompleteSource = [System.Windows.Forms.AutoCompleteSource]::ListItems
$form.Controls.Add($combo)

$btnBrowseProfile = New-Object System.Windows.Forms.Button
$btnBrowseProfile.Location = New-Object System.Drawing.Point(268, 388)
$btnBrowseProfile.Size = New-Object System.Drawing.Size(72, 25)
$btnBrowseProfile.Text = "Browse..."
$form.Controls.Add($btnBrowseProfile)

$lblProfileHint = New-Object System.Windows.Forms.Label
$lblProfileHint.Location = New-Object System.Drawing.Point(346, 393)
$lblProfileHint.AutoSize = $true
$lblProfileHint.Text = ""
$form.Controls.Add($lblProfileHint)

$btnRefresh = New-Object System.Windows.Forms.Button
$btnRefresh.Location = New-Object System.Drawing.Point(400, 388)
$btnRefresh.Size = New-Object System.Drawing.Size(95, 25)
$btnRefresh.Text = "Refresh"
$form.Controls.Add($btnRefresh)

$lblFlimageWarning = New-Object System.Windows.Forms.Label
$lblFlimageWarning.Location = New-Object System.Drawing.Point(12, 420)
$lblFlimageWarning.Size = New-Object System.Drawing.Size(488, 18)
$lblFlimageWarning.Text = "Close FLIMage before Save or Restore."
$lblFlimageWarning.ForeColor = [System.Drawing.Color]::FromArgb(192, 0, 0)
$lblFlimageWarning.Visible = $false
$form.Controls.Add($lblFlimageWarning)

$btnSave = New-Object System.Windows.Forms.Button
$btnSave.Location = New-Object System.Drawing.Point(115, 444)
$btnSave.Size = New-Object System.Drawing.Size(175, 32)
$btnSave.Text = "Save current"
$form.Controls.Add($btnSave)

$btnRestore = New-Object System.Windows.Forms.Button
$btnRestore.Location = New-Object System.Drawing.Point(300, 444)
$btnRestore.Size = New-Object System.Drawing.Size(195, 32)
$btnRestore.Text = "Restore (before launch)"
$form.Controls.Add($btnRestore)

$lblStatus = New-Object System.Windows.Forms.Label
$lblStatus.Location = New-Object System.Drawing.Point(12, 484)
$lblStatus.Size = New-Object System.Drawing.Size(488, 72)
$lblStatus.Text = "Ready. Default scope: All."
$form.Controls.Add($lblStatus)
$lblFlimageWarning.BringToFront()

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
    if ($parts.Count -eq 0) { return "(none)" }
    return ($parts -join ", ")
}

$chkAll.Add_CheckedChanged({
    if ($script:SyncingCheckboxes) { return }
    Set-AllComponentCheckboxes $chkAll.Checked
})

foreach ($chk in $script:ComponentCheckboxes.Values) {
    $chk.Add_CheckedChanged({ Update-AllCheckboxFromComponents })
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

function Update-ProfileNameHint {
    Update-ProfilesBaseFromUi
    $name = $combo.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($name)) {
        $lblProfileHint.Text = ""
        return
    }
    $storage = Resolve-ProfileStorageRoot $name
    if ($storage -and $storage.Kind -eq 'auto_backup') {
        $lblProfileHint.Text = "AutoBackup"
        $lblProfileHint.ForeColor = [System.Drawing.Color]::FromArgb(0, 102, 153)
    } elseif ($storage) {
        $lblProfileHint.Text = ""
    } else {
        $lblProfileHint.Text = "New profile"
        $lblProfileHint.ForeColor = [System.Drawing.Color]::FromArgb(192, 0, 0)
    }
}

function Refresh-ProfileList {
    Sync-ManagerConfigFromUi
    if (-not (Test-Path -LiteralPath $script:ProfilesBase)) {
        New-Item -ItemType Directory -Path $script:ProfilesBase -Force | Out-Null
    }
    $selected = $combo.Text
    $combo.Items.Clear()
    foreach ($name in Get-ProfileNames) {
        [void]$combo.Items.Add($name)
    }
    if (-not [string]::IsNullOrWhiteSpace($selected)) {
        $combo.Text = $selected
    } elseif ($combo.Items.Count -gt 0) {
        $combo.SelectedIndex = 0
    }
    $initOk = if (Test-InitFilesPathValid (Get-InitFilesPath)) { "OK" } else { "INVALID" }
    Update-Status ("Init_Files ({0}): {1}`nProfiles: {2}. Scope: {3}." -f $initOk, (Get-InitFilesPath), $script:ProfilesBase, (Get-SelectedComponentSummary))
    Update-FlimageRunningWarning
    Update-ProfileNameHint
}

$combo.Add_TextChanged({ Update-ProfileNameHint })
$combo.Add_SelectedIndexChanged({ Update-ProfileNameHint })

function Get-SelectedProfileName {
    $name = $combo.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($name)) {
        throw "Enter or select a profile name."
    }
    return $name
}

function Confirm-NotRunning {
    if (-not (Test-FLIMageRunning)) { return $true }
    [System.Windows.Forms.MessageBox]::Show(
        "FLIMage appears to be running. Close it before Save or Restore.",
        "Warning",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Warning
    ) | Out-Null
    return $false
}

$btnBrowseInit.Add_Click({
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = "Select FLIMage Init_Files folder"
    Update-InitFilesPathFromUi
    if (Test-Path -LiteralPath (Get-InitFilesPath)) {
        $dlg.SelectedPath = Get-InitFilesPath
    }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtInit.Text = $dlg.SelectedPath
        if (-not (Test-InitFilesPathValid $dlg.SelectedPath)) {
            [System.Windows.Forms.MessageBox]::Show(
                "This folder does not look like a valid Init_Files folder yet.",
                "Warning",
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

$btnBrowseProfile.Add_Click({
    Sync-ManagerConfigFromUi
    $script:BrowseDialogArchivedProfiles = $false
    $script:BrowseDialogRenamedProfile = $null
    $script:BrowseDialogRestoredFromBackup = $false
    $picked = Show-ProfileBrowseDialog -InitialSelection $combo.Text.Trim()
    Refresh-ProfileList
    if ($script:BrowseDialogRestoredFromBackup) {
        Update-ProfileNameHint
    } elseif ($picked) {
        $combo.Text = $picked
        Update-ProfileNameHint
    } elseif ($script:BrowseDialogRenamedProfile) {
        $combo.Text = $script:BrowseDialogRenamedProfile
        Update-ProfileNameHint
    } elseif ($script:BrowseDialogArchivedProfiles -and $combo.Text.Trim()) {
        if (-not (Test-ProfileExists $combo.Text.Trim())) {
            $combo.Text = ''
            Update-ProfileNameHint
        }
    }
})

$btnSave.Add_Click({
    if (-not (Confirm-NotRunning)) { return }
    try {
        if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) {
            throw "Init_Files folder is missing or invalid. Set Init_Files folder first."
        }
        Sync-ManagerConfigFromUi
        $name = Get-SelectedProfileName
        $isNewProfile = -not (Test-ProfileExists $name)
        if (-not $isNewProfile) {
            if (-not (Confirm-OverwriteProfileSave -ProfileName $name)) { return }
        }
        $components = Get-SelectedComponents
        $syncDefaultsInProfile = $false
        if ($components.initRoot) {
            $syncResult = Invoke-DefaultSyncPromptOnSave -InitFilesPath (Get-InitFilesPath)
            if ($syncResult.Action -eq 'Cancel') { return }
            if ($syncResult.Action -eq 'Yes') { $syncDefaultsInProfile = $true }
        }
        $saveResult = Save-Profile -ProfileName $name -Components $components
        if ($saveResult -and $saveResult.NoChanges) {
            [System.Windows.Forms.MessageBox]::Show(
                "No changes were detected in the selected scope.`nThe profile was not modified and no changelog entry was added.",
                "No changes",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
            return
        }
        $syncSummary = $null
        if ($syncDefaultsInProfile) {
            $profileInit = Join-Path (Get-ProfileRoot $name) "Init_Files"
            if (Test-Path -LiteralPath $profileInit) {
                $syncSummary = Sync-DefaultSettingsAcrossVersions -InitFilesPath $profileInit
            }
        }
        Refresh-ProfileList
        $scope = Get-SelectedComponentSummary
        $changelogPath = Get-ProfileSaveChangelogPath -ProfileRoot (Get-ProfileRoot $name)
        if ($isNewProfile) {
            $saveMsg = "Saved as a new profile.`n`nProfile: $name`nScope: $scope"
            $saveTitle = "New profile saved"
        } else {
            $saveMsg = "Profile overwritten successfully.`n`nProfile: $name`nScope: $scope"
            if (Test-Path -LiteralPath $changelogPath) {
                $saveMsg += "`n`nChangelog appended to:`n$changelogPath"
            }
            $saveTitle = "Profile saved"
        }
        if ($syncSummary) {
            $masterSaved = $syncSummary.MasterSavedAt.ToString('g')
            $saveMsg += "`n`nDefault sync in profile (master: $($syncSummary.MasterName), last saved $masterSaved):"
            $saveMsg += "`n$($syncSummary.FilesUpdated) file(s), $($syncSummary.KeysUpdated) shared key(s) updated in the profile copy."
            $saveMsg += "`nLive Init_Files was not modified."
        }
        [System.Windows.Forms.MessageBox]::Show(
            $saveMsg,
            $saveTitle,
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            "Error",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$btnRestore.Add_Click({
    if (-not (Confirm-NotRunning)) { return }
    try {
        if (-not (Test-InitFilesPathValid (Get-InitFilesPath))) {
            throw "Init_Files folder is missing or invalid. Set Init_Files folder first."
        }
        Sync-ManagerConfigFromUi
        $name = Get-SelectedProfileName
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
        Update-Status ("Restored: {0}. Scope: {1}. Pre-restore backup saved. Start FLIMage." -f $name, $scope)
        [System.Windows.Forms.MessageBox]::Show(
            "Profile '$name' has been restored.`n`nScope: $scope`n`nPre-restore backup saved to:`n$preRestorePath`n`nYou can start FLIMage now.",
            "Restore complete",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    } catch {
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            "Error",
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
})

[void](Invoke-AutoBackupIfNeeded)
Refresh-ProfileList
$script:FlimageWatchTimer.Start()
[void]$form.ShowDialog()