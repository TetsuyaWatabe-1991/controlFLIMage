# Automated test for per-profile history backup (_history/slot_00..09).
# No GUI required — loads functions from FLIMageProfileManager.bat and uses temp folders.

$ErrorActionPreference = 'Stop'

function Write-TestLog {
    param([string]$Message, [string]$Level = 'INFO')
    $color = switch ($Level) {
        'PASS' { 'Green' }
        'FAIL' { 'Red' }
        'WARN' { 'Yellow' }
        default { 'Gray' }
    }
    Write-Host ("[{0}] {1}" -f $Level, $Message) -ForegroundColor $color
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        throw "ASSERT FAILED: $Message"
    }
    Write-TestLog "PASS: $Message" -Level PASS
}

function Get-ProfileManagerFunctionsScriptPath {
    $batPath = Join-Path $PSScriptRoot 'FLIMageProfileManager.bat'
    if (-not (Test-Path -LiteralPath $batPath)) {
        throw "Missing FLIMageProfileManager.bat at $batPath"
    }
    $lines = [System.IO.File]::ReadAllLines($batPath)
    $endIndex = -1
    for ($i = 0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match '^# --- GUI ---' -or $lines[$i] -match '^\$form = New-Object System\.Windows\.Forms\.Form') {
            $endIndex = $i - 1
            break
        }
    }
    if ($endIndex -lt 0) {
        throw 'Could not find GUI section marker in FLIMageProfileManager.bat'
    }
    $code = [string]::Join([Environment]::NewLine, $lines[14..$endIndex])
    $tmp = Join-Path $env:TEMP ("FLIMageProfileManager_funcs_{0}.ps1" -f ([guid]::NewGuid().ToString('N')))
    $enc = New-Object System.Text.UTF8Encoding $true
    [System.IO.File]::WriteAllText($tmp, $code, $enc)
    return $tmp
}

function Get-TestComponents {
    @{
        initRoot       = $true
        settings       = $false
        windowsInfo    = $false
        uncaging       = $false
        initOther      = $false
        appDataRoaming = $false
        appDataLocal   = $false
    }
}

function Set-LiveDefaultContent {
    param(
        [string]$InitFilesPath,
        [string]$BuildTag
    )
    # InitFilesPath is the Init_Files folder (same as Get-InitFilesPath in Profile Manager).
    New-Item -ItemType Directory -Path $InitFilesPath -Force | Out-Null
    $defaultPath = Join-Path $InitFilesPath 'Default-4_0_4.txt'
    Set-Content -LiteralPath $defaultPath -Value "build=$BuildTag" -Encoding UTF8 -NoNewline
}

function Get-ProfileDefaultBuildTag {
    param([string]$ProfileRoot)
    $path = Join-Path $ProfileRoot 'Init_Files\Default-4_0_4.txt'
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    if ($raw -match 'build=(.+)') { return $Matches[1].Trim() }
    return $raw
}

function Get-HistorySlotBuildTags {
    param([string]$ProfileRoot)
    $base = Join-Path $ProfileRoot '_history'
    $tags = @{}
    for ($i = 0; $i -lt $script:ProfileHistoryMaxCount; $i++) {
        $slotRoot = Join-Path $base ("slot_{0:D2}" -f $i)
        if (-not (Test-Path -LiteralPath $slotRoot)) { continue }
        $tags["slot_{0:D2}" -f $i] = Get-ProfileDefaultBuildTag -ProfileRoot $slotRoot
    }
    return $tags
}

function Get-HistoryOccupiedSlotCount {
    param([string]$ProfileRoot)
    (Get-HistorySlotBuildTags -ProfileRoot $ProfileRoot).Count
}

$script:ProfileManagerFuncsPath = Get-ProfileManagerFunctionsScriptPath
try {
    . $script:ProfileManagerFuncsPath
    if (-not (Get-Command Save-Profile -ErrorAction SilentlyContinue)) {
        throw 'Failed to load Save-Profile from FLIMageProfileManager.bat'
    }
}
catch {
    Remove-Item -LiteralPath $script:ProfileManagerFuncsPath -Force -ErrorAction SilentlyContinue
    throw
}

$testRoot = Join-Path $env:TEMP ("FLIMageProfileHistoryTest_{0}" -f ([guid]::NewGuid().ToString('N')))
$script:ProfilesBase = Join-Path $testRoot 'Profiles'
$script:InitFilesPath = Join-Path $testRoot 'Init_Files'
$profileName = 'HistoryTestProfile'
$profileRoot = Join-Path $script:ProfilesBase $profileName
$components = Get-TestComponents

New-Item -ItemType Directory -Path $script:ProfilesBase -Force | Out-Null
Write-TestLog "Test root: $testRoot"

try {
    # --- Test 1: first save does not create history ---
    Set-LiveDefaultContent -InitFilesPath $script:InitFilesPath -BuildTag '1'
    $r1 = Save-Profile -ProfileName $profileName -Components $components
    Assert-True (-not $r1.NoChanges) 'First save completes'
    Assert-True (Test-Path -LiteralPath (Join-Path $profileRoot 'manifest.json')) 'manifest.json exists after first save'
    $countAfterFirst = Get-HistoryOccupiedSlotCount -ProfileRoot $profileRoot
    Assert-True ($countAfterFirst -eq 0) "First save creates no history slots (got $countAfterFirst)"

    # --- Test 2: no-change save does not add history ---
    $rNoChange = Save-Profile -ProfileName $profileName -Components $components
    Assert-True ($rNoChange.NoChanges) 'No-change save returns NoChanges'
    $countAfterNoChange = Get-HistoryOccupiedSlotCount -ProfileRoot $profileRoot
    Assert-True ($countAfterNoChange -eq 0) 'No-change save does not add history'

    # --- Test 3: each changing save adds history until 10 slots ---
    $expectedSlotsAfterSave = 0
    for ($n = 2; $n -le 12; $n++) {
        Set-LiveDefaultContent -InitFilesPath $script:InitFilesPath -BuildTag ([string]$n)
        Start-Sleep -Milliseconds 50
        $rn = Save-Profile -ProfileName $profileName -Components $components
        Assert-True (-not $rn.NoChanges) "Save with build=$n is not empty"
        $expectedSlotsAfterSave = [Math]::Min(($n - 1), $script:ProfileHistoryMaxCount)
        $actual = Get-HistoryOccupiedSlotCount -ProfileRoot $profileRoot
        Assert-True ($actual -eq $expectedSlotsAfterSave) (
            "After save build=$n expect $expectedSlotsAfterSave history slot(s), got $actual"
        )
    }

    $tags = Get-HistorySlotBuildTags -ProfileRoot $profileRoot
    Assert-True ($tags.Count -eq 10) "Exactly 10 occupied slots after 12 saves (got $($tags.Count))"

    $tagValues = @($tags.Values | Where-Object { $_ })
    $uniqueTags = @($tagValues | Select-Object -Unique)
    Assert-True ($uniqueTags.Count -eq 10) (
        "All 10 slots have distinct build tags (unique=$($uniqueTags.Count), tags=$($tagValues -join ', '))"
    )

    # Slots should hold builds 2..11 (pre-overwrite): save k backs up build k-1.
    for ($k = 2; $k -le 11; $k++) {
        Assert-True ($tagValues -contains ([string]$k)) "History contains snapshot of build=$k"
    }

    # --- Test 4: 13th save rotates ring (overwrites oldest) ---
    Set-LiveDefaultContent -InitFilesPath $script:InitFilesPath -BuildTag '13'
    Start-Sleep -Milliseconds 50
    $r13 = Save-Profile -ProfileName $profileName -Components $components
    Assert-True (-not $r13.NoChanges) '13th save completes'
    Assert-True ((Get-ProfileDefaultBuildTag -ProfileRoot $profileRoot) -eq '13') 'Profile has build=13'

    $tagsAfter13 = Get-HistorySlotBuildTags -ProfileRoot $profileRoot
    Assert-True ($tagsAfter13.Count -eq 10) 'Still 10 slots after ring overwrite'
    $valuesAfter13 = @($tagsAfter13.Values)
    Assert-True ($valuesAfter13 -contains '12') 'Ring retains build=12 snapshot'
    Assert-True ($valuesAfter13 -notcontains '2') 'Oldest snapshot build=2 was overwritten'

    # --- Test 5: Browse list helper sees 10 entries ---
    $entries = @(Get-ProfileHistoryListEntries -ProfileName $profileName)
    Assert-True ($entries.Count -eq 10) "Get-ProfileHistoryListEntries returns 10 (got $($entries.Count))"
    Assert-True ($entries[0].Kind -eq 'profile_history') 'Entry kind is profile_history'
    Assert-True ($entries[0].RestoreTarget -eq $profileName) 'RestoreTarget is profile name'

    # --- Test 6: manifest on slot marks profile_history ---
    $sampleSlot = Join-Path $profileRoot '_history\slot_00'
    if (Test-Path -LiteralPath $sampleSlot) {
        $mj = Get-Content (Join-Path $sampleSlot 'manifest.json') -Raw | ConvertFrom-Json
        Assert-True ($mj.profile_kind -eq 'profile_history') 'Slot manifest profile_kind=profile_history'
        Assert-True ($mj.history_source_profile -eq $profileName) 'Slot manifest history_source_profile set'
    }

    Write-Host ''
    Write-TestLog 'All profile history backup tests passed.' -Level PASS
    exit 0
}
catch {
    Write-Host ''
    Write-TestLog $_.Exception.Message -Level FAIL
    if ($_.ScriptStackTrace) {
        Write-Host $_.ScriptStackTrace -ForegroundColor DarkRed
    }
    exit 1
}
finally {
    if ($testRoot -and (Test-Path -LiteralPath $testRoot)) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
        Write-TestLog "Cleaned up $testRoot"
    }
    if ($script:ProfileManagerFuncsPath -and (Test-Path -LiteralPath $script:ProfileManagerFuncsPath)) {
        Remove-Item -LiteralPath $script:ProfileManagerFuncsPath -Force -ErrorAction SilentlyContinue
    }
}
