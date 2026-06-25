# FLIMage motor / turret pipe GUI (PowerShell + WinForms). Olympus IX2-UCB via SendOlympusCommand.
# Pipe logic matches test_flimage_pipe_connect.ps1 (no background threads).

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Import-SystemIoPortsAssembly {
    try {
        $null = [System.IO.Ports.SerialPort]
        return
    }
    catch {
    }

    $dllPaths = @(
        (Join-Path $PSHOME 'System.IO.Ports.dll')
        (Join-Path ([System.Runtime.InteropServices.RuntimeEnvironment]::GetRuntimeDirectory()) 'System.IO.Ports.dll')
    )
    foreach ($dllPath in $dllPaths) {
        if (-not (Test-Path -LiteralPath $dllPath)) { continue }
        try {
            Add-Type -Path $dllPath -ErrorAction Stop
            $null = [System.IO.Ports.SerialPort]
            return
        }
        catch {
        }
    }

    $nugetDll = Get-ChildItem -Path (Join-Path $env:USERPROFILE '.nuget\packages\system.io.ports') -Recurse -Filter 'System.IO.Ports.dll' -ErrorAction SilentlyContinue |
        Sort-Object { [version]($_.Directory.Parent.Name) } -Descending |
        Select-Object -First 1
    if ($nugetDll) {
        Add-Type -Path $nugetDll.FullName -ErrorAction Stop
        $null = [System.IO.Ports.SerialPort]
        return
    }

    foreach ($assemblyName in @(
            'System.IO.Ports, Version=8.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51'
            'System.IO.Ports, Version=7.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51'
            'System.IO.Ports, Version=4.0.0.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51'
            'System.IO.Ports')) {
        try {
            Add-Type -AssemblyName $assemblyName -ErrorAction Stop
            $null = [System.IO.Ports.SerialPort]
            return
        }
        catch {
        }
    }

    throw @"
System.IO.Ports is required for ROE serial (COM port).
Use olympus_turret_pipe_gui_ps.bat (Windows PowerShell 5.1), or install the package:
  Install-Package System.IO.Ports -ProviderName NuGet -Scope CurrentUser -Force
"@
}

Import-SystemIoPortsAssembly

. (Join-Path $PSScriptRoot 'FlimagePipeClient.ps1')

Add-Type @"
using System;
using System.Runtime.InteropServices;

public sealed class GlobalNumpadHook : IDisposable
{
    public event Action<string, int> NumpadEvent;
    public Func<bool> ShouldCaptureKey;
    public IntPtr GuiWindowHandle = IntPtr.Zero;
    public string KeyLayout = "Numpad";

    private const int WH_KEYBOARD_LL = 13;
    private const int WM_KEYDOWN = 0x0100;
    private const int WM_KEYUP = 0x0101;
    private const int WM_SYSKEYDOWN = 0x0104;
    private const int WM_SYSKEYUP = 0x0105;

    private IntPtr _hookId = IntPtr.Zero;
    private readonly LowLevelKeyboardProc _proc;
    private bool _disposed;

    private delegate IntPtr LowLevelKeyboardProc(int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern IntPtr SetWindowsHookEx(int idHook, LowLevelKeyboardProc lpfn, IntPtr hMod, uint dwThreadId);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool UnhookWindowsHookEx(IntPtr hhk);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern IntPtr GetModuleHandle(string lpModuleName);

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern bool IsChild(IntPtr hWndParent, IntPtr hWnd);

    public GlobalNumpadHook()
    {
        _proc = HookCallback;
    }

    public bool IsGuiForeground()
    {
        if (GuiWindowHandle == IntPtr.Zero) { return false; }
        IntPtr fg = GetForegroundWindow();
        if (fg == GuiWindowHandle) { return true; }
        return IsChild(GuiWindowHandle, fg);
    }

    public void Start()
    {
        if (_hookId != IntPtr.Zero) { return; }
        using (var curProcess = System.Diagnostics.Process.GetCurrentProcess())
        using (var curModule = curProcess.MainModule)
        {
            _hookId = SetWindowsHookEx(WH_KEYBOARD_LL, _proc, GetModuleHandle(curModule.ModuleName), 0);
        }
    }

    public void Stop()
    {
        if (_hookId == IntPtr.Zero) { return; }
        UnhookWindowsHookEx(_hookId);
        _hookId = IntPtr.Zero;
    }

    public void Dispose()
    {
        if (_disposed) { return; }
        Stop();
        _disposed = true;
    }

    private static string MapVirtualKey(int vkCode, string keyLayout)
    {
        if (string.Equals(keyLayout, "Cursor", StringComparison.OrdinalIgnoreCase))
        {
            switch (vkCode)
            {
                case 0x25: return "XMinus";
                case 0x27: return "XPlus";
                case 0x28: return "YMinus";
                case 0x26: return "YPlus";
                case 0xDB: return "ZMinus";
                case 0xDD: return "ZPlus";
                case 0xBC: return "PresetFiner";
                case 0xBE: return "PresetCoarser";
                default: return null;
            }
        }

        switch (vkCode)
        {
            case 0x64: return "XMinus";
            case 0x66: return "XPlus";
            case 0x62: return "YMinus";
            case 0x68: return "YPlus";
            case 0x6B: return "ZPlus";
            case 0x6D: return "ZMinus";
            case 0x6F: return "PresetFiner";
            case 0x6A: return "PresetCoarser";
            default: return null;
        }
    }

    private IntPtr HookCallback(int nCode, IntPtr wParam, IntPtr lParam)
    {
        if (nCode >= 0)
        {
            int vkCode = Marshal.ReadInt32(lParam);
            string keyName = MapVirtualKey(vkCode, KeyLayout);
            if (keyName != null)
            {
                bool isDown = wParam == (IntPtr)WM_KEYDOWN || wParam == (IntPtr)WM_SYSKEYDOWN;
                bool isUp = wParam == (IntPtr)WM_KEYUP || wParam == (IntPtr)WM_SYSKEYUP;
                if (isDown || isUp)
                {
                    if (IsGuiForeground())
                    {
                        return CallNextHookEx(_hookId, nCode, wParam, lParam);
                    }

                    var filter = ShouldCaptureKey;
                    if (filter != null && !filter())
                    {
                        return CallNextHookEx(_hookId, nCode, wParam, lParam);
                    }

                    var handler = NumpadEvent;
                    if (handler != null) { handler(keyName, isDown ? 1 : 0); }
                    return (IntPtr)1;
                }
            }
        }
        return CallNextHookEx(_hookId, nCode, wParam, lParam);
    }
}
"@

$TurretMin = 1
$TurretMax = 6
$MotorVelMin = 101
$MotorVelMax = 9999
$MotorVelDefault = 2000
$SettingsFile = Join-Path $PSScriptRoot "olympus_turret_pipe_gui.settings.txt"
$PresetNames = @('1', '2', '3', '4', '5')
$PresetLegacyNameMap = @{
    Fine   = '2'
    Mid    = '3'
    Coarse = '5'
}
$JogKeyLayouts = @('Numpad', 'Cursor')
$GuiMargin = 10
$GuiInnerWidth = 372
$script:CollapsibleSections = New-Object System.Collections.Generic.List[object]
$script:LayoutTopY = 42
$script:LayoutSectionGap = 6
$script:LayoutBottomMargin = 12
$script:CollapsedSectionHeight = 26
$script:SavedGuiSettings = @{}

$script:OlympusObjectiveParkZUm = 0.0
$script:OlympusTurretWaitMs = 500
$script:OlympusZMoveTimeoutMs = 60000
$script:OlympusZPositionToleranceUm = 2.0
$script:OlympusObjectiveSetTimeoutMs = 45000
$script:OlympusPanelControlMode = 'COMPUTER'
$script:SuppressOlympusPanelControlUiEvent = $false

$script:PipeR = $null
$script:PipeW = $null
$script:PipeConnected = $false
$script:Busy = $false
$script:JogKeysDown = @{}
$script:PresetKeysDown = @{}
$script:JogKeyLayout = 'Numpad'
$script:JogActionNames = @('XMinus', 'XPlus', 'YMinus', 'YPlus', 'ZPlus', 'ZMinus')
$script:PresetActionNames = @('PresetFiner', 'PresetCoarser')
$script:GlobalNumpadHook = $null
$script:HeldJogDx = 0.0
$script:HeldJogDy = 0.0
$script:HeldJogDz = 0.0
$script:HeldJogVx = 0.0
$script:HeldJogVy = 0.0
$script:HeldJogVz = 0.0
$script:MotorUsesVectorJog = $false
$script:NumpadJogEnabled = $true
$script:PresetDialogOpen = $false
$script:SuppressKeyLayoutComboEvent = $false
$script:ActivePreset = '3'
$script:LastVelocity = $MotorVelDefault

function New-DefaultPresetTable {
    return @{
        '1' = @{
            KeyStepXY = 0.05; KeyStepZ = 0.05
            KnobStepXY = 0.05; KnobStepZ = 0.05
            VelXY = 0.005; VelZ = 0.003
        }
        '2' = @{
            KeyStepXY = 0.10; KeyStepZ = 0.05
            KnobStepXY = 0.10; KnobStepZ = 0.05
            VelXY = 0.010; VelZ = 0.005
        }
        '3' = @{
            KeyStepXY = 1.00; KeyStepZ = 0.50
            KnobStepXY = 1.00; KnobStepZ = 0.50
            VelXY = 0.050; VelZ = 0.025
        }
        '4' = @{
            KeyStepXY = 5.00; KeyStepZ = 2.50
            KnobStepXY = 5.00; KnobStepZ = 2.50
            VelXY = 0.100; VelZ = 0.050
        }
        '5' = @{
            KeyStepXY = 10.0; KeyStepZ = 5.00
            KnobStepXY = 10.0; KnobStepZ = 5.00
            VelXY = 0.200; VelZ = 0.100
        }
    }
}

$script:Presets = New-DefaultPresetTable
$script:FilterLabels = @{}
$script:ObjectiveLabels = @{}
$script:OlympusTurretsAvailable = $false
$script:RoeConnected = $false
$script:RoeComPort = ''
$script:RoeSerial = $null
$script:RoeLineBuffer = ''
$script:RoeHostPingMs = 400
$script:MotorInvertX = $false
$script:MotorInvertY = $false
$script:MotorInvertZ = $false
$script:MotorSwapXY = $false
$script:RoeClockwisePositive = $false
$script:RoeEncTicks = @{}
$script:RoeMoveInFlight = $false
$script:RoePendingMove = $null
$script:MotorPosRefreshPending = $false
$script:LastMotorActivityUtc = [datetime]::MinValue
$script:MotorPosRefreshQuietMs = 500
$script:MotorPosDeadX = 0.0
$script:MotorPosDeadY = 0.0
$script:MotorPosDeadZ = 0.0
$script:MotorPosDeadSynced = $false
$RoeFirmwareId = 'roe_encoder_nano'
$RoeBaudRate = 115200

function Get-DefaultTurretLabel {
    param([int]$Position)
    return "$Position"
}

function Initialize-TurretLabelDefaults {
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        if (-not $script:FilterLabels.ContainsKey($p)) {
            $script:FilterLabels[$p] = Get-DefaultTurretLabel $p
        }
        if (-not $script:ObjectiveLabels.ContainsKey($p)) {
            $script:ObjectiveLabels[$p] = Get-DefaultTurretLabel $p
        }
    }
}

function Get-DefaultGuiSettings {
    $settings = @{
        LastVelocity     = $MotorVelDefault
        ActivePreset     = '3'
        NumpadJogEnabled = $true
        JogKeyLayout     = 'Numpad'
        RoeComPort       = ''
        RoeHostPingMs    = 400
        MotorInvertX     = 'false'
        MotorInvertY     = 'false'
        MotorInvertZ     = 'false'
        MotorSwapXY      = 'false'
        RoeClockwisePositive = 'false'
        'Olympus.ObjectiveParkZUm' = 0
        'Olympus.TurretWaitMs'  = 500
        'Olympus.ZMoveTimeoutMs' = 60000
        'Olympus.ZPositionToleranceUm' = 2
        'Olympus.ObjectiveSetTimeoutMs' = 45000
        'Olympus.PanelControlMode' = 'COMPUTER'
        'SectionExpanded.Roe' = 'false'
        'SectionExpanded.Filter' = 'true'
        'SectionExpanded.Objective' = 'true'
        'SectionExpanded.LightPath' = 'true'
        'SectionExpanded.Panel' = 'true'
        'SectionExpanded.Motor' = 'true'
        'SectionExpanded.PipeCommand' = 'false'
        'SectionExpanded.Log' = 'true'
    }
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $settings["Filter.P$p"] = Get-DefaultTurretLabel $p
        $settings["Objective.P$p"] = Get-DefaultTurretLabel $p
    }
    foreach ($name in $PresetNames) {
        $preset = (New-DefaultPresetTable)[$name]
        $settings["$name.KeyStepXY"] = $preset.KeyStepXY
        $settings["$name.KeyStepZ"] = $preset.KeyStepZ
        $settings["$name.KnobStepXY"] = $preset.KnobStepXY
        $settings["$name.KnobStepZ"] = $preset.KnobStepZ
        $settings["$name.VelXY"] = $preset.VelXY
        $settings["$name.VelZ"] = $preset.VelZ
    }
    return $settings
}

function Import-GuiSettings {
    $settings = Get-DefaultGuiSettings
    if (-not (Test-Path $SettingsFile)) {
        return $settings
    }

    $loaded = @{}
    foreach ($line in Get-Content -Path $SettingsFile -ErrorAction SilentlyContinue) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $parts = $trimmed.Split('=', 2)
        if ($parts.Count -lt 2) { continue }
        $loaded[$parts[0].Trim()] = $parts[1].Trim()
    }

    foreach ($key in @($loaded.Keys)) {
        $settings[$key] = $loaded[$key]
    }
    return $settings
}

function Apply-GuiSettings {
    param([hashtable]$Settings)

    $script:LastVelocity = [double]$Settings.LastVelocity
    if ($script:LastVelocity -lt $MotorVelMin) { $script:LastVelocity = $MotorVelMin }
    if ($script:LastVelocity -gt $MotorVelMax) { $script:LastVelocity = $MotorVelMax }

    $preset = [string]$Settings.ActivePreset
    if ($PresetLegacyNameMap.ContainsKey($preset)) {
        $preset = $PresetLegacyNameMap[$preset]
    }
    if ($PresetNames -contains $preset) {
        $script:ActivePreset = $preset
    }

    $numpadText = [string]$Settings.NumpadJogEnabled
    $script:NumpadJogEnabled = ($numpadText -match '^(?i)(1|true|yes|on)$')

    $layoutText = [string]$Settings.JogKeyLayout
    if ($JogKeyLayouts -contains $layoutText) {
        $script:JogKeyLayout = $layoutText
    }

    $script:RoeComPort = [string]$Settings.RoeComPort
    if ($Settings.ContainsKey('RoeHostPingMs')) {
        try {
            $pingMs = [int]$Settings.RoeHostPingMs
            if ($pingMs -ge 200 -and $pingMs -le 5000) {
                $script:RoeHostPingMs = $pingMs
            }
        }
        catch { }
    }

    $script:MotorInvertX = (Get-SettingsBool -Settings $Settings -Key 'MotorInvertX' -LegacyKey 'RoeInvertX')
    $script:MotorInvertY = (Get-SettingsBool -Settings $Settings -Key 'MotorInvertY' -LegacyKey 'RoeInvertY')
    $script:MotorInvertZ = (Get-SettingsBool -Settings $Settings -Key 'MotorInvertZ' -LegacyKey 'RoeInvertZ')
    $script:MotorSwapXY = (Get-SettingsBool -Settings $Settings -Key 'MotorSwapXY' -LegacyKey 'RoeSwapXY')
    $script:RoeClockwisePositive = (Get-SettingsBool -Settings $Settings -Key 'RoeClockwisePositive')

    try {
        $script:OlympusObjectiveParkZUm = [double]$Settings['Olympus.ObjectiveParkZUm']
    }
    catch {
        $script:OlympusObjectiveParkZUm = 0.0
    }
    try {
        $script:OlympusTurretWaitMs = [int]$Settings['Olympus.TurretWaitMs']
    }
    catch {
        $script:OlympusTurretWaitMs = 500
    }
    if ($script:OlympusTurretWaitMs -lt 0) { $script:OlympusTurretWaitMs = 0 }
    try {
        $script:OlympusZMoveTimeoutMs = [int]$Settings['Olympus.ZMoveTimeoutMs']
    }
    catch {
        if ($Settings.ContainsKey('Olympus.EscapeTimeoutMs')) {
            $script:OlympusZMoveTimeoutMs = [int]$Settings['Olympus.EscapeTimeoutMs']
        }
        else {
            $script:OlympusZMoveTimeoutMs = 60000
        }
    }
    if ($script:OlympusZMoveTimeoutMs -lt 1000) { $script:OlympusZMoveTimeoutMs = 1000 }
    try {
        $script:OlympusZPositionToleranceUm = [double]$Settings['Olympus.ZPositionToleranceUm']
    }
    catch {
        $script:OlympusZPositionToleranceUm = 2.0
    }
    if ($script:OlympusZPositionToleranceUm -lt 0.1) { $script:OlympusZPositionToleranceUm = 0.1 }
    try {
        $script:OlympusObjectiveSetTimeoutMs = [int]$Settings['Olympus.ObjectiveSetTimeoutMs']
    }
    catch {
        $script:OlympusObjectiveSetTimeoutMs = 45000
    }
    if ($script:OlympusObjectiveSetTimeoutMs -lt 1000) { $script:OlympusObjectiveSetTimeoutMs = 1000 }
    if ($Settings.ContainsKey('Olympus.PanelControlMode') -and -not [string]::IsNullOrWhiteSpace($Settings['Olympus.PanelControlMode'])) {
        $script:OlympusPanelControlMode = Normalize-OlympusPanelControlModeToken $Settings['Olympus.PanelControlMode']
    }
    else {
        $script:OlympusPanelControlMode = 'COMPUTER'
    }

    Initialize-JogKeyState

    $script:Presets = New-DefaultPresetTable
    $legacyTierByName = @{ Fine = '2'; Mid = '3'; Coarse = '5' }
    foreach ($name in $PresetNames) {
        $legacyName = $null
        if ($name -eq '2') { $legacyName = 'Fine' }
        elseif ($name -eq '3') { $legacyName = 'Mid' }
        elseif ($name -eq '5') { $legacyName = 'Coarse' }

        foreach ($field in @('KeyStepXY', 'KeyStepZ', 'KnobStepXY', 'KnobStepZ', 'VelXY', 'VelZ')) {
            $loaded = $false
            $newKey = "$name.$field"
            if ($Settings.ContainsKey($newKey)) {
                try {
                    $script:Presets[$name][$field] = [double]$Settings[$newKey]
                    $loaded = $true
                }
                catch { }
            }
            if ($loaded -or [string]::IsNullOrWhiteSpace($legacyName)) { continue }

            $legacyField = switch ($field) {
                'KeyStepXY' { 'StepXY' }
                'KeyStepZ'  { 'StepZ' }
                'KnobStepXY' { 'StepXY' }
                'KnobStepZ'  { 'StepZ' }
                'VelXY'     { 'VelXY' }
                'VelZ'      { 'VelZ' }
            }
            $legacyKey = "$legacyName.$legacyField"
            if ($Settings.ContainsKey($legacyKey)) {
                try { $script:Presets[$name][$field] = [double]$Settings[$legacyKey] } catch { }
            }
        }
    }

    $script:FilterLabels = @{}
    $script:ObjectiveLabels = @{}
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $filterKey = "Filter.P$p"
        $objKey = "Objective.P$p"
        if ($Settings.ContainsKey($filterKey) -and -not [string]::IsNullOrWhiteSpace($Settings[$filterKey])) {
            $script:FilterLabels[$p] = [string]$Settings[$filterKey]
        }
        else {
            $script:FilterLabels[$p] = Get-DefaultTurretLabel $p
        }
        if ($Settings.ContainsKey($objKey) -and -not [string]::IsNullOrWhiteSpace($Settings[$objKey])) {
            $script:ObjectiveLabels[$p] = [string]$Settings[$objKey]
        }
        else {
            $script:ObjectiveLabels[$p] = Get-DefaultTurretLabel $p
        }
    }
}

function Export-GuiSettings {
    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    Initialize-TurretLabelDefaults
    $lines = @(
        '# FLIMage pipe GUI settings (auto-saved next to the .ps1 script)',
        '# Filter.P1..P6 and Objective.P1..P6 are turret position labels',
        ("LastVelocity={0}" -f [int]$script:LastVelocity),
        ("ActivePreset={0}" -f $script:ActivePreset),
        ("NumpadJogEnabled={0}" -f ($(if ($script:NumpadJogEnabled) { 'true' } else { 'false' }))),
        ("JogKeyLayout={0}" -f $script:JogKeyLayout),
        ("RoeComPort={0}" -f $script:RoeComPort),
        ("RoeHostPingMs={0}" -f $script:RoeHostPingMs),
        ("MotorInvertX={0}" -f ($(if ($script:MotorInvertX) { 'true' } else { 'false' }))),
        ("MotorInvertY={0}" -f ($(if ($script:MotorInvertY) { 'true' } else { 'false' }))),
        ("MotorInvertZ={0}" -f ($(if ($script:MotorInvertZ) { 'true' } else { 'false' }))),
        ("MotorSwapXY={0}" -f ($(if ($script:MotorSwapXY) { 'true' } else { 'false' }))),
        ("RoeClockwisePositive={0}" -f ($(if ($script:RoeClockwisePositive) { 'true' } else { 'false' }))),
        ("Olympus.ObjectiveParkZUm={0}" -f $script:OlympusObjectiveParkZUm.ToString('F1', $ci)),
        ("Olympus.TurretWaitMs={0}" -f $script:OlympusTurretWaitMs),
        ("Olympus.ZMoveTimeoutMs={0}" -f $script:OlympusZMoveTimeoutMs),
        ("Olympus.ZPositionToleranceUm={0}" -f $script:OlympusZPositionToleranceUm.ToString('F1', $ci)),
        ("Olympus.ObjectiveSetTimeoutMs={0}" -f $script:OlympusObjectiveSetTimeoutMs),
        ("Olympus.PanelControlMode={0}" -f $script:OlympusPanelControlMode)
    )
    foreach ($section in $script:CollapsibleSections) {
        $lines += ("SectionExpanded.{0}={1}" -f $section.SettingsKey, ($(if ($section.Expanded) { 'true' } else { 'false' })))
    }
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $lines += ("Filter.P{0}={1}" -f $p, $script:FilterLabels[$p])
        $lines += ("Objective.P{0}={1}" -f $p, $script:ObjectiveLabels[$p])
    }
    foreach ($name in $PresetNames) {
        $preset = $script:Presets[$name]
        $lines += ("{0}.KeyStepXY={1}" -f $name, $preset.KeyStepXY.ToString('F2', $ci))
        $lines += ("{0}.KeyStepZ={1}" -f $name, $preset.KeyStepZ.ToString('F2', $ci))
        $lines += ("{0}.KnobStepXY={1}" -f $name, $preset.KnobStepXY.ToString('F2', $ci))
        $lines += ("{0}.KnobStepZ={1}" -f $name, $preset.KnobStepZ.ToString('F2', $ci))
        $lines += ("{0}.VelXY={1}" -f $name, $preset.VelXY.ToString('F3', $ci))
        $lines += ("{0}.VelZ={1}" -f $name, $preset.VelZ.ToString('F3', $ci))
    }
    try {
        Set-Content -Path $SettingsFile -Value $lines -Encoding UTF8
    }
    catch {
        if ($null -ne $logBox) { Add-Log "!! Failed to save settings: $($_.Exception.Message)" }
    }
}

function Clear-KeyboardFocus {
    $form.ActiveControl = $null
    [void]$form.Focus()
}

function Register-FocusClearOnClick {
    param([System.Windows.Forms.Control]$Control)
    $Control.Add_MouseDown({
            param($sender, $eventArgs)
            if ($eventArgs.Button -eq [System.Windows.Forms.MouseButtons]::Left) {
                Clear-KeyboardFocus
            }
        })
}

function Get-SectionExpandedFromSettings {
    param(
        [string]$SettingsKey,
        [bool]$DefaultExpanded = $true
    )

    $key = "SectionExpanded.$SettingsKey"
    if ($script:SavedGuiSettings.ContainsKey($key)) {
        return (Get-SettingsBool -Settings $script:SavedGuiSettings -Key $key)
    }
    return $DefaultExpanded
}

function Register-CollapsibleSection {
    param(
        [System.Windows.Forms.GroupBox]$Group,
        [int]$ExpandedHeight,
        [string]$SettingsKey,
        [bool]$DefaultExpanded = $true
    )

    $expanded = Get-SectionExpandedFromSettings -SettingsKey $SettingsKey -DefaultExpanded $DefaultExpanded

    $toggle = New-Object System.Windows.Forms.Button
    $toggle.Size = New-Object System.Drawing.Size 24, 20
    $toggle.Location = New-Object System.Drawing.Point ($GuiInnerWidth - 30), 1
    $toggle.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $toggle.TabStop = $false
    $toggle.Font = New-Object System.Drawing.Font (
        $toggle.Font.FontFamily,
        9.0,
        [System.Drawing.FontStyle]::Bold
    )
    $Group.Controls.Add($toggle)
    $toggle.BringToFront()

    $contentControls = @()
    foreach ($child in $Group.Controls) {
        if ($child -ne $toggle) {
            $contentControls += $child
        }
    }

    $section = [PSCustomObject]@{
        Group           = $Group
        Toggle          = $toggle
        ExpandedHeight  = $ExpandedHeight
        SettingsKey     = $SettingsKey
        Expanded        = $expanded
        ContentControls = $contentControls
    }
    [void]$script:CollapsibleSections.Add($section)

    $toggle.Add_Click({
            param($sender, $eventArgs)
            $sec = $sender.Tag
            Set-CollapsibleSectionExpanded -Section $sec -Expanded (-not $sec.Expanded)
        })
    $toggle.Tag = $section

    Set-CollapsibleSectionExpanded -Section $section -Expanded $expanded -SkipLayout
}

function Set-CollapsibleSectionExpanded {
    param(
        $Section,
        [bool]$Expanded,
        [switch]$SkipLayout
    )

    $Section.Expanded = $Expanded
    foreach ($ctrl in $Section.ContentControls) {
        $ctrl.Visible = $Expanded
    }
    $Section.Toggle.Text = if ($Expanded) { '-' } else { '+' }
    if (-not $SkipLayout) {
        Update-CollapsibleLayout
        Export-GuiSettings
    }
}

function Get-CollapsibleSectionLayoutHeight {
    param($Section)

    if (-not $Section.Expanded) {
        return $script:CollapsedSectionHeight
    }

    $maxBottom = 24
    foreach ($ctrl in $Section.ContentControls) {
        if (-not $ctrl.Visible) { continue }
        $bottom = $ctrl.Location.Y + $ctrl.Height
        if ($bottom -gt $maxBottom) {
            $maxBottom = $bottom
        }
    }

    $toggleBottom = $Section.Toggle.Location.Y + $Section.Toggle.Height
    if ($toggleBottom -gt $maxBottom) {
        $maxBottom = $toggleBottom
    }

    $computed = $maxBottom + 8
    if ($computed -gt $Section.ExpandedHeight) {
        return $computed
    }
    return $Section.ExpandedHeight
}

function Update-CollapsibleLayout {
    if ($null -eq $form -or $script:CollapsibleSections.Count -eq 0) { return }

    $form.SuspendLayout()
    try {
        $y = $script:LayoutTopY
        foreach ($section in $script:CollapsibleSections) {
            $height = Get-CollapsibleSectionLayoutHeight -Section $section
            $section.Group.Location = New-Object System.Drawing.Point $GuiMargin, $y
            $section.Group.Size = New-Object System.Drawing.Size $GuiInnerWidth, $height
            $y += $height + $script:LayoutSectionGap
        }

        $clientHeight = $y - $script:LayoutSectionGap + $script:LayoutBottomMargin
        $formWidth = $GuiInnerWidth + ($GuiMargin * 2) + 16
        $borderHeight = $form.Height - $form.ClientSize.Height
        if ($borderHeight -lt 8) { $borderHeight = 39 }
        $form.ClientSize = New-Object System.Drawing.Size ($GuiInnerWidth + ($GuiMargin * 2)), $clientHeight
        $form.Size = New-Object System.Drawing.Size $formWidth, ($clientHeight + $borderHeight)

        $minClient = $script:LayoutTopY + ($script:CollapsedSectionHeight * $script:CollapsibleSections.Count) +
            (($script:LayoutSectionGap * ($script:CollapsibleSections.Count - 1)) + $script:LayoutBottomMargin)
        $form.MinimumSize = New-Object System.Drawing.Size $formWidth, ($minClient + $borderHeight)
    }
    finally {
        $form.ResumeLayout($true)
    }
}

function Initialize-CollapsibleLayout {
    Register-CollapsibleSection -Group $roeGroup -ExpandedHeight 62 -SettingsKey 'Roe' -DefaultExpanded $false
    Register-CollapsibleSection -Group $filterGroup -ExpandedHeight 56 -SettingsKey 'Filter' -DefaultExpanded $true
    Register-CollapsibleSection -Group $objGroup -ExpandedHeight 56 -SettingsKey 'Objective' -DefaultExpanded $true
    Register-CollapsibleSection -Group $prismGroup -ExpandedHeight 56 -SettingsKey 'LightPath' -DefaultExpanded $true
    Register-CollapsibleSection -Group $panelGroup -ExpandedHeight 56 -SettingsKey 'Panel' -DefaultExpanded $true
    Register-CollapsibleSection -Group $motorGroup -ExpandedHeight 200 -SettingsKey 'Motor' -DefaultExpanded $true
    Register-CollapsibleSection -Group $cmdGroup -ExpandedHeight 118 -SettingsKey 'PipeCommand' -DefaultExpanded $false
    Register-CollapsibleSection -Group $logGroup -ExpandedHeight 180 -SettingsKey 'Log' -DefaultExpanded $true
    Update-CollapsibleLayout
}

function Get-Timestamp {
    return (Get-Date).ToString("HH:mm:ss.fff")
}

function Get-OlympusReplyPayload {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return '' }
    if ($Reply -match '^OlympusReply,\s*(.+)$') {
        return $Matches[1].Trim()
    }
    if ($Reply -match '(?i)^Error:\s*(.+)$') {
        return $Matches[1].Trim()
    }
    return $Reply.Trim()
}

function Get-OlympusIx2PositionValue {
    param(
        [string]$Reply,
        [string]$CommandToken = ''
    )

    if ($Reply -match '(?i)^Error') { return $null }
    $payload = Get-OlympusReplyPayload -Reply $Reply
    if ([string]::IsNullOrWhiteSpace($payload)) { return $null }

    if (-not [string]::IsNullOrWhiteSpace($CommandToken)) {
        $token = $CommandToken.Trim()
        $escaped = [regex]::Escape($token)

        # IX2 query/set replies such as "1OB 4" or "1MU 4".
        if ($payload -match ('(?i)' + $escaped + '\s+([+-]?\d+)')) {
            return [int]$Matches[1]
        }

        # IX2 set ack such as "1MU +004" (position after '+').
        if ($payload -match ('(?i)' + $escaped + '\s+\+(\d+)')) {
            return [int]$Matches[1]
        }

        # Set ack without position digits (e.g. "1OB +") — do not treat the leading "1" in "1OB" as position.
        if ($payload -match ('(?i)^' + $escaped + '\b')) {
            return $null
        }
    }

    # Generic fallback: last numeric field (skip when token prefix would be misread as position).
    $numberMatches = [regex]::Matches($payload, '(?<!\d)[+-]?\d+')
    if ($numberMatches.Count -gt 0) {
        return [int]$numberMatches[$numberMatches.Count - 1].Value
    }
    return $null
}

function Test-OlympusIx2SetAckWithoutPosition {
    param(
        [string]$Reply,
        [string]$CommandToken
    )

    if (-not (Test-OlympusIx2Success -Reply $Reply)) { return $false }
    if ([string]::IsNullOrWhiteSpace($CommandToken)) { return $false }

    $payload = Get-OlympusReplyPayload -Reply $Reply
    if ([string]::IsNullOrWhiteSpace($payload)) { return $false }

    $escaped = [regex]::Escape($CommandToken.Trim())
    return ($payload -match ('(?i)^' + $escaped + '\s*\+\s*$'))
}

function Test-OlympusIx2TurretSetCommand {
    param([string]$Ix2Command)

    if ([string]::IsNullOrWhiteSpace($Ix2Command)) { return $false }
    return ($Ix2Command.Trim() -match '^(?i)(1OB|1MU|1PRISM)\s+\d')
}

function Resolve-OlympusTurretPositionFromReply {
    param(
        [string]$Reply,
        [string]$CommandToken
    )

    $value = Get-OlympusIx2PositionValue -Reply $Reply -CommandToken $CommandToken
    if ($null -ne $value) { return $value }

    if (Test-OlympusIx2SetAckWithoutPosition -Reply $Reply -CommandToken $CommandToken) {
        $queryReply = Invoke-OlympusIx2Query -Ix2Command ("{0}?" -f $CommandToken.Trim())
        if ($null -ne $queryReply) {
            return Get-OlympusIx2PositionValue -Reply $queryReply -CommandToken $CommandToken
        }
    }

    return $null
}

function Get-OlympusNumericPosition {
    param([string]$Reply)
    return Get-OlympusIx2PositionValue -Reply $Reply
}

function Update-OlympusLabelsFromPipeExchange {
    param(
        [string]$Command,
        [string]$Reply
    )

    $ix2 = $null
    if ($Command -match '(?i)^SendOlympusCommand,\s*(.+)$') {
        $ix2 = $Matches[1].Trim()
    }
    elseif ($Command -match '(?i)^(1OB|1MU|1PRISM)\b') {
        $ix2 = $Command.Trim()
    }
    if ([string]::IsNullOrWhiteSpace($ix2)) { return }

    $ix2Upper = $ix2.ToUpperInvariant()
    if ($ix2Upper -eq '1OB?' -or $ix2Upper -match '^1OB\s') {
        Update-ObjectiveLabel -Reply $Reply
    }
    elseif ($ix2Upper -eq '1MU?' -or $ix2Upper -match '^1MU\s') {
        Update-FilterLabel -Reply $Reply
    }
    elseif ($ix2Upper -eq '1PRISM?' -or $ix2Upper -match '^1PRISM\s') {
        Update-PrismLabel -Reply $Reply
    }
}

function Get-OlympusTurretWaitMs {
    if ($null -ne $script:OlympusTurretWaitMs) {
        return [int]$script:OlympusTurretWaitMs
    }
    return 500
}

function Build-OlympusTurretSetSequence {
    param([string]$Ix2SetCommand)
    return ,$Ix2SetCommand.Trim()
}

function Test-OlympusIx2Success {
    param([string]$Reply)
    if ($Reply -match '(?i)^Error') { return $false }
    $payload = Get-OlympusReplyPayload -Reply $Reply
    if ($payload -match '!') { return $false }
    if ($payload -match '\+') { return $true }
    return $true
}

function Invoke-OlympusIx2Query {
    param([string]$Ix2Command)
    $ix2 = $Ix2Command.Trim()
    if ($ix2 -eq '') { return $null }
    return Send-FlimagePipeCommand -Command ("SendOlympusCommand, $ix2")
}

function Get-FlimageAbsPositionUm {
    param([switch]$Quiet)

    if (-not $script:PipeConnected) { return $null }
    try {
        if (-not $Quiet) { Add-Log '>> GetCurrentPosition' }
        $reply = Send-FlimagePipeCommand -Command 'GetCurrentPosition'
        if (-not $Quiet) { Add-Log "<< $reply" }
        if ($reply -match '(?i)^Error') { return $null }
        return Get-MotorPositionFromReply -Reply $reply
    }
    catch {
        return $null
    }
}

function Invoke-FlimageMoveRelativeWait {
    param(
        [double]$DeltaX = 0,
        [double]$DeltaY = 0,
        [double]$DeltaZ = 0,
        [switch]$Quiet
    )

    if (-not $script:PipeConnected) {
        throw 'FLIMage pipe not connected'
    }

    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    $cmd = ('MoveMotorRelative, {0}, {1}, {2}' -f `
        $DeltaX.ToString($ci), `
        $DeltaY.ToString($ci), `
        $DeltaZ.ToString($ci))
    if (-not $Quiet) { Add-Log ">> $cmd" }
    $reply = Send-FlimagePipeCommand -Command $cmd
    if (-not $Quiet) { Add-Log "<< $reply" }
    if ($reply -match '(?i)^Error') {
        throw $reply
    }
    Update-MotorPosLabel $reply
    return $reply
}

function Invoke-FlimageSetAbsolutePositionWait {
    param(
        [double]$X,
        [double]$Y,
        [double]$Z,
        [switch]$Quiet
    )

    if (-not $script:PipeConnected) {
        throw 'FLIMage pipe not connected'
    }

    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    $cmd = ('SetMotorPosition, {0}, {1}, {2}' -f `
        $X.ToString($ci), `
        $Y.ToString($ci), `
        $Z.ToString($ci))
    if (-not $Quiet) { Add-Log ">> $cmd" }
    $reply = Send-FlimagePipeCommand -Command $cmd
    if (-not $Quiet) { Add-Log "<< $reply" }
    if ($reply -match '(?i)^Error') {
        throw $reply
    }
    Update-MotorPosLabel $reply
    return $reply
}

function Wait-FlimageZPositionUm {
    param(
        [double]$TargetZUm,
        [double]$ToleranceUm,
        [int]$TimeoutMs,
        [int]$PollMs = 200,
        [string]$Label = 'Z move'
    )

    $deadline = [datetime]::UtcNow.AddMilliseconds($TimeoutMs)
    while ([datetime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds $PollMs
        $pos = Get-FlimageAbsPositionUm -Quiet
        if ($null -eq $pos) { continue }
        if ([math]::Abs($pos[2] - $TargetZUm) -le $ToleranceUm) {
            Add-Log ("{0} confirmed (Z={1:F2} um)" -f $Label, $pos[2])
            Update-MotorPosLabel ("CurrentPosition, {0}, {1}, {2}" -f $pos[0], $pos[1], $pos[2])
            return $true
        }
    }
    return $false
}

function Wait-OlympusTurretAtPosition {
    param(
        [string]$CommandToken,
        [int]$TargetPos,
        [int]$TimeoutMs,
        [int]$PollMs = 500
    )

    $deadline = [datetime]::UtcNow.AddMilliseconds($TimeoutMs)
    while ([datetime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds $PollMs
        $reply = Invoke-OlympusIx2Query -Ix2Command ("{0}?" -f $CommandToken)
        if ($null -eq $reply) { continue }
        $pos = Get-OlympusIx2PositionValue -Reply $reply -CommandToken $CommandToken
        if ($null -ne $pos -and $pos -eq $TargetPos) {
            Add-Log ("Turret {0} reached position {1}" -f $CommandToken, $TargetPos)
            return $true
        }
    }
    return $false
}

function Send-OlympusObjectiveSetWithEscape {
    param(
        [int]$Position,
        [scriptblock]$OnSuccess = $null
    )

    if ($Position -lt $TurretMin -or $Position -gt $TurretMax) {
        throw ("Objective position must be {0}-{1}" -f $TurretMin, $TurretMax)
    }

    $zTimeout = [int]$script:OlympusZMoveTimeoutMs
    $zTol = [double]$script:OlympusZPositionToleranceUm
    $parkZ = [double]$script:OlympusObjectiveParkZUm
    $turretTimeout = [int]$script:OlympusObjectiveSetTimeoutMs

    $pos = Get-FlimageAbsPositionUm
    if ($null -eq $pos) {
        throw 'Could not read FLIMage motor position (GetCurrentPosition)'
    }
    $x = $pos[0]
    $y = $pos[1]
    $zSaved = $pos[2]

    if ([math]::Abs($zSaved - $parkZ) -gt $zTol) {
        Add-Log ("Objective change -> {0}: SetMotorPosition Z={1:F2} um (from {2:F2})..." -f $Position, $parkZ, $zSaved)
        Invoke-FlimageSetAbsolutePositionWait -X $x -Y $y -Z $parkZ -Quiet
        if (-not (Wait-FlimageZPositionUm -TargetZUm $parkZ -ToleranceUm $zTol -TimeoutMs $zTimeout -Label 'Z park')) {
            throw ("Z did not reach park position ({0} um) within {1} ms" -f $parkZ, $zTimeout)
        }
    }

    Add-Log ("Objective change -> {0}: 1OB {0}..." -f $Position)
    $setReply = Send-OlympusIx2WithLog -Ix2Command ("1OB {0}" -f $Position)
    if (-not (Test-OlympusIx2Success -Reply $setReply)) {
        throw ("1OB set failed: {0}" -f (Get-OlympusReplyPayload -Reply $setReply))
    }
    if (-not (Wait-OlympusTurretAtPosition -CommandToken '1OB' -TargetPos $Position -TimeoutMs $turretTimeout)) {
        throw ("Objective turret did not reach position {0} within {1} ms" -f $Position, $turretTimeout)
    }

    if ([math]::Abs($zSaved - $parkZ) -gt $zTol) {
        Add-Log ("Objective change -> {0}: SetMotorPosition restore Z={1:F2} um..." -f $Position, $zSaved)
        Invoke-FlimageSetAbsolutePositionWait -X $x -Y $y -Z $zSaved -Quiet
        if (-not (Wait-FlimageZPositionUm -TargetZUm $zSaved -ToleranceUm $zTol -TimeoutMs $zTimeout -Label 'Z restore')) {
            Add-Log "!! Z restore timed out; focus may still be at park position"
        }
    }

    $queryReply = Invoke-OlympusIx2Query -Ix2Command '1OB?'
    if ($null -ne $queryReply) {
        Update-ObjectiveLabel -Reply $queryReply
    }
    else {
        Update-ObjectiveLabel -Reply $setReply
    }
    if ($null -ne $OnSuccess) {
        & $OnSuccess $setReply
    }
    return $setReply
}

function Send-OlympusIx2WithLog {
    param(
        [string]$Ix2Command,
        [switch]$AllowError
    )
    $ix2 = $Ix2Command.Trim()
    if ($ix2 -eq '') { throw 'Empty IX2-UCB command' }
    $pipeCmd = "SendOlympusCommand, $ix2"
    Add-Log ">> $pipeCmd"
    $reply = Send-FlimagePipeCommand -Command $pipeCmd
    Add-Log "<< $reply"
    if ($reply -match '(?i)^Error' -and -not $AllowError) {
        throw $reply
    }
    if (-not (Test-OlympusIx2TurretSetCommand -Ix2Command $ix2)) {
        Update-OlympusLabelsFromPipeExchange -Command $pipeCmd -Reply $reply
    }
    return $reply
}

function Normalize-OlympusPanelControlModeToken {
    param([string]$Token)

    if ([string]::IsNullOrWhiteSpace($Token)) { return 'COMPUTER' }
    $normalized = $Token.Trim().ToUpperInvariant() -replace '\s', ''
    switch ($normalized) {
        'ON' { return 'MANUAL+COMPUTER' }
        'OFF' { return 'COMPUTER' }
        'MANUAL+COMPUTER' { return 'MANUAL+COMPUTER' }
        'MANUALCOMPUTER' { return 'MANUAL+COMPUTER' }
        'COMPUTER' { return 'COMPUTER' }
        'REMOTE' { return 'COMPUTER' }
        'PC' { return 'COMPUTER' }
        'LOCAL' { return 'MANUAL+COMPUTER' }
        'MANUAL' { return 'MANUAL+COMPUTER' }
        'PANEL' { return 'MANUAL+COMPUTER' }
        default { return 'COMPUTER' }
    }
}

function Get-OlympusPanelControlFromReply {
    param([string]$Reply)

    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    if ($Reply -match '(?i)^OlympusPanelControl\s*,\s*(.+)') {
        return (Normalize-OlympusPanelControlModeToken $Matches[1].Trim())
    }
    return $null
}

$script:PanelControlOptions = @(
    @{ Mode = 'MANUAL+COMPUTER'; Label = 'Manual + Computer' },
    @{ Mode = 'COMPUTER'; Label = 'Computer' }
)

function Initialize-PanelControlCombo {
    if ($null -eq $panelCombo) { return }
    $panelCombo.Items.Clear()
    foreach ($opt in $script:PanelControlOptions) {
        [void]$panelCombo.Items.Add($opt.Label)
    }
}

function Sync-PanelComboToMode {
    param([string]$Mode)

    if ($null -eq $panelCombo) { return }
    $modeNorm = Normalize-OlympusPanelControlModeToken $Mode
    for ($i = 0; $i -lt $script:PanelControlOptions.Count; $i++) {
        if ($script:PanelControlOptions[$i].Mode -eq $modeNorm) {
            $script:SuppressOlympusPanelControlUiEvent = $true
            try { $panelCombo.SelectedIndex = $i }
            finally { $script:SuppressOlympusPanelControlUiEvent = $false }
            return
        }
    }
}

function Get-PanelComboMode {
    if ($null -eq $panelCombo -or $panelCombo.SelectedIndex -lt 0) {
        return 'COMPUTER'
    }
    if ($panelCombo.SelectedIndex -ge $script:PanelControlOptions.Count) {
        return 'COMPUTER'
    }
    return [string]$script:PanelControlOptions[$panelCombo.SelectedIndex].Mode
}

function Update-OlympusPanelControlUi {
    param([string]$Mode)

    if ($null -eq $panelCombo) { return }
    $modeNorm = Normalize-OlympusPanelControlModeToken $Mode
    $script:OlympusPanelControlMode = $modeNorm
    Sync-PanelComboToMode -Mode $modeNorm
}

function Invoke-OlympusPanelControlSet {
    param(
        [string]$Mode,
        [switch]$Quiet
    )

    if (-not $script:PipeConnected) {
        throw 'FLIMage pipe not connected'
    }

    $modeNorm = Normalize-OlympusPanelControlModeToken $Mode
    $pipeMode = switch ($modeNorm) {
        'COMPUTER' { 'COMPUTER' }
        default { 'MANUAL+COMPUTER' }
    }
    $cmd = "SetOlympusPanelControl, $pipeMode"
    if (-not $Quiet) { Add-Log ">> $cmd" }
    $reply = Send-FlimagePipeCommand -Command $cmd
    if (-not $Quiet) { Add-Log "<< $reply" }
    if ($reply -match '(?i)^Error') {
        throw $reply
    }
    $applied = Get-OlympusPanelControlFromReply -Reply $reply
    if ($null -ne $applied) {
        Update-OlympusPanelControlUi -Mode $applied
    }
    Export-GuiSettings
    return $reply
}

function Sync-OlympusPanelControlFromFlimage {
    param([switch]$Quiet)

    if (-not $script:PipeConnected) { return }
    try {
        if (-not $Quiet) { Add-Log '>> GetOlympusPanelControl' }
        $reply = Send-FlimagePipeCommand -Command 'GetOlympusPanelControl'
        if (-not $Quiet) { Add-Log "<< $reply" }
        $mode = Get-OlympusPanelControlFromReply -Reply $reply
        if ($null -ne $mode) {
            Update-OlympusPanelControlUi -Mode $mode
        }
    }
    catch {
        if (-not $Quiet) { Add-Log "!! GetOlympusPanelControl failed: $($_.Exception.Message)" }
    }
}

function Send-OlympusIx2SequenceWithLog {
    param(
        [string[]]$Ix2Commands,
        [scriptblock]$OnLastReply = $null
    )

    if ($null -eq $Ix2Commands -or $Ix2Commands.Count -eq 0) {
        throw 'No IX2-UCB commands'
    }

    $reply = $null
    foreach ($ix2 in $Ix2Commands) {
        if ([string]::IsNullOrWhiteSpace($ix2)) { continue }
        $reply = Send-OlympusIx2WithLog -Ix2Command $ix2
    }

    $waitMs = Get-OlympusTurretWaitMs
    if ($waitMs -gt 0) {
        Start-Sleep -Milliseconds $waitMs
    }

    if ($null -ne $OnLastReply -and $null -ne $reply) {
        & $OnLastReply $reply
    }
    return $reply
}

function Get-TurretValue {
    param([string]$Reply, [string]$Token)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    $pattern = "{0}\s*,\s*(-?\d+)" -f [regex]::Escape($Token)
    $match = [regex]::Match($Reply, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) { return $null }
    return [int]$match.Groups[1].Value
}

function Get-StateAssignmentValue {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    if ($Reply -match '=\s*"?([^"\r\n]+)"?\s*$') {
        return $Matches[1].Trim()
    }
    return $null
}

function Test-MotorHwSupportsOlympusTurrets {
    param([string]$MotorHwName)
    if ([string]::IsNullOrWhiteSpace($MotorHwName)) { return $false }
    return ($MotorHwName -match '(?i)olympus')
}

function Update-RoeStatusLabel {
    if ($null -eq $roeStatusLabel) { return }
    if ($script:RoeConnected) {
        $roeStatusLabel.Text = ("ROE: {0}  -  host mode" -f $script:RoeComPort)
        $roeStatusLabel.ForeColor = [System.Drawing.Color]::DarkGreen
    }
    else {
        $roeStatusLabel.Text = 'ROE: disconnected'
        $roeStatusLabel.ForeColor = [System.Drawing.Color]::DimGray
    }
}

function Write-RoeLine {
    param([string]$Line)
    if (-not (Test-RoeSerialOpen)) { return }
    $script:RoeSerial.WriteLine($Line)
}

function Test-RoeSerialOpen {
    return ($null -ne $script:RoeSerial) -and $script:RoeSerial.IsOpen
}

function Read-RoeLines {
    if (-not (Test-RoeSerialOpen)) { return @() }

    $lines = @()
    try {
        $chunk = $script:RoeSerial.ReadExisting()
        if ([string]::IsNullOrEmpty($chunk)) { return @() }

        $script:RoeLineBuffer += $chunk
        $parts = $script:RoeLineBuffer -split "`r?`n", -1
        if ($parts.Count -gt 1) {
            for ($i = 0; $i -lt ($parts.Count - 1); $i++) {
                $trimmed = $parts[$i].Trim()
                if ($trimmed) { $lines += $trimmed }
            }
            $script:RoeLineBuffer = $parts[$parts.Count - 1]
        }
    }
    catch {
        Add-Log ("!! ROE read failed: {0}" -f $_.Exception.Message)
        Stop-RoeConnection -Quiet
    }
    return $lines
}

function Test-RoePortIdentity {
    param(
        [string]$PortName,
        [int]$TimeoutMs = 2000
    )

    $port = New-Object System.IO.Ports.SerialPort $PortName, $RoeBaudRate, 'None', 8, 'One'
    $port.NewLine = "`n"
    $port.ReadTimeout = 200
    $port.WriteTimeout = 500
    $port.DtrEnable = $true
    $port.RtsEnable = $true

    try {
        $port.Open()
        Start-Sleep -Milliseconds 150
        $port.WriteLine('PING')

        $buffer = ''
        $deadline = [datetime]::UtcNow.AddMilliseconds($TimeoutMs)
        while ([datetime]::UtcNow -lt $deadline) {
            try {
                $buffer += $port.ReadExisting()
            }
            catch { }
            if ($buffer -match 'ROE_NANO' -and $buffer -match 'PONG|OK READY|roe_encoder') {
                return $true
            }
            if ($buffer -match 'PONG') {
                return $true
            }
            Start-Sleep -Milliseconds 40
        }
        return ($buffer -match 'ROE_NANO')
    }
    catch {
        return $false
    }
    finally {
        if ($port.IsOpen) { $port.Close() }
        $port.Dispose()
    }
}

function Find-RoeEncoderPort {
    $knownPorts = [System.IO.Ports.SerialPort]::GetPortNames()
    if (-not [string]::IsNullOrWhiteSpace($script:RoeComPort)) {
        if ($script:RoeComPort -in $knownPorts) {
            return $script:RoeComPort
        }
        Add-Log ("!! Saved RoeComPort {0} is not present" -f $script:RoeComPort)
    }

    foreach ($portName in $knownPorts | Sort-Object) {
        if (Test-RoePortIdentity -PortName $portName) {
            return $portName
        }
    }
    return $null
}

function Wait-RoeBootBanner {
    param([int]$TimeoutMs = 3000)

    $deadline = [datetime]::UtcNow.AddMilliseconds($TimeoutMs)
    $sawBanner = $false
    while ([datetime]::UtcNow -lt $deadline) {
        foreach ($line in (Read-RoeLines)) {
            Process-RoeSerialLine $line
            if ($line -match 'ROE_NANO') { $sawBanner = $true }
            if ($sawBanner -and $line -match '^OK READY') { return $true }
        }
        Start-Sleep -Milliseconds 30
    }
    return $sawBanner
}

function Confirm-RoeHostMode {
    param([int]$TimeoutMs = 2000)

    $script:RoeHostModeConfirmed = $false
    Write-RoeLine 'HOST ON'
    $deadline = [datetime]::UtcNow.AddMilliseconds($TimeoutMs)
    while ([datetime]::UtcNow -lt $deadline) {
        foreach ($line in (Read-RoeLines)) {
            Process-RoeSerialLine $line
            if ($line -match '^OK HOST ON') {
                $script:RoeHostModeConfirmed = $true
                Add-Log 'ROE << OK HOST ON'
                return $true
            }
            if ($line -match '^ERR') {
                Add-Log ("ROE << {0}" -f $line)
            }
        }
        Start-Sleep -Milliseconds 30
    }
    return $false
}

function Test-SettingsBool {
    param([string]$Text)
    return ($Text -match '^(?i)(1|true|yes|on)$')
}

function Get-SettingsBool {
    param(
        [hashtable]$Settings,
        [string]$Key,
        [string]$LegacyKey = ''
    )

    if ($Settings.ContainsKey($Key)) {
        return (Test-SettingsBool ([string]$Settings[$Key]))
    }
    if ($LegacyKey -and $Settings.ContainsKey($LegacyKey)) {
        return (Test-SettingsBool ([string]$Settings[$LegacyKey]))
    }
    return $false
}

function Apply-MotorAxisMapping {
    param(
        [double]$DeltaX = 0,
        [double]$DeltaY = 0,
        [double]$DeltaZ = 0
    )

    $dx = $DeltaX
    $dy = $DeltaY
    $dz = $DeltaZ
    if ($script:MotorInvertX) { $dx = -$dx }
    if ($script:MotorInvertY) { $dy = -$dy }
    if ($script:MotorInvertZ) { $dz = -$dz }
    if ($script:MotorSwapXY) {
        $swap = $dx
        $dx = $dy
        $dy = $swap
    }
    return @{ Dx = $dx; Dy = $dy; Dz = $dz }
}

function Get-RoeEncoderTickSign {
    param([int]$Direction)
    if ($Direction -eq 0) { return 0 }
    if ($script:RoeClockwisePositive) { return -$Direction }
    return $Direction
}

function Register-RoeEncTick {
    param(
        [string]$Axis,
        [int]$Direction
    )

    if ($Direction -eq 0) { return }
    # ENC value from Arduino is a direction hint only (+/-). Step size comes from the active GUI preset.
    $detentSign = if ($Direction -gt 0) { 1 } else { -1 }
    $signed = Get-RoeEncoderTickSign -Direction $detentSign
    if ($signed -eq 0) { return }
    Register-MotorActivity
    if (-not $script:RoeEncTicks.ContainsKey($Axis)) {
        $script:RoeEncTicks[$Axis] = 0
    }
    $script:RoeEncTicks[$Axis] += $signed
}

function Build-RoeMoveDeltasFromAccumulatedTicks {
    $dx = 0.0
    $dy = 0.0
    $dz = 0.0
    $steps = Get-ActiveKnobStepValues

    foreach ($axis in @($script:RoeEncTicks.Keys)) {
        $detents = [double]$script:RoeEncTicks[$axis]
        if ($detents -eq 0) { continue }
        switch ($axis) {
            'X' { $dx += $detents * $steps.StepXY }
            'Y' { $dy += $detents * $steps.StepXY }
            'Z' { $dz += $detents * $steps.StepZ }
        }
    }

    $script:RoeEncTicks = @{}
    return (Apply-MotorAxisMapping -DeltaX $dx -DeltaY $dy -DeltaZ $dz)
}

function Invoke-RoeMotorMove {
    param(
        [double]$DeltaX,
        [double]$DeltaY,
        [double]$DeltaZ
    )

    if ($DeltaX -eq 0 -and $DeltaY -eq 0 -and $DeltaZ -eq 0) { return }
    if (-not $script:PipeConnected) {
        $now = [datetime]::UtcNow
        if ($null -eq $script:RoeLastPipeWarnUtc -or ($now - $script:RoeLastPipeWarnUtc).TotalSeconds -ge 5) {
            $script:RoeLastPipeWarnUtc = $now
            Add-Log '!! ROE encoder: connect FLIMage (pipe) to move stage'
        }
        return
    }

    if ($script:RoeMoveInFlight) {
        if ($null -eq $script:RoePendingMove) {
            $script:RoePendingMove = @{ Dx = 0.0; Dy = 0.0; Dz = 0.0 }
        }
        $script:RoePendingMove.Dx += $DeltaX
        $script:RoePendingMove.Dy += $DeltaY
        $script:RoePendingMove.Dz += $DeltaZ
        return
    }

    $script:RoeMoveInFlight = $true
    Register-MotorActivity
    try {
        Invoke-MotorJogQuick -DeltaX $DeltaX -DeltaY $DeltaY -DeltaZ $DeltaZ -Quiet -FromRoe
    }
    finally {
        $script:RoeMoveInFlight = $false
        if ($null -ne $script:RoePendingMove) {
            $pending = $script:RoePendingMove
            $script:RoePendingMove = $null
            Invoke-RoeMotorMove -DeltaX $pending.Dx -DeltaY $pending.Dy -DeltaZ $pending.Dz
        }
        else {
            Request-MotorPosRefreshAfterIdle
        }
    }
}

function Flush-RoeEncoderMoves {
    if ($script:RoeEncTicks.Count -eq 0) { return }
    $deltas = Build-RoeMoveDeltasFromAccumulatedTicks
    Invoke-RoeMotorMove -DeltaX $deltas.Dx -DeltaY $deltas.Dy -DeltaZ $deltas.Dz
}

function Process-RoeSerialLine {
    param([string]$Line)

    if ($Line -match '^ENC\s+([XYZ])=(-?\d+)$') {
        Register-RoeEncTick -Axis $Matches[1] -Direction ([int]$Matches[2])
        return
    }

    if ($Line -notmatch '^(PONG|OK|STAT|HOST)') {
        Add-Log ("ROE << {0}" -f $Line)
    }
}

function Invoke-RoePollTick {
    if (-not $script:RoeConnected) { return }

    do {
        $lines = Read-RoeLines
        if ($lines.Count -eq 0) { break }
        foreach ($line in $lines) {
            Process-RoeSerialLine $line
        }
    } while ($true)

    Flush-RoeEncoderMoves
    Invoke-AbsPositionRefreshIfIdle
}

function Register-MotorActivity {
    $script:LastMotorActivityUtc = [datetime]::UtcNow
    $script:MotorPosRefreshPending = $false
    if ($null -ne $script:MotorPosRefreshTimer) {
        $script:MotorPosRefreshTimer.Stop()
    }
}

function Request-MotorPosRefreshAfterIdle {
    $script:MotorPosRefreshPending = $true
    $script:LastMotorActivityUtc = [datetime]::UtcNow
    if ($null -ne $script:MotorPosRefreshTimer) {
        $script:MotorPosRefreshTimer.Start()
    }
}

function Show-AxisMappingDialog {
    Stop-AllHeldJogKeys

    $mappingSnapshot = @{
        MotorInvertX          = $script:MotorInvertX
        MotorInvertY          = $script:MotorInvertY
        MotorInvertZ          = $script:MotorInvertZ
        MotorSwapXY           = $script:MotorSwapXY
        RoeClockwisePositive  = $script:RoeClockwisePositive
    }

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = 'Axis mapping'
    $dlg.FormBorderStyle = 'FixedDialog'
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.StartPosition = 'CenterParent'
    $dlg.ClientSize = New-Object System.Drawing.Size 340, 300
    $dlg.KeyPreview = $true

    $stageGroup = New-Object System.Windows.Forms.GroupBox
    $stageGroup.Text = 'Stage / keyboard jog (shared)'
    $stageGroup.Location = New-Object System.Drawing.Point 12, 12
    $stageGroup.Size = New-Object System.Drawing.Size 316, 130
    $dlg.Controls.Add($stageGroup)

    $chkInvertX = New-Object System.Windows.Forms.CheckBox
    $chkInvertX.Text = 'Invert X'
    $chkInvertX.Location = New-Object System.Drawing.Point 16, 24
    $chkInvertX.AutoSize = $true
    $chkInvertX.Checked = $script:MotorInvertX
    $stageGroup.Controls.Add($chkInvertX)

    $chkInvertY = New-Object System.Windows.Forms.CheckBox
    $chkInvertY.Text = 'Invert Y'
    $chkInvertY.Location = New-Object System.Drawing.Point 16, 48
    $chkInvertY.AutoSize = $true
    $chkInvertY.Checked = $script:MotorInvertY
    $stageGroup.Controls.Add($chkInvertY)

    $chkInvertZ = New-Object System.Windows.Forms.CheckBox
    $chkInvertZ.Text = 'Invert Z'
    $chkInvertZ.Location = New-Object System.Drawing.Point 16, 72
    $chkInvertZ.AutoSize = $true
    $chkInvertZ.Checked = $script:MotorInvertZ
    $stageGroup.Controls.Add($chkInvertZ)

    $chkSwapXy = New-Object System.Windows.Forms.CheckBox
    $chkSwapXy.Text = 'Swap X and Y'
    $chkSwapXy.Location = New-Object System.Drawing.Point 16, 96
    $chkSwapXy.AutoSize = $true
    $chkSwapXy.Checked = $script:MotorSwapXY
    $stageGroup.Controls.Add($chkSwapXy)

    $roeGroup = New-Object System.Windows.Forms.GroupBox
    $roeGroup.Text = 'ROE encoder knob sense'
    $roeGroup.Location = New-Object System.Drawing.Point 12, 150
    $roeGroup.Size = New-Object System.Drawing.Size 316, 88
    $dlg.Controls.Add($roeGroup)

    $radioAnticlockwise = New-Object System.Windows.Forms.RadioButton
    $radioAnticlockwise.Text = 'Anticlockwise = positive (default)'
    $radioAnticlockwise.Location = New-Object System.Drawing.Point 16, 24
    $radioAnticlockwise.AutoSize = $true
    $radioAnticlockwise.Checked = -not $script:RoeClockwisePositive
    $roeGroup.Controls.Add($radioAnticlockwise)

    $radioClockwise = New-Object System.Windows.Forms.RadioButton
    $radioClockwise.Text = 'Clockwise = positive'
    $radioClockwise.Location = New-Object System.Drawing.Point 16, 48
    $radioClockwise.AutoSize = $true
    $radioClockwise.Checked = $script:RoeClockwisePositive
    $roeGroup.Controls.Add($radioClockwise)

    $script:SuppressAxisMappingLiveUpdate = $false
    $applyLiveAxisMapping = {
            if ($script:SuppressAxisMappingLiveUpdate) { return }
            $script:MotorInvertX = $chkInvertX.Checked
            $script:MotorInvertY = $chkInvertY.Checked
            $script:MotorInvertZ = $chkInvertZ.Checked
            $script:MotorSwapXY = $chkSwapXy.Checked
            $script:RoeClockwisePositive = $radioClockwise.Checked
        }

    foreach ($chk in @($chkInvertX, $chkInvertY, $chkInvertZ, $chkSwapXy)) {
        $chk.Add_CheckedChanged($applyLiveAxisMapping)
    }
    foreach ($radio in @($radioAnticlockwise, $radioClockwise)) {
        $radio.Add_CheckedChanged({
                param($sender, $eventArgs)
                if ($script:SuppressAxisMappingLiveUpdate) { return }
                if (-not $sender.Checked) { return }
                & $applyLiveAxisMapping
            })
    }

    $note = New-Object System.Windows.Forms.Label
    $note.Text = 'Stage mapping applies to numpad/cursor jog and ROE. Knob sense is ROE-only.'
    $note.Location = New-Object System.Drawing.Point 12, 244
    $note.Size = New-Object System.Drawing.Size 316, 28
    $note.ForeColor = [System.Drawing.Color]::DimGray
    $dlg.Controls.Add($note)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = 'OK'
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $btnOk.Location = New-Object System.Drawing.Point 172, 268
    $btnOk.Width = 72
    $dlg.Controls.Add($btnOk)
    $dlg.AcceptButton = $btnOk

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = 'Cancel'
    $btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $btnCancel.Location = New-Object System.Drawing.Point 252, 268
    $btnCancel.Width = 72
    $dlg.Controls.Add($btnCancel)
    $dlg.CancelButton = $btnCancel

    $result = $dlg.ShowDialog($form)

    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
        & $applyLiveAxisMapping
        Export-GuiSettings
        Add-Log ("Axis mapping: invX={0} invY={1} invZ={2} swapXY={3} ROE={4}" -f `
                $script:MotorInvertX, $script:MotorInvertY, $script:MotorInvertZ, $script:MotorSwapXY, `
                ($(if ($script:RoeClockwisePositive) { 'clockwise+' } else { 'anticlockwise+' })))
    }
    else {
        $script:MotorInvertX = $mappingSnapshot.MotorInvertX
        $script:MotorInvertY = $mappingSnapshot.MotorInvertY
        $script:MotorInvertZ = $mappingSnapshot.MotorInvertZ
        $script:MotorSwapXY = $mappingSnapshot.MotorSwapXY
        $script:RoeClockwisePositive = $mappingSnapshot.RoeClockwisePositive
    }
}

function Start-RoeConnection {
    if ($script:RoeConnected) { return $true }
    if ($script:Busy) {
        Add-Log '!! Busy, cannot connect ROE'
        return $false
    }

    Set-Busy $true
    try {
        $portName = Find-RoeEncoderPort
        if (-not $portName) {
            Add-Log '!! ROE Nano not found on any COM port'
            return $false
        }

        $port = New-Object System.IO.Ports.SerialPort $portName, $RoeBaudRate, 'None', 8, 'One'
        $port.NewLine = "`n"
        $port.ReadTimeout = 50
        $port.WriteTimeout = 500
        $port.DtrEnable = $true
        $port.RtsEnable = $true
        $script:RoeSerial = $port
        $script:RoeComPort = $portName
        $script:RoeLineBuffer = ''
        $script:RoeHostModeConfirmed = $false

        $port.Open()
        Start-Sleep -Milliseconds 150

        if (-not (Wait-RoeBootBanner)) {
            Add-Log '!! ROE boot banner not seen within 3 s (check USB / firmware)'
        }

        if (-not (Confirm-RoeHostMode)) {
            Add-Log '!! ROE HOST ON not acknowledged  -  knob ENC lines will not be sent'
            Stop-RoeConnection -Quiet
            return $false
        }

        $script:RoeConnected = $true

        if ($null -ne $roePingTimer) {
            $roePingTimer.Interval = $script:RoeHostPingMs
            $roePingTimer.Start()
        }
        if ($null -ne $roePollTimer) { $roePollTimer.Start() }

        Update-RoeStatusLabel
        Export-GuiSettings
        Add-Log ("ROE connected on {0} (host mode; Zaber paused)" -f $portName)
        return $true
    }
    catch {
        Add-Log ("!! ROE connect failed: {0}" -f $_.Exception.Message)
        Stop-RoeConnection -Quiet
        return $false
    }
    finally {
        Set-Busy $false
    }
}

function Stop-RoeConnection {
    param([switch]$Quiet)

    if ($null -ne $roePingTimer) { $roePingTimer.Stop() }
    if ($null -ne $roePollTimer) { $roePollTimer.Stop() }

    if (Test-RoeSerialOpen) {
        try {
            if ($script:RoeHostModeConfirmed) {
                Write-RoeLine 'HOST OFF'
                Start-Sleep -Milliseconds 60
            }
        }
        catch { }
    }

    if ($null -ne $script:RoeSerial) {
        try {
            if ($script:RoeSerial.IsOpen) { $script:RoeSerial.Close() }
            $script:RoeSerial.Dispose()
        }
        catch { }
        $script:RoeSerial = $null
    }

    $wasConnected = $script:RoeConnected
    $script:RoeConnected = $false
    $script:RoeEncTicks = @{}
    $script:RoePendingMove = $null
    $script:RoeMoveInFlight = $false
    Update-RoeStatusLabel

    if ($wasConnected -and -not $Quiet) {
        Add-Log 'ROE disconnected (Arduino returns to Zaber mode)'
    }
}

function Format-TurretComboText {
    param(
        [int]$Position,
        [hashtable]$Labels
    )
    $label = if ($Labels.ContainsKey($Position)) { $Labels[$Position] } else { Get-DefaultTurretLabel $Position }
    if ([string]::IsNullOrWhiteSpace($label)) { $label = Get-DefaultTurretLabel $Position }
    return ("{0}: {1}" -f $Position, $label)
}

function Format-TurretCurrentText {
    param([int]$Position)
    return ("Cur: {0}" -f $Position)
}

function Update-TurretComboItems {
    param(
        [System.Windows.Forms.ComboBox]$Combo,
        [hashtable]$Labels
    )
    $selectedPos = Get-ComboTurretPosition $Combo
    $Combo.Items.Clear()
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        [void]$Combo.Items.Add((Format-TurretComboText -Position $p -Labels $Labels))
    }
    Sync-TurretComboToPosition -Combo $Combo -Position $selectedPos
}

function Sync-TurretComboToPosition {
    param(
        [System.Windows.Forms.ComboBox]$Combo,
        [int]$Position
    )
    $idx = $Position - $TurretMin
    if ($idx -ge 0 -and $idx -lt $Combo.Items.Count) {
        $Combo.SelectedIndex = $idx
    }
}

function Get-ComboTurretPosition {
    param([System.Windows.Forms.ComboBox]$Combo)
    if ($Combo.SelectedIndex -lt 0) { return $TurretMin }
    return $TurretMin + $Combo.SelectedIndex
}

function Initialize-JogKeyState {
    $script:JogKeysDown = @{}
    $script:PresetKeysDown = @{}
    foreach ($name in $script:JogActionNames) {
        $script:JogKeysDown[$name] = $false
    }
    foreach ($name in $script:PresetActionNames) {
        $script:PresetKeysDown[$name] = $false
    }
}

function Sync-GlobalHookKeyLayout {
    if ($null -ne $script:GlobalNumpadHook) {
        $script:GlobalNumpadHook.KeyLayout = $script:JogKeyLayout
    }
}

function Set-JogKeyLayout {
    param([string]$Layout)

    if ($JogKeyLayouts -notcontains $Layout) { return }
    if ($script:JogKeyLayout -eq $Layout) { return }

    Stop-AllHeldJogKeys
    $script:JogKeyLayout = $Layout
    Sync-GlobalHookKeyLayout
    Update-JogKeyLayoutUi
    Export-GuiSettings
    if ($null -ne $logBox) {
        Add-Log ("Jog key layout: {0}" -f $script:JogKeyLayout)
    }
}

function Update-JogKeyLayoutUi {
    if ($null -ne $keyLayoutCombo) {
        $idx = [array]::IndexOf($JogKeyLayouts, $script:JogKeyLayout)
        if ($idx -ge 0) {
            $script:SuppressKeyLayoutComboEvent = $true
            $keyLayoutCombo.SelectedIndex = $idx
            $script:SuppressKeyLayoutComboEvent = $false
        }
    }
    if ($null -ne $numpadHintLabel) {
        $numpadHintLabel.Text = "Capture ON = jog keys work when another window is focused"
    }
}

function Get-JogActionFromKeyCode {
    param([System.Windows.Forms.Keys]$KeyCode)

    if ($script:JogKeyLayout -eq 'Cursor') {
        switch ($KeyCode) {
            ([System.Windows.Forms.Keys]::Left) { return 'XMinus' }
            ([System.Windows.Forms.Keys]::Right) { return 'XPlus' }
            ([System.Windows.Forms.Keys]::Down) { return 'YMinus' }
            ([System.Windows.Forms.Keys]::Up) { return 'YPlus' }
            ([System.Windows.Forms.Keys]::OemOpenBrackets) { return 'ZMinus' }
            ([System.Windows.Forms.Keys]::OemCloseBrackets) { return 'ZPlus' }
        }
        return $null
    }

    switch ($KeyCode) {
        ([System.Windows.Forms.Keys]::NumPad4) { return 'XMinus' }
        ([System.Windows.Forms.Keys]::NumPad6) { return 'XPlus' }
        ([System.Windows.Forms.Keys]::NumPad2) { return 'YMinus' }
        ([System.Windows.Forms.Keys]::NumPad8) { return 'YPlus' }
        ([System.Windows.Forms.Keys]::Add) { return 'ZPlus' }
        ([System.Windows.Forms.Keys]::Subtract) { return 'ZMinus' }
        default { return $null }
    }
}

function Get-PresetActionFromKeyCode {
    param([System.Windows.Forms.Keys]$KeyCode)

    if ($script:JogKeyLayout -eq 'Cursor') {
        switch ($KeyCode) {
            ([System.Windows.Forms.Keys]::Oemcomma) { return 'PresetFiner' }
            ([System.Windows.Forms.Keys]::OemPeriod) { return 'PresetCoarser' }
        }
        return $null
    }

    switch ($KeyCode) {
        ([System.Windows.Forms.Keys]::Divide) { return 'PresetFiner' }
        ([System.Windows.Forms.Keys]::Multiply) { return 'PresetCoarser' }
        default { return $null }
    }
}

function Get-JogActionFromKeyEvent {
    param([System.Windows.Forms.KeyEventArgs]$EventArgs)
    return Get-JogActionFromKeyCode -KeyCode $EventArgs.KeyCode
}

function Get-PresetActionFromKeyEvent {
    param([System.Windows.Forms.KeyEventArgs]$EventArgs)
    return Get-PresetActionFromKeyCode -KeyCode $EventArgs.KeyCode
}

function Get-KeyMapHelpRows {
    if ($script:JogKeyLayout -eq 'Cursor') {
        return @{
            HoldRows = @(
                @('Left', 'X minus'),
                @('Right', 'X plus'),
                @('Down', 'Y minus'),
                @('Up', 'Y plus'),
                @('[', 'Z minus'),
                @(']', 'Z plus')
            )
            TapRows = @(
                @(',', 'Finer step tier (1-5)'),
                @('.', 'Coarser step tier (1-5)')
            )
            Notes = "Capture ON: keys work when another window is focused.`r`nWhen keyboard/ROE motion stops: Abs XYZ updates immediately.`r`nStep tiers 1-5: keys use Key XY/Z; ROE knob uses Knob XY/Z.`r`nASI / ZoZoLab tiers = jog speed (mm/s).`r`nOlympus: Light path (1PRISM) in GUI  -  1=eyepiece, 2=camera."
        }
    }

    return @{
        HoldRows = @(
            @('NumPad 4', 'X minus'),
            @('NumPad 6', 'X plus'),
            @('NumPad 2', 'Y minus'),
            @('NumPad 8', 'Y plus'),
            @('NumPad +', 'Z plus'),
            @('NumPad -', 'Z minus')
        )
        TapRows = @(
            @('NumPad /', 'Finer step tier (1-5)'),
            @('NumPad *', 'Coarser step tier (1-5)')
        )
        Notes = "Numpad ON: keys work when another window is focused.`r`nWhen keyboard/ROE motion stops: Abs XYZ updates immediately.`r`nStep tiers 1-5: keys use Key XY/Z; ROE knob uses Knob XY/Z.`r`nASI / ZoZoLab tiers = jog speed (mm/s).`r`nOlympus: Light path (1PRISM) in GUI  -  1=eyepiece, 2=camera."
    }
}

function Test-JogKeyInputAllowed {
    if (-not (Test-CanAcceptNumpadJog)) { return $false }
    $focused = $form.ActiveControl
    if ($null -ne $cmdBox -and $focused -eq $cmdBox) { return $false }
    if ($null -ne $focused -and $focused -is [System.Windows.Forms.TextBoxBase]) { return $false }
    return $true
}

function Get-NumpadJogKeyName {
    param([System.Windows.Forms.KeyEventArgs]$EventArgs)
    return Get-JogActionFromKeyEvent -EventArgs $EventArgs
}

function Get-NumpadPresetKeyName {
    param([System.Windows.Forms.KeyEventArgs]$EventArgs)
    return Get-PresetActionFromKeyEvent -EventArgs $EventArgs
}

function Test-CanAcceptNumpadJog {
    if (-not $script:NumpadJogEnabled) { return $false }
    if ($script:PresetDialogOpen) { return $false }
    if ($script:Busy) { return $false }
    return $true
}

function Stop-AllHeldJogKeys {
    foreach ($name in $script:JogActionNames) {
        $script:JogKeysDown[$name] = $false
    }
    foreach ($name in $script:PresetActionNames) {
        $script:PresetKeysDown[$name] = $false
    }
    if ($null -ne $jogRepeatTimer) { $jogRepeatTimer.Stop() }
    if ($null -ne $script:MotorPosRefreshTimer) { $script:MotorPosRefreshTimer.Stop() }
    if ($script:MotorUsesVectorJog) {
        Invoke-MotorVectorStop -Quiet
    }
    Request-MotorPosRefreshAfterIdle
}

function Test-MotorMotionIdle {
    if (Test-AnyJogKeyDown) { return $false }
    if ($script:RoeMoveInFlight) { return $false }
    if ($script:RoeEncTicks.Count -gt 0) { return $false }
    if ($null -ne $jogRepeatTimer -and $jogRepeatTimer.Enabled -and (Test-AnyJogKeyDown)) {
        return $false
    }
    return $true
}

function Invoke-AbsPositionRefreshQuiet {
    if ($script:Busy) { return }
    if (-not $script:PipeConnected) { return }
    Sync-MotorPosFromFlimage -Quiet
}

function Invoke-AbsPositionRefreshIfIdle {
    if (-not $script:MotorPosRefreshPending) { return }
    if (-not (Test-MotorMotionIdle)) { return }
    $quietMs = [int]$script:MotorPosRefreshQuietMs
    if ($quietMs -lt 1) { $quietMs = 500 }
    $elapsedMs = ([datetime]::UtcNow - $script:LastMotorActivityUtc).TotalMilliseconds
    if ($elapsedMs -lt $quietMs) { return }

    $script:MotorPosRefreshPending = $false
    if ($null -ne $script:MotorPosRefreshTimer) {
        $script:MotorPosRefreshTimer.Stop()
    }
    Invoke-AbsPositionRefreshQuiet
}

function Convert-ToNumpadIsDown {
    param([object]$Value)
    if ($Value -is [bool]) { return $Value }
    if ($null -eq $Value) { return $false }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [double]) {
        return ($Value -ne 0)
    }
    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) { return $false }
    return [System.Convert]::ToBoolean($text)
}

function Invoke-NumpadKeyEvent {
    param(
        [string]$KeyName,
        [object]$IsDown
    )

    $isDownEvent = Convert-ToNumpadIsDown $IsDown
    if (-not (Test-JogKeyInputAllowed)) { return }

    if ($KeyName -eq 'PresetFiner' -or $KeyName -eq 'PresetCoarser') {
        if ($isDownEvent) {
            if ($script:PresetKeysDown[$KeyName]) { return }
            $script:PresetKeysDown[$KeyName] = $true
            $direction = if ($KeyName -eq 'PresetFiner') { -1 } else { 1 }
            [void](Switch-StepPreset -Direction $direction)
        }
        else {
            $script:PresetKeysDown[$KeyName] = $false
            Request-MotorPosRefreshAfterIdle
        }
        return
    }

    if ($isDownEvent) {
        if ($script:JogKeysDown[$KeyName]) { return }
        $script:JogKeysDown[$KeyName] = $true
        Register-MotorActivity
        Start-HeldJogIfNeeded
    }
    else {
        $script:JogKeysDown[$KeyName] = $false
        if ($script:MotorUsesVectorJog) {
            if (Test-AnyJogKeyDown) {
                Start-HeldJogIfNeeded
            }
            else {
                Invoke-MotorVectorStop -Quiet
            }
        }
        else {
            Stop-HeldJogIfIdle
        }
        Request-MotorPosRefreshAfterIdle
    }
}

function Start-GlobalNumpadHook {
    if ($null -eq $script:GlobalNumpadHook) {
        $script:GlobalNumpadHook = New-Object GlobalNumpadHook
        $script:GlobalNumpadHook.ShouldCaptureKey = [Func[bool]] { Test-JogKeyInputAllowed }
        $script:GlobalNumpadHook.add_NumpadEvent({
                param([string]$keyName, [int]$isDownFlag)
                if ($form.IsDisposed) { return }
                if ([string]::IsNullOrWhiteSpace($keyName)) { return }
                $kCopy = [string]$keyName
                $dCopy = [int]$isDownFlag
                $form.BeginInvoke([System.Action]{
                        Invoke-NumpadKeyEvent -KeyName $kCopy -IsDown $dCopy
                    }.GetNewClosure())
            })
    }
    if ($null -ne $form -and -not $form.IsDisposed) {
        $script:GlobalNumpadHook.GuiWindowHandle = $form.Handle
    }
    Sync-GlobalHookKeyLayout
    $script:GlobalNumpadHook.Start()
}

function Stop-GlobalNumpadHook {
    if ($null -ne $script:GlobalNumpadHook) {
        $script:GlobalNumpadHook.Stop()
    }
}

function Sync-GlobalNumpadHookState {
    if ($script:NumpadJogEnabled) { Start-GlobalNumpadHook }
    else { Stop-GlobalNumpadHook }
}

function Switch-StepPreset {
    param([int]$Direction)

    $idx = [array]::IndexOf($PresetNames, $script:ActivePreset)
    if ($idx -lt 0) { return $false }
    $newIdx = $idx + $Direction
    if ($newIdx -lt 0 -or $newIdx -ge $PresetNames.Count) { return $false }

    $script:ActivePreset = $PresetNames[$newIdx]
    Update-StepPresetDisplay
    Export-GuiSettings
    Add-Log ("Step tier: {0}" -f $script:ActivePreset)
    if ($script:MotorUsesVectorJog -and (Test-AnyJogKeyDown)) {
        Start-HeldJogIfNeeded
    }
    return $true
}

$script:SavedGuiSettings = Import-GuiSettings
Apply-GuiSettings $script:SavedGuiSettings

$form = New-Object System.Windows.Forms.Form
$form.Text = "FLIMage Motor Pipe GUI"
$form.Size = New-Object System.Drawing.Size ($GuiInnerWidth + ($GuiMargin * 2) + 16), 918
$form.MinimumSize = New-Object System.Drawing.Size ($GuiInnerWidth + ($GuiMargin * 2) + 16), 878
$form.StartPosition = "CenterScreen"
$form.KeyPreview = $true

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.AutoSize = $false
$statusLabel.Location = New-Object System.Drawing.Point $GuiMargin, 10
$statusLabel.Size = New-Object System.Drawing.Size ($GuiInnerWidth - 170), 32
$statusLabel.Text = "Status: Disconnected"
$form.Controls.Add($statusLabel)

$btnConnect = New-Object System.Windows.Forms.Button
$btnConnect.Text = "Connect"
$btnConnect.Location = New-Object System.Drawing.Point ($GuiMargin + $GuiInnerWidth - 158), 8
$btnConnect.Width = 74
$form.Controls.Add($btnConnect)

$btnDisconnect = New-Object System.Windows.Forms.Button
$btnDisconnect.Text = "Disconnect"
$btnDisconnect.Location = New-Object System.Drawing.Point ($GuiMargin + $GuiInnerWidth - 78), 8
$btnDisconnect.Width = 74
$form.Controls.Add($btnDisconnect)

$roeGroup = New-Object System.Windows.Forms.GroupBox
$roeGroup.Text = "Sutter ROE (encoder)"
$roeGroup.Location = New-Object System.Drawing.Point $GuiMargin, 42
$roeGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 62
$form.Controls.Add($roeGroup)

$roeStatusLabel = New-Object System.Windows.Forms.Label
$roeStatusLabel.Text = "ROE: disconnected"
$roeStatusLabel.Location = New-Object System.Drawing.Point 8, 16
$roeStatusLabel.Size = New-Object System.Drawing.Size ($GuiInnerWidth - 16), 18
$roeStatusLabel.AutoEllipsis = $true
$roeStatusLabel.ForeColor = [System.Drawing.Color]::DimGray
$roeGroup.Controls.Add($roeStatusLabel)

$btnRoeDetails = New-Object System.Windows.Forms.Button
$btnRoeDetails.Text = "Details"
$btnRoeDetails.Location = New-Object System.Drawing.Point 8, 34
$btnRoeDetails.Width = 72
$roeGroup.Controls.Add($btnRoeDetails)

$btnConnectRoe = New-Object System.Windows.Forms.Button
$btnConnectRoe.Text = "Connect ROE"
$btnConnectRoe.Location = New-Object System.Drawing.Point 86, 34
$btnConnectRoe.Width = 100
$roeGroup.Controls.Add($btnConnectRoe)

$btnDisconnectRoe = New-Object System.Windows.Forms.Button
$btnDisconnectRoe.Text = "Disconnect"
$btnDisconnectRoe.Location = New-Object System.Drawing.Point 192, 34
$btnDisconnectRoe.Width = 82
$roeGroup.Controls.Add($btnDisconnectRoe)

$roePollTimer = New-Object System.Windows.Forms.Timer
$roePollTimer.Interval = 30
$roePollTimer.Add_Tick({ Invoke-RoePollTick })

$roePingTimer = New-Object System.Windows.Forms.Timer
$roePingTimer.Interval = 400
$roePingTimer.Add_Tick({
        if (-not $script:RoeConnected) { return }
        Write-RoeLine 'PING'
    })

$logGroup = New-Object System.Windows.Forms.GroupBox
$logGroup.Text = "Log"
$logGroup.Location = New-Object System.Drawing.Point $GuiMargin, 680
$logGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 180
$form.Controls.Add($logGroup)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ScrollBars = "Vertical"
$logBox.ReadOnly = $true
$logBox.WordWrap = $false
$logBox.Font = New-Object System.Drawing.Font "Consolas", 8
$logBox.Location = New-Object System.Drawing.Point 8, 18
$logBox.Size = New-Object System.Drawing.Size ($GuiInnerWidth - 20), 120
$logGroup.Controls.Add($logBox)

$btnClearLog = New-Object System.Windows.Forms.Button
$btnClearLog.Text = "Clear"
$btnClearLog.Location = New-Object System.Drawing.Point ($GuiInnerWidth - 68), 146
$btnClearLog.Width = 56
$logGroup.Controls.Add($btnClearLog)

function Add-Log {
    param([string]$Line)
    $logBox.AppendText("[$(Get-Timestamp)] $Line`r`n")
    $logBox.SelectionStart = $logBox.Text.Length
    $logBox.ScrollToCaret()
    [System.Windows.Forms.Application]::DoEvents()
}

function Set-Status {
    param([bool]$Connected, [string]$Detail = "")
    if ($Connected) { $text = "Status: Connected" }
    else { $text = "Status: Disconnected" }
    if ($Detail) { $text += " - $Detail" }
    $statusLabel.Text = $text
}

function Test-MotorHwUsesVectorJog {
    param([string]$MotorHwName)
    if ([string]::IsNullOrWhiteSpace($MotorHwName)) { return $false }
    if ($MotorHwName -match 'ASI' -and $MotorHwName -match 'MS2000') { return $true }
    if ($MotorHwName -match '(?i)zozolab') { return $true }
    return $false
}

function Set-MotorJogMode {
    param(
        [bool]$UsesVectorJog,
        [string]$MotorHwName = ""
    )
    $script:MotorUsesVectorJog = $UsesVectorJog
    Update-JogModeUi -MotorHwName $MotorHwName
}

function Update-JogModeUi {
    param([string]$MotorHwName = "")
    if ($null -ne $presetLabel) {
        if ($script:MotorUsesVectorJog) {
            $presetLabel.Text = "Speed:"
        }
        else {
            $presetLabel.Text = "Step:"
        }
    }
    Update-StepPresetDisplay
    if ($script:MotorUsesVectorJog -and $MotorHwName) {
        Add-Log ("Vector jog enabled ({0}); presets use mm/s" -f $MotorHwName)
    }
}

function Get-ActiveKeyStepValues {
    $preset = $script:Presets[$script:ActivePreset]
    return @{
        StepXY = [double]$preset.KeyStepXY
        StepZ  = [double]$preset.KeyStepZ
    }
}

function Get-ActiveKnobStepValues {
    $preset = $script:Presets[$script:ActivePreset]
    return @{
        StepXY = [double]$preset.KnobStepXY
        StepZ  = [double]$preset.KnobStepZ
    }
}

function Get-ActiveStepValues {
    return (Get-ActiveKeyStepValues)
}

function Get-ActiveJogSpeeds {
    $preset = $script:Presets[$script:ActivePreset]
    if ($script:MotorUsesVectorJog) {
        return @{
            SpeedXY = [double]$preset.VelXY
            SpeedZ  = [double]$preset.VelZ
        }
    }
    $steps = Get-ActiveStepValues
    return @{
        SpeedXY = [double]$steps.StepXY
        SpeedZ  = [double]$steps.StepZ
    }
}

function Update-StepPresetDisplay {
    $preset = $script:Presets[$script:ActivePreset]
    if ($script:MotorUsesVectorJog) {
        $stepDisplayLabel.Text = ("Tier {0}: XY={1:F3} Z={2:F3} mm/s" -f $script:ActivePreset, $preset.VelXY, $preset.VelZ)
    }
    else {
        $stepDisplayLabel.Text = ('Tier {0}: Key XY={1:F2} Z={2:F2} | Knob XY={3:F2} Z={4:F2} um' -f `
            $script:ActivePreset, $preset.KeyStepXY, $preset.KeyStepZ, $preset.KnobStepXY, $preset.KnobStepZ)
    }
    foreach ($name in $PresetNames) {
        if ($null -ne $presetRadioButtons[$name]) {
            $presetRadioButtons[$name].Checked = ($script:ActivePreset -eq $name)
        }
    }
}

function New-NumpadHelpKeyGrid {
    param(
        [object[]]$Rows,
        [int]$Height
    )

    $grid = New-Object System.Windows.Forms.DataGridView
    $grid.ReadOnly = $true
    $grid.AllowUserToAddRows = $false
    $grid.AllowUserToDeleteRows = $false
    $grid.AllowUserToResizeRows = $false
    $grid.AllowUserToResizeColumns = $false
    $grid.RowHeadersVisible = $false
    $grid.ColumnHeadersHeightSizeMode = 'DisableResizing'
    $grid.AutoSizeColumnsMode = 'Fill'
    $grid.AutoSizeRowsMode = 'AllCells'
    $grid.MultiSelect = $false
    $grid.TabStop = $false
    $grid.BorderStyle = 'FixedSingle'
    $grid.BackgroundColor = [System.Drawing.SystemColors]::Control
    $grid.GridColor = [System.Drawing.SystemColors]::ControlDark
    $grid.ColumnHeadersDefaultCellStyle.Font = New-Object System.Drawing.Font(
        $grid.Font,
        [System.Drawing.FontStyle]::Bold
    )
    $grid.DefaultCellStyle.SelectionBackColor = $grid.DefaultCellStyle.BackColor
    $grid.DefaultCellStyle.SelectionForeColor = $grid.DefaultCellStyle.ForeColor
    $grid.Height = $Height
    $grid.Width = 396

    [void]$grid.Columns.Add('Key', 'Key')
    [void]$grid.Columns.Add('Action', 'Action')
    $grid.Columns['Key'].FillWeight = 35
    $grid.Columns['Action'].FillWeight = 65

    foreach ($row in $Rows) {
        [void]$grid.Rows.Add($row[0], $row[1])
    }
    $grid.ClearSelection()

    return $grid
}

function Show-NumpadHelpDialog {
    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = "Key map ($($script:JogKeyLayout))"
    $dlg.FormBorderStyle = 'FixedDialog'
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.StartPosition = 'CenterParent'
    $dlg.ClientSize = New-Object System.Drawing.Size 420, 418
    $dlg.KeyPreview = $true

    $contentWidth = 396
    $left = 12
    $y = 12

    $holdCaption = New-Object System.Windows.Forms.Label
    $holdCaption.Text = "Hold (release key to stop jog)"
    $holdCaption.Location = New-Object System.Drawing.Point $left, $y
    $holdCaption.Size = New-Object System.Drawing.Size $contentWidth, 18
    $holdCaption.Font = New-Object System.Drawing.Font(
        $holdCaption.Font,
        [System.Drawing.FontStyle]::Bold
    )
    $dlg.Controls.Add($holdCaption)
    $y += 22

    $mapRows = Get-KeyMapHelpRows
    $holdGrid = New-NumpadHelpKeyGrid -Height 158 -Rows $mapRows.HoldRows
    $holdGrid.Location = New-Object System.Drawing.Point $left, $y
    $holdGrid.Width = $contentWidth
    $dlg.Controls.Add($holdGrid)
    $y += 166

    $tapCaption = New-Object System.Windows.Forms.Label
    $tapCaption.Text = "Tap once"
    $tapCaption.Location = New-Object System.Drawing.Point $left, $y
    $tapCaption.Size = New-Object System.Drawing.Size $contentWidth, 18
    $tapCaption.Font = New-Object System.Drawing.Font(
        $tapCaption.Font,
        [System.Drawing.FontStyle]::Bold
    )
    $dlg.Controls.Add($tapCaption)
    $y += 22

    $tapGrid = New-NumpadHelpKeyGrid -Height 78 -Rows $mapRows.TapRows
    $tapGrid.Location = New-Object System.Drawing.Point $left, $y
    $tapGrid.Width = $contentWidth
    $dlg.Controls.Add($tapGrid)
    $y += 86

    $notes = New-Object System.Windows.Forms.Label
    $notes.Text = $mapRows.Notes
    $notes.Location = New-Object System.Drawing.Point $left, $y
    $notes.Size = New-Object System.Drawing.Size $contentWidth, 68
    $dlg.Controls.Add($notes)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = "OK"
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $btnOk.Width = 76
    $btnOk.Height = 28
    $btnOk.Location = New-Object System.Drawing.Point 332, 380
    $dlg.Controls.Add($btnOk)
    $dlg.AcceptButton = $btnOk

    [void]$dlg.ShowDialog($form)
}

function Update-NumpadJogToggleUi {
    if ($script:NumpadJogEnabled) {
        $btnNumpadJogToggle.Text = "Numpad: ON"
        $btnNumpadJogToggle.BackColor = [System.Drawing.Color]::LightGreen
        $btnNumpadJogToggle.ForeColor = [System.Drawing.Color]::DarkGreen
    }
    else {
        $btnNumpadJogToggle.Text = "Numpad: OFF"
        $btnNumpadJogToggle.BackColor = [System.Drawing.Color]::MistyRose
        $btnNumpadJogToggle.ForeColor = [System.Drawing.Color]::DarkRed
        Stop-AllHeldJogKeys
    }
    $btnNumpadJogToggle.Font = New-Object System.Drawing.Font(
        $btnNumpadJogToggle.Font.FontFamily,
        8.5,
        [System.Drawing.FontStyle]::Bold
    )
    if ($form.IsHandleCreated) {
        Sync-GlobalNumpadHookState
    }
}

function Get-LightPathValueFromReply {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    foreach ($token in @('OlympusLightPath', 'OlympusLightPathDone', 'LightPath')) {
        $val = Get-TurretValue -Reply $Reply -Token $token
        if ($null -ne $val) { return $val }
    }
    $val = Get-OlympusIx2PositionValue -Reply $Reply -CommandToken '1PRISM'
    if ($null -ne $val) { return $val }
    return $null
}

function Format-LightPathCurrentText {
    param([int]$Position)
    switch ($Position) {
        1 { return 'Cur: Eyepiece' }
        2 { return 'Cur: Camera' }
        default { return ("Cur: {0}" -f $Position) }
    }
}

function Update-PrismLabel {
    param([string]$Reply)
    $pos = Get-LightPathValueFromReply -Reply $Reply
    if ($null -ne $pos) {
        $prismPosLabel.Text = Format-LightPathCurrentText -Position $pos
        if ($pos -eq 1) {
            $prismRadioEyepiece.Checked = $true
        }
        elseif ($pos -eq 2) {
            $prismRadioCamera.Checked = $true
        }
    }
    elseif ($Reply -match '(?i)^Error') {
        $prismPosLabel.Text = 'Cur: ERR'
    }
}

function Stop-PrismRefreshTimer {
    if ($null -ne $script:PrismRefreshTimer) {
        $script:PrismRefreshTimer.Stop()
    }
    $script:PrismRefreshAttempt = 0
}

function Start-PrismGetRetryTimer {
    param(
        [int]$MaxAttempts = 5,
        [int]$IntervalMs = 1000
    )

    if (-not $script:PipeConnected) { return }

    Stop-PrismRefreshTimer
    $script:PrismRefreshAttempt = 0
    if ($null -eq $script:PrismRefreshTimer) {
        $script:PrismRefreshTimer = New-Object System.Windows.Forms.Timer
        $script:PrismRefreshTimer.Add_Tick({
                if (-not $script:PipeConnected) {
                    Stop-PrismRefreshTimer
                    return
                }
                if ($script:Busy) { return }

                try {
                    $reply = Send-OlympusIx2WithLog -Ix2Command '1PRISM?'
                    Update-PrismLabel $reply
                    $pos = Get-LightPathValueFromReply -Reply $reply
                    if ($null -ne $pos -and $pos -ge 1 -and $pos -le 2) {
                        Stop-PrismRefreshTimer
                        return
                    }
                    if ($reply -notmatch '(?i)^Error') {
                        Stop-PrismRefreshTimer
                        return
                    }
                }
                catch {
                    Add-Log ("!! Prism refresh failed: {0}" -f $_.Exception.Message)
                }

                $script:PrismRefreshAttempt++
                if ($script:PrismRefreshAttempt -ge $script:PrismRefreshMaxAttempts) {
                    Stop-PrismRefreshTimer
                }
            })
    }

    $script:PrismRefreshMaxAttempts = $MaxAttempts
    $script:PrismRefreshTimer.Interval = $IntervalMs
    $script:PrismRefreshTimer.Start()
}

function Start-PrismLabelRefreshWithRetry {
    param(
        [string]$InitialReply = '',
        [switch]$VerifyAfterSet
    )

    Update-PrismLabel $InitialReply
    if ($InitialReply -match '(?i)^Error') {
        Start-PrismGetRetryTimer
        return
    }

    if ($VerifyAfterSet) {
        Start-PrismGetRetryTimer
        return
    }

    $pos = Get-LightPathValueFromReply -Reply $InitialReply
    if ($null -eq $pos -or $pos -lt 1 -or $pos -gt 2) {
        Start-PrismGetRetryTimer
    }
}

function Get-SelectedLightPathPosition {
    if ($prismRadioEyepiece.Checked) { return 1 }
    if ($prismRadioCamera.Checked) { return 2 }
    return 1
}

function Set-OlympusTurretControlsEnabled {
    param(
        [bool]$Enabled,
        [string]$MotorHwName = ""
    )

    $script:OlympusTurretsAvailable = $Enabled
    $filterGroup.Text = if ($Enabled) { "Filter (1MU)" } else { "Filter (1MU) - off" }
    $objGroup.Text = if ($Enabled) { "Objective (1OB)" } else { "Objective (1OB) - off" }
    $prismGroup.Text = if ($Enabled) { "Light path (1PRISM)" } else { "Light path (1PRISM) - off" }
    if ($null -ne $panelGroup) {
        $panelGroup.Text = if ($Enabled) { "Control mode" } else { "Control mode - off" }
    }

    $controls = @(
        $filterCombo, $btnFilterGet, $btnFilterSet, $btnFilterLabels,
        $objCombo, $btnObjGet, $btnObjSet, $btnObjLabels,
        $prismRadioEyepiece, $prismRadioCamera, $btnPrismGet, $btnPrismSet,
        $panelCombo, $btnPanelGet, $btnPanelSet
    )
    foreach ($ctrl in $controls) {
        if ($null -ne $ctrl) {
            $ctrl.Enabled = $Enabled -and (-not $script:Busy)
        }
    }
    if ($script:CollapsibleSections.Count -gt 0) {
        Update-CollapsibleLayout
    }
}

function Show-StepPresetTableEditor {
    $script:PresetDialogOpen = $true
    Stop-AllHeldJogKeys

    $isVector = $script:MotorUsesVectorJog
    $ci = [System.Globalization.CultureInfo]::InvariantCulture

    $dlg = New-Object System.Windows.Forms.Form
    if ($isVector) {
        $dlg.Text = 'Step tier speeds (ASI)'
        $clientWidth = 340
    }
    else {
        $dlg.Text = 'Step tier settings'
        $clientWidth = 520
    }
    $dlg.FormBorderStyle = 'FixedDialog'
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.StartPosition = 'CenterParent'
    $dlg.ClientSize = New-Object System.Drawing.Size $clientWidth, 280
    $dlg.KeyPreview = $true

    $grid = New-Object System.Windows.Forms.DataGridView
    $grid.Location = New-Object System.Drawing.Point 12, 12
    $grid.Size = New-Object System.Drawing.Size ($clientWidth - 24), 190
    $grid.AllowUserToAddRows = $false
    $grid.AllowUserToDeleteRows = $false
    $grid.AllowUserToResizeRows = $false
    $grid.RowHeadersVisible = $false
    $grid.ColumnHeadersHeightSizeMode = 'DisableResizing'
    $grid.AutoSizeColumnsMode = 'Fill'
    $grid.MultiSelect = $false
    $grid.BorderStyle = 'FixedSingle'
    $grid.ColumnHeadersDefaultCellStyle.Font = New-Object System.Drawing.Font(
        $grid.Font,
        [System.Drawing.FontStyle]::Bold
    )
    $dlg.Controls.Add($grid)

    [void]$grid.Columns.Add('Tier', 'Tier')
    $grid.Columns['Tier'].ReadOnly = $true
    $grid.Columns['Tier'].FillWeight = 12

    if ($isVector) {
        [void]$grid.Columns.Add('VelXY', 'XY speed (mm/s)')
        [void]$grid.Columns.Add('VelZ', 'Z speed (mm/s)')
        $grid.Columns['VelXY'].FillWeight = 44
        $grid.Columns['VelZ'].FillWeight = 44
    }
    else {
        [void]$grid.Columns.Add('KeyStepXY', 'Key XY (um)')
        [void]$grid.Columns.Add('KeyStepZ', 'Key Z (um)')
        [void]$grid.Columns.Add('KnobStepXY', 'Knob XY (um)')
        [void]$grid.Columns.Add('KnobStepZ', 'Knob Z (um)')
        foreach ($colName in @('KeyStepXY', 'KeyStepZ', 'KnobStepXY', 'KnobStepZ')) {
            $grid.Columns[$colName].FillWeight = 22
        }
    }

    foreach ($name in $PresetNames) {
        $preset = $script:Presets[$name]
        if ($isVector) {
            [void]$grid.Rows.Add(
                $name,
                $preset.VelXY.ToString('F3', $ci),
                $preset.VelZ.ToString('F3', $ci)
            )
        }
        else {
            [void]$grid.Rows.Add(
                $name,
                $preset.KeyStepXY.ToString('F2', $ci),
                $preset.KeyStepZ.ToString('F2', $ci),
                $preset.KnobStepXY.ToString('F2', $ci),
                $preset.KnobStepZ.ToString('F2', $ci)
            )
        }
    }

    $noteLabel = New-Object System.Windows.Forms.Label
    if ($isVector) {
        $noteLabel.Text = 'Edit all tiers at once. Numpad jog disabled while open.'
    }
    else {
        $noteLabel.Text = 'Key = keyboard jog; Knob = ROE encoder. Numpad jog disabled while open.'
    }
    $noteLabel.Location = New-Object System.Drawing.Point 12, 208
    $noteLabel.Size = New-Object System.Drawing.Size ($clientWidth - 24), 32
    $noteLabel.ForeColor = [System.Drawing.Color]::DimGray
    $dlg.Controls.Add($noteLabel)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = 'OK'
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $btnOk.Location = New-Object System.Drawing.Point ($clientWidth - 178), 244
    $dlg.Controls.Add($btnOk)
    $dlg.AcceptButton = $btnOk

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = 'Cancel'
    $btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $btnCancel.Location = New-Object System.Drawing.Point ($clientWidth - 97), 244
    $dlg.Controls.Add($btnCancel)
    $dlg.CancelButton = $btnCancel

    $result = $dlg.ShowDialog($form)
    $script:PresetDialogOpen = $false

    if ($result -ne [System.Windows.Forms.DialogResult]::OK) { return }

    $updated = @{}
    foreach ($name in $PresetNames) {
        $updated[$name] = @{}
        foreach ($key in $script:Presets[$name].Keys) {
            $updated[$name][$key] = $script:Presets[$name][$key]
        }
    }
    foreach ($row in $grid.Rows) {
        if ($row.IsNewRow) { continue }
        $tier = [string]$row.Cells['Tier'].Value
        if (-not ($PresetNames -contains $tier)) { continue }

        if ($isVector) {
            $velXY = 0.0
            $velZ = 0.0
            if (-not [double]::TryParse([string]$row.Cells['VelXY'].Value, [System.Globalization.NumberStyles]::Float, $ci, [ref]$velXY)) {
                [System.Windows.Forms.MessageBox]::Show($form, "Invalid XY speed for tier $tier.", 'Step tiers', 'OK', 'Warning') | Out-Null
                return
            }
            if (-not [double]::TryParse([string]$row.Cells['VelZ'].Value, [System.Globalization.NumberStyles]::Float, $ci, [ref]$velZ)) {
                [System.Windows.Forms.MessageBox]::Show($form, "Invalid Z speed for tier $tier.", 'Step tiers', 'OK', 'Warning') | Out-Null
                return
            }
            if ($velXY -lt 0.001 -or $velXY -gt 2.0 -or $velZ -lt 0.001 -or $velZ -gt 2.0) {
                [System.Windows.Forms.MessageBox]::Show($form, "Tier $tier speed must be 0.001-2.0 mm/s.", 'Step tiers', 'OK', 'Warning') | Out-Null
                return
            }
            $updated[$tier].VelXY = $velXY
            $updated[$tier].VelZ = $velZ
        }
        else {
            $keyXY = 0.0
            $keyZ = 0.0
            $knobXY = 0.0
            $knobZ = 0.0
            $fields = @(
                @{ Name = 'KeyStepXY'; Ref = [ref]$keyXY; Label = 'Key XY' }
                @{ Name = 'KeyStepZ'; Ref = [ref]$keyZ; Label = 'Key Z' }
                @{ Name = 'KnobStepXY'; Ref = [ref]$knobXY; Label = 'Knob XY' }
                @{ Name = 'KnobStepZ'; Ref = [ref]$knobZ; Label = 'Knob Z' }
            )
            foreach ($field in $fields) {
                $parsed = 0.0
                if (-not [double]::TryParse([string]$row.Cells[$field.Name].Value, [System.Globalization.NumberStyles]::Float, $ci, [ref]$parsed)) {
                    [System.Windows.Forms.MessageBox]::Show($form, "Invalid $($field.Label) for tier $tier.", 'Step tiers', 'OK', 'Warning') | Out-Null
                    return
                }
                if ($parsed -lt 0.01 -or $parsed -gt 1000) {
                    [System.Windows.Forms.MessageBox]::Show($form, "Tier $tier $($field.Label) must be 0.01-1000 um.", 'Step tiers', 'OK', 'Warning') | Out-Null
                    return
                }
                $field.Ref.Value = $parsed
            }
            $updated[$tier].KeyStepXY = $keyXY
            $updated[$tier].KeyStepZ = $keyZ
            $updated[$tier].KnobStepXY = $knobXY
            $updated[$tier].KnobStepZ = $knobZ
        }
    }

    foreach ($name in $PresetNames) {
        foreach ($key in $updated[$name].Keys) {
            $script:Presets[$name][$key] = $updated[$name][$key]
        }
    }

    Update-StepPresetDisplay
    Export-GuiSettings
    if ($isVector) {
        Add-Log 'Updated step tier speeds (ASI)'
        if (Test-AnyJogKeyDown) { Start-HeldJogIfNeeded }
    }
    else {
        Add-Log 'Updated step tier table (Key/Knob XY/Z)'
    }
}

function Show-TurretLabelsEditor {
    param(
        [string]$Kind,
        [hashtable]$Labels
    )

    $script:PresetDialogOpen = $true
    Stop-AllHeldJogKeys

    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = "Set $Kind labels"
    $dlg.FormBorderStyle = 'FixedDialog'
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.StartPosition = 'CenterParent'
    $dlg.ClientSize = New-Object System.Drawing.Size 300, 220
    $dlg.KeyPreview = $true

    $editBoxes = @{}
    $y = 12
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $lbl = New-Object System.Windows.Forms.Label
        $lbl.Text = ("Pos {0}:" -f $p)
        $lbl.Location = New-Object System.Drawing.Point 12, ($y + 4)
        $lbl.Width = 42
        $dlg.Controls.Add($lbl)

        $tb = New-Object System.Windows.Forms.TextBox
        $tb.Text = if ($Labels.ContainsKey($p)) { $Labels[$p] } else { Get-DefaultTurretLabel $p }
        $tb.Location = New-Object System.Drawing.Point 58, $y
        $tb.Width = 220
        $dlg.Controls.Add($tb)
        $editBoxes[$p] = $tb
        $y += 28
    }

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = "OK"
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $btnOk.Location = New-Object System.Drawing.Point 122, 186
    $dlg.Controls.Add($btnOk)
    $dlg.AcceptButton = $btnOk

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = "Cancel"
    $btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $btnCancel.Location = New-Object System.Drawing.Point 203, 186
    $dlg.Controls.Add($btnCancel)
    $dlg.CancelButton = $btnCancel

    $result = $dlg.ShowDialog($form)
    $script:PresetDialogOpen = $false

    if ($result -ne [System.Windows.Forms.DialogResult]::OK) { return $false }

    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $text = $editBoxes[$p].Text.Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            $Labels[$p] = Get-DefaultTurretLabel $p
        }
        else {
            $Labels[$p] = $text
        }
    }
    Export-GuiSettings
    Add-Log ("Updated {0} turret labels" -f $Kind)
    return $true
}

function Get-MotorVelocityFromReply {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    foreach ($token in @("MotorVelocityDone", "MotorVelocity")) {
        $pattern = "{0}\s*,\s*(-?\d+(?:\.\d+)?)" -f [regex]::Escape($token)
        $match = [regex]::Match($Reply, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($match.Success) {
            return [double]$match.Groups[1].Value
        }
    }
    return $null
}

function Update-MotorVelocityField {
    param([string]$Reply)
    $vel = Get-MotorVelocityFromReply -Reply $Reply
    if ($null -ne $vel) {
        if ($vel -lt $MotorVelMin) { $vel = $MotorVelMin }
        if ($vel -gt $MotorVelMax) { $vel = $MotorVelMax }
        $velNumeric.Value = [decimal]$vel
        $script:LastVelocity = $vel
    }
}

function Set-MotorVelocityFromGui {
    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    $vel = [double]$velNumeric.Value
    $cmd = "SetMotorVelocity, {0}" -f $vel.ToString($ci)
    Send-CommandWithLog -Command $cmd -OnReply {
        param($r)
        Update-MotorVelocityField $r
        Export-GuiSettings
    }
}

function Format-MotorAxisUm {
    param(
        [double]$Value,
        [int]$IntegerDigits
    )

    $ci = [System.Globalization.CultureInfo]::InvariantCulture
    $text = ([math]::Round($Value, 1)).ToString('F1', $ci)
    $dotIndex = $text.IndexOf('.')
    if ($dotIndex -lt 0) {
        return $text.PadLeft($IntegerDigits, ' ')
    }

    $intPart = $text.Substring(0, $dotIndex)
    $decPart = $text.Substring($dotIndex + 1)
    $sign = ''
    if ($intPart.StartsWith('-', [StringComparison]::Ordinal)) {
        $sign = '-'
        $intPart = $intPart.Substring(1)
    }

    $intPadded = ($sign + $intPart.PadLeft($IntegerDigits, ' '))
    return ('{0}.{1}' -f $intPadded, $decPart)
}

function Initialize-MotorPosDeadReckoning {
    $script:MotorPosDeadX = 0.0
    $script:MotorPosDeadY = 0.0
    $script:MotorPosDeadZ = 0.0
    $script:MotorPosDeadSynced = $false
    if ($null -ne $motorPosLabel) {
        $motorPosLabel.Text = 'Pos: --'
    }
}

function Set-MotorPosDeadReckoning {
    param(
        [double]$X,
        [double]$Y,
        [double]$Z
    )
    $script:MotorPosDeadX = [math]::Round($X, 2)
    $script:MotorPosDeadY = [math]::Round($Y, 2)
    $script:MotorPosDeadZ = [math]::Round($Z, 2)
    $script:MotorPosDeadSynced = $true
    Update-MotorPosLabelFromDeadReckoning
}

function Add-MotorPosDeadReckoningDelta {
    param(
        [double]$DeltaX = 0,
        [double]$DeltaY = 0,
        [double]$DeltaZ = 0
    )
    if (-not $script:MotorPosDeadSynced) { return }
    if ([math]::Abs($DeltaX) -lt 1e-12 -and [math]::Abs($DeltaY) -lt 1e-12 -and [math]::Abs($DeltaZ) -lt 1e-12) {
        return
    }
    $script:MotorPosDeadX = [math]::Round($script:MotorPosDeadX + $DeltaX, 2)
    $script:MotorPosDeadY = [math]::Round($script:MotorPosDeadY + $DeltaY, 2)
    $script:MotorPosDeadZ = [math]::Round($script:MotorPosDeadZ + $DeltaZ, 2)
    Update-MotorPosLabelFromDeadReckoning
}

function Update-MotorPosLabelFromDeadReckoning {
    if ($null -eq $motorPosLabel) { return }
    if (-not $script:MotorPosDeadSynced) {
        $motorPosLabel.Text = 'Pos: --'
        return
    }

    $xText = Format-MotorAxisUm -Value $script:MotorPosDeadX -IntegerDigits 6
    $yText = Format-MotorAxisUm -Value $script:MotorPosDeadY -IntegerDigits 6
    $zText = Format-MotorAxisUm -Value $script:MotorPosDeadZ -IntegerDigits 5
    $motorPosLabel.Text = ('Pos (um){0}X={1}  Y={2}  Z={3}' -f `
        [Environment]::NewLine, $xText, $yText, $zText)
}

function Sync-MotorPosFromFlimage {
    param([switch]$Quiet)

    if (-not $script:PipeConnected) { return }
    try {
        if (-not $Quiet) { Add-Log '>> GetCurrentPosition (sync)' }
        $reply = Send-FlimagePipeCommand -Command 'GetCurrentPosition' -Quiet:$Quiet
        if (-not $Quiet) { Add-Log "<< $reply" }
        Update-MotorPosLabel $reply
    }
    catch {
        if (-not $Quiet) { Add-Log "!! Position sync failed: $($_.Exception.Message)" }
    }
}

function Get-ActualMotorPositionFromReply {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    foreach ($token in @(
            "CurrentPosition", "CurrentPosition_um",
            "MoveMotorRelativeDone", "SetMotorPositionDone", "MotorVectorStopped")) {
        $pattern = "{0}\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)" -f [regex]::Escape($token)
        $match = [regex]::Match($Reply, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($match.Success) {
            return @(
                [double]$match.Groups[1].Value,
                [double]$match.Groups[2].Value,
                [double]$match.Groups[3].Value
            )
        }
    }
    return $null
}

function Get-MotorPositionFromReply {
    param([string]$Reply)
    return (Get-ActualMotorPositionFromReply -Reply $Reply)
}

function Update-MotorPosLabel {
    param([string]$Reply)
    $pos = Get-ActualMotorPositionFromReply -Reply $Reply
    if ($null -ne $pos) {
        Set-MotorPosDeadReckoning -X $pos[0] -Y $pos[1] -Z $pos[2]
    }
    elseif ($Reply -match '(?i)^Error') {
        $motorPosLabel.Text = 'Pos: ERR'
    }
}

function Invoke-MotorVectorJog {
    param([switch]$Quiet)

    if ($script:Busy) { return }
    if (-not (Test-CanAcceptNumpadJog)) { return }
    if (-not $script:MotorUsesVectorJog) { return }

    try {
        if (-not $script:PipeConnected) {
            if (-not $Quiet) { Add-Log "Connecting for vector jog..." }
            $ok = Connect-FlimagePipes -Log { param($line) if (-not $Quiet) { Add-Log $line } }
            Set-Status $ok
            if (-not $ok) { return }
        }

        $mapped = Apply-MotorAxisMapping -DeltaX $script:HeldJogVx -DeltaY $script:HeldJogVy -DeltaZ $script:HeldJogVz
        $ci = [System.Globalization.CultureInfo]::InvariantCulture
        $cmd = "SetMotorVectorVelocity, {0}, {1}, {2}" -f `
            $mapped.Dx.ToString($ci), `
            $mapped.Dy.ToString($ci), `
            $mapped.Dz.ToString($ci)
        Register-MotorActivity
        if (-not $Quiet) { Add-Log ">> $cmd" }
        $reply = Send-FlimagePipeCommand -Command $cmd
        if (-not $Quiet) { Add-Log "<< $reply" }
        if ($reply -match '(?i)^Error') {
            if (-not $Quiet) { Add-Log "!! Vector jog failed: $reply" }
        }
    }
    catch {
        Set-Status $false $_.Exception.Message
        if (-not $Quiet) { Add-Log "!! $($_.Exception.Message)" }
    }
}

function Invoke-MotorVectorStop {
    param([switch]$Quiet)

    if (-not $script:PipeConnected) { return }

    try {
        if (-not $Quiet) { Add-Log ">> StopMotorVector" }
        $reply = Send-FlimagePipeCommand -Command "StopMotorVector"
        if (-not $Quiet) { Add-Log "<< $reply" }
        Request-MotorPosRefreshAfterIdle
    }
    catch {
        if (-not $Quiet) { Add-Log "!! StopMotorVector failed: $($_.Exception.Message)" }
    }
}

function Invoke-MotorJogQuick {
    param(
        [double]$DeltaX = 0,
        [double]$DeltaY = 0,
        [double]$DeltaZ = 0,
        [switch]$Quiet,
        [switch]$FromRoe
    )

    if (-not $FromRoe) {
        if ($script:Busy) { return }
        if (-not (Test-CanAcceptNumpadJog)) { return }
    }
    elseif ($script:Busy) {
        return
    }

    if (-not $FromRoe) {
        $mapped = Apply-MotorAxisMapping -DeltaX $DeltaX -DeltaY $DeltaY -DeltaZ $DeltaZ
        $DeltaX = $mapped.Dx
        $DeltaY = $mapped.Dy
        $DeltaZ = $mapped.Dz
    }

    try {
        if (-not $script:PipeConnected) {
            if (-not $Quiet) { Add-Log "Connecting for jog..." }
            $ok = Connect-FlimagePipes -Log { param($line) if (-not $Quiet) { Add-Log $line } }
            Set-Status $ok
            if (-not $ok) { return }
        }

        $ci = [System.Globalization.CultureInfo]::InvariantCulture
        $cmd = "MoveMotorRelativeQuick, {0}, {1}, {2}" -f `
            $DeltaX.ToString($ci), $DeltaY.ToString($ci), $DeltaZ.ToString($ci)
        Register-MotorActivity
        if (-not $Quiet) { Add-Log ">> $cmd" }
        $reply = Send-FlimagePipeCommand -Command $cmd
        if (-not $Quiet) { Add-Log "<< $reply" }
        if ($reply -match '(?i)^Error') {
            Request-MotorPosRefreshAfterIdle
            return
        }
        if (-not $script:MotorPosDeadSynced) {
            Sync-MotorPosFromFlimage -Quiet
        }
        if ($script:MotorPosDeadSynced) {
            Add-MotorPosDeadReckoningDelta -DeltaX $DeltaX -DeltaY $DeltaY -DeltaZ $DeltaZ
        }
    }
    catch {
        Set-Status $false $_.Exception.Message
        if (-not $Quiet) { Add-Log "!! $($_.Exception.Message)" }
    }
}

function Update-HeldJogVector {
    $speeds = Get-ActiveJogSpeeds
    $stepXY = $speeds.SpeedXY
    $stepZ = $speeds.SpeedZ
    $script:HeldJogDx = 0.0
    $script:HeldJogDy = 0.0
    $script:HeldJogDz = 0.0
    if ($script:JogKeysDown['XMinus']) { $script:HeldJogDx -= $stepXY }
    if ($script:JogKeysDown['XPlus']) { $script:HeldJogDx += $stepXY }
    if ($script:JogKeysDown['YMinus']) { $script:HeldJogDy -= $stepXY }
    if ($script:JogKeysDown['YPlus']) { $script:HeldJogDy += $stepXY }
    if ($script:JogKeysDown['ZPlus']) { $script:HeldJogDz += $stepZ }
    if ($script:JogKeysDown['ZMinus']) { $script:HeldJogDz -= $stepZ }
}

function Update-HeldJogVelocity {
    $speeds = Get-ActiveJogSpeeds
    $speedXY = $speeds.SpeedXY
    $speedZ = $speeds.SpeedZ
    $script:HeldJogVx = 0.0
    $script:HeldJogVy = 0.0
    $script:HeldJogVz = 0.0
    if ($script:JogKeysDown['XMinus']) { $script:HeldJogVx -= $speedXY }
    if ($script:JogKeysDown['XPlus']) { $script:HeldJogVx += $speedXY }
    if ($script:JogKeysDown['YMinus']) { $script:HeldJogVy -= $speedXY }
    if ($script:JogKeysDown['YPlus']) { $script:HeldJogVy += $speedXY }
    if ($script:JogKeysDown['ZPlus']) { $script:HeldJogVz += $speedZ }
    if ($script:JogKeysDown['ZMinus']) { $script:HeldJogVz -= $speedZ }
}

function Test-AnyJogKeyDown {
    foreach ($name in $script:JogActionNames) {
        if ($script:JogKeysDown[$name]) { return $true }
    }
    return $false
}

function Start-HeldJogIfNeeded {
    if (-not (Test-CanAcceptNumpadJog)) {
        Stop-AllHeldJogKeys
        return
    }
    if ($script:MotorUsesVectorJog) {
        Update-HeldJogVelocity
        if (-not (Test-AnyJogKeyDown)) {
            return
        }
        Invoke-MotorVectorJog -Quiet
        if ($null -ne $jogRepeatTimer) { $jogRepeatTimer.Stop() }
        return
    }

    Update-HeldJogVector
    if (-not (Test-AnyJogKeyDown)) {
        $jogRepeatTimer.Stop()
        return
    }
    Invoke-MotorJogQuick -DeltaX $script:HeldJogDx -DeltaY $script:HeldJogDy -DeltaZ $script:HeldJogDz -Quiet
    if (-not $jogRepeatTimer.Enabled) {
        $jogRepeatTimer.Start()
    }
}

function Stop-HeldJogIfIdle {
    if (-not (Test-AnyJogKeyDown)) {
        $jogRepeatTimer.Stop()
        Request-MotorPosRefreshAfterIdle
    }
}

function Set-Busy {
    param([bool]$IsBusy)
    $script:Busy = $IsBusy
    $btnConnect.Enabled = -not $IsBusy
    $btnDisconnect.Enabled = -not $IsBusy
    if ($null -ne $btnConnectRoe) { $btnConnectRoe.Enabled = -not $IsBusy }
    if ($null -ne $btnDisconnectRoe) { $btnDisconnectRoe.Enabled = -not $IsBusy }
    if ($null -ne $btnSend) { $btnSend.Enabled = -not $IsBusy }
    if ($null -ne $btnMotorRefresh) { $btnMotorRefresh.Enabled = -not $IsBusy }
    if ($null -ne $btnMotorVelGet) { $btnMotorVelGet.Enabled = -not $IsBusy }
    if ($null -ne $btnMotorVelSet) { $btnMotorVelSet.Enabled = -not $IsBusy }
    if ($null -ne $btnSetPreset) { $btnSetPreset.Enabled = -not $IsBusy }
    if ($null -ne $btnNumpadJogToggle) { $btnNumpadJogToggle.Enabled = -not $IsBusy }
    if ($null -ne $btnNumpadHelp) { $btnNumpadHelp.Enabled = -not $IsBusy }
    if ($null -ne $velNumeric) { $velNumeric.Enabled = -not $IsBusy }
    foreach ($name in $PresetNames) {
        if ($null -ne $presetRadioButtons[$name]) {
            $presetRadioButtons[$name].Enabled = -not $IsBusy
        }
    }
    $turretControls = @(
        $filterCombo, $btnFilterGet, $btnFilterSet, $btnFilterLabels,
        $objCombo, $btnObjGet, $btnObjSet, $btnObjLabels,
        $prismRadioEyepiece, $prismRadioCamera, $btnPrismGet, $btnPrismSet,
        $panelCombo, $btnPanelGet, $btnPanelSet
    )
    foreach ($ctrl in $turretControls) {
        if ($null -ne $ctrl) {
            $ctrl.Enabled = $script:OlympusTurretsAvailable -and (-not $IsBusy)
        }
    }
    if ($IsBusy) { $form.Cursor = [System.Windows.Forms.Cursors]::WaitCursor }
    else { $form.Cursor = [System.Windows.Forms.Cursors]::Default }
    [System.Windows.Forms.Application]::DoEvents()
}

function Send-CommandWithLog {
    param(
        [string]$Command,
        [scriptblock]$OnReply = $null
    )

    if ($script:Busy) {
        Add-Log "!! Busy, wait for current operation"
        return
    }

    Set-Busy $true
    try {
        if (-not $script:PipeConnected) {
            Add-Log "Connecting..."
            $ok = Connect-FlimagePipes -Log { param($line) Add-Log $line }
            Set-Status $ok
            if (-not $ok) {
                Add-Log "Connect failed"
                return
            }
            Add-Log "Connected"
        }

        Add-Log ">> $Command"
        if ($Command -match '(?i)^SendOlympusCommand,\s*1OB\s+(\d+)\s*$') {
            $objPos = [int]$Matches[1]
            $reply = Send-OlympusObjectiveSetWithEscape -Position $objPos
            if ($null -ne $OnReply) { & $OnReply $reply }
            return
        }

        $reply = Send-FlimagePipeCommand -Command $Command
        Add-Log "<< $reply"
        Update-OlympusLabelsFromPipeExchange -Command $Command -Reply $reply
        if ($null -ne $OnReply) { & $OnReply $reply }
    }
    catch {
        Set-Status $false $_.Exception.Message
        Add-Log "!! $($_.Exception.Message)"
    }
    finally {
        Set-Busy $false
    }
}

$filterGroup = New-Object System.Windows.Forms.GroupBox
$filterGroup.Text = "Filter (1MU)"
$filterGroup.Location = New-Object System.Drawing.Point $GuiMargin, 110
$filterGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 56
$form.Controls.Add($filterGroup)

$filterPosLabel = New-Object System.Windows.Forms.Label
$filterPosLabel.Text = "Cur: --"
$filterPosLabel.Location = New-Object System.Drawing.Point 8, 18
$filterPosLabel.Width = 44
$filterGroup.Controls.Add($filterPosLabel)

$filterCombo = New-Object System.Windows.Forms.ComboBox
$filterCombo.DropDownStyle = 'DropDownList'
$filterCombo.Location = New-Object System.Drawing.Point 54, 16
$filterCombo.Width = 168
$filterGroup.Controls.Add($filterCombo)

$btnFilterGet = New-Object System.Windows.Forms.Button
$btnFilterGet.Text = "Get"
$btnFilterGet.Location = New-Object System.Drawing.Point 228, 14
$btnFilterGet.Width = 36
$filterGroup.Controls.Add($btnFilterGet)

$btnFilterSet = New-Object System.Windows.Forms.Button
$btnFilterSet.Text = "Set"
$btnFilterSet.Location = New-Object System.Drawing.Point 268, 14
$btnFilterSet.Width = 36
$filterGroup.Controls.Add($btnFilterSet)

$btnFilterLabels = New-Object System.Windows.Forms.Button
$btnFilterLabels.Text = "Labels"
$btnFilterLabels.Location = New-Object System.Drawing.Point 308, 14
$btnFilterLabels.Width = 44
$filterGroup.Controls.Add($btnFilterLabels)

$objGroup = New-Object System.Windows.Forms.GroupBox
$objGroup.Text = "Objective (1OB)"
$objGroup.Location = New-Object System.Drawing.Point $GuiMargin, 172
$objGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 56
$form.Controls.Add($objGroup)

$objPosLabel = New-Object System.Windows.Forms.Label
$objPosLabel.Text = "Cur: --"
$objPosLabel.Location = New-Object System.Drawing.Point 8, 18
$objPosLabel.Width = 44
$objGroup.Controls.Add($objPosLabel)

$objCombo = New-Object System.Windows.Forms.ComboBox
$objCombo.DropDownStyle = 'DropDownList'
$objCombo.Location = New-Object System.Drawing.Point 54, 16
$objCombo.Width = 168
$objGroup.Controls.Add($objCombo)

$btnObjGet = New-Object System.Windows.Forms.Button
$btnObjGet.Text = "Get"
$btnObjGet.Location = New-Object System.Drawing.Point 228, 14
$btnObjGet.Width = 36
$objGroup.Controls.Add($btnObjGet)

$btnObjSet = New-Object System.Windows.Forms.Button
$btnObjSet.Text = "Set"
$btnObjSet.Location = New-Object System.Drawing.Point 268, 14
$btnObjSet.Width = 36
$objGroup.Controls.Add($btnObjSet)

$btnObjLabels = New-Object System.Windows.Forms.Button
$btnObjLabels.Text = "Labels"
$btnObjLabels.Location = New-Object System.Drawing.Point 308, 14
$btnObjLabels.Width = 44
$objGroup.Controls.Add($btnObjLabels)

$prismGroup = New-Object System.Windows.Forms.GroupBox
$prismGroup.Text = "Light path (1PRISM) - off"
$prismGroup.Location = New-Object System.Drawing.Point $GuiMargin, 234
$prismGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 56
$form.Controls.Add($prismGroup)

$prismPosLabel = New-Object System.Windows.Forms.Label
$prismPosLabel.Text = "Cur: --"
$prismPosLabel.Location = New-Object System.Drawing.Point 8, 18
$prismPosLabel.Width = 88
$prismGroup.Controls.Add($prismPosLabel)

$prismRadioEyepiece = New-Object System.Windows.Forms.RadioButton
$prismRadioEyepiece.Text = "Eyepiece"
$prismRadioEyepiece.AutoSize = $true
$prismRadioEyepiece.Location = New-Object System.Drawing.Point 100, 18
$prismRadioEyepiece.Checked = $true
$prismGroup.Controls.Add($prismRadioEyepiece)

$prismRadioCamera = New-Object System.Windows.Forms.RadioButton
$prismRadioCamera.Text = "Camera"
$prismRadioCamera.AutoSize = $true
$prismRadioCamera.Location = New-Object System.Drawing.Point 188, 18
$prismGroup.Controls.Add($prismRadioCamera)

$btnPrismGet = New-Object System.Windows.Forms.Button
$btnPrismGet.Text = "Get"
$btnPrismGet.Location = New-Object System.Drawing.Point 268, 14
$btnPrismGet.Width = 36
$prismGroup.Controls.Add($btnPrismGet)

$btnPrismSet = New-Object System.Windows.Forms.Button
$btnPrismSet.Text = "Set"
$btnPrismSet.Location = New-Object System.Drawing.Point 308, 14
$btnPrismSet.Width = 36
$prismGroup.Controls.Add($btnPrismSet)

$panelGroup = New-Object System.Windows.Forms.GroupBox
$panelGroup.Text = "Control mode - off"
$panelGroup.Location = New-Object System.Drawing.Point $GuiMargin, 290
$panelGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 56
$form.Controls.Add($panelGroup)

$panelCombo = New-Object System.Windows.Forms.ComboBox
$panelCombo.DropDownStyle = 'DropDownList'
$panelCombo.Location = New-Object System.Drawing.Point 8, 16
$panelCombo.Width = 214
$panelGroup.Controls.Add($panelCombo)
Initialize-PanelControlCombo

$btnPanelGet = New-Object System.Windows.Forms.Button
$btnPanelGet.Text = "Get"
$btnPanelGet.Location = New-Object System.Drawing.Point 228, 14
$btnPanelGet.Width = 36
$panelGroup.Controls.Add($btnPanelGet)

$btnPanelSet = New-Object System.Windows.Forms.Button
$btnPanelSet.Text = "Set"
$btnPanelSet.Location = New-Object System.Drawing.Point 268, 14
$btnPanelSet.Width = 36
$panelGroup.Controls.Add($btnPanelSet)

$motorGroup = New-Object System.Windows.Forms.GroupBox
$motorGroup.Text = "Motor jog"
$motorGroup.Location = New-Object System.Drawing.Point $GuiMargin, 348
$motorGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 200
$form.Controls.Add($motorGroup)

$script:MotorPosRefreshTimer = New-Object System.Windows.Forms.Timer
$script:MotorPosRefreshTimer.Interval = 50
$script:MotorPosRefreshTimer.Add_Tick({
        Invoke-AbsPositionRefreshIfIdle
    })

$jogRepeatTimer = New-Object System.Windows.Forms.Timer
$jogRepeatTimer.Interval = 45
$jogRepeatTimer.Add_Tick({
        if ($script:MotorUsesVectorJog) { return }
        if (-not (Test-CanAcceptNumpadJog)) {
            Stop-AllHeldJogKeys
            return
        }
        if (-not (Test-AnyJogKeyDown)) {
            $jogRepeatTimer.Stop()
            Request-MotorPosRefreshAfterIdle
            return
        }
        Update-HeldJogVector
        Invoke-MotorJogQuick -DeltaX $script:HeldJogDx -DeltaY $script:HeldJogDy -DeltaZ $script:HeldJogDz -Quiet
    })

$motorPosLabel = New-Object System.Windows.Forms.Label
$motorPosLabel.Text = "Pos: --"
$motorPosLabel.Font = New-Object System.Drawing.Font('Consolas', 9)
$motorPosLabel.Location = New-Object System.Drawing.Point 8, 18
$motorPosLabel.Size = New-Object System.Drawing.Size 276, 36
$motorPosLabel.AutoSize = $false
$motorGroup.Controls.Add($motorPosLabel)

$btnMotorRefresh = New-Object System.Windows.Forms.Button
$btnMotorRefresh.Text = "Get pos"
$btnMotorRefresh.Location = New-Object System.Drawing.Point 290, 16
$btnMotorRefresh.Width = 58
$motorGroup.Controls.Add($btnMotorRefresh)

$presetLabel = New-Object System.Windows.Forms.Label
$presetLabel.Text = "Tier:"
$presetLabel.Location = New-Object System.Drawing.Point 8, 54
$presetLabel.AutoSize = $true
$motorGroup.Controls.Add($presetLabel)

$presetRadioButtons = @{}
$presetX = 44
foreach ($presetName in $PresetNames) {
    $radio = New-Object System.Windows.Forms.RadioButton
    $radio.Text = $presetName
    $radio.AutoSize = $true
    $radio.Location = New-Object System.Drawing.Point $presetX, 52
    $radio.Tag = $presetName
    $motorGroup.Controls.Add($radio)
    $presetRadioButtons[$presetName] = $radio
    $presetX += 32
    $radio.Add_CheckedChanged({
            if (-not $this.Checked) { return }
            $script:ActivePreset = [string]$this.Tag
            Update-StepPresetDisplay
            Export-GuiSettings
            if ($script:MotorUsesVectorJog -and (Test-AnyJogKeyDown)) {
                Start-HeldJogIfNeeded
            }
        })
}

$btnSetPreset = New-Object System.Windows.Forms.Button
$btnSetPreset.Text = "Edit"
$btnSetPreset.Location = New-Object System.Drawing.Point 210, 50
$btnSetPreset.Width = 40
$motorGroup.Controls.Add($btnSetPreset)

$stepDisplayLabel = New-Object System.Windows.Forms.Label
$stepDisplayLabel.Text = "XY=-- Z=--"
$stepDisplayLabel.Location = New-Object System.Drawing.Point 8, 76
$stepDisplayLabel.Size = New-Object System.Drawing.Size 356, 18
$stepDisplayLabel.AutoSize = $false
$motorGroup.Controls.Add($stepDisplayLabel)

$velLabel = New-Object System.Windows.Forms.Label
$velLabel.Text = "Vel:"
$velLabel.Location = New-Object System.Drawing.Point 8, 100
$velLabel.AutoSize = $true
$motorGroup.Controls.Add($velLabel)

$velNumeric = New-Object System.Windows.Forms.NumericUpDown
$velNumeric.DecimalPlaces = 0
$velNumeric.Increment = 100
$velNumeric.Minimum = $MotorVelMin
$velNumeric.Maximum = $MotorVelMax
$velNumeric.Value = [decimal]$script:LastVelocity
$velNumeric.Location = New-Object System.Drawing.Point 42, 98
$velNumeric.Width = 64
$motorGroup.Controls.Add($velNumeric)

$btnMotorVelSet = New-Object System.Windows.Forms.Button
$btnMotorVelSet.Text = "Set"
$btnMotorVelSet.Location = New-Object System.Drawing.Point 110, 96
$btnMotorVelSet.Width = 36
$motorGroup.Controls.Add($btnMotorVelSet)

$btnMotorVelGet = New-Object System.Windows.Forms.Button
$btnMotorVelGet.Text = "Get"
$btnMotorVelGet.Location = New-Object System.Drawing.Point 150, 96
$btnMotorVelGet.Width = 36
$motorGroup.Controls.Add($btnMotorVelGet)

$btnNumpadJogToggle = New-Object System.Windows.Forms.Button
$keyLayoutLabel = New-Object System.Windows.Forms.Label
$keyLayoutLabel.Text = "Keys:"
$keyLayoutLabel.Location = New-Object System.Drawing.Point 8, 122
$keyLayoutLabel.AutoSize = $true
$motorGroup.Controls.Add($keyLayoutLabel)

$keyLayoutCombo = New-Object System.Windows.Forms.ComboBox
$keyLayoutCombo.DropDownStyle = 'DropDownList'
$keyLayoutCombo.Location = New-Object System.Drawing.Point 48, 120
$keyLayoutCombo.Width = 100
foreach ($layoutName in $JogKeyLayouts) {
    [void]$keyLayoutCombo.Items.Add($layoutName)
}
$keyLayoutCombo.Add_SelectedIndexChanged({
        if ($script:SuppressKeyLayoutComboEvent) { return }
        if ($null -eq $keyLayoutCombo -or $keyLayoutCombo.SelectedIndex -lt 0) { return }
        Set-JogKeyLayout -Layout ([string]$keyLayoutCombo.SelectedItem)
    })
$motorGroup.Controls.Add($keyLayoutCombo)
Update-JogKeyLayoutUi

$btnNumpadJogToggle.Location = New-Object System.Drawing.Point 8, 148
$btnNumpadJogToggle.Width = 88
$btnNumpadJogToggle.Height = 24
$motorGroup.Controls.Add($btnNumpadJogToggle)

$btnNumpadHelp = New-Object System.Windows.Forms.Button
$btnNumpadHelp.Text = "Key map"
$btnNumpadHelp.Location = New-Object System.Drawing.Point 102, 148
$btnNumpadHelp.Width = 72
$btnNumpadHelp.Height = 24
$motorGroup.Controls.Add($btnNumpadHelp)

$btnAxisMapping = New-Object System.Windows.Forms.Button
$btnAxisMapping.Text = "Axis map"
$btnAxisMapping.Location = New-Object System.Drawing.Point 180, 148
$btnAxisMapping.Width = 72
$btnAxisMapping.Height = 24
$motorGroup.Controls.Add($btnAxisMapping)

$numpadHintLabel = New-Object System.Windows.Forms.Label
$numpadHintLabel.Text = "Capture ON = jog keys work when another window is focused"
$numpadHintLabel.Location = New-Object System.Drawing.Point 8, 176
$numpadHintLabel.Size = New-Object System.Drawing.Size 356, 16
$numpadHintLabel.AutoSize = $false
$numpadHintLabel.Font = New-Object System.Drawing.Font $numpadHintLabel.Font.FontFamily, 8
$numpadHintLabel.ForeColor = [System.Drawing.Color]::DimGray
$motorGroup.Controls.Add($numpadHintLabel)

$cmdGroup = New-Object System.Windows.Forms.GroupBox
$cmdGroup.Text = "Pipe command"
$cmdGroup.Location = New-Object System.Drawing.Point $GuiMargin, 556
$cmdGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 118
$form.Controls.Add($cmdGroup)

$cmdBox = New-Object System.Windows.Forms.TextBox
$cmdBox.Location = New-Object System.Drawing.Point 8, 18
$cmdBox.Width = ($GuiInnerWidth - 88)
$cmdGroup.Controls.Add($cmdBox)

$btnSend = New-Object System.Windows.Forms.Button
$btnSend.Text = "Send"
$btnSend.Location = New-Object System.Drawing.Point ($GuiInnerWidth - 72), 16
$btnSend.Width = 56
$cmdGroup.Controls.Add($btnSend)

$quickPanel = New-Object System.Windows.Forms.FlowLayoutPanel
$quickPanel.Location = New-Object System.Drawing.Point 8, 46
$quickPanel.Size = New-Object System.Drawing.Size ($GuiInnerWidth - 16), 28
$quickPanel.WrapContents = $false
$cmdGroup.Controls.Add($quickPanel)

foreach ($quick in @(
        @{ Text = "Rel XYZ"; Cmd = "GetRelativeXYZ" },
        @{ Text = "Abs XYZ"; Cmd = "GetCurrentPosition" },
        @{ Text = "Version"; Cmd = "GetVersion" }
    )) {
    $qb = New-Object System.Windows.Forms.Button
    $qb.Text = $quick.Text
    $qb.AutoSize = $true
    $qb.Tag = $quick.Cmd
    $quickPanel.Controls.Add($qb)
    $qb.Add_Click({
            $cmdBox.Text = $this.Tag
            Send-CommandWithLog -Command $this.Tag
        })
}

Initialize-CollapsibleLayout

function Update-FilterLabel {
    param([string]$Reply)
    $value = Get-TurretValue -Reply $Reply -Token "FilterTurret"
    if ($null -eq $value) { $value = Resolve-OlympusTurretPositionFromReply -Reply $Reply -CommandToken '1MU' }
    if ($null -ne $value) {
        $filterPosLabel.Text = Format-TurretCurrentText -Position $value
        Sync-TurretComboToPosition -Combo $filterCombo -Position $value
    }
    elseif ($Reply -match '(?i)^Error') { $filterPosLabel.Text = "Cur: ERR" }
}

function Update-ObjectiveLabel {
    param([string]$Reply)
    $value = Get-TurretValue -Reply $Reply -Token "ObjectiveTurret"
    if ($null -eq $value) { $value = Resolve-OlympusTurretPositionFromReply -Reply $Reply -CommandToken '1OB' }
    if ($null -ne $value) {
        $objPosLabel.Text = Format-TurretCurrentText -Position $value
        Sync-TurretComboToPosition -Combo $objCombo -Position $value
    }
    elseif ($Reply -match '(?i)^Error') { $objPosLabel.Text = "Cur: ERR" }
}

Update-TurretComboItems -Combo $filterCombo -Labels $script:FilterLabels
Update-TurretComboItems -Combo $objCombo -Labels $script:ObjectiveLabels
Update-StepPresetDisplay
Update-NumpadJogToggleUi
Update-OlympusPanelControlUi -Mode $script:OlympusPanelControlMode
Set-OlympusTurretControlsEnabled -Enabled $false -MotorHwName "not connected"

$panelCombo.Add_SelectedIndexChanged({
        if ($script:SuppressOlympusPanelControlUiEvent) { return }
        if (-not $script:PipeConnected) { return }
        if ($script:Busy) { return }
        Set-Busy $true
        try { Invoke-OlympusPanelControlSet -Mode (Get-PanelComboMode) }
        catch { Add-Log "!! $($_.Exception.Message)" }
        finally { Set-Busy $false }
    })

$btnPanelGet.Add_Click({
        if ($script:Busy) { return }
        if (-not $script:PipeConnected) {
            Add-Log '!! Connect FLIMage first'
            return
        }
        Set-Busy $true
        try { Sync-OlympusPanelControlFromFlimage }
        finally { Set-Busy $false }
    })

$btnPanelSet.Add_Click({
        if ($script:Busy) { return }
        if (-not $script:PipeConnected) {
            Add-Log '!! Connect FLIMage first'
            return
        }
        Set-Busy $true
        try { Invoke-OlympusPanelControlSet -Mode (Get-PanelComboMode) }
        catch { Add-Log "!! $($_.Exception.Message)" }
        finally { Set-Busy $false }
    })

$btnRoeDetails.Add_Click({ Show-AxisMappingDialog })

$btnAxisMapping.Add_Click({ Show-AxisMappingDialog })

$btnConnectRoe.Add_Click({ [void](Start-RoeConnection) })

$btnDisconnectRoe.Add_Click({ Stop-RoeConnection })

$btnConnect.Add_Click({
        if ($script:Busy) { return }
        Set-Busy $true
        try {
            Add-Log "Connecting..."
            $ok = Connect-FlimagePipes -Log { param($line) Add-Log $line }
            if ($ok) {
                Add-Log ">> GetVersion"
                $reply = Send-FlimagePipeCommand -Command "GetVersion"
                Add-Log "<< $reply"

                Add-Log ">> State.Init.MotorHWName?"
                $motorReply = Send-FlimagePipeCommand -Command "State.Init.MotorHWName?"
                Add-Log "<< $motorReply"
                $motorName = Get-StateAssignmentValue $motorReply
                if ([string]::IsNullOrWhiteSpace($motorName)) { $motorName = "unknown" }
                $hasOlympus = Test-MotorHwSupportsOlympusTurrets -MotorHwName $motorName
                Set-OlympusTurretControlsEnabled -Enabled $hasOlympus -MotorHwName $motorName
                Set-MotorJogMode -UsesVectorJog (Test-MotorHwUsesVectorJog -MotorHwName $motorName) -MotorHwName $motorName
                Set-Status $true ("Motor=$motorName")
                Sync-MotorPosFromFlimage -Quiet

                Add-Log ">> GetMotorVelocity"
                $velReply = Send-FlimagePipeCommand -Command "GetMotorVelocity"
                Add-Log "<< $velReply (saved vel kept until Get)"

                if ($hasOlympus) {
                    try {
                        Invoke-OlympusPanelControlSet -Mode $script:OlympusPanelControlMode -Quiet
                    }
                    catch {
                        Add-Log "!! Panel mode: $($_.Exception.Message)"
                        Update-OlympusPanelControlUi -Mode $script:OlympusPanelControlMode
                    }
                    Sync-OlympusPanelControlFromFlimage -Quiet

                    $filterReply = Send-OlympusIx2WithLog -Ix2Command '1MU?'
                    Update-FilterLabel $filterReply
                    $objReply = Send-OlympusIx2WithLog -Ix2Command '1OB?'
                    Update-ObjectiveLabel $objReply
                    $prismReply = Send-OlympusIx2WithLog -Ix2Command '1PRISM?'
                    Update-PrismLabel $prismReply
                }
            }
            else {
                Set-Status $false "open FLIMage and retry"
                Add-Log "Connect failed"
            }
        }
        catch {
            Set-Status $false $_.Exception.Message
            Add-Log "!! $($_.Exception.Message)"
        }
        finally {
            Set-Busy $false
        }
    })

$btnDisconnect.Add_Click({
        Stop-PrismRefreshTimer
        if ($script:MotorUsesVectorJog) {
            Invoke-MotorVectorStop -Quiet
        }
        Initialize-MotorPosDeadReckoning
        Close-FlimagePipes
        Set-Status $false
        Set-MotorJogMode -UsesVectorJog $false
        Set-OlympusTurretControlsEnabled -Enabled $false -MotorHwName "disconnected"
        Add-Log "Disconnected"
    })

$btnFilterGet.Add_Click({
        Send-CommandWithLog -Command "SendOlympusCommand, 1MU?" -OnReply { param($r) Update-FilterLabel $r }
    })

$btnFilterSet.Add_Click({
        $value = Get-ComboTurretPosition $filterCombo
        if ($script:Busy) {
            Add-Log "!! Busy, wait for current operation"
            return
        }
        Set-Busy $true
        try {
            if (-not $script:PipeConnected) {
                Add-Log "Connecting..."
                $ok = Connect-FlimagePipes -Log { param($line) Add-Log $line }
                Set-Status $ok
                if (-not $ok) {
                    Add-Log "Connect failed"
                    return
                }
                Add-Log "Connected"
            }
            Send-OlympusIx2SequenceWithLog -Ix2Commands (Build-OlympusTurretSetSequence -Ix2SetCommand ("1MU {0}" -f $value)) -OnLastReply {
                param($r)
                Update-FilterLabel -Reply $r
            }
        }
        catch {
            Set-Status $false $_.Exception.Message
            Add-Log "!! $($_.Exception.Message)"
        }
        finally {
            Set-Busy $false
        }
    })

$btnFilterLabels.Add_Click({
        if (Show-TurretLabelsEditor -Kind 'Filter' -Labels $script:FilterLabels) {
            Update-TurretComboItems -Combo $filterCombo -Labels $script:FilterLabels
        }
    })

$btnObjGet.Add_Click({
        Send-CommandWithLog -Command "SendOlympusCommand, 1OB?" -OnReply { param($r) Update-ObjectiveLabel $r }
    })

$btnObjSet.Add_Click({
        $value = Get-ComboTurretPosition $objCombo
        if ($script:Busy) {
            Add-Log "!! Busy, wait for current operation"
            return
        }
        Set-Busy $true
        try {
            if (-not $script:PipeConnected) {
                Add-Log "Connecting..."
                $ok = Connect-FlimagePipes -Log { param($line) Add-Log $line }
                Set-Status $ok
                if (-not $ok) {
                    Add-Log "Connect failed"
                    return
                }
                Add-Log "Connected"
            }
            Send-OlympusObjectiveSetWithEscape -Position $value
        }
        catch {
            Set-Status $false $_.Exception.Message
            Add-Log "!! $($_.Exception.Message)"
        }
        finally {
            Set-Busy $false
        }
    })

$btnObjLabels.Add_Click({
        if (Show-TurretLabelsEditor -Kind 'Objective' -Labels $script:ObjectiveLabels) {
            Update-TurretComboItems -Combo $objCombo -Labels $script:ObjectiveLabels
        }
    })

$btnPrismGet.Add_Click({
        Send-CommandWithLog -Command "SendOlympusCommand, 1PRISM?" -OnReply {
            param($r)
            Update-PrismLabel $r
            if ($r -match '(?i)^Error') { Start-PrismGetRetryTimer }
        }
    })

$btnPrismSet.Add_Click({
        $value = Get-SelectedLightPathPosition
        Send-CommandWithLog -Command ("SendOlympusCommand, 1PRISM {0}" -f $value) -OnReply {
            param($r)
            Start-PrismLabelRefreshWithRetry -InitialReply $r -VerifyAfterSet
        }
    })

$btnSend.Add_Click({
        $command = $cmdBox.Text.Trim()
        if ($command) { Send-CommandWithLog -Command $command }
    })

$cmdBox.Add_KeyDown({
        if ($_.KeyCode -eq [System.Windows.Forms.Keys]::Enter) {
            $command = $cmdBox.Text.Trim()
            if ($command) { Send-CommandWithLog -Command $command }
            $_.SuppressKeyPress = $true
        }
    })

$btnClearLog.Add_Click({ $logBox.Clear() })

$btnMotorRefresh.Add_Click({
        Send-CommandWithLog -Command "GetCurrentPosition" -OnReply { param($r) Update-MotorPosLabel $r }
    })

$btnMotorVelGet.Add_Click({
        Send-CommandWithLog -Command "GetMotorVelocity" -OnReply {
            param($r)
            Update-MotorVelocityField $r
            Export-GuiSettings
        }
    })

$btnMotorVelSet.Add_Click({ Set-MotorVelocityFromGui })

$btnSetPreset.Add_Click({
        Show-StepPresetTableEditor
    })

$btnNumpadHelp.Add_Click({ Show-NumpadHelpDialog })

$btnNumpadJogToggle.Add_Click({
        $script:NumpadJogEnabled = -not $script:NumpadJogEnabled
        Update-NumpadJogToggleUi
        Export-GuiSettings
        Add-Log ("Numpad jog {0}" -f ($(if ($script:NumpadJogEnabled) { 'enabled' } else { 'disabled' })))
    })

$velNumeric.Add_ValueChanged({
        $script:LastVelocity = [double]$velNumeric.Value
    })

$form.Add_Load({
        if ($null -ne $script:GlobalNumpadHook) {
            $script:GlobalNumpadHook.GuiWindowHandle = $form.Handle
        }
        Update-JogKeyLayoutUi
        Sync-GlobalHookKeyLayout
        Sync-GlobalNumpadHookState
        Update-RoeStatusLabel
        if ($null -ne $roePingTimer) { $roePingTimer.Interval = $script:RoeHostPingMs }
    })

$form.Add_KeyDown({
        if (-not (Test-JogKeyInputAllowed)) { return }

        $presetKeyName = Get-NumpadPresetKeyName -EventArgs $_
        if ($null -ne $presetKeyName) {
            $_.SuppressKeyPress = $true
            Clear-KeyboardFocus
            if (-not $script:PresetKeysDown[$presetKeyName]) {
                Invoke-NumpadKeyEvent -KeyName $presetKeyName -IsDown $true
            }
            return
        }

        $jogKeyName = Get-NumpadJogKeyName -EventArgs $_
        if ($null -eq $jogKeyName) { return }

        $_.SuppressKeyPress = $true
        Clear-KeyboardFocus
        Invoke-NumpadKeyEvent -KeyName $jogKeyName -IsDown $true
    })

$form.Add_KeyUp({
        $presetKeyName = Get-NumpadPresetKeyName -EventArgs $_
        if ($null -ne $presetKeyName) {
            Invoke-NumpadKeyEvent -KeyName $presetKeyName -IsDown $false
            return
        }

        $jogKeyName = Get-NumpadJogKeyName -EventArgs $_
        if ($null -eq $jogKeyName) { return }
        Invoke-NumpadKeyEvent -KeyName $jogKeyName -IsDown $false
    })

Register-FocusClearOnClick $form
Register-FocusClearOnClick $motorGroup
Register-FocusClearOnClick $roeGroup
Register-FocusClearOnClick $filterGroup
Register-FocusClearOnClick $objGroup
Register-FocusClearOnClick $prismGroup
Register-FocusClearOnClick $panelGroup
Register-FocusClearOnClick $cmdGroup
Register-FocusClearOnClick $logGroup
Register-FocusClearOnClick $statusLabel

$form.Add_Load({
        Update-CollapsibleLayout
    })

$form.Add_FormClosed({
        if ($null -ne $jogRepeatTimer) { $jogRepeatTimer.Stop() }
        if ($null -ne $script:MotorPosRefreshTimer) { $script:MotorPosRefreshTimer.Stop() }
        Stop-RoeConnection -Quiet
        Stop-GlobalNumpadHook
        if ($null -ne $script:GlobalNumpadHook) {
            $script:GlobalNumpadHook.Dispose()
            $script:GlobalNumpadHook = $null
        }
        $script:LastVelocity = [double]$velNumeric.Value
        Export-GuiSettings
        Close-FlimagePipes
    })

[void]$form.ShowDialog()
