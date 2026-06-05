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

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class RemoteControlCloser
{
    private delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    private static extern bool PostMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

    private const uint WM_CLOSE = 0x0010;
    private static IntPtr _target = IntPtr.Zero;

    private static bool Callback(IntPtr hWnd, IntPtr lParam)
    {
        var sb = new StringBuilder(512);
        GetWindowText(hWnd, sb, sb.Capacity);
        if (sb.ToString().Contains("Remote control & script"))
            _target = hWnd;
        return true;
    }

    public static bool CloseRemoteControlWindow()
    {
        _target = IntPtr.Zero;
        EnumWindows(Callback, IntPtr.Zero);
        if (_target == IntPtr.Zero)
            return false;
        PostMessage(_target, WM_CLOSE, IntPtr.Zero, IntPtr.Zero);
        return true;
    }
}
"@

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
$HandshakeCode = "FLIMage"
$ReadPipeName = "FLIMageR"
$WritePipeName = "FLIMageW"
$ComInitDir = Join-Path $env:USERPROFILE "Documents\FLIMage\Init_Files\COM"
$ComInitFile = Join-Path $ComInitDir "COM_method.txt"
$ConnectTimeoutMs = 500
$ConnectRetries = 3
$SettingsFile = Join-Path $PSScriptRoot "olympus_turret_pipe_gui.settings.txt"
$PresetNames = @('Fine', 'Mid', 'Coarse')
$JogKeyLayouts = @('Numpad', 'Cursor')
$GuiMargin = 10
$GuiInnerWidth = 372

$script:PipeR = $null
$script:PipeW = $null
$script:PipeConnected = $false
$script:PipeLock = New-Object object
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
$script:ActivePreset = 'Mid'
$script:LastVelocity = $MotorVelDefault
$script:Presets = @{
    Fine   = @{ StepXY = 0.10; StepZ = 0.05; VelXY = 0.010; VelZ = 0.005 }
    Mid    = @{ StepXY = 1.00; StepZ = 0.50; VelXY = 0.050; VelZ = 0.025 }
    Coarse = @{ StepXY = 10.0; StepZ = 5.00; VelXY = 0.200; VelZ = 0.100 }
}
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
        ActivePreset     = 'Mid'
        NumpadJogEnabled = $true
        JogKeyLayout     = 'Numpad'
        RoeComPort       = ''
        RoeHostPingMs    = 400
        MotorInvertX     = 'false'
        MotorInvertY     = 'false'
        MotorInvertZ     = 'false'
        MotorSwapXY      = 'false'
        RoeClockwisePositive = 'false'
        'Fine.StepXY'    = 0.10
        'Fine.StepZ'     = 0.05
        'Fine.VelXY'     = 0.010
        'Fine.VelZ'      = 0.005
        'Mid.StepXY'     = 1.00
        'Mid.StepZ'      = 0.50
        'Mid.VelXY'      = 0.050
        'Mid.VelZ'       = 0.025
        'Coarse.StepXY'  = 10.0
        'Coarse.StepZ'   = 5.00
        'Coarse.VelXY'   = 0.200
        'Coarse.VelZ'    = 0.100
        'Olympus.EscapeCommand' = ''
        'Olympus.TurretWaitMs'  = 500
    }
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $settings["Filter.P$p"] = Get-DefaultTurretLabel $p
        $settings["Objective.P$p"] = Get-DefaultTurretLabel $p
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

    if ($Settings.ContainsKey('Olympus.EscapeCommand')) {
        $script:OlympusEscapeCommand = [string]$Settings['Olympus.EscapeCommand']
    }
    else {
        $script:OlympusEscapeCommand = ''
    }
    try {
        $script:OlympusTurretWaitMs = [int]$Settings['Olympus.TurretWaitMs']
    }
    catch {
        $script:OlympusTurretWaitMs = 500
    }
    if ($script:OlympusTurretWaitMs -lt 0) { $script:OlympusTurretWaitMs = 0 }

    Initialize-JogKeyState

    foreach ($name in $PresetNames) {
        $xyKey = "$name.StepXY"
        $zKey = "$name.StepZ"
        $velXyKey = "$name.VelXY"
        $velZKey = "$name.VelZ"
        if ($Settings.ContainsKey($xyKey)) {
            try { $script:Presets[$name].StepXY = [double]$Settings[$xyKey] } catch { }
        }
        if ($Settings.ContainsKey($zKey)) {
            try { $script:Presets[$name].StepZ = [double]$Settings[$zKey] } catch { }
        }
        if ($Settings.ContainsKey($velXyKey)) {
            try { $script:Presets[$name].VelXY = [double]$Settings[$velXyKey] } catch { }
        }
        if ($Settings.ContainsKey($velZKey)) {
            try { $script:Presets[$name].VelZ = [double]$Settings[$velZKey] } catch { }
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
        ("Fine.StepXY={0}" -f $script:Presets.Fine.StepXY.ToString('F2', $ci)),
        ("Fine.StepZ={0}" -f $script:Presets.Fine.StepZ.ToString('F2', $ci)),
        ("Fine.VelXY={0}" -f $script:Presets.Fine.VelXY.ToString('F3', $ci)),
        ("Fine.VelZ={0}" -f $script:Presets.Fine.VelZ.ToString('F3', $ci)),
        ("Mid.StepXY={0}" -f $script:Presets.Mid.StepXY.ToString('F2', $ci)),
        ("Mid.StepZ={0}" -f $script:Presets.Mid.StepZ.ToString('F2', $ci)),
        ("Mid.VelXY={0}" -f $script:Presets.Mid.VelXY.ToString('F3', $ci)),
        ("Mid.VelZ={0}" -f $script:Presets.Mid.VelZ.ToString('F3', $ci)),
        ("Coarse.StepXY={0}" -f $script:Presets.Coarse.StepXY.ToString('F2', $ci)),
        ("Coarse.StepZ={0}" -f $script:Presets.Coarse.StepZ.ToString('F2', $ci)),
        ("Coarse.VelXY={0}" -f $script:Presets.Coarse.VelXY.ToString('F3', $ci)),
        ("Coarse.VelZ={0}" -f $script:Presets.Coarse.VelZ.ToString('F3', $ci)),
        ("Olympus.EscapeCommand={0}" -f $script:OlympusEscapeCommand),
        ("Olympus.TurretWaitMs={0}" -f $script:OlympusTurretWaitMs)
    )
    for ($p = $TurretMin; $p -le $TurretMax; $p++) {
        $lines += ("Filter.P{0}={1}" -f $p, $script:FilterLabels[$p])
        $lines += ("Objective.P{0}={1}" -f $p, $script:ObjectiveLabels[$p])
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

function Get-Timestamp {
    return (Get-Date).ToString("HH:mm:ss.fff")
}

function Read-PipeString([System.IO.Stream]$Stream) {
    $b1 = $Stream.ReadByte()
    $b2 = $Stream.ReadByte()
    if ($b1 -lt 0 -or $b2 -lt 0) { return $null }
    $len = ($b1 * 256) + $b2
    if ($len -le 0) { return "" }
    $buffer = New-Object byte[] $len
    $offset = 0
    while ($offset -lt $len) {
        $read = $Stream.Read($buffer, $offset, $len - $offset)
        if ($read -le 0) { break }
        $offset += $read
    }
    return [System.Text.Encoding]::UTF8.GetString($buffer, 0, $offset)
}

function Write-PipeString([System.IO.Stream]$Stream, [string]$Text) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $len = $bytes.Length
    if ($len -gt 65535) { $len = 65535 }
    $Stream.WriteByte([byte][math]::Floor($len / 256))
    $Stream.WriteByte([byte]($len -band 255))
    if ($len -gt 0) { $Stream.Write($bytes, 0, $len) }
    $Stream.Flush()
}

function Invoke-PipeHandshake {
    param(
        [System.IO.Pipes.NamedPipeClientStream]$Stream,
        [string]$PipeLabel
    )
    $first = Read-PipeString $Stream
    if ($first -ne $HandshakeCode) {
        return @{ Ok = $false; Detail = "$PipeLabel handshake step 1 failed (got '$first')" }
    }
    Write-PipeString $Stream $HandshakeCode
    $second = Read-PipeString $Stream
    if ($second -ne "Connected") {
        return @{ Ok = $false; Detail = "$PipeLabel handshake step 2 failed (got '$second')" }
    }
    return @{ Ok = $true; Detail = "" }
}

function Start-FlimagePipeServer {
    if (-not (Test-Path $ComInitDir)) {
        New-Item -ItemType Directory -Path $ComInitDir -Force | Out-Null
    }
    Set-Content -Path $ComInitFile -Value "PIPE" -Encoding ascii -NoNewline
}

function Close-FlimagePipes {
    foreach ($pipe in @($script:PipeW, $script:PipeR)) {
        if ($null -eq $pipe) { continue }
        try {
            if ($pipe.IsConnected) { $pipe.Close() }
            $pipe.Dispose()
        }
        catch { }
    }
    $script:PipeW = $null
    $script:PipeR = $null
    $script:PipeConnected = $false
}

function Reset-FlimageRemoteControl {
    param([scriptblock]$Log = $null)
    if ($null -ne $Log) { & $Log "Closing Remote control & script, then restarting PIPE server..." }
    $closed = [RemoteControlCloser]::CloseRemoteControlWindow()
    if ($null -ne $Log) {
        if ($closed) { & $Log "Remote control window close message sent." }
        else { & $Log "Remote control window not found." }
    }
    Start-Sleep -Milliseconds 150
    Start-FlimagePipeServer
    Start-Sleep -Milliseconds 250
}

function Connect-FlimagePipesOnce {
    param([scriptblock]$Log = $null)

    Close-FlimagePipes
    Start-FlimagePipeServer
    Start-Sleep -Milliseconds 150

    $pipeR = New-Object System.IO.Pipes.NamedPipeClientStream ".", $ReadPipeName, ([System.IO.Pipes.PipeDirection]::InOut)
    $pipeW = New-Object System.IO.Pipes.NamedPipeClientStream ".", $WritePipeName, ([System.IO.Pipes.PipeDirection]::InOut)

    try {
        $pipeR.Connect($ConnectTimeoutMs)
        $pipeW.Connect($ConnectTimeoutMs)
    }
    catch {
        if ($null -ne $Log) { & $Log "!! Pipe connect failed: $($_.Exception.Message)" }
        $pipeR.Dispose()
        $pipeW.Dispose()
        return $false
    }

    $hsR = Invoke-PipeHandshake -Stream $pipeR -PipeLabel $ReadPipeName
    if (-not $hsR.Ok) {
        if ($null -ne $Log) { & $Log $hsR.Detail }
        $pipeR.Dispose(); $pipeW.Dispose()
        return $false
    }

    $hsW = Invoke-PipeHandshake -Stream $pipeW -PipeLabel $WritePipeName
    if (-not $hsW.Ok) {
        if ($null -ne $Log) { & $Log $hsW.Detail }
        $pipeR.Dispose(); $pipeW.Dispose()
        return $false
    }

    $script:PipeR = $pipeR
    $script:PipeW = $pipeW
    $script:PipeConnected = $true
    return $true
}

function Connect-FlimagePipes {
    param([scriptblock]$Log = $null)

    if (Connect-FlimagePipesOnce -Log $Log) { return $true }

    for ($attempt = 2; $attempt -le $ConnectRetries; $attempt++) {
        if ($null -ne $Log) { & $Log "Connect attempt $attempt/$ConnectRetries" }
        Reset-FlimageRemoteControl -Log $Log
        if (Connect-FlimagePipesOnce -Log $Log) { return $true }
    }
    return $false
}

function Send-FlimagePipeCommand {
    param([string]$Command)
    if (-not $script:PipeConnected -or $null -eq $script:PipeW) {
        throw "Not connected to FLIMage pipe"
    }
    [System.Threading.Monitor]::Enter($script:PipeLock)
    try {
        Write-PipeString $script:PipeW $Command
        $reply = Read-PipeString $script:PipeW
        if ($null -eq $reply) {
            $script:PipeConnected = $false
            throw "Connection problem"
        }
        return $reply
    }
    finally {
        [System.Threading.Monitor]::Exit($script:PipeLock)
    }
}


function Get-OlympusReplyPayload {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return '' }
    if ($Reply -match '^OlympusReply,\s*(.+)$') {
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
        $tokenPattern = ('(?i)' + [regex]::Escape($CommandToken.Trim()) + '\s+([+-]?\d+)')
        if ($payload -match $tokenPattern) {
            return [int]$Matches[1]
        }
    }

    # IX2 replies such as "1OB 4" or "1MU +004" — position is the last numeric field.
    $numberMatches = [regex]::Matches($payload, '[+-]?\d+')
    if ($numberMatches.Count -gt 0) {
        return [int]$numberMatches[$numberMatches.Count - 1].Value
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

function Get-OlympusEscapeCommand {
    if ($null -ne $script:OlympusEscapeCommand) {
        return [string]$script:OlympusEscapeCommand
    }
    return ''
}

function Get-OlympusTurretWaitMs {
    if ($null -ne $script:OlympusTurretWaitMs) {
        return [int]$script:OlympusTurretWaitMs
    }
    return 500
}

function Build-OlympusTurretSetSequence {
    param([string]$Ix2SetCommand)
    $escapeCmd = Get-OlympusEscapeCommand
    $steps = @()
    if (-not [string]::IsNullOrWhiteSpace($escapeCmd)) {
        $steps += $escapeCmd.Trim()
    }
    $steps += $Ix2SetCommand.Trim()
    if (-not [string]::IsNullOrWhiteSpace($escapeCmd)) {
        $steps += $escapeCmd.Trim()
    }
    return ,$steps
}

function Send-OlympusIx2WithLog {
    param([string]$Ix2Command)
    $ix2 = $Ix2Command.Trim()
    if ($ix2 -eq '') { throw 'Empty IX2-UCB command' }
    $pipeCmd = "SendOlympusCommand, $ix2"
    Add-Log ">> $pipeCmd"
    $reply = Send-FlimagePipeCommand -Command $pipeCmd
    Add-Log "<< $reply"
    if ($reply -match '(?i)^Error') {
        throw $reply
    }
    Update-OlympusLabelsFromPipeExchange -Command $pipeCmd -Reply $reply
    return $reply
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

    $signed = Get-RoeEncoderTickSign -Direction $Direction
    if ($signed -eq 0) { return }
    if (-not $script:RoeEncTicks.ContainsKey($Axis)) {
        $script:RoeEncTicks[$Axis] = 0
    }
    $script:RoeEncTicks[$Axis] += $signed
}

function Build-RoeMoveDeltasFromAccumulatedTicks {
    $preset = $script:Presets[$script:ActivePreset]
    $dx = 0.0
    $dy = 0.0
    $dz = 0.0

    foreach ($axis in @($script:RoeEncTicks.Keys)) {
        $ticks = [int]$script:RoeEncTicks[$axis]
        if ($ticks -eq 0) { continue }
        switch ($axis) {
            'X' { $dx += $preset.StepXY * $ticks }
            'Y' { $dy += $preset.StepXY * $ticks }
            'Z' { $dz += $preset.StepZ * $ticks }
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
                @(',', 'Finer speed/step preset'),
                @('.', 'Coarser speed/step preset')
            )
            Notes = "Capture ON: keys work when another window is focused.`r`nRelease all jog keys: position refresh after 0.5 s idle.`r`nASI / ZoZoLab presets = speed (mm/s); others = step (um).`r`nOlympus: Light path (1PRISM) in GUI  -  1=eyepiece, 2=camera."
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
            @('NumPad /', 'Finer speed/step preset'),
            @('NumPad *', 'Coarser speed/step preset')
        )
        Notes = "Numpad ON: keys work when another window is focused.`r`nRelease all jog keys: position refresh after 0.5 s idle.`r`nASI / ZoZoLab presets = speed (mm/s); others = step (um).`r`nOlympus: Light path (1PRISM) in GUI  -  1=eyepiece, 2=camera."
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
    if ($script:MotorUsesVectorJog) {
        Invoke-MotorVectorStop -Quiet
    }
    Stop-AbsPositionIdleTimer
}

function Stop-AbsPositionIdleTimer {
    if ($null -ne $absPositionIdleTimer) { $absPositionIdleTimer.Stop() }
}

function Restart-AbsPositionIdleTimer {
    Stop-AbsPositionIdleTimer
    if (Test-AnyJogKeyDown) { return }
    if (-not $script:NumpadJogEnabled) { return }
    if ($null -ne $absPositionIdleTimer) { $absPositionIdleTimer.Start() }
}

function Invoke-AbsPositionRefreshQuiet {
    if ($script:Busy) { return }

    try {
        if (-not $script:PipeConnected) { return }
        $reply = Send-FlimagePipeCommand -Command "GetCurrentPosition"
        Update-MotorPosLabel $reply
        Add-Log ("<< Abs XYZ (idle): {0}" -f $reply)
    }
    catch {
        Add-Log "!! Abs XYZ idle failed: $($_.Exception.Message)"
    }
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

    Stop-AbsPositionIdleTimer

    if ($KeyName -eq 'PresetFiner' -or $KeyName -eq 'PresetCoarser') {
        if ($isDownEvent) {
            if ($script:PresetKeysDown[$KeyName]) { return }
            $script:PresetKeysDown[$KeyName] = $true
            $direction = if ($KeyName -eq 'PresetFiner') { -1 } else { 1 }
            [void](Switch-StepPreset -Direction $direction)
        }
        else {
            $script:PresetKeysDown[$KeyName] = $false
            Restart-AbsPositionIdleTimer
        }
        return
    }

    if ($isDownEvent) {
        if ($script:JogKeysDown[$KeyName]) { return }
        $script:JogKeysDown[$KeyName] = $true
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
        Restart-AbsPositionIdleTimer
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
    Add-Log ("Step preset: {0}" -f $script:ActivePreset)
    if ($script:MotorUsesVectorJog -and (Test-AnyJogKeyDown)) {
        Start-HeldJogIfNeeded
    }
    return $true
}

Apply-GuiSettings (Import-GuiSettings)

$form = New-Object System.Windows.Forms.Form
$form.Text = "FLIMage Motor Pipe GUI"
$form.Size = New-Object System.Drawing.Size ($GuiInnerWidth + ($GuiMargin * 2) + 16), 866
$form.MinimumSize = New-Object System.Drawing.Size ($GuiInnerWidth + ($GuiMargin * 2) + 16), 826
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
$logGroup.Location = New-Object System.Drawing.Point $GuiMargin, 608
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

function Get-ActiveStepValues {
    $preset = $script:Presets[$script:ActivePreset]
    return @{
        StepXY = [double]$preset.StepXY
        StepZ  = [double]$preset.StepZ
    }
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
    $speeds = Get-ActiveJogSpeeds
    if ($script:MotorUsesVectorJog) {
        $stepDisplayLabel.Text = ("XY={0:F3} Z={1:F3} mm/s" -f $speeds.SpeedXY, $speeds.SpeedZ)
    }
    else {
        $stepDisplayLabel.Text = ("XY={0:F2} Z={1:F2} um" -f $speeds.SpeedXY, $speeds.SpeedZ)
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
    $reason = if ($Enabled) { "" } else { "no Olympus ($MotorHwName)" }
    $filterGroup.Text = if ($Enabled) { "Filter (1MU)" } else { "Filter (1MU) - off" }
    $objGroup.Text = if ($Enabled) { "Objective (1OB)" } else { "Objective (1OB) - off" }
    $prismGroup.Text = if ($Enabled) { "Light path (1PRISM)" } else { "Light path (1PRISM) - off" }

    $controls = @(
        $filterCombo, $btnFilterGet, $btnFilterSet, $btnFilterLabels,
        $objCombo, $btnObjGet, $btnObjSet, $btnObjLabels,
        $prismRadioEyepiece, $prismRadioCamera, $btnPrismGet, $btnPrismSet
    )
    foreach ($ctrl in $controls) {
        if ($null -ne $ctrl) {
            $ctrl.Enabled = $Enabled -and (-not $script:Busy)
        }
    }
}

function Show-StepPresetEditor {
    param([string]$PresetName)

    $script:PresetDialogOpen = $true
    Stop-AllHeldJogKeys

    $preset = $script:Presets[$PresetName]
    $isVector = $script:MotorUsesVectorJog
    $dlg = New-Object System.Windows.Forms.Form
    if ($isVector) {
        $dlg.Text = "Set $PresetName speed (ASI)"
    }
    else {
        $dlg.Text = "Set $PresetName step"
    }
    $dlg.FormBorderStyle = 'FixedDialog'
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.StartPosition = 'CenterParent'
    $dlg.ClientSize = New-Object System.Drawing.Size 260, 130
    $dlg.KeyPreview = $true

    $xyLabel = New-Object System.Windows.Forms.Label
    if ($isVector) {
        $xyLabel.Text = "XY speed (mm/s):"
        $xyNumeric = New-Object System.Windows.Forms.NumericUpDown
        $xyNumeric.DecimalPlaces = 3
        $xyNumeric.Increment = 0.005
        $xyNumeric.Minimum = 0.001
        $xyNumeric.Maximum = 2.0
        $xyNumeric.Value = [decimal]$preset.VelXY
    }
    else {
        $xyLabel.Text = "Step XY (um):"
        $xyNumeric = New-Object System.Windows.Forms.NumericUpDown
        $xyNumeric.DecimalPlaces = 2
        $xyNumeric.Increment = 0.1
        $xyNumeric.Minimum = 0.01
        $xyNumeric.Maximum = 1000
        $xyNumeric.Value = [decimal]$preset.StepXY
    }
    $xyLabel.Location = New-Object System.Drawing.Point 12, 18
    $xyLabel.AutoSize = $true
    $dlg.Controls.Add($xyLabel)

    $xyNumeric.Location = New-Object System.Drawing.Point 110, 16
    $xyNumeric.Width = 90
    $dlg.Controls.Add($xyNumeric)

    $zLabel = New-Object System.Windows.Forms.Label
    if ($isVector) {
        $zLabel.Text = "Z speed (mm/s):"
    }
    else {
        $zLabel.Text = "Step Z (um):"
    }
    $zLabel.Location = New-Object System.Drawing.Point 12, 52
    $zLabel.AutoSize = $true
    $dlg.Controls.Add($zLabel)

    $zNumeric = New-Object System.Windows.Forms.NumericUpDown
    if ($isVector) {
        $zNumeric.DecimalPlaces = 3
        $zNumeric.Increment = 0.005
        $zNumeric.Minimum = 0.001
        $zNumeric.Maximum = 2.0
        $zNumeric.Value = [decimal]$preset.VelZ
    }
    else {
        $zNumeric.DecimalPlaces = 2
        $zNumeric.Increment = 0.1
        $zNumeric.Minimum = 0.01
        $zNumeric.Maximum = 1000
        $zNumeric.Value = [decimal]$preset.StepZ
    }
    $zNumeric.Location = New-Object System.Drawing.Point 110, 50
    $zNumeric.Width = 90
    $dlg.Controls.Add($zNumeric)

    $noteLabel = New-Object System.Windows.Forms.Label
    $noteLabel.Text = "Numpad jog disabled while open."
    $noteLabel.Location = New-Object System.Drawing.Point 12, 78
    $noteLabel.AutoSize = $true
    $noteLabel.ForeColor = [System.Drawing.Color]::DimGray
    $dlg.Controls.Add($noteLabel)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = "OK"
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $btnOk.Location = New-Object System.Drawing.Point 92, 98
    $dlg.Controls.Add($btnOk)
    $dlg.AcceptButton = $btnOk

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = "Cancel"
    $btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $btnCancel.Location = New-Object System.Drawing.Point 173, 98
    $dlg.Controls.Add($btnCancel)
    $dlg.CancelButton = $btnCancel

    $result = $dlg.ShowDialog($form)
    $script:PresetDialogOpen = $false

    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
        if ($isVector) {
            $script:Presets[$PresetName].VelXY = [double]$xyNumeric.Value
            $script:Presets[$PresetName].VelZ = [double]$zNumeric.Value
            Update-StepPresetDisplay
            Export-GuiSettings
            Add-Log ("Updated {0}: XY={1:F3} Z={2:F3} mm/s" -f $PresetName, $preset.VelXY, $preset.VelZ)
            if (Test-AnyJogKeyDown) { Start-HeldJogIfNeeded }
        }
        else {
            $script:Presets[$PresetName].StepXY = [double]$xyNumeric.Value
            $script:Presets[$PresetName].StepZ = [double]$zNumeric.Value
            Update-StepPresetDisplay
            Export-GuiSettings
            Add-Log ("Updated {0}: XY={1:F2}, Z={2:F2} um" -f $PresetName, $preset.StepXY, $preset.StepZ)
        }
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

function Get-MotorPositionFromReply {
    param([string]$Reply)
    if ([string]::IsNullOrWhiteSpace($Reply)) { return $null }
    foreach ($token in @(
            "MoveMotorRelativeQueued", "MoveMotorRelativeDone", "CurrentPosition",
            "SetMotorPositionDone", "MotorVectorStopped")) {
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

function Update-MotorPosLabel {
    param([string]$Reply)
    $pos = Get-MotorPositionFromReply -Reply $Reply
    if ($null -ne $pos) {
        $motorPosLabel.Text = ("Pos (um)`r`nX={0:F1}  Y={1:F1}  Z={2:F1}" -f $pos[0], $pos[1], $pos[2])
    }
    elseif ($Reply -match '(?i)^Error') {
        $motorPosLabel.Text = "Pos: ERR"
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
        Update-MotorPosLabel $reply
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
        if (-not $Quiet) { Add-Log ">> $cmd" }
        $reply = Send-FlimagePipeCommand -Command $cmd
        if (-not $Quiet) { Add-Log "<< $reply" }
        Update-MotorPosLabel $reply
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
        $prismRadioEyepiece, $prismRadioCamera, $btnPrismGet, $btnPrismSet
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

$motorGroup = New-Object System.Windows.Forms.GroupBox
$motorGroup.Text = "Motor jog"
$motorGroup.Location = New-Object System.Drawing.Point $GuiMargin, 296
$motorGroup.Size = New-Object System.Drawing.Size $GuiInnerWidth, 200
$form.Controls.Add($motorGroup)

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
            return
        }
        Update-HeldJogVector
        Invoke-MotorJogQuick -DeltaX $script:HeldJogDx -DeltaY $script:HeldJogDy -DeltaZ $script:HeldJogDz -Quiet
    })

$absPositionIdleTimer = New-Object System.Windows.Forms.Timer
$absPositionIdleTimer.Interval = 500
$absPositionIdleTimer.Add_Tick({
        Stop-AbsPositionIdleTimer
        if (Test-AnyJogKeyDown) { return }
        if (-not (Test-CanAcceptNumpadJog)) { return }
        Invoke-AbsPositionRefreshQuiet
    })

$motorPosLabel = New-Object System.Windows.Forms.Label
$motorPosLabel.Text = "Pos: --"
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
$presetLabel.Text = "Step:"
$presetLabel.Location = New-Object System.Drawing.Point 8, 54
$presetLabel.AutoSize = $true
$motorGroup.Controls.Add($presetLabel)

$presetRadioButtons = @{}
$presetX = 48
foreach ($presetName in $PresetNames) {
    $radio = New-Object System.Windows.Forms.RadioButton
    $radio.Text = $presetName
    $radio.AutoSize = $true
    $radio.Location = New-Object System.Drawing.Point $presetX, 52
    $radio.Tag = $presetName
    $motorGroup.Controls.Add($radio)
    $presetRadioButtons[$presetName] = $radio
    $presetX += 54
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
$btnSetPreset.Text = "Set"
$btnSetPreset.Location = New-Object System.Drawing.Point 210, 50
$btnSetPreset.Width = 36
$motorGroup.Controls.Add($btnSetPreset)

$stepDisplayLabel = New-Object System.Windows.Forms.Label
$stepDisplayLabel.Text = "XY=-- Z=--"
$stepDisplayLabel.Location = New-Object System.Drawing.Point 8, 76
$stepDisplayLabel.Size = New-Object System.Drawing.Size 340, 18
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
$cmdGroup.Location = New-Object System.Drawing.Point $GuiMargin, 504
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

function Update-FilterLabel {
    param([string]$Reply)
    $value = Get-TurretValue -Reply $Reply -Token "FilterTurret"
    if ($null -eq $value) { $value = Get-OlympusIx2PositionValue -Reply $Reply -CommandToken '1MU' }
    if ($null -ne $value) {
        $filterPosLabel.Text = Format-TurretCurrentText -Position $value
        Sync-TurretComboToPosition -Combo $filterCombo -Position $value
    }
    elseif ($Reply -match '(?i)^Error') { $filterPosLabel.Text = "Cur: ERR" }
}

function Update-ObjectiveLabel {
    param([string]$Reply)
    $value = Get-TurretValue -Reply $Reply -Token "ObjectiveTurret"
    if ($null -eq $value) { $value = Get-OlympusIx2PositionValue -Reply $Reply -CommandToken '1OB' }
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
Set-OlympusTurretControlsEnabled -Enabled $false -MotorHwName "not connected"

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

                Add-Log ">> GetMotorVelocity"
                $velReply = Send-FlimagePipeCommand -Command "GetMotorVelocity"
                Add-Log "<< $velReply (saved vel kept until Get)"

                if ($hasOlympus) {
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
                $done = Get-OlympusIx2PositionValue -Reply $r -CommandToken '1MU'
                if ($null -ne $done) {
                    $filterPosLabel.Text = Format-TurretCurrentText -Position $done
                    Sync-TurretComboToPosition -Combo $filterCombo -Position $done
                }
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
            Send-OlympusIx2SequenceWithLog -Ix2Commands (Build-OlympusTurretSetSequence -Ix2SetCommand ("1OB {0}" -f $value)) -OnLastReply {
                param($r)
                $done = Get-OlympusIx2PositionValue -Reply $r -CommandToken '1OB'
                if ($null -ne $done) {
                    $objPosLabel.Text = Format-TurretCurrentText -Position $done
                    Sync-TurretComboToPosition -Combo $objCombo -Position $done
                }
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
        Show-StepPresetEditor -PresetName $script:ActivePreset
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
Register-FocusClearOnClick $cmdGroup
Register-FocusClearOnClick $logGroup
Register-FocusClearOnClick $statusLabel

$form.Add_FormClosed({
        if ($null -ne $jogRepeatTimer) { $jogRepeatTimer.Stop() }
        if ($null -ne $absPositionIdleTimer) { $absPositionIdleTimer.Stop() }
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
