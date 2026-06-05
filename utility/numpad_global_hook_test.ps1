# Minimal test GUI for global numpad hook + foreground-window handling.
# Usage:
#   .\numpad_global_hook_test.ps1
#   .\numpad_global_hook_test.ps1 -LogFile C:\temp\numpad_test.log -AutoExitSec 120

param(
    [string]$LogFile = "",
    [int]$AutoExitSec = 0
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public sealed class NumpadHookTestHook : IDisposable
{
    public event Action<string, int> NumpadEvent;
    public IntPtr GuiWindowHandle = IntPtr.Zero;

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

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    public NumpadHookTestHook()
    {
        _proc = HookCallback;
    }

    public static string DescribeWindow(IntPtr hwnd)
    {
        if (hwnd == IntPtr.Zero) { return "(none)"; }
        var sb = new StringBuilder(256);
        GetWindowText(hwnd, sb, sb.Capacity);
        string title = sb.ToString();
        if (string.IsNullOrWhiteSpace(title)) { title = "(no title)"; }
        return string.Format("0x{0:X} {1}", hwnd.ToInt64(), title);
    }

    public bool IsGuiForeground()
    {
        if (GuiWindowHandle == IntPtr.Zero) { return false; }
        IntPtr fg = GetForegroundWindow();
        if (fg == GuiWindowHandle) { return true; }
        return IsChild(GuiWindowHandle, fg);
    }

    public string ForegroundDescription()
    {
        return DescribeWindow(GetForegroundWindow());
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

    private static string MapVirtualKey(int vkCode)
    {
        switch (vkCode)
        {
            case 0x62: return "NumPad2";
            case 0x64: return "NumPad4";
            case 0x66: return "NumPad6";
            case 0x68: return "NumPad8";
            case 0x6B: return "Add";
            case 0x6D: return "Subtract";
            case 0x6F: return "Divide";
            case 0x6A: return "Multiply";
            default: return null;
        }
    }

    private IntPtr HookCallback(int nCode, IntPtr wParam, IntPtr lParam)
    {
        if (nCode >= 0)
        {
            int vkCode = Marshal.ReadInt32(lParam);
            string keyName = MapVirtualKey(vkCode);
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

$script:Hook = $null
$script:HookEnabled = $true
$script:FormKeyCount = 0
$script:HookKeyCount = 0

function Write-TestLog {
    param([string]$Line)
    $stamp = (Get-Date).ToString("HH:mm:ss.fff")
    $text = "[$stamp] $Line"
    if ($null -ne $logBox) {
        $logBox.AppendText("$text`r`n")
        $logBox.SelectionStart = $logBox.Text.Length
        $logBox.ScrollToCaret()
    }
    if ($LogFile) {
        Add-Content -Path $LogFile -Value $text -Encoding UTF8
    }
    [System.Windows.Forms.Application]::DoEvents()
}

function Update-ForegroundLabel {
    if ($null -eq $script:Hook) { return }
    $fgLabel.Text = "Foreground: $($script:Hook.ForegroundDescription())"
    $guiFgLabel.Text = "GUI is foreground: $($script:Hook.IsGuiForeground())"
}

function Invoke-HookKeyEvent {
    param(
        [string]$KeyName,
        [int]$IsDownFlag
    )
    $down = ($IsDownFlag -ne 0)
    $script:HookKeyCount++
    Write-TestLog ("HOOK {0} {1} (fg={2})" -f $KeyName, ($(if ($down) { 'down' } else { 'up' })), $script:Hook.ForegroundDescription())
}

function Start-Hook {
    if ($null -eq $script:Hook) {
        $script:Hook = New-Object NumpadHookTestHook
        $script:Hook.add_NumpadEvent({
                param([string]$keyName, [int]$isDownFlag)
                if ($form.IsDisposed) { return }
                if ([string]::IsNullOrWhiteSpace($keyName)) { return }
                $kCopy = [string]$keyName
                $dCopy = [int]$isDownFlag
                $form.BeginInvoke([System.Action]{
                        Invoke-HookKeyEvent -KeyName $kCopy -IsDownFlag $dCopy
                        Update-ForegroundLabel
                    }.GetNewClosure())
            })
    }
    $script:Hook.GuiWindowHandle = $form.Handle
    if ($script:HookEnabled) { $script:Hook.Start() }
    else { $script:Hook.Stop() }
    Update-ForegroundLabel
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "Numpad global hook TEST"
$form.Size = New-Object System.Drawing.Size 520, 420
$form.StartPosition = "CenterScreen"
$form.KeyPreview = $true

$infoLabel = New-Object System.Windows.Forms.Label
$infoLabel.Location = New-Object System.Drawing.Point 12, 12
$infoLabel.Size = New-Object System.Drawing.Size 480, 36
$infoLabel.Text = "Focus another window (Notepad), press numpad. Hook should log HOOK lines."
$form.Controls.Add($infoLabel)

$fgLabel = New-Object System.Windows.Forms.Label
$fgLabel.Location = New-Object System.Drawing.Point 12, 50
$fgLabel.Size = New-Object System.Drawing.Size 480, 18
$fgLabel.Text = "Foreground: --"
$form.Controls.Add($fgLabel)

$guiFgLabel = New-Object System.Windows.Forms.Label
$guiFgLabel.Location = New-Object System.Drawing.Point 12, 72
$guiFgLabel.Size = New-Object System.Drawing.Size 480, 18
$guiFgLabel.Text = "GUI is foreground: --"
$form.Controls.Add($guiFgLabel)

$countLabel = New-Object System.Windows.Forms.Label
$countLabel.Location = New-Object System.Drawing.Point 12, 94
$countLabel.Size = New-Object System.Drawing.Size 480, 18
$countLabel.Text = "Form keys: 0   Hook keys: 0"
$form.Controls.Add($countLabel)

$btnFocusNotepad = New-Object System.Windows.Forms.Button
$btnFocusNotepad.Text = "Focus Notepad"
$btnFocusNotepad.Location = New-Object System.Drawing.Point 12, 120
$btnFocusNotepad.Width = 100
$form.Controls.Add($btnFocusNotepad)

$btnFocusSelf = New-Object System.Windows.Forms.Button
$btnFocusSelf.Text = "Focus this GUI"
$btnFocusSelf.Location = New-Object System.Drawing.Point 118, 120
$btnFocusSelf.Width = 100
$form.Controls.Add($btnFocusSelf)

$btnToggleHook = New-Object System.Windows.Forms.Button
$btnToggleHook.Text = "Hook: ON"
$btnToggleHook.Location = New-Object System.Drawing.Point 224, 120
$btnToggleHook.Width = 80
$btnToggleHook.BackColor = [System.Drawing.Color]::LightGreen
$form.Controls.Add($btnToggleHook)

$btnClear = New-Object System.Windows.Forms.Button
$btnClear.Text = "Clear log"
$btnClear.Location = New-Object System.Drawing.Point 310, 120
$btnClear.Width = 80
$form.Controls.Add($btnClear)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Vertical"
$logBox.Font = New-Object System.Drawing.Font "Consolas", 9
$logBox.Location = New-Object System.Drawing.Point 12, 154
$logBox.Size = New-Object System.Drawing.Size 480, 210
$form.Controls.Add($logBox)

$fgTimer = New-Object System.Windows.Forms.Timer
$fgTimer.Interval = 500
$fgTimer.Add_Tick({ Update-ForegroundLabel })

function Update-CountLabel {
    $countLabel.Text = "Form keys: $script:FormKeyCount   Hook keys: $script:HookKeyCount"
}

$btnFocusNotepad.Add_Click({
        $np = Get-Process notepad -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $np) {
            Start-Process notepad | Out-Null
            Start-Sleep -Milliseconds 400
            $np = Get-Process notepad -ErrorAction SilentlyContinue | Select-Object -First 1
        }
        if ($null -ne $np) {
            Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class WinFocus {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@
            [WinFocus]::ShowWindow($np.MainWindowHandle, 9) | Out-Null
            [WinFocus]::SetForegroundWindow($np.MainWindowHandle) | Out-Null
            Write-TestLog "Focused Notepad"
        }
        Update-ForegroundLabel
    })

$btnFocusSelf.Add_Click({
        $form.Activate()
        [void]$form.Focus()
        Write-TestLog "Focused test GUI"
        Update-ForegroundLabel
    })

$btnToggleHook.Add_Click({
        $script:HookEnabled = -not $script:HookEnabled
        if ($script:HookEnabled) {
            $btnToggleHook.Text = "Hook: ON"
            $btnToggleHook.BackColor = [System.Drawing.Color]::LightGreen
            $script:Hook.Start()
        }
        else {
            $btnToggleHook.Text = "Hook: OFF"
            $btnToggleHook.BackColor = [System.Drawing.Color]::MistyRose
            $script:Hook.Stop()
        }
        Write-TestLog ("Hook {0}" -f ($(if ($script:HookEnabled) { 'enabled' } else { 'disabled' })))
    })

$btnClear.Add_Click({ $logBox.Clear() })

$form.Add_Load({
        if ($LogFile) {
            Set-Content -Path $LogFile -Value "# numpad hook test log" -Encoding UTF8
        }
        Start-Hook
        $fgTimer.Start()
        Write-TestLog "Ready. GUI handle=$($form.Handle)"
    })

$form.Add_KeyDown({
        switch ($_.KeyCode) {
            ([System.Windows.Forms.Keys]::NumPad2) { $name = 'NumPad2' }
            ([System.Windows.Forms.Keys]::NumPad4) { $name = 'NumPad4' }
            ([System.Windows.Forms.Keys]::NumPad6) { $name = 'NumPad6' }
            ([System.Windows.Forms.Keys]::NumPad8) { $name = 'NumPad8' }
            ([System.Windows.Forms.Keys]::Add) { $name = 'Add' }
            ([System.Windows.Forms.Keys]::Subtract) { $name = 'Subtract' }
            ([System.Windows.Forms.Keys]::Divide) { $name = 'Divide' }
            ([System.Windows.Forms.Keys]::Multiply) { $name = 'Multiply' }
            default { return }
        }
        $_.SuppressKeyPress = $true
        $script:FormKeyCount++
        Write-TestLog ("FORM {0} down" -f $name)
        Update-CountLabel
    })

$form.Add_FormClosed({
        $fgTimer.Stop()
        if ($null -ne $script:Hook) {
            $script:Hook.Dispose()
            $script:Hook = $null
        }
    })

if ($AutoExitSec -gt 0) {
    $exitTimer = New-Object System.Windows.Forms.Timer
    $exitTimer.Interval = ($AutoExitSec * 1000)
    $exitTimer.Add_Tick({
            $exitTimer.Stop()
            $form.Close()
        })
    $form.Add_Load({ $exitTimer.Start() })
}

[void]$form.ShowDialog()
