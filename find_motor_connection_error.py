# -*- coding: utf-8 -*-
"""
Dismiss the "Motor connection problem!!" MessageBox that FLIMage shows on
startup when use_motor is on but the stage failed to connect
(FLIMageMain.cs, around the MotorCtrl(...) construction). That MessageBox is
modal on FLIMage's main thread, so it blocks the splash screen from ever
reaching the main window -- open_FLIMage() would otherwise just sit in its
"waiting for FLIMage to open..." loop forever.

Same detection style as find_yes_overwrite_warning.py: enumerate FLIMage's
own top-level windows, match the unnamed dialog by its Static child text,
then click its (only) button.
"""
import win32gui
import win32con
import win32process
import psutil

EXPECTED_TEXT = "Motor connection problem!!"

def find_flimage_process_ids():
    return {p.info['pid'] for p in psutil.process_iter(['pid', 'name']) if "FLIMage" in p.info['name']}

def get_flimage_windows():
    flimage_pids = find_flimage_process_ids()
    hwnds = []

    def callback(hwnd, _):
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid in flimage_pids:
            hwnds.append(hwnd)

    win32gui.EnumWindows(callback, None)
    return hwnds

def window_contains_expected_static_text(hwnd):
    matched = []

    def child_callback(child_hwnd, _):
        class_name = win32gui.GetClassName(child_hwnd)
        text = win32gui.GetWindowText(child_hwnd)
        if class_name == "Static" and text.strip() == EXPECTED_TEXT:
            matched.append(True)

    win32gui.EnumChildWindows(hwnd, child_callback, None)
    return bool(matched)

def click_ok_in_window(hwnd):
    # MessageBox.Show(string) with no custom buttons has exactly one button
    # (OK), so click whatever single Button child is there instead of
    # matching exact text (label/mnemonic can vary by locale).
    def child_callback(child_hwnd, _):
        class_name = win32gui.GetClassName(child_hwnd)
        if class_name == "Button":
            text = win32gui.GetWindowText(child_hwnd)
            print(f"Clicking {text!r} button: HWND={child_hwnd}")
            win32gui.SendMessage(child_hwnd, win32con.BM_CLICK, 0, 0)
    win32gui.EnumChildWindows(hwnd, child_callback, None)

def close_motor_connection_error():
    for hwnd in get_flimage_windows():
        if win32gui.GetWindowText(hwnd) == "":
            if window_contains_expected_static_text(hwnd):
                print("Matched motor connection error dialog.")
                click_ok_in_window(hwnd)
                return True
    return False


if __name__ == "__main__":
    success = close_motor_connection_error()
    print(success)
