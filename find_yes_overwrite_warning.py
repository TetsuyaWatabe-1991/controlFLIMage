import win32gui
import win32con
import win32process
import psutil

EXPECTED_TEXT = "File already exist! Do you want to overwrite?"

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

def click_yes_in_window(hwnd,find_yes=True):
    def child_callback(child_hwnd, _):
        text = win32gui.GetWindowText(child_hwnd)
        class_name = win32gui.GetClassName(child_hwnd)
        if class_name == "Button" and text.strip() == ("&Yes" if find_yes else "&No"):
            print(f"Clicking {text} button: HWND={child_hwnd}")
            win32gui.SendMessage(child_hwnd, win32con.BM_CLICK, 0, 0)
    win32gui.EnumChildWindows(hwnd, child_callback, None)

def close_overwrite_warning(find_yes=True):
    for hwnd in get_flimage_windows():
        if win32gui.GetWindowText(hwnd) == "":
            # print(f"\n[Checking unnamed window] HWND: {hwnd}")
            if window_contains_expected_static_text(hwnd):
                print("Matched expected static text.")
                click_yes_in_window(hwnd,find_yes)
                return True
            else:
                pass
    return False


if __name__ == "__main__":
    # Search all unnamed FLIMage windows for the correct static text and click '&Yes'
    success = close_overwrite_warning(find_yes=True)
    print(success)