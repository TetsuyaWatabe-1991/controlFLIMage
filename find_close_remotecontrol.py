# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:37:38 2022

@author: yasudalab
"""
import win32gui
import win32con
# import pyautogui

def windowEnumerationHandler(hwnd, top_windows):
    top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

def window_exists(window_name):
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    window_name_list = [each_hwnd_name[1] for each_hwnd_name in top_windows]
    return window_name in window_name_list

def window_exists_startswith(window_name_header):
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    window_name_list = [each_hwnd_name[1] for each_hwnd_name in top_windows]
    # if window's name starts with window_name_header, return True
    return any(window_name.startswith(window_name_header) for window_name in window_name_list)

def close_remote_control():
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    
    hwnd = 123456789
    for i in top_windows:
        if 'Remote control & script' in i[1]:            
            hwnd = i[0]
    
    print("hwnd",hwnd)
    
    if hwnd!=123456789:    
        
        win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except:
            print("  Failed <- win32gui.SetForegroundWindow(hwnd) ")
            pass
        # oripos=pyautogui.position()
        # pyautogui.moveTo(right-30, top + 20)
        # pyautogui.click()
        # win32gui.SetWindowPos(hwnd,win32con.HWND_NOTOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        # pyautogui.moveTo(oripos)
        
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

        
def close_realtime_plot():
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    
    
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    
    for i in top_windows:
        if i[1]=='Realtime plot':
            hwnd = i[0]
            win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except:
                print("  Failed <- win32gui.SetForegroundWindow(hwnd) ")
                pass
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    
if __name__ == '__main__':
    # if False:
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    
    for i in top_windows:
        if 'FLIM' in i[1]:
            print(i)