# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:28:47 2023

@author: yasudalab
"""

import tkinter as tk
import time
import threading

class TextWindow(threading.Thread):
    def __init__(self):
        self.running=False
        # self.run()
        
    # def callback(self):
    #     self.root.destroy()
        
    # def close(self):
    #     print("CLOSE TextWindow")
    #     self.running=False
    #     self.callback()
      
    # def run(self):
    #     pass
    
    def run(self):
        # self.root = tk.Tk()
        # self.root.geometry("200x100")
        # self.root.protocol("WM_DELETE_WINDOW", self.close)
        # self.textvar=tk.StringVar()
        # self.textvar.set("INFO")
        # self.TextShow = tk.Label(self.root, width=15, font=("Arial",18,""),
        #                           textvariable=self.textvar)
        # self.TextShow.pack(side="top")
        pass
    # def udpate(self,text):
    #     print(text)
    def udpate(self,text):
        # if self.running==False:
            # print("No window for countdown")
        print(text)
        return

        
        
class TextWindow_original(threading.Thread):
    def __init__(self):
        self.running=True
        self.run()
        
    def callback(self):
        self.root.destroy()
        
    def close(self):
        print("CLOSE TextWindow")
        self.running=False
        self.callback()
      
    # def run(self):
    #     pass
    
    def run(self):
        self.root = tk.Tk()
        self.root.geometry("200x100")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.textvar=tk.StringVar()
        self.textvar.set("INFO")
        self.TextShow = tk.Label(self.root, width=15, font=("Arial",18,""),
                                  textvariable=self.textvar)
        self.TextShow.pack(side="top")
    # def udpate(self,text):
    #     print(text)
    def udpate(self,text):
        if self.running==False:
            print("No window for countdown")
            print(text)
            return
        
        self.textvar.set(text)
        self.root.update()
        
        
#### EXAMPLE
if __name__ == '__main__':
    TxtWind= TextWindow()
    time.sleep(1)
    for i in range(5):
        TxtWind.udpate(str(i))
        print(i)
        time.sleep(2)
    TxtWind.close()
