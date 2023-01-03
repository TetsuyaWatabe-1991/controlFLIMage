# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:46:00 2023

@author: yasudalab
"""

import PySimpleGUI as sg
import time

class TextWindow():
    def __init__(self,Text = 'test'):        
        self.Text = 'test'
        self.breaknow= False
        self.maxcount = 50
        self.start()
        
    def start(self):
        layout = [[sg.Text(self.Text, 
                           key = 'TEXT',
                           font='Arial 30', size=(13, 2))]]
        self.window = sg.Window("Information", layout, finalize=True)
        
    def udpate(self,text):
        event, values = self.window.read(timeout=10)
        self.window['TEXT'].update(text)
        event, values = self.window.read(timeout=10)

    def close(self):
        event, values = self.window.read(timeout=10)
        self.window['TEXT'].update("CLOSE")
        event, values = self.window.read(timeout=200)
        self.window.close()
        
#### EXAMPLE
if __name__ == '__main__':
    TxtWind= TextWindow(Text = 'INFO')
    for i in range(5):
        TxtWind.udpate(str(i))
        print(i)
        time.sleep(2)
    TxtWind.close()



