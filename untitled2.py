# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:50:33 2022

@author: yasudalab
"""


from datetime import datetime
from time import sleep

interval_sec=2
each_acquisition_from=datetime.now()
for i in range(1000):
    each_acquisition_len=(datetime.now()-each_acquisition_from)
    if each_acquisition_len.seconds>=interval_sec:
        print("each_acquisition_len",each_acquisition_len.seconds)
        print("each_acquisition_len",each_acquisition_len)
        break
    sleep(0.01)