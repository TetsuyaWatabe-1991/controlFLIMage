# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:46:15 2023

@author: yasudalab
"""

import requests

def line_notification(message = "notification",
                      token = 'yc9r4Djg76SGHv1y92njJF8slF2gprjHO4hF6zuMVyI'):
    payload = {'message' : message}
    r = requests.post('https://notify-api.line.me/api/notify'
                    , headers={'Authorization' : 'Bearer {}'.format(token)}
                    , params = payload)
    
    print("Sent message.")