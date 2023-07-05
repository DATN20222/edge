import pickle
import struct
import cv2
import numpy as np
import time
from datetime import timedelta, datetime

import socket
import _thread
import serial
import json
from config import config
import requests

ip="192.168.55.1"
rabbitmq="192.168.0.106" #ip server chạy rabbitmq

def ReadData(nameThread):
    global humidity
    global temperature
    global gas

    ser = serial.Serial(
        port= '/dev/ttyACM0',
        baurate = 9600
    )
    try:
        while True:
            s = ser.readline()
            data = s.decode("utf-8")
            j = json.loads(data)
            humidity = j["humidity"]
            temperature = j["temperature"]
            gas = j["gas"]
            number = j["code"]
            # user = get_data('http://{0}:8800/accounts/bycode/{1}'.format(config.server_ip, number))
            user = requests.get("http://{0}:8800/accounts/bycode/{1}".format(config.server_ip, number))
            print(user.json())
            ser.write("{0}".format(user.name))
            
    except KeyboardInterrupt:
        print("error")

_thread.start_new_thread(ReadData, ("ReadData",))
# def get_data(self, api):
#         response = requests.get(f"{api}")
#         if response.status_code == 200:
#             print("sucessfully fetched the data")
#             self.formatted_print(response.json())
#         else:
#             print(f"Hello person, there's a {response.status_code} error with your request")