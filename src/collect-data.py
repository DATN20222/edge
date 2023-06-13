import pickle
import struct
import cv2
import numpy as np
import time
from datetime import timedelta, datetime
import socket

import serial
import json
import pika, sys, os
from datetime import datetime
import requests

ip="192.168.55.1"
rabbitmq="192.168.0.106"

def ReadData():

    ser = serial.Serial(
        port= '/dev/ttyACM0',
        baudrate = 115200
    )
    try:
        while True:
            s = ser.readline()
            data = s.decode("utf-8")
            # j = json.loads(data)
            # humidity = j["humidity"]
            # temperature = j["temperature"]
            # gas = j["gas"]
            # number = j["code"]
            # print(number)
            number = 1
            # user = get_data('http://{0}:8800/accounts/bycode/{1}'.format(config.server_ip, number))
            user = requests.get("http://{0}:8800/accounts/bycode/{1}".format("192.168.0.106", number))
            print(user.json())
            name = user.json()['name']
            ser.write(name.encode())
            break
            
    except KeyboardInterrupt:
        print("error")

ReadData()
