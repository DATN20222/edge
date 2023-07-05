import serial
import json

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
            print(j)
    except KeyboardInterrupt:
        print("error")