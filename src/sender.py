import base64
import numpy as np
import pika
import sys
import os
import json
import time
from config import config
import cv2


def send_frame(frame):
    start_time = time.time()
    url = os.environ.get("CLOUDAMQP_URL", f"amqp://admin:admin@{config.server_ip}:5672")
    params = pika.URLParameters(url)
    params.socket_timeout = 5
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue="q-3")
    send_frame = cv2.resize(frame, config.send_frame_reso)
    _, send_frame = cv2.imencode('.jpeg', send_frame)
    send_frame = send_frame.tobytes()
    image_byte = base64.b64encode(send_frame)
    data = {
        "ip": config.jetson_ip,
        "image": str(image_byte),
        "type": 1,
    }
    message = json.dumps(data)
    channel.basic_publish(exchange="", routing_key="hello", body=message)
    print('send frame time', time.time()-start_time, 's')


def send_feature(tracked_objects):
    for o in tracked_objects:
        #send ft
        pass
