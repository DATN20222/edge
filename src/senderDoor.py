import base64
import numpy as np
import sys
import os
import json
import time
from config import config
import pika
from utils.general import LOGGER

def sendDoor(tracked_objects, number):
    haveEmbedding = False
    # Create connection
    print('Creating connection...')
    url = os.environ.get("CLOUDAMQP_URL", f"amqp://admin:admin@{config.server_ip}:5672")
    params = pika.URLParameters(url)
    #params.socket_timeout = 5
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue="q-3")
    print('Connection established')
    LOGGER.info("Sender")
    start_time = time.time()
    for o in tracked_objects:
        if o.last_detection.embedding is not None:
            print(np.array(o.last_detection.data))
            posi = np.array(o.last_detection.data)

            # if(posi[0] > 0.25 and posi[1])
            data = {
                "ip": config.jetson_ip,
                "userId": o.id,
                "code": number,
                "position": base64.binascii.b2a_base64(np.array(o.last_detection.data)).decode("ascii"),
                "vector": base64.binascii.b2a_base64(o.last_detection.embedding).decode("ascii"),
                "type": 3,
            }
            message = json.dumps(data)
            channel.basic_publish(exchange="", routing_key="q-2", body=message) 
            haveEmbedding = True
            print('send features time', time.time()-start_time, 's')

    connection.close()
    return haveEmbedding