import base64
import numpy as np
import sys
import os
import json
import time
from config import config
import pika
from utils.general import LOGGER

def iouArea(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    area1 = (xmax1-xmin1)*(ymax1-ymin1)
    area2 = (xmax2-xmin2)*(ymax2-ymin2)
    smaller_area = min(area1, area2)
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    if xmin>xmax or ymin>ymax:
        return 0
    else:
        return (xmax-xmin)*(ymax-ymin)/smaller_area

def check_position(human_bbox, ori_shape, predefined_bbox=[0.35, 0, 0.65, 0.9], threshold=0.5): #can set predefined box
    h, w = ori_shape
    human_bbox = [human_bbox[0]*w, human_bbox[1]*h, human_bbox[2]*w, human_bbox[3]*h]
    predefined_bbox = [predefined_bbox[0]*w, predefined_bbox[1]*h, predefined_bbox[2]*w, predefined_bbox[3]*h]
    return True if iouArea(human_bbox, predefined_bbox) > threshold else False
    
def sendDoor(tracked_objects, number, ori_shape):
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
            if check_position(np.array(o.last_detection.data), ori_shape):
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
                break

    connection.close()
    return haveEmbedding
