import torch
from utils import TryExcept
from utils.augmentations import letterbox
from utils.general import (LOGGER, ROOT, Profile, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, embedding_distance)
from utils.torch_utils import select_device
import numpy as np
from config import config
import cv2
import os
from norfair import Tracker, Video, draw_points, draw_tracked_objects, Detection
from norfair.filter import OptimizedKalmanFilterFactory
from models import DetectBackend, BodyFeatureExtractBackend
import time
from senderDoor import sendDoor
import pika
# import _thread
import serial
import json
import requests

# Load detection model
device = select_device(config.device)
model = DetectBackend(config.weights, device=device, data=config.classes, fp16=config.fp16)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, config.height, config.width))  # warmup

# Load extracting model
body_model = BodyFeatureExtractBackend(config.body_extract_model)
body_model.warmup()

count, dt = -1, (Profile(), Profile(), Profile(), Profile())
frame_time, ft_time = 0, 0
LOGGER.info('Creating Tracker...')
tracker = Tracker(
        initialization_delay=10,
        distance_function="euclidean",
        hit_counter_max=config.hit_counter_max,
        filter_factory=OptimizedKalmanFilterFactory(),
        distance_threshold=config.distance_threshold,
        past_detections_length=config.past_detections_length,
        reid_distance_function=embedding_distance,
        reid_distance_threshold=config.reid_distance_threshold,
        reid_hit_counter_max=config.reid_hit_counter_max,
        )


# def ReadData(nameThread):
#     while True:
#         try:   
#             s = ser.readline()
#             data = s.decode("utf-8")
#             j = json.loads(data)
#             print(j)   
#         except KeyboardInterrupt:
#             print("error")

# Read video input
cap = cv2.VideoCapture(config.source)
print("Test")
# cap = cv2.VideoCapture(0)
print('Camera Ready?', cap.isOpened())
if cap.isOpened() == False:
    os._exit(1)


# _thread.start_new_thread(ReadData, ("Read Data",))




# if config.draw:
#     video = Video(input_path=config.source, output_path='./out.mp4')

LOGGER.info('Start running...')
#load serial
# ser = serial.Serial(port= '/dev/ttyACM0', baudrate=115200)
# time.sleep(12)

while cap.isOpened():
    try:
        LOGGER.info('Read data')
        ser = serial.Serial(port= '/dev/ttyACM0', baudrate=115200)
        while True:
            s = ser.readline()
            LOGGER.info("Parse data")
            data = s.decode("utf-8")
            j = json.loads(data)
            number = j["code"]
            LOGGER.info("number")
            # number = 1
            user = requests.get("http://{0}:8800/accounts/bycode/{1}".format(config.server_ip, number))
            name = user.json()['name']
            LOGGER.info(name)
            break
        while True:
            start_time = time.time()
            ret, ori_im = cap.read()
            if ret == False:
                break
            count += 1
            if count % config.skip_period == 0:
            # Detection preprocess
                with dt[0]:
                    im = letterbox(ori_im, (config.height, config.width), stride=stride, auto=False)[0]
                    im = im[np.newaxis, ...]
                    im = im[..., ::-1].transpose((0, 3, 1, 2))
                    im = np.ascontiguousarray(im)
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Detection Inference
                with dt[1]:
                    pred = model(im)
                
                # NMS
                with dt[2]:
                    det = non_max_suppression(pred, config.conf_thres, config.iou_thres, 0, False, max_det=config.max_det)[0]

                s = ''
                s += '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    dect_ls = []
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], ori_im.shape).round()
                    with dt[3]:
                        for *xyxy, conf, cls in det:
                            xmin, ymin, xmax, ymax = xyxy
                            xmin, ymin, xmax, ymax = round(xmin.item()), round(ymin.item()), round(xmax.item()), round(ymax.item())
                            if (ymax-ymin)/(xmax-xmin) > 10 or (ymax-ymin)/(xmax-xmin) < 0.8:
                                det_pred = Detection(
                                        points=np.vstack(
                                            (
                                                [xmin, ymin],
                                                [xmax, ymin],
                                                [xmin, ymax],
                                                [xmax, ymax],
                                                )
                                            ),
                                        label=names[int(cls)],
                                        embedding=None,
                                        )
                            else:
                                det_pred = Detection(
                                    points=np.vstack(
                                        (
                                            [xmin, ymin],
                                            [xmax, ymin],
                                            [xmin, ymax],
                                            [xmax, ymax],
                                            )
                                        ),
                                    data=[xmin/ori_im.shape[1], ymin/ori_im.shape[0], xmax/ori_im.shape[1], ymax/ori_im.shape[0]],
                                    label=names[int(cls)],
                                    embedding=body_model.extract(ori_im[ymin:ymax, xmin:xmax]),
                                )
                            dect_ls.append(det_pred)
                            LOGGER.info("Det_pred")
                            LOGGER.info(det_pred)
                            LOGGER.info("Det_ls")
                            LOGGER.info(dect_ls)
                        tracked_objects = tracker.update(detections=dect_ls)
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                else:
                    with dt[3]:
                        tracked_objects = tracker.update(period=config.skip_period)
            else:
                with dt[3]:
                    tracked_objects = tracker.update()
                # time.sleep(1)
                # send_frame(ori_im, 80, 20, 200, len(tracked_objects))
                
            # sendDoor(tracked_objects, number)
            LOGGER.info(tracked_objects)
            LOGGER.info(det)
            LOGGER.info(f"Total time: {(time.time()-start_time) * 1E3}ms")
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(tracked_objects) else '(no detections), '}{dt[0].dt * 1E3:.1f}ms, {dt[1].dt * 1E3:.1f}ms, {dt[2].dt * 1E3:.1f}ms, {dt[3].dt * 1E3:.1f}ms, {1/(dt[0].dt+dt[1].dt+dt[2].dt+dt[3].dt):.1f}fps")
            if len(tracked_objects):
                haveEmbedding = sendDoor(tracked_objects, number)
                   
                tracked_objects = []
                det = []
                tracker.tracked_objects = []
                if (haveEmbedding): 
                    break

    except KeyboardInterrupt:
        break
cap.release()
del cap


