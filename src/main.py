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
#from sender import send_frame, send_feature
import pika


# Read video input
cap = cv2.VideoCapture(config.source)
print('Camera Ready?', cap.isOpened())
if cap.isOpened() == False:
    os._exit(1)


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
        initialization_delay=config.initialization_delay,
        distance_function="iou",
        hit_counter_max=config.hit_counter_max,
        filter_factory=OptimizedKalmanFilterFactory(),
        distance_threshold=config.distance_threshold,
        past_detections_length=config.past_detections_length,
        reid_distance_function=embedding_distance,
        reid_distance_threshold=config.reid_distance_threshold,
        reid_hit_counter_max=config.reid_hit_counter_max,
        )

if config.draw:
    video = Video(input_path=config.source, output_path='./out.mp4')

LOGGER.info('Start running...')
while cap.isOpened():
    try:
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
                        if (ymax-ymin)/(xmax-xmin) > 10 or (ymax-ymin)/(xmax-xmin) < 0.9:
                            det_pred = Detection(
                                    points=np.vstack(
                                        (
                                            [xmin, ymin],
                                            [xmax, ymax],
                                            )
                                        ),
                                    data=[xmin/ori_im.shape[1], ymin/ori_im.shape[0], xmax/ori_im.shape[1], ymax/ori_im.shape[0]],
                                    label=names[int(cls)],
                                    embedding=None,
                                    )
                        else:
                            det_pred = Detection(
                                    points=np.vstack(
                                        (
                                            [xmin, ymin],
                                            [xmax, ymax],
                                            )
                                        ),
                                    data=[xmin/ori_im.shape[1], ymin/ori_im.shape[0], xmax/ori_im.shape[1], ymax/ori_im.shape[0]],
                                    label=names[int(cls)],
                                    embedding=body_model.extract(ori_im[ymin:ymax, xmin:xmax]),
                                    )
                        dect_ls.append(det_pred)
                    tracked_objects = tracker.update(detections=dect_ls, period=config.skip_period)
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
        if config.draw:
            try:
                draw_points(ori_im, dect_ls)
            except:
                draw_points(ori_im, [])
            for track in tracked_objects:
                print('age:', track.age, 'hit_counter:', track.hit_counter, 'reid_hit_counter:', track.reid_hit_counter, 'id:', track.id)
            draw_tracked_objects(ori_im, tracked_objects)
            frame_with_border = np.ones(
                shape=(
                    ori_im.shape[0] + 2 * 10,
                    ori_im.shape[1] + 2 * 10,
                    ori_im.shape[2],
                ),
                dtype=ori_im.dtype,
            )
            frame_with_border *= 254
            frame_with_border[
                10:-10, 10:-10
            ] = ori_im
            video.write(frame_with_border)
        LOGGER.info(f"Total time: {(time.time()-start_time) * 1E3}ms")
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[0].dt * 1E3:.1f}ms, {dt[1].dt * 1E3:.1f}ms, {dt[2].dt * 1E3:.1f}ms, {dt[3].dt * 1E3:.1f}ms, {1/(dt[0].dt+dt[1].dt+dt[2].dt+dt[3].dt):.1f}fps")
    except KeyboardInterrupt:
        break
cap.release()
del cap

