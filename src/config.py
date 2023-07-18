class Config:
    def __init__(self):
        self.skip_period = 1
        self.draw = False
        #Models
        self.weights = 'weights/yolov5n_16_384.engine' #weights of the detection model
        self.body_extract_model = 'weights/reid_fp32.trt' #weights of the extraction model
        self.device = 0 #gpu0
        self.source = 1
        self.height = 384
        self.width = 384
        self.fp16 = True
        self.classes = 'classes.yaml'
        self.conf_thres = 0.3
        self.iou_thres = 0.35
        self.max_det = 200
        #Track
        self.initialization_delay = 10
        self.hit_counter_max = 50
        self.distance_threshold = 40
        self.past_detections_length = 10
        self.reid_distance_threshold = 0.13
        self.reid_hit_counter_max = 150
        #Send
        self.frame_interval = 2 #seconds
        self.feature_interval = 0.5 #seconds
        self.server_ip = '192.168.50.117'
        self.send_frame_reso = (341, 192)
        self.jetson_ip = '172.168.1.10'

config = Config()
