class Config:
    def __init__(self):
        self.weights = 'weights/yolov5n_16_384.engine'
        self.body_extract_model = 'weights/model_trt'
        self.device = 0
        self.source = 'cam2.mp4'
        self.height = 384
        self.width = 384
        self.fp16 = True
        self.classes = 'classes.yaml'
        self.conf_thres = 0.35
        self.iou_thres = 0.35
        self.max_det = 50
        #Track
        self.hit_counter_max = 15
        self.distance_threshold = 20
        self.past_detections_length = 20
        self.reid_distance_threshold = 0.25
        self.reid_hit_counter_max = 200


config = Config()
