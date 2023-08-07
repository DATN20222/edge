class Config:
    def __init__(self):
        self.skip_period = 1 #skip detect period
        self.draw = False #draw
        #Models
        self.weights = 'weights/yolov5n_16_384.engine' #weights of the detection model
        self.body_extract_model = 'weights/reid_fp32.trt' #weights of the extraction model
        self.device = 0 #gpu0
        self.source = 1 #input source
        self.height = 384 #input height of detection model
        self.width = 384 #input width of detection
        self.fp16 = True #float16 infer
        self.classes = 'classes.yaml' #class
        self.conf_thres = 0.45 #confidence for detection model
        self.iou_thres = 0.45 #iou for nms
        self.max_det = 200
        #Track
        self.initialization_delay = 10 #delay to initialize new track object
        self.hit_counter_max = 50
        self.distance_threshold = 0.6
        self.past_detections_length = 10 
        self.reid_distance_threshold = 0.2
        self.reid_hit_counter_max = 100
        #Send
        self.frame_interval = 2 #seconds
        self.feature_interval = 1 #seconds
        self.server_ip = '192.168.50.131'
        self.send_frame_reso = (341, 192) #resolution for send frame
        self.jetson_ip = '172.168.1.12'

config = Config()
