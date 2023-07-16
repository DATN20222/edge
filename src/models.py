import numpy as np
import torch
import torch.nn as nn
import cv2
from utils.general import yaml_load, check_version, LOGGER
from collections import OrderedDict, namedtuple
from utils.torch_utils import smart_inference_mode
import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0


class DetectBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), data=None, fp16=False):

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        LOGGER.info(f'Loading {w} for TensorRT inference...')
        if device.type == 'cpu':
            device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = fp16  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        self.__dict__.update(locals())  # assign all variables to self
    
    @smart_inference_mode()
    def forward(self, im):
        # YOLOv5 MultiBackend inference
        if im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        for _ in range(1):  #
            self.forward(im)  # warmup


# import tensorflow as tf
# from tensorflow.python.saved_model import tag_constants
# class BodyFeatureExtractBackend():
#     def __init__(self, weights='saved_model'):
#         LOGGER.info('Loading extracting model...')
#         self.model = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
#         self.img_size = (64, 128)

#     def warmup(self):
#         LOGGER.info('Warming up extracting model...')
#         for i in range(20):
#             im_tmp = np.random.randint(0, 255, (np.random.randint(20, 400), np.random.randint(20, 400), 3), "uint8")
#             self.extract(im_tmp)

#     def preprocess(self, im):
#         im = cv2.resize(im, self.img_size)
#         im = im[np.newaxis, ...]
#         im = tf.cast(im, tf.float32)
#         return im
    
#     def extract(self, im):
#         return self.model.signatures['serving_default'](self.preprocess(im))['output_1'][0]


import pycuda.driver as cuda
import pycuda.autoinit
class BodyFeatureExtractBackend():
    def __init__(self, weights='reid_fp32.trt'):
        LOGGER.info('Loading extracting model...')
        self.img_size = (192, 64)
        f = open(weights, "rb")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.output = np.empty([1, 224], dtype = np.float32)
        temp_batch = self.preprocess(np.random.randint(0, 255, (np.random.randint(20, 400), np.random.randint(20, 400), 3), "uint8"))
        self.d_input = cuda.mem_alloc(1 * temp_batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def warmup(self):
        LOGGER.info('Warming up extracting model...')
        for i in range(10):
            im_tmp = np.random.randint(0, 255, (np.random.randint(20, 400), np.random.randint(20, 400), 3), "uint8")
            

    def preprocess(self, img):
        desired_h, desired_w = self.img_size
        img = cv2.resize(img, (desired_w, round(img.shape[0]*desired_w/img.shape[1])))
        if img.shape[0] >= desired_h:
                img = cv2.resize(img, (desired_w, desired_h))
        else:
                img = cv2.copyMakeBorder(img, 0, desired_h-img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = img[np.newaxis, ...]
        return img.astype(np.float32)
    
    def extract(self, im):
        im = self.preprocess(im)
        cuda.memcpy_htod_async(self.d_input, im, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.output