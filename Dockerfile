#Download base image tensorflow2.7 tensorrt for jetpack 4.6.1
FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3
#Update ubuntu software
RUN apt-get update && apt-get install -y libfreetype6-dev libjpeg-dev zlib1g-dev libopenblas-base libopenmpi-dev libomp-dev git vim
#Upgrade pip
RUN pip3 install --upgrade pip
#Install yolov5 dependencies
RUN pip install Cython
RUN wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
RUN pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl && rm -rf torch-1.10.0-cp36-cp36m-linux_aarch64.whl
RUN pip install gitpython>=3.1.20
RUN pip install matplotlib>=3.3
RUN pip install numpy>=1.18.5
RUN pip install Pillow>=7.1.2
RUN pip install psutil
RUN pip install PyYAML>=5.3.1
RUN pip install requests>=2.23.0
RUN pip install scipy>=1.4.1
RUN pip install thop>=0.1.1
RUN pip install tqdm>=4.64.0
RUN pip install tensorboard>=2.4.1
RUN pip install pandas>=1.1.4
RUN pip install seaborn>=0.11.0
RUN pip install setuptools>=59.6.0
RUN git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
RUN cd torchvision && python3 setup.py install && cd .. && rm -rf torchvision
RUN wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
RUN pip install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
RUN rm -rf onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
RUN pip install onnx==1.11.0
#Install opencv
RUN pip install opencv-contrib-python==4.6.0.66
#Install other dependencies
RUN pip install pika
RUN pip install pybase64
