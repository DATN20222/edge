# Build from base l4t r32.7.1 image
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.7.1
FROM ${BASE_IMAGE}

#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /opt


#
# OpenCV - https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh
#
ARG OPENCV_VERSION="4.6.0"

# install build dependencies
COPY scripts/opencv_install_deps.sh opencv_install_deps.sh
RUN ./opencv_install_deps.sh

# OpenCV looks for the cuDNN version in cudnn_version.h, but it's been renamed to cudnn_version_v8.h
RUN ln -s /usr/include/$(uname -i)-linux-gnu/cudnn_version_v8.h /usr/include/$(uname -i)-linux-gnu/cudnn_version.h

# architecture-specific build options
ARG CUDA_ARCH_BIN=""
ARG ENABLE_NEON="OFF"

# clone and configure OpenCV repo
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    echo "configuring OpenCV ${OPENCV_VERSION}, CUDA_ARCH_BIN=${CUDA_ARCH_BIN}, ENABLE_NEON=${ENABLE_NEON}" && \
    cmake \
        -D CPACK_BINARY_DEB=ON \
	   -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
	   -D BUILD_opencv_java=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -D CUDA_ARCH_PTX= \
        -D CUDA_FAST_MATH=ON \
        -D CUDNN_INCLUDE_DIR=/usr/include/$(uname -i)-linux-gnu \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
	   -D WITH_EIGEN=ON \
        -D ENABLE_NEON=${ENABLE_NEON} \
        -D OPENCV_DNN_CUDA=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_OPENGL=ON \
	   -D WITH_OPENCL=OFF \
	   -D WITH_IPP=OFF \
        -D WITH_TBB=ON \
	   -D BUILD_TIFF=ON \
	   -D BUILD_PERF_TESTS=OFF \
	   -D BUILD_TESTS=OFF \
	   ../
	   
RUN cd opencv/build && make -j$(nproc)
RUN cd opencv/build && make install
RUN cd opencv/build && make package

RUN cd opencv/build && tar -czvf OpenCV-${OPENCV_VERSION}-$(uname -i).tar.gz *.deb
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
#Install other dependencies
RUN pip install pika
RUN pip install pybase64
RUN pip install rich
RUN pip install filterpy
RUN pip install pyserial
RUN pip install tf2onnx
RUN pip install packaging
