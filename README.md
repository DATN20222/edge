# EdgeAI
---
This is the implementation of AI models and services on edge devices.

Tested on Jetson Nano 4GB

## 0. Prerequisite
---
Check for L4T version of Jetson Nano.

This implementation is for L4T R32.7.x version.
```shell
cat /etc/nv_tegra_release
```

## 1. Installation
---
Tips: Use jtop to turn on fan and maximize power consumption.

### Docker Default Runtime

To enable access to the CUDA compiler (nvcc) during `docker build` operations, add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```

You will then want to restart the Docker service or reboot your system before proceeding.
```shell
sudo systemctl restart docker.service
```
### Build Docker
Firstly, clone the code of this repo.
```shell
git clone https://github.com/DATN20222/edge.git
cd edge
```

Build the docker.
```shell
sudo docker build -t edgeai .
```
It takes from 30 minutes to more than an hour to finish. Therefore, grab a cup of coffee and watch TV :D.

## 2. Usage
---

### Create Container
```shell
sudo docker run --runtime nvidia -it --rm --network host --device /dev/video0 --device /dev/video1 -v /path/to/mount/folder:/path/to/mount/folder/inside/docker edgeai:latest
```
- Use '--device' to add the usb cameras to docker. Check for /dev/video* and add the corresponding to the command.
- Use '-v' to mount the working folder

### Run Code
Modify `config.py` file to change the settings. Change the `source` attribute for the appropriate input, it takes a path to a video or number(0, 1, 2, ...) for the corresponding usb cameras.

Current implementation will only produce a result video `output.mp4` after execution.
```shell
python3 main.py
```
