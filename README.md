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
Tips: Use jtop to turn on fan and maximize power consumption.
---
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

## 2. Usage
---
