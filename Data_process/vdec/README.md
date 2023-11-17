# Video Decoding using DVPP
VDEC is a DVPP based module and it is used to decode videos. Please open the `jupyter-notebook` for a quick demo

## Overview
VDEC module encapsulates video stream decoding interfaces, including video stream frame cut. Because of the VDEC is based on DVPP, It is recommended that you learn some basic knowledge about DVPP.

The DVPP(Digital Vision Pre-Processing) is an image pre-processing hardware acceleration module provided by Ascend 310. DVPP module preprocesses images through encoding, decoding, and format conversion. DVPP converts the video or image data input from the system memory and network into a format supported by the Ascend AI Processors before neural network computing by the Da Vinci Architecture.

Preprocessing for video datasets can be done on NPU or CPU. If you are using video-preprocessing on the CPU for example using OpenCV, these processes will be run on the CPU. And since we know OpenCV has high memory consumption, the same preprocessing can be done using DVPP with lower consumption and faster on NPU.

## Getting Started

| **Module** | **CANN Version** | 
|---|---|
| VDEC | 5.1.RC2  | 
<details> <summary> Working on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name acl_vdec -p 9595:4545 \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /PATH/pyacl_samples:/workspace/pyacl_samples \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
ascendhub.huawei.com/public-ascendhub/infer-modelzoo:22.0.RC2 /bin/bash
```
    
```bash
apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        cmake \
        zlib1g \
        zlib1g-dev \
        openssl \
        libsqlite3-dev \
        libssl-dev \
        libffi-dev \
        unzip \
        pciutils \
        net-tools \
        libblas-dev \
        gfortran \
        libblas3 \
        libopenblas-dev \
        libbz2-dev \
        build-essential \
	    ffmpeg \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```
```bash
rm -rf /usr/local/python3.9.2

wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz --no-check-certificate && \
    tar -zxvf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared && make -j && make install && \
    cd .. && \
    rm -r -d Python-3.7.5 && rm Python-3.7.5.tgz && \
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH && \
    export PATH=/usr/local/python3.7.5/bin:$PATH

pip3 install --upgrade pip
pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py jupyter jupyterlab sympy
```    
</details>

## Preparing Video Data
For this stage, it is recommended to use docker environment to avoid affecting the development environment. VDEC module supports only H.264 types of videos. if your input video data is not H.264 compression format, you need to convert it. For this process, ffmpeg package is highly recommended. The conversion steps are as showed below;

If the video data is not MP4 format, you should convert it first. As an example, we are doing this process for MOV. 

```bash
ffmpeg -i video.mov -q:v 0 video.mp4
```

Then, you can convert to H.264 format as shown below.

```bash
ffmpeg -i video.mp4 -codec copy -bsf: h264_mp4toannexb -f h264 video.h264
```

Install dependencies;
- opencv-python
- numpy
- Pillow
- av

```bash
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):9595