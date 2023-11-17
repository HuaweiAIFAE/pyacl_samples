# TensorFlow YOLOv3 Object Detection
Please open the `jupyter notebook` for a quick demo.

YOLOv3 is one of the fast, light-weight real-time object detection system and models pretrained on the COCO dataset. | [Read More](https://pjreddie.com/darknet/yolo/) | [Paper](https://arxiv.org/abs/1804.02767) | [Original Github Repository](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FYunYang1994%2Ftensorflow-yolov3)

<img src="https://pjreddie.com/media/image/yologo_2.png" height="200" alt="prcurve"/>


## Overview
`TensorFlow` implementation for **YOLOv3** (You Only Look Once, Version 3) is a **real-time object detection algorithm** that detects objects in images, videos or  live streams. The YOLO machine learning algorithm build by Deep Convolutional Neural Networks to detect an object. 

<img src="../../Common/data/yolo_result.png" alt="prcurve"/>


## Getting started

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| YOLOv3| 5.1.RC2  | Download pretrained model [YOLOv3](https://www.hiascend.com/en/software/modelzoo/models/detail/1/8320c01a25974c6eb7cd117d0af3cc30)


<details> <summary> Work on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name tf_yolov3 -p 9595:4545 \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /PATH/pyacl_samples:/workspace/pyacl_samples \
ascendhub.huawei.com/public-ascendhub/infer-modelzoo:22.0.RC2 /bin/bash
```
```bash
rm -rf /usr/local/python3.9.2 # if your python version > 3.7.5

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
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```
</details>

## Convert Your Model

## PB model -> Ascend OM format
For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_download.sh` file will download the pretrained model automatically. After downloading, the `.pb` model will be in model directory.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/TensorFlow/YoloV3/model
bash model_download.sh
```

```bash
atc --model=yolov3_int8_tf.pb \
    --framework=3 \
    --input_shape="input/input_data:1,416,416,3" \
    --output=./yolov3 \
    --insert_op_conf=./aipp_yolov3_tf.cfg \
    --soc_version=Ascend310\
    --out_nodes="pred_sbbox/concat_2:0;pred_mbbox/concat_2:0;pred_lbbox/concat_2:0"
```

Install dependencies;
- opencv-python
- numpy
- Pillow

```
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):9595

