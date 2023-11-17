# MindSpore YOLOv5 Example

Please open the `jupyter-notebook` for a quick demo |[Original Repository](https://gitee.com/ktuna/mind-spore-yolov5/tree/master)

<img src="https://pjreddie.com/media/image/yologo_2.png" height="200" alt="prcurve"/>

## Overview

YOLOv5 is one of the most popular object detecion AI, incorporating lessons learned and best practices evolved over thousands of hours of research and development. This version developed by Huawei's AI framework Mindspore for best performance with Huawei Ascend NPU's in every stage.

<img src="../../Common/data/yolo_result.png" alt="prcurve"/>

## Getting Started

Download appropriate **YOLOv5 model** from the following link and put it in the _model_ folder. 

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| YOLOv5 | 5.1.RC2 | Download pretrained model [YOLOv5](https://onebox.huawei.com/p/dad426ea028637e90fdef4f7a272e8cf) |

<details> <summary> Work on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name mindspore_yolov5 -p 6565:4545 \
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

### CKPT model -> AIR format -> Ascend OM format

For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_convert.sh` file will do every model conversion stage automatically. After conversion you should have the `.air` model in your model path. If you want to change model input sizes you need to customize `model_convert.sh` file.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/MindSpore/YoloV5/model
bash model_convert.sh
```

```bash
atc --model=yolov5s.air \
    --framework=1 \
    --output=yolov5s \
    --soc_version=Ascend310
```

Install dependencies;

- numpy
- opencv_python
- Pillow

```
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):6565