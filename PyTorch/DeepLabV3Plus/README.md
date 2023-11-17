# PyTorch Deeplabv3 Plus

Please open the `jupyter-notebook` for a quick demo |[Original Github Repository](https://github.com/open-mmlab/mmsegmentation)|

## Overview

DeepLabv3+ is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (such as a person, a dog, a cat and so on) to every pixel in the input image.

## Getting started

Download following **DepLabV3+ ONNX model** from the link and put it in the _model_ folder. 

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| DeepLabV3+ | 5.1.RC2  | Download pretrained model [DeepLabV3+](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/model/ATC%20DeepLabV3%2B%20%28FP16%29%20from%20Pytorch%20-%20Ascend310/zh/1.1/DeepLabV3%2B.zip) |

<details> <summary> Work on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name deeplabv3plus -p 6565:4545 \
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


## Convert Your Model

### ONNX format -> Ascend om format
For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_convert.sh` file will do every model conversion stage automatically. After conversion you should have the `.onnx` model in your model path.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/PyTorch/DeepLabV3Plus/model
bash model_convert.sh
```

```bash
atc --model=deeplabv3_plus_res101-sim.onnx \
    --framework=5 \
    --output=deeplabv3plus513_310 \
    --soc_version=Ascend310 \
    --output_type=FP16 \
    --input_shape="actual_input_1:1,3,513,513" 
    
```   

Install dependencies;
- opencv-python-headless
- matplotlib
- numpy
- Pillow

```
pip install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):6565

## Citation
```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```