# PyTorch Deep Text Recognition

Please open the `jupyter-notebook` for a quick demo | [Paper](https://arxiv.org/pdf/1904.01906.pdf) | [Original Github Repository](https://github.com/clovaai/deep-text-recognition-benchmark) |

## Overview

`PyTorch` implementation for two-stage Scene Text Recognition (STR), that most existing STR models fit into.

Using this framework, texts detected by [PyTorch-CRAFT](../Craft) can identification.

<img alt="teaser" src="../../Common/data/figures/deep_text_reco.jpg">

## Getting started

Download following **None-ResNet-None-CTC PT model** from the link and put it in the _model_ folder. 

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| None-ResNet-None-CTC | 5.1.RC2  | Download pretrained model [None-ResNet-None-CTC](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0)

<details> <summary> Working on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name deeptextrecog -p 6565:4545 \
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
        lzma \
        liblzma-dev \
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

### PT model -> ONNX format -> Ascend om format

For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_convert.sh` file will do model conversion stage automatically. After conversion you should have the **.onnx** model in your `model` path.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/PyTorch/DeepTextRecognition/model

bash model_convert.sh
```

```bash
atc --model=None-ResNet-None-CTC.onnx \
    --framework=5 \
    --output=None-ResNet-None-CTC \
    --soc_version=Ascend310
```

Install dependencies;
- opencv-python-headless
- functools
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

**Note :** Do not forget to give the boxes_path including the polygon coordinates detected by PyTorch-CRAFT for demo.

## Citation
```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```

## Resources
[1] https://github.com/axinc-ai/ailia-models/issues/302

[2] https://colab.research.google.com/github/tulasiram58827/ocr_tflite/blob/main/colabs/Deep_Text_Recognition_TFLite.ipynb#scrollTo=OzhcDAc5kvJK

[3] https://github.com/clovaai/deep-text-recognition-benchmark/issues/76