# Tensorflow FaceNet

Please open the `jupyter-notebook` for a quick demo | [Paper](https://arxiv.org/abs/1503.03832) | [Original Github Repository](https://github.com/davidsandberg/facenet)

## Overview
`FaceNet` is a general-purpose system that can be used for face verification (is it the same person?), recognition (who is this person?), and cluster (how to find similar people?). FaceNet uses a convolutional neural network to map images into Euclidean space. The spatial distance is directly related to the image similarity. The spatial distance between different images of the same person is small, and the spatial distance between images of different persons is large. As long as the mapping is determined, face recognition becomes simple. FaceNet directly uses the loss function of the triplets-based LMNN (large margin nearest neighbor) to train the neural network. The network directly outputs a 512-dimensional vector space. The triples we selected contain two matched face thumbnails and one unmatched face thumbnail. The objective of the loss function is to distinguish positive and negative classes by distance boundary.


## Getting started

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| FaceNet | 5.1.RC2  | Download pretrained model [FaceNet](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Facenet_for_ACL.zip)

<details> <summary> Working on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.


```bash
sudo docker run -it -u root --rm --name facenet_infer -p 6565:4545 \
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

## PB model -> Ascend OM format 
For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_download.sh` file will download the pretrained model automatically. After downloading, the `.pb` model will be in model directory.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/TensorFlow/FaceNet/model
bash model_download.sh
```

```bash
atc --model=facenet_tf.pb \
      --framework=3 \
      --output=facenet_tf  \
      --soc_version=Ascend310 \
      --input_shape="input:1,160,160,3"
```

Install dependencies;

- opencv-python-headless
- numpy

```
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):6565