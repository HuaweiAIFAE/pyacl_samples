# OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
Please open the `jupyter-notebook` for a quick demo | [Paper](https://arxiv.org/abs/1812.08008) | [Original Github Repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |

## Overview
Realtime multi-person 2D pose estimation is a key component in enabling machines to have an understanding of people in images and videos. In this work, we present a realtime approach to detect the 2D pose of multiple people in an image. The proposed method uses a nonparametric representation, which we refer to as Part Affinity Fields (PAFs), to learn to associate body parts with individuals in the image. This bottom-up system achieves high accuracy and realtime performance, regardless of the number of people in the image. In previous work, PAFs and body part location estimation were refined simultaneously across training stages. We demonstrate that a PAF-only refinement rather than both PAF and body part location refinement results in a substantial increase in both runtime performance and accuracy. We also present the first combined body and foot keypoint detector, based on an internal annotated foot dataset that we have publicly released. We show that the combined detector not only reduces the inference time compared to running them sequentially, but also maintains the accuracy of each component individually. This work has culminated in the release of OpenPose, the first open-source realtime system for multi-person 2D pose detection, including body, foot, hand, and facial keypoints.

## Getting started

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| OpenPose | 5.1.RC2  | Download pretrained model [OpenPose](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2021-12-30_tf/ATC%20OpenPose%28FP16%29%20from%20TensorFlow%20-%20Ascend310/zh/1.1/ATC%20OpenPose%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip)

<details> <summary> Work on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name openpose -p 6565:4545 \
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

#### PB model -> Ascend OM format
For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_download.sh` file will download the pretrained model automatically. After downloading, the `.pb` model will be in model directory.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/TensorFlow/OpenPose/model
bash model_download.sh
```

```bash
atc --model=OpenPose_for_TensorFlow_BatchSize_1.pb \
    --framework=3 \
    --output=openpose_no_aipp \
    --soc_version=Ascend310 \
    --input_shape="image:1,368,656,3"
```

Install dependencies;
- opencv-python-headless
- numpy
- Pillow


```bash
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):6565

## Citation
```
@misc{https://doi.org/10.48550/arxiv.1812.08008,
  doi = {10.48550/ARXIV.1812.08008},
  url = {https://arxiv.org/abs/1812.08008},
  author = {Cao, Zhe and Hidalgo, Gines and Simon, Tomas and Wei, Shih-En and Sheikh, Yaser},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  publisher = {arXiv},
  year = {2018},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```