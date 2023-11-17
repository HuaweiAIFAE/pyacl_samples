# Tensorflow Online Inference FaceNet
Please open the `jupyter-notebook` for a quick demo |[Paper](https://arxiv.org/abs/1503.03832)| [Original Github Repository](https://github.com/davidsandberg/facenet)|

## Overview
FaceNet is a general-purpose system that can be used for face verification (is it the same person?), recognition (who is this person?), and cluster (how to find similar people?). FaceNet uses a convolutional neural network to map images into Euclidean space. The spatial distance is directly related to the image similarity. The spatial distance between different images of the same person is small, and the spatial distance between images of different persons is large. As long as the mapping is determined, face recognition becomes simple. FaceNet directly uses the loss function of the triplets-based LMNN (large margin nearest neighbor) to train the neural network. The network directly outputs a 512-dimensional vector space. The triples we selected contain two matched face thumbnails and one unmatched face thumbnail. The objective of the loss function is to distinguish positive and negative classes by distance boundary.

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| FaceNet| 5.1.RC2  | Download pretrained model [FaceNet](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Facenet_for_ACL.zip)

<details> <summary> Work on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --name onlineFaceNet -p 1515:4545 \
--device=/dev/davinci1 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /PATH/pyacl_samples:/workspace/pyacl_samples \
ascendhub.huawei.com/public-ascendhub/tensorflow-modelzoo:22.0.RC2-ubuntu18.04 /bin/bash
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
source /usr/local/Ascend/nnae/set_env.sh
```


#### Fix [libascend_hal.so](https://bbs.huaweicloud.com/blogs/344623): cannot open shared object file
```bash
cp /usr/local/Ascend/driver/lib64/libascend_hal.so /usr/local/lib/

vim /etc/ld.so.conf
```
```bash
{vim /etc/ld.so.conf}
# Add the following to another line:
/usr/local/lib/

# Save & Exit
:wq
```
```bash
ldconfig
```
</details>

## Getting started
Download the pretrained model.
And then download the `PB` file.

```bash
bash get_model.sh
```

Finaly, open `jupyter-notebook` and run the code for demo


```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):6565

CLI execution command.
```bash
python3 main.py --model_path ./models/facenet_tf.pb --input_tensor_name input:0 --output_tensor_name embeddings:0 --image_path ./facenet_data
```

## Citation
```
@inproceedings{Schroff_2015,
    year = 2015,
        month = {jun},
        publisher = {{IEEE}
},
        author = {Florian Schroff and Dmitry Kalenichenko and James Philbin},
        title = {{FaceNet}: A unified embedding for face recognition and clustering},
        booktitle = {2015 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
```

