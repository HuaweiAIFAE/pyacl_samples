# Deep Face Recognition with ArcFace in TensorFlow
Please open the `jupyter-notebook` for a quick demo.

DeepFace is Face Recognition and facial attribute analysis system created by a research group at Facebook. | [Read More](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) | [Original Github Repository](https://github.com/serengil/deepface)

<img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="300" height="300" alt="prcurve"/>

## Getting started

| **Model** | **CANN Version** | **How to Obtain** |
|---|---|---|
| ArcFace | 5.1.RC2  | Download pretrained model [ArcFace](https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5)


<details> <summary> Work on docker environment (<i>click to expand</i>)</summary>

Start your docker environment.

```bash
sudo docker run -it -u root --rm --name tf_deepface -p 9595:4545 \
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

## H5 model -> ONNX format -> Ascend OM format

For this stages it is recommended to use the docker environment to avoid affecting the development environment. The `model_convert.sh` file will do model conversion stage automatically. After conversion you should have the **.onnx** model in your `model` path.

```bash
cd <root_path_of_pyacl_samples>/pyacl_samples/TensorFlow/DeepFace/model/export

bash model_convert.sh
```

```bash
atc --model=arcface.onnx \
    --framework=5 \
    --input_shape="input_1:1,112,112,3" \
    --output=./arcface \
    --soc_version=Ascend310 \
    --out_nodes="embedding:0"
```

Install dependencies;
- opencv-python
- numpy

```
pip3 install -r requirements.txt
```

Finaly, open `jupyter-notebook` and run the code for demo

```bash
jupyter-notebook --port 4545 --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter-notebook will open in (localhost):9595








