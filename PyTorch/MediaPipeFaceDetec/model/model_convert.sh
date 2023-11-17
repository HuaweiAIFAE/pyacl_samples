#!/bin/bash

# clone necessary repository
echo "Downloading BlazeFace repository"
git clone https://github.com/hollance/BlazeFace-PyTorch.git

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertTfliteOnnx
source convertTfliteOnnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "[ENV] Virtual Environment Preparation Done!"

# copy necessary file to repo
cp -r onnx_export.py tflite_pthExport.py BlazeFace-PyTorch/

# open repo
cd BlazeFace-PyTorch/
# get model
wget https://github.com/google/mediapipe/raw/v0.7.12/mediapipe/models/face_detection_back.tflite --no-check-certificate

# convert tf-lite model to pt model
echo "[MODEL] Conversion Starting!"
python3 tflite_pthExport.py
# convert pt model to onnx model
python3 onnx_export.py

# mv onnx model
mv blazefaceback.onnx ../

# deactivate venv
deactivate
# Delete virtual environment
cd ..
rm -r convertTfliteOnnx

# Install Mindspore-ascend-310-x86
pip3 install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/ascend/x86_64/mindspore_ascend-1.8.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# MindConverter
pip3 install onnx~=1.8.0 onnxoptimizer~=0.1.2 onnxruntime~=1.5.2 protobuf==3.20.0
pip3 install torch==1.8.2+cpu -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.7.0/MindInsight/any/mindconverter-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# convert onnx model to ckpt model 
mindconverter --model_file blazefaceback.onnx
# copy ckpt model
cp ms_export.py output/

# go output file
cd output
# convert ckpt model to air model
python3 ms_export.py

# move the air model to model folder
mv air_blazeface_back.air ../
# remove unnecessary files
cd ..
rm -r BlazeFace-PyTorch/ output/ blazefaceback.onnx
echo "[MODEL] Conversion Done!"