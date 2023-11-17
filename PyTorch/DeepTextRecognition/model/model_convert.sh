#!/bin/bash

# clone necessary repository
echo "Downloading Deep Text Recognition Benchmark repository"
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git

# download pretrained model
curl -L -o None-ResNet-None-CTC.pth https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth?dl=1

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "[ENV] Virtual Environment Preparation Done!"

# copy necessary file to repo
echo "[MODEL] Conversion Starting!"
cp -r onnx_export.py model.py None-ResNet-None-CTC.pth deep-text-recognition-benchmark

# open repo
cd deep-text-recognition-benchmark

# convert pt model to onnx model
python3 onnx_export.py

# mv onnx model
mv None-ResNet-None-CTC.onnx ../

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

# remove unnecessary files
rm -r deep-text-recognition-benchmark/
echo "[MODEL] Conversion Done!"