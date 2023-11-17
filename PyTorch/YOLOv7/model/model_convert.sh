#!/bin/bash

# Clone original yolov7 repo
git clone https://github.com/WongKinYiu/yolov7.git

# Download pretrained model file 
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt --no-check-certificate

cd ./yolov7

# Create virtual python environment and install dependencies
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convert_onnx
source convert_onnx/bin/activate
pip3 install --upgrade pip 
pip3 install --upgrade setuptools 
pip3 install -r ../requirements.txt 
echo "[ENV] Virtual Environment Preparation Done!"

# Export PT -> ONNX
echo "[MODEL] Conversion Starting!"
python3 export.py --weights ../yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.5 --conf-thres 0.4 --img-size 640 640 --max-wh 640

# Deactivate virtual environment
deactivate

# Remove unnecessary files and folders
cd .. && rm yolov7.torchscript.ptl && rm yolov7.torchscript.pt && rm -r ./yolov7 && rm yolov7.pt

echo "[MODEL] Conversion Done!"
