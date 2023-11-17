#!/bin/bash

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertAirOnnx --system-site-packages
source convertAirOnnx/bin/activate

# open repo
cd export/

# install necessary python libs
python -m pip install --upgrade pip --quiet
pip install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py sympy

pip install -r requirements.txt 
echo "[ENV] Virtual Environment Preparation Done!"

# convert air model to onnx model
echo "[MODEL] Conversion Starting!"
python3 export.py --ckpt_file ../yolov5s_260_COCO2017.ckpt --file_name yolov5s --file_format AIR

mv yolov5s_yolov5s.air ../yolov5s.air

# deactivate venv
deactivate
cd ../

# delete virtual environment
rm -r convertAirOnnx

echo "[MODEL] Conversion Done!"
