#!/bin/bash

# yolov5 model type (s,m,x) 
pt_model=$1

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# copy necessary file to repo
cp ${pt_model} export/

# open repo
cd export/

# install necessary python libs
pip3 install --upgrade pip 
pip3 install -r requirements.txt 

echo "[ENV] Virtual Environment Preparation Done!"

python -m pip install --upgrade pip
pip3 install -r requirements.txt

# convert pt model to onnx model
echo "[MODEL] Conversion Starting!"
python3 onnx_export.py --weights ${pt_model}

# mv onnx model
mv "${pt_model%.*}.onnx"  ../

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

echo "[MODEL] Conversion Done!"
