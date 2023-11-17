#!/bin/bash

# download pretrained model
wget  --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies 'https://docs.google.com/uc?export=download&id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ" -O yolov4.pth  && rm -rf /tmp/cookies.txt


# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "[ENV] Virtual Environment Preparation Done!"

# copy necessary file to repo
cp yolov4.pth export/
cp ../../../Common/data/person.jpg export/

# open repo
cd export/

# convert pt model to onnx model
echo "[MODEL] Conversion Starting!"
python3 onnx_export.py yolov4.pth person.jpg 80 608 608

# mv onnx model
mv yolov4.onnx ../

# remove unnecessary files
rm -r yolov4.pth person.jpg

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx && rm yolov4.pth

echo "[MODEL] Conversion Done!"