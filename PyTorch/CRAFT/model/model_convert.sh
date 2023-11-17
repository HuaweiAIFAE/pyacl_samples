#!/bin/bash

# clone necessary repository
echo "Downloading CRAFT repository"
git clone https://github.com/clovaai/CRAFT-pytorch

# Download model
wget "https://drive.google.com/uc?export=download&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ" -O craft_mlt_25k.pth --no-check-certificate

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "[ENV] Virtual Environment Preparation Done!"

# copy necessary file to repo
cp -r onnx_export.py craft_mlt_25k.pth CRAFT-pytorch

# open repo
cd CRAFT-pytorch

# convert pt model to onnx model
echo "[MODEL] Conversion Starting!"
python3 onnx_export.py

# mv onnx model
mv craft.onnx ../

# deactivate venv
deactivate
cd ../
# delete virtual environment
rm -r convertPt2Onnx

# remove unnecessary files
sudo rm -r CRAFT-pytorch/
echo "[MODEL] Conversion Done!"