#!/bin/bash

# prepering environment
update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertTfOnnx
source convertTfOnnx/bin/activate

# open repo
cd export/

# install necessary python libs
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet 
echo "[ENV] Virtual Environment Preparation Done!"

# convert pt model to onnx model
echo "[MODEL] Conversion Starting!"
model_name='arcface'
python onnx_export.py --model $model_name --output $model_name

# deactivate venv
deactivate
mv arcface.onnx ../

# Delete virtual environment
rm -r convertTfOnnx

echo "[MODEL] Conversion Done!"