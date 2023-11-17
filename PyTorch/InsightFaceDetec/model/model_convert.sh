#!/bin/bash

# clone necessary repository
echo "Downloading IndsightFace repository"
git clone https://github.com/deepinsight/insightface.git

# download pretrained model
curl -L -o model.pth https://hrqazw.dm.files.1drv.com/y4mVVE2xX7BEswodNl8DuN4D5iwlAmEVdxjgCu6-mX4ui62B5WZ-33R492NsRJjlq_3QCtVIjUbgTfiJxI8iF2RP9iFBSCitgCxcoaug4eBrRkaTKZCICQn5SqMskDUpIMSpQJ6ALHVdsrpA9l8Q2nTohRZwcOxaW1_0rZgwCwpASt7CtrGwg5hSPYwkVakKjdjl7Uo_fC3sZ43uwpZmgtd8w

# change model name
mv model.pth scrfd_34g.pth

# create python virtual environment
echo "[ENV] Virtual Environment Preparation Starting!"
python3 -m venv convertPt2Onnx
source convertPt2Onnx/bin/activate

# install necessary python libs
python -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "[ENV] Virtual Environment Preparation Done!"

# copy necessary file to repo
cp -r onnx_export.py scrfd_34g.py scrfd_34g.pth insightface/detection/scrfd
cp ../../../Common/data/peoples.jpg insightface/detection/scrfd

# open repo
cd insightface/detection/scrfd

# convert pt model to onnx model
echo "[MODEL] Conversion Starting!"
python3 onnx_export.py --config scrfd_34g.py --weights scrfd_34g.pth --input_img peoples.jpg --simplify

# mv onnx model
mv scrfd_34g_shape640x640.onnx ../../../

# deactivate venv
deactivate
cd ../../../
# delete virtual environment
rm -r convertPt2Onnx

# remove unnecessary files
rm -r insightface/
echo "[MODEL] Conversion Done!"