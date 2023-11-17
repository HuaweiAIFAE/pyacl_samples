#!/bin/bash

echo "[MODEL] Download Started!"
# Download PB pretrained model
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/model/YoloV3/zh/1.1.0/yolov3.zip --no-check-certificate

# Unzip the model file
unzip yolov3.zip

# Copy PB model to ./model directory
cp "yolov3(int8)"/yolov3_int8_tf.pb .

# Remove unnecessary files and folders.
rm -r  yolov3.zip "yolov3(int8)"
echo "[MODEL] Download Complated!"
