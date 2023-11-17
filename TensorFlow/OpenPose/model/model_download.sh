#!/bin/bash

echo "[MODEL] Download Started!"
# Download PB pretrained model
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2021-12-30_tf/ATC%20OpenPose%28FP16%29%20from%20TensorFlow%20-%20Ascend310/zh/1.1/ATC%20OpenPose%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip --no-check-certificate

# Unzip the model file
unzip ATC\ OpenPose\(FP16\)\ from\ TensorFlow\ -\ Ascend310.zip

# Change directory
cd ./OpenPose_for_ACL/models/

# Copy PB model to ./model directory
cp OpenPose_for_TensorFlow_BatchSize_1.pb ../../

cd ../../
# Remove unnecessary files and folders.
rm -r ./OpenPose_for_ACL && rm ATC\ OpenPose\(FP16\)\ from\ TensorFlow\ -\ Ascend310.zip
echo "[MODEL] Download Complated!"