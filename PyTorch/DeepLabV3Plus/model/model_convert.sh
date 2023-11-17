#!/bin/bash

# Download model
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/model/ATC%20DeepLabV3%2B%20%28FP16%29%20from%20Pytorch%20-%20Ascend310/zh/1.1/DeepLabV3%2B.zip --no-check-certificate

# change file name
mv DeepLabV3+.zip DeepLabV3Plus.zip

# unzip DeepLab
unzip DeepLabV3Plus.zip

# change directory name
mv DeepLabV3+ DeepLabV3Plus

cd DeepLabV3Plus

mv deeplabv3_plus_res101-sim.onnx ../

# delete unnecessary folder
cd .. && rm -r ./DeepLabV3Plus

echo "[MODEL] Conversion Done!"