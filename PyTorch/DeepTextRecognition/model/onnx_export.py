import onnx
import torch

from model import Model
from argparse import Namespace
from collections import OrderedDict


def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]  
        new_state_dict[key] = value
    return new_state_dict

# inital parameters
opt = Namespace(imgH = 32, imgW = 100, num_fiducial = 20, input_channel = 1, output_channel = 512,
                hidden_size = 256, num_class = 37, batch_max_length = 25, Transformation = None, 
                FeatureExtraction = "ResNet", SequenceModeling = None, Prediction = "CTC",
                gpu_enabled = False, saved_model = "./None-ResNet-None-CTC.pth",
                image_folder = "./result")

# set model
model = Model(opt)
print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
      opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
      opt.SequenceModeling, opt.Prediction)

# load model
print('loading pretrained model from %s' % opt.saved_model)
model.load_state_dict(copyStateDict(torch.load(opt.saved_model, map_location='cpu')))

# create random data
x = torch.randn(1, opt.input_channel, opt.imgH, opt.imgW)
model.eval()

# get file name
f = opt.saved_model.replace('.pth', '.onnx') 

# trace export
torch.onnx.export(model, x, f, export_params=True, verbose=False, opset_version=11,  
                  input_names = ['input'], # the model's input names
                  output_names = ['output'], # the model's output names
                  )

# Checks
onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
print('ONNX export success, saved as %s' % f)