import torch
from blazeface import BlazeFace


# load net
net = BlazeFace(back_model=True).to(torch.device("cpu"))
net.load_weights("blazefaceback.pth")

# load data
img = torch.zeros((1, 3, 256, 256)) # BCHW

# trace export
torch.onnx.export(net, img, 'blazefaceback.onnx', export_params=True, verbose=True, opset_version=11)