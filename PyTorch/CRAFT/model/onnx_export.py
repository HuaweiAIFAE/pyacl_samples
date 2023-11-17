import torch
from craft import CRAFT
from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# load net
net = CRAFT() # initialize

net.load_state_dict(copyStateDict(torch.load('craft_mlt_25k.pth', map_location='cpu')))
net.eval()

# load data
img = torch.zeros((1, 3, 736, 1280)) # BCHW

# trace export
torch.onnx.export(net, img, 'craft.onnx', export_params=True, verbose=True, opset_version=11)