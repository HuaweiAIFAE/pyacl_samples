import torch
import numpy as np
from tflite import Model
from blazeface import BlazeFace
from collections import OrderedDict


def get_shape(tensor):
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def print_graph(graph):
    for i in range(0, graph.TensorsLength()):
        tensor = graph.Tensors(i)
        print("%3d %30s %d %2d %s" % (i, tensor.Name(), tensor.Type(), tensor.Buffer(), 
                                      get_shape(graph.Tensors(i))))


def get_parameters(graph):
    parameters = {}
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        if tensor.Buffer() > 0:
            name = tensor.Name().decode("utf8")
            parameters[name] = tensor.Buffer()
    return parameters


def get_weights(model, graph, tensor_dict, tensor_name):
    i = tensor_dict[tensor_name]
    tensor = graph.Tensors(i)
    buffer = tensor.Buffer()
    shape = get_shape(tensor)
    assert(tensor.Type() == 0 or tensor.Type() == 1)  # FLOAT32
    
    W = model.Buffers(buffer).DataAsNumpy()
    if tensor.Type() == 0:
        W = W.view(dtype=np.float32)
    elif tensor.Type() == 1:
        W = W.view(dtype=np.float16)
    W = W.reshape(shape)
    return W


def get_probable_names(graph):
    probable_names = []
    for i in range(0, graph.TensorsLength()):
        tensor = graph.Tensors(i)
        if tensor.Buffer() > 0 and (tensor.Type() == 0 or tensor.Type() == 1):
            probable_names.append(tensor.Name().decode("utf-8"))
    return probable_names


def get_convert(net, probable_names):
    convert = {}
    i = 0
    for name, params in net.state_dict().items():
        convert[name] = probable_names[i]
        i += 1
    return convert


def build_state_dict(model, graph, tensor_dict, net, convert):
    new_state_dict = OrderedDict()

    for dst, src in convert.items():
        W = get_weights(model, graph, tensor_dict, src)
        print(dst, src, W.shape, net.state_dict()[dst].shape)

        if W.ndim == 4:
            if W.shape[0] == 1:
                W = W.transpose((3, 0, 1, 2))  # depthwise conv
            else:
                W = W.transpose((0, 3, 1, 2))  # regular conv
    
        new_state_dict[dst] = torch.from_numpy(W)
    return new_state_dict


#### >>> BACK <<< ####
back_data = open("./face_detection_back.tflite", "rb").read()

back_model = Model.GetRootAsModel(back_data, 0)
back_subgraph = back_model.Subgraphs(0)
back_subgraph.Name()
print_graph(back_subgraph)

back_tensor_dict = {(back_subgraph.Tensors(i).Name().decode("utf8")): i 
               for i in range(back_subgraph.TensorsLength())}
back_parameters = get_parameters(back_subgraph)
print(len(back_parameters))

W = get_weights(back_model, back_subgraph, back_tensor_dict, "conv2d/Kernel")
b = get_weights(back_model, back_subgraph, back_tensor_dict, "conv2d/Bias")
print(W.shape, b.shape)

back_net = BlazeFace(back_model=True)
print(back_net)

back_probable_names = get_probable_names(back_subgraph)
back_convert = get_convert(back_net, back_probable_names)
back_state_dict = build_state_dict(back_model, back_subgraph, back_tensor_dict, back_net, back_convert)
back_net.load_state_dict(back_state_dict, strict=True)

torch.save(back_net.state_dict(), "blazefaceback.pth")