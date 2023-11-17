import mindspore
import numpy as np
# Replace the following classpath according to the actual situation.
from blazefaceback import MindSporeModel as MindSporeNetwork


network = MindSporeNetwork()
param_dict = mindspore.load_checkpoint('blazefaceback.ckpt')
mindspore.load_param_into_net(network, param_dict)

# Generate input data
input_data = mindspore.numpy.zeros([1, 3, 256, 256], mindspore.float32)

# Generate the AIR file.
mindspore.export(network, mindspore.Tensor(input_data), file_name='air_blazeface_back', file_format='AIR')