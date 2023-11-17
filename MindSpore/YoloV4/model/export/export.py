import os
import numpy as np
from mindspore import context, Tensor, float32
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from utils.yolo import YOLOV4CspDarkNet53
from model.config import config
from model.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)
    ts_shape = config.testing_shape

    network = YOLOV4CspDarkNet53()
    network.set_train(False)

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)

    input_data = Tensor(np.zeros([config.batch_size, 3, ts_shape, ts_shape]),float32)

    export(network, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
    run_export()
