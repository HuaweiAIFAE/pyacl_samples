import mindspore as ms

from model.yolo import YOLOV5s_Infer
from utils.config import config
from utils.moxing_adapter import moxing_wrapper, modelarts_export_preprocess


@moxing_wrapper(pre_process=modelarts_export_preprocess, pre_args=[config])
def run_export():
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        ms.set_context(device_id=config.device_id)

    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    config.file_name = config.file_name + '_' + config.yolov5_version

    network = YOLOV5s_Infer(config.test_img_shape[0], version=dict_version[config.yolov5_version])
    network.set_train(False)

    param_dict = ms.load_checkpoint(config.ckpt_file)
    ms.load_param_into_net(network, param_dict)

    ts = config.test_img_shape[0] // 2
    input_data = ms.numpy.zeros([config.batch_size, 12, ts, ts], ms.float32)

    ms.export(network, input_data, file_name=config.file_name, file_format=config.file_format)
    print('==========success export===============')

if __name__ == "__main__":
    run_export()
