import cv2
import numpy as np
import acl
from src.postprocessing import draw_label


def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:]
    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:3]
            
    print("=" * 50)
    print("[Model] class Model init resource stage success")
    return model_input_height,model_input_width


def preprocessing(picPath,model_desc):
    model_input_height,model_input_width = get_sizes(model_desc)
    bgr_img_ = cv2.imread(picPath).astype(np.uint8)

    img = cv2.resize(bgr_img_, (model_input_width, model_input_height))
    img=img.astype(np.float32, copy=False)

    img[:, :, 0] -= 104
    img[:, :, 0] = img[:, :, 0] / 57.375
    img[:, :, 1] -= 117
    img[:, :, 1] = img[:, :, 1] / 57.120
    img[:, :, 2] -= 123
    img[:, :, 2] = img[:, :, 2] / 58.395
    img = img.transpose([2, 0, 1]).copy()
    return img


def postprocessing(result_list, pic):
    result = draw_label(pic, result_list[0].squeeze())
    return result