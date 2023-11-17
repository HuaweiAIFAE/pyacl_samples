import numpy as np
import cv2
import acl
from src.postprocessing import decode_bbox, nms


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

def preprocessing(image,model_desc):
    model_input_height,model_input_width = get_sizes(model_desc)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    image = np.expand_dims(img, 0)
    data = np.concatenate((image[..., ::2, ::2],
                           image[..., 1::2, ::2],
                           image[..., ::2, 1::2],
                           image[..., 1::2, 1::2]),
                          axis=1)
    return data


def postprocessing(outputs, img, coco_labels):
    img_shape = tuple(reversed(np.array(img, dtype=np.float32).shape))[1:]
    results = decode_bbox(outputs,img_shape,coco_labels)
    det_boxes = nms(results)
    return det_boxes
