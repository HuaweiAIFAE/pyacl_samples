import acl
import numpy as np

from cv2 import resize, INTER_AREA
from src.postprocessing import scale_nms_reshape


def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)

    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:])
    print("=" * 95)
    print("model output size", output_size)
    
    for i in range(output_size):
        print("output ", i)
        print("model output dims", acl.mdl.get_output_dims(model_desc, i))
        print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
        # model_output_height = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
    print("=" * 95)
    print("[Model] class Model init resource stage success")
    
    return model_input_height,model_input_width


def preprocessing(img_path,model_desc): # 1) pre-processing stage
    model_input_height, model_input_width = get_sizes(model_desc)
    img_resized = resize(img_path[:, :, ::-1], (model_input_height, model_input_width), 
                            interpolation = INTER_AREA) # bgr to rgb (color space change) & resize
    img_resized = img_resized.transpose(2, 0, 1)  # [h, w, c] to [c, h, w]
    print("[PreProc] img_resized shape", img_resized.shape)

    image_np_expanded = np.expand_dims(img_resized, axis=0)  # NCHW
    image_np_expanded = image_np_expanded.astype('float32') / 255.0 # Converts the image pixels to the range [-1, 1]
    print("[PreProc] image_np_expanded shape", image_np_expanded.shape)
    # image_np_expanded = np.ascontiguousarray(image_np_expanded)
    
    return image_np_expanded, model_input_height, model_input_width


def postprocessing(output, img, conf_thresh=0.4, nms_thresh=0.45 ):
    boxes = scale_nms_reshape(output, img, conf_thresh, nms_thresh)

    return boxes