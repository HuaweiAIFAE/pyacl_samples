import numpy as np
import acl
import cv2
from src.postprocessing import postprocess_boxes, letterbox_resize

def get_sizes(model_desc, show_model=False):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    model_input_height,model_input_width = acl.mdl.get_input_dims(model_desc, 0)[0]['dims'][1:3]
    if show_model == True:
        for i in range(input_size):
            print("input ", i)
            print("model input dims", acl.mdl.get_input_dims(model_desc, i))
            print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        print("=" * 95)
        print("model output size", output_size)
        for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
        print("=" * 95)
        return model_input_width, model_input_height
    else:
        return model_input_width, model_input_height


def preprocessing(img,model_desc):
    model_width, model_height = get_sizes(model_desc)
    img = cv2.resize(img, (model_width, model_height), interpolation = cv2.INTER_AREA)
    image_height, image_width = img.shape[:2]
    img_resized = letterbox_resize(img, model_width, model_height)[:, :, ::-1]
    img_resized = np.ascontiguousarray(img_resized)
    return img_resized
                  
    
def postprocessing(infer_output, bgr_img, labels, model_desc):
    box_num = infer_output[1][0, 0]
    box_info = infer_output[0].flatten() 
    model_width, model_height = get_sizes(model_desc)
    bboxes = postprocess_boxes(box_num, box_info ,bgr_img, model_width, model_height, labels)
    return bboxes                  
                  

def construct_image_info(model_desc):
    model_width, model_height = get_sizes(model_desc)
    image_info = np.array([model_width, model_height, 
                           model_width, model_height], 
                           dtype = np.float32) 
    return image_info
