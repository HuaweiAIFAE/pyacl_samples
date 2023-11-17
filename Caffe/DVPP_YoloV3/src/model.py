import numpy as np
import acl
from src.postprocessing import postprocess_boxes

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


def preprocessing(image, dvpp, model_desc):
    model_input_width, model_input_height = get_sizes(model_desc,show_model=True)
    image_input = image.copy_to_dvpp()
    yuv_image = dvpp.jpegd(image_input)
    print("[Image] Image decoded.")
    resized_image = dvpp.crop_and_paste(yuv_image, image.width, image.height, model_input_width, model_input_height)
    print("[Image] YUV image resized.")
    return resized_image


def postprocessing(result_list, img, model_desc):
    model_input_width, model_input_height = get_sizes(model_desc)
    image_shape = [img.width, img.height]
    bboxes = postprocess_boxes(result_list, image_shape, model_input_height,model_input_width)
    return bboxes


def construct_image_info(model_desc):
    model_width, model_height = get_sizes(model_desc)
    image_info = np.array([model_width, model_height, 
                           model_width, model_height], 
                           dtype = np.float32) 
    return image_info


