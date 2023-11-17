import numpy as np
import acl
from PIL import Image
from src.postprocessing import letterbox, non_max_suppression

def get_sizes(model_desc, returnType):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    if returnType:
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
    else:
        for i in range(input_size):
            model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:]
        return model_input_height, model_input_width


def preprocessing(img, model_desc):
    model_input_height, model_input_width = get_sizes(model_desc, True)
    image_org = Image.fromarray(img)
    image = letterbox(image_org, model_input_height, model_input_width)
    image = image.astype(np.float32)
    image = image / 255
    image = image.transpose(2, 0, 1).copy()
    return image, image_org

def postprocessing(infer_output, origin_img, labels, model_desc):
    model_input_height, model_input_width = get_sizes(model_desc, False)
    result_return = non_max_suppression(infer_output, origin_img, model_input_width, model_input_height, labels)   
    return result_return
