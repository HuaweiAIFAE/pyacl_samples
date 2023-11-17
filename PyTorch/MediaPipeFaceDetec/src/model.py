import acl
import numpy as np
from cv2 import resize, INTER_AREA


def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)

    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:]
    print("=" * 95)
    print("model output size", output_size)
    
    for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
            # model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:3]
    print("=" * 95)
    print("[Model] class Model init resource stage success")
    
    return model_input_height,model_input_width


def preprocessing(org_img,model_desc):
    model_input_height, model_input_width = get_sizes(model_desc)
    img_resized = resize(org_img[:, :, ::-1], (model_input_height, model_input_width), 
                        interpolation = INTER_AREA) # bgr to rgb (color space change) & resize
    img_resized = img_resized.transpose(2, 0, 1)  # [h, w, c] to [c, h, w]
    print("[PreProc] img_resized shape", img_resized.shape)

    image_np_expanded = np.expand_dims(img_resized, axis=0)  # NCHW
    image_np_expanded = image_np_expanded.astype('float32') / 127.5 - 1.0 # Converts the image pixels to the range [-1, 1]
    print("[PreProc] image_np_expanded shape", image_np_expanded.shape)
    image_np_expanded = np.ascontiguousarray(image_np_expanded)
    
    return image_np_expanded