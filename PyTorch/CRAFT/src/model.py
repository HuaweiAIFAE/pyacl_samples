import acl
import numpy as np

from src.imgproc import loadImage, resize_aspect_ratio, \
                    normalizeMeanVariance


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
        if i ==0:
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
            # model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:3]
    print("=" * 95)
    print("[Model] class Model init resource stage success")

    return model_input_height,model_input_width


def preprocessing(img_path,model_desc):
    model_input_height, model_input_width = get_sizes(model_desc)
    image = loadImage(img_path)
    # resize
    mag_ratio = round((model_input_width / model_input_height), 1)
    img_resized,  ratio_h, ratio_w = resize_aspect_ratio(image, (model_input_width, model_input_height), 1, mag_ratio)

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2,0,1))    # [h, w, c] to [c, h, w]               
    image_np_expanded = np.expand_dims(x, axis=0)   # [c, h, w] to [b, c, h, w]
    print("[PreProc] image_np_expanded shape:", image_np_expanded.shape)
    
    img_resized = np.ascontiguousarray(image_np_expanded)
    
    return img_resized, ratio_h, ratio_w