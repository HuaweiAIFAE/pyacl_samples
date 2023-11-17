import acl
import numpy as np
from cv2 import resize, INTER_NEAREST

def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])
    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
        print("output ", i)
        print("model output dims", acl.mdl.get_output_dims(model_desc, i))
        print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
        model_output = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
    print("=" * 50)
    print("[Model] class Model init resource stage success")
    return model_input_height,model_input_width,model_output

def preprocessing(img,model_desc):
    model_input_height, model_input_width ,_  = get_sizes(model_desc)
    img_resized = letterbox_resize(img, model_input_width, model_input_height)[:, :, ::-1]
    img_resized = (img_resized / 255).astype(np.float32)
    img_resized = np.ascontiguousarray(img_resized)
    return img_resized.astype(np.float32)

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]
    
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)
    
    img = resize(img, (resize_w, resize_h), interpolation=INTER_NEAREST)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    return image_padded