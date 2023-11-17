import acl
import cv2
import numpy as np

def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
    print("=" * 50)
    print("[Model] class Model init resource stage success")


def resize_image(img, size):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else \
                    cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask.fill(128)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask.fill(128)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def preprocessing(img,model_desc):
    get_sizes(model_desc)
    img_resized =  resize_image(img,(160, 160))[:, :, ::-1]
    img_resized = img_resized.transpose(2, 0, 1)  # BGR to RGB, to 3x640x640

    image_np = np.array(img_resized, dtype=np.float32) / 255.0
    image_np_expanded = np.expand_dims(image_np, axis=0)  # NCHW
    return image_np_expanded

