import acl
import numpy as np
from cv2 import resize, INTER_LINEAR
from src.postprocess import non_max_suppression, scale_coords

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
    print("=" * 95)
    print("[Model] class Model init resource stage success")
    
    return model_input_height,model_input_width

def preprocessing(img,model_desc):
    model_input_height, model_input_width = get_sizes(model_desc)
    print("model_h,model_w",model_input_height, model_input_width)
    img_resized = resize(img, (model_input_width, model_input_height), interpolation=INTER_LINEAR)
    print("resized",img.shape)
    img = img_resized.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    print("transpose",img.shape)
    img = np.ascontiguousarray(img)
    print("cont array",img.shape)
    img = img.astype('float32') / 255.0 # Converts the image pixels to the range [-1, 1]
    
    return img,img_resized,model_input_width,model_input_height

def postprocessing(prediction,img,model_input_width,model_input_height):
    pred = non_max_suppression(prediction, conf_thres=0.45, iou_thres=0.50, classes=None, agnostic=False)
    bboxes = []
    src_img = img
    for i, det in enumerate(pred):  # detections per image
        # Rescale boxes from img_size to im0 size
        if det is not None:
            det[:, :4] = scale_coords((model_input_width, model_input_height), det[:, :4], src_img.shape).round()
            for *xyxy, conf, cls in det:
                bboxes.append([*xyxy, conf, int(cls)])
        else:
            pass
    return bboxes

