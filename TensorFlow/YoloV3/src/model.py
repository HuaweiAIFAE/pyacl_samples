import acl
import numpy as np
from src.postprocessing import letterbox_resize, nms, postprocess_boxes

def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])
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
    image_height, image_width = img.shape[:2]
    img_resized = letterbox_resize(img, model_input_width, model_input_height)[:, :, ::-1]
    img_resized = np.ascontiguousarray(img_resized)
    print("img_resized shape", img_resized.shape)
    return img_resized

def postprocessing(model, result_list, original_image, score_threshold, iou_threshold):
    '''
    postprocessing(model, result_list, original_image, 0.3, 0.45)
    '''
    pred_sbbox = result_list[0].reshape(-1, 85)
    pred_mbbox = result_list[1].reshape(-1, 85)
    pred_lbbox = result_list[2].reshape(-1, 85)
    pred_bbox = np.concatenate([pred_sbbox, \
                                pred_mbbox, \
                                pred_lbbox], axis=0)
    original_image_size = original_image.shape[:2]
    bboxes = postprocess_boxes(pred_bbox, original_image_size, acl.mdl.get_input_dims(model._model_desc, 0)[0]['dims'][1], score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    return bboxes