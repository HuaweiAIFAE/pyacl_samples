import acl
from src.postprocessing import non_max_suppression, scale_coords, letterbox

def get_sizes(model_desc, show_model=False):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    if show_model == True:
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
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
        print("=" * 95)
        return model_input_height,model_input_width,model_output_height,model_output_width
    else:
        for i in range(input_size):
            model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])
        for i in range(output_size):
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
        return model_input_height,model_input_width,model_output_height,model_output_width


def preprocessing(org_img,model_desc): 
    model_input_height, model_input_width, \
    model_output_height,model_output_width = get_sizes(model_desc,show_model=True)
    # bgr to rgb (color space change) & resize
    img_resized = letterbox(org_img[:, :, ::-1], (model_input_width, 
                    model_input_height))[0] 
    print("[PreProc] img_resized shape", img_resized.shape)
    return img_resized

def postprocessing(result_list, img, model_desc):
    model_input_height, model_input_width, \
    model_output_height,model_output_width = get_sizes(model_desc)
    
    feature_maps =result_list[0].reshape(-1,model_output_height, model_output_width)
    # apply non-maximum suppression to remove overlapping detections
    pred = non_max_suppression(feature_maps, conf_thres=0.5, iou_thres=0.6, 
                            classes=None, agnostic=False)
    # Process detections
    bboxes = []
    for i, det in enumerate(pred):  # detections per image
        # Rescale boxes from img_size to im0 size
        if det is not None:
            det[:, :4] = scale_coords((model_input_width, model_input_height), 
                        det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                bboxes.append([*xyxy, conf, int(cls)])
        else:
            pass
    return bboxes
        
