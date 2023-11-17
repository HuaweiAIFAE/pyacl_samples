import acl
import cv2
import numpy as np

from src.postprocessing import PostProcess


def get_size(model_desc):
    model_output_dims = []
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)

    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][2:]
        # model_input_element_number = acl.mdl.get_input_dims(model_desc, i)[0]['dimCount']
    print("=" * 95)
    print("model output size", output_size)

    for i in range(output_size): 
        print("output ", i)
        print("model output dims", acl.mdl.get_output_dims(model_desc, i))
        print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
        model_output_dims.append(acl.mdl.get_output_dims(model_desc, i)[0]['dims'])
        model_output_element_number = acl.mdl.get_output_dims(model_desc, i)[0]['dimCount']
    print("=" * 95)
    print("[Model] class Model init resource stage success")
    
    return model_input_height, model_input_width, model_output_dims,model_output_element_number, output_size


def preprocessing(img, model_desc, nms_thresh = 0.4):
    model_input_height, model_input_width, model_output_dims, \
    model_output_element_number, output_size = get_size(model_desc)

    im_ratio = float(img.shape[0]) / img.shape[1]

    input_size = model_input_width, model_input_height
    model_ratio = float(input_size[1]) / input_size[0]
    
    if im_ratio>model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    
    # resize   
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
    det_img[:new_height, :new_width, :] = resized_img
    
    input_size = tuple(det_img.shape[0:2][::-1])
    blob = cv2.dnn.blobFromImage(det_img, 1.0/128, input_size, 
                                (127.5, 127.5, 127.5), swapRB=True)
    
    # preprocessing
    print("[PreProc] image_np_expanded shape:", blob.shape)
    img_resized = np.ascontiguousarray(blob)

    # set inital parameters according to model input size
    batched = False
    if model_output_element_number == 3:
        batched = True
        
    # set initial parameters according to model output size
    use_kps = False
    num_anchors = feat_stride_fpn = fmc = None
    if output_size == 6:
        fms = 1
        fmb = 2
        feat_stride_fpn = [8, 16, 32]
        num_anchors = 2
    elif output_size == 9:
        fms = 1
        fmb = 2
        feat_stride_fpn = [8, 16, 32]
        num_anchors = 2
        use_kps = True
    elif output_size == 10:
        fms = 1
        fmb = 2
        feat_stride_fpn = [8, 16, 32, 64, 128]
        num_anchors = 1
    elif output_size == 15:
        fms = 1
        fmb = 2
        feat_stride_fpn = [8, 16, 32, 64, 128]
        num_anchors = 1
        use_kps = True
    
    # creat postprocess object 
    postprocess = PostProcess (model_output_dims, feat_stride_fpn, nms_thresh, 
                            fms, fmb, num_anchors, batched, use_kps)
    
    return img_resized, postprocess, blob, det_scale


def postprocessing(output_data, postprocess, blob, det_scale, thresh = 0.5):
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        bboxes, kpss = postprocess.detect(output_data, det_scale, input_height, input_width, thresh)
                
        return bboxes, kpss