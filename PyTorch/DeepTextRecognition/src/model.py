import acl
import cv2
import numpy as np
from functools import cmp_to_key
from src.postprocessing import Process
from src.box_utils import BBox, compare_alldims


def _read_bboxes(boxes_path, image):
        # read coordinates of extracted bounding boxes from text file
        with open(boxes_path, 'r') as bbox_f:
            box_coords = [line.rstrip('\n') for line in bbox_f]
        vertical_positions = np.argsort(np.asarray([int(y0.split(',')[1]) for y0 in box_coords]))
        
        # create bounding box objects
        bboxes = []
        
        # save box coordinates from EAST output which is pixel location of each vertex of rectangle
        print("[INFO] reading text boxes . . .")
        for i in vertical_positions:
            line = box_coords[i]
            box_coord = [int(i) for i in line.split(',')]
            bboxes.append(BBox(box_coord, 1, 0, image))
        bboxes_sorted = sorted(bboxes, key=cmp_to_key(compare_alldims))
        
        print('[INFO]  {} text boxes found.'.format(len(bboxes_sorted)))
        
        return bboxes_sorted


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


def preprocessing(path_dict,model_desc,char_list):
    model_input_height, model_input_width = get_sizes(model_desc)
    org_img = cv2.imread(path_dict['img_path'])
    bboxes_sorted = _read_bboxes(path_dict['boxes_path'], org_img)
    # pre-processing
    print("[INFO] recognizing texts . . .")
    # crop textboxes from image
    cropped_imgs = []
    for box in bboxes_sorted:
        # crop each rectangle, enlarged by horizontal offset value, from original image
        cropped_imgs.append(box.get_crop(0, 0))
    process = Process(char_list, bboxes_sorted)
    
    return process, cropped_imgs, model_input_height, model_input_width, bboxes_sorted


def run(model,path_dict,model_desc,char_list):
    
    process, cropped_imgs, model_input_height, \
    model_input_width, bboxes_sorted = preprocessing(path_dict,model_desc,char_list)
    
    for i, cropped_img in enumerate(cropped_imgs):
        img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)/127.5 - 1.0
        img_resized = cv2.resize(img_gray,(model_input_width, model_input_height ),
                                interpolation=cv2.INTER_CUBIC)
        img_np_expanded = np.expand_dims(np.float32(img_resized), (0,1))

        # print("[PreProc] image_np_expanded shape:", img_np_expanded.shape)
        img_resized = np.ascontiguousarray(img_np_expanded)
        data = img_resized
        
        result_list = model.execute([data,])
        
        preds = np.array(result_list)
        preds = preds[0]
        # print('PRED SHAPE', preds.shape)

        process.text_boxes(preds, i) # recognize corresponding text for each text box

    print("[INFO] recognition done . . .")
    
    if len(bboxes_sorted)==0:
        return len(bboxes_sorted)

    # concatenate text boxes that mistakenly detected as two words but actually is one word
    bboxes_sorted = process.get_concat_same() 

    return bboxes_sorted