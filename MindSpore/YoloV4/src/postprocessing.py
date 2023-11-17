import numpy as np

def letterbox(image, net_h, net_w):
    img_w, img_h = image.size[0:2]

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h

    image_ = image.resize((new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    return new_image

def get_anchor_list():
    stride_list = [32, 16, 8]
    anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
    anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
    anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
    anchor_list = [anchors_1, anchors_2, anchors_3]
    return anchor_list

def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres, class_num):
    res = []
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue
            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res


def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio, class_num):
    _, h, w, _, _ = conv_output.shape 
    conv_output = conv_output.transpose(0, 1, 2, 3, -1)
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)    
    pred = pred[pred[:, 4] >= 0.65]

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes

def convert_labels(label_list, labels):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def non_max_suppression(infer_output, origin_img, model_input_width, model_input_height, labels, iou_threshold=0.8):
    result_return = dict()

    img_h = origin_img.size[1]
    img_w = origin_img.size[0]
    
    scale = min(float(model_input_width) / float(img_w), float(model_input_height) / float(img_h))
    
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    shift_x_ratio = (model_input_width - new_w) / 2.0 / model_input_width
    shift_y_ratio = (model_input_height - new_h) / 2.0 / model_input_height
    
    class_number = len(labels)+1 
    num_channel = 3 * (class_number + 5)
    
    x_scale = model_input_width / float(new_w)
    y_scale = model_input_height / float(new_h)
    
    all_boxes = [[] for ix in range(class_number)]
    anchor_list = get_anchor_list()
    
    for ix in range(3):    
        pred = infer_output[ix]
        anchors = anchor_list[ix]
        boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio, class_number)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]
    
    res = apply_nms(all_boxes, iou_threshold, class_number)
    
    bboxes = []
    if not res:
        return bboxes
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4], labels)
        picked_score = new_res[:, 5]
        
        for k in range(new_res[:,0].shape[0]):
            bboxes.append([new_res[k, 0:4].tolist(), new_res[k, 5], new_res[k, 4]])
            
        return bboxes
