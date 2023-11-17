import numpy as np
from collections import defaultdict


def decode_bbox(outputs,img_shape,coco_labels,ignore_threshold=0.6):
    ori_w, ori_h = img_shape[:2]
    num_classes = len(coco_labels)
    coco_catIds = np.arange(0,num_classes)
    results = defaultdict(list)
    multi_label_thresh = 0.6
    for out_item in outputs:
        # 52, 52, 3, 85
        x = ori_w * out_item[..., 0].reshape(-1)
        y = ori_h * out_item[..., 1].reshape(-1)
        w = ori_w * out_item[..., 2].reshape(-1)
        h = ori_h * out_item[..., 3].reshape(-1)
        conf = out_item[..., 4:5]
        cls_emb = out_item[..., 5:]
        cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
        x_top_left = x - w / 2.
        y_top_left = y - h / 2.
        cls_emb = cls_emb.reshape(-1, num_classes)
        confidence = conf.reshape(-1, 1) * cls_emb
        flag = cls_emb > multi_label_thresh
        flag = flag.nonzero()
        for i, j in zip(*flag):
            confi = confidence[i][j]
            if confi < ignore_threshold:
                continue
            x_lefti, y_lefti = max(0, x_top_left[i]), max(0, y_top_left[i])
            wi, hi = min(w[i], ori_w), min(h[i], ori_h)
            coco_clsi = coco_catIds[j]
            results[coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
    return results
            
def nms(results):
    det_boxes = []
    for clsi in results:
        dets = results[clsi]
        dets = np.array(dets)
        keep_index = _diou_nms(dets)
        keep_box = [[[dets[i][0].astype(float),
                   dets[i][1].astype(float),
                   dets[i][0].astype(float) + dets[i][2].astype(float),
                   dets[i][1].astype(float) + dets[i][3].astype(float)],dets[i][4].astype(float),int(clsi)]  for i in keep_index]
        det_boxes.extend(keep_box)
    return det_boxes


def _diou_nms(dets, thresh=0.45):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = x1 + dets[:, 2]
    y2 = y1 + dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        center_x1 = (x1[i] + x2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        out_max_x = np.maximum(x2[i], x2[order[1:]])
        out_max_y = np.maximum(y2[i], y2[order[1:]])
        out_min_x = np.minimum(x1[i], x1[order[1:]])
        out_min_y = np.minimum(y1[i], y1[order[1:]])
        outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
        diou = ovr - inter_diag / outer_diag
        diou = np.clip(diou, -1, 1)
        inds = np.where(diou <= thresh)[0]
        order = order[inds + 1]
    return keep            