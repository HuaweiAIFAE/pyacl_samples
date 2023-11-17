import cv2
import numpy as np

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]
    resize_ratio = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img
    return image_padded


def postprocess_boxes(box_num, box_info ,bgr_img, model_width, model_height, labels):
    scalex = bgr_img.shape[1] / model_width
    scaley = bgr_img.shape[0] / model_height
    bboxes = []
    for n in range(int(box_num)):
        ids = int(box_info[5 * int(box_num) + n]) 
        label = labels[ids] 
        score = box_info[4 * int(box_num)+n]
        top_left_x = box_info[0 * int(box_num) + n] * scalex
        top_left_y = box_info[1 * int(box_num) + n] * scaley
        bottom_right_x = box_info[2 * int(box_num) + n] * scalex
        bottom_right_y = box_info[3 * int(box_num) + n] * scaley
        print(" % s: class % d, box % d % d % d % d, score % f" % (
            label, ids, top_left_x, top_left_y, 
            bottom_right_x, bottom_right_y, score))
        bboxes.append([top_left_x, top_left_y, bottom_right_x,
                      bottom_right_y, score, ids])
    return bboxes