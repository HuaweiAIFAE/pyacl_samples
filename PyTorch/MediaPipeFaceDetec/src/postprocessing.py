import numpy as np
import warnings

# suppress warnings
warnings.filterwarnings('ignore')


def sigmoid(x0):
    s = 1 / (1 + np.exp(-x0))
    return s


def decode_boxes(raw_boxes, anchors, input_size):
    """
    Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / input_size[0] * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / input_size[1] * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / input_size[0] * anchors[:, 2]
    h = raw_boxes[..., 3] / input_size[1] * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset    ] / input_size[0] * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / input_size[1] * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset    ] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


### IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py ###
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = np.size(box_a, 0)
    B = np.size(box_b, 0)
    
    max_xy = np.minimum(np.resize(np.expand_dims(box_a[:, 2:], axis=1), (A, B, 2)),
                                np.resize(np.expand_dims(box_b[:, 2:], axis=0), (A, B, 2)))
    min_xy =  np.maximum(np.resize(np.expand_dims(box_a[:, :2], axis=1), (A, B, 2)),
                                np.resize(np.expand_dims(box_b[:, :2], axis=0), (A, B, 2)))
    
    inter = np.clip((max_xy - min_xy), 0, None)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    
    area_a = np.resize(np.expand_dims((box_a[:, 2]-box_a[:, 0])*
                                      (box_a[:, 3]-box_a[:, 1]), axis=1), inter.shape)  # [A,B]
    area_b = np.resize(np.expand_dims((box_b[:, 0]-box_b[:, 0])*
                                      (box_b[:, 3]-box_b[:, 1]), axis=1), inter.shape)  # [A,B]
    union = area_a + area_b - inter
    
    return (inter / union)  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return np.squeeze(jaccard(np.expand_dims(box, axis=0), other_boxes), axis=0)


def detect(detections, img):
    '''
    The input image should be 128x128 for the short_range model and 
    256x256 for the full_range model. BlazeFace will not automatically 
    resize the image, you have to do this yourself!
    '''
    #print("Found %d faces" % front_detections.shape[0])
    bboxes = []
    landmarks = []
    for i in range(detections.shape[0]):
        ymin = int((detections[i, 0] * img.shape[0]).item())
        xmin = int((detections[i, 1] * img.shape[1]).item())
        ymax = int((detections[i, 2] * img.shape[0]).item())
        xmax = int((detections[i, 3] * img.shape[1]).item())
        bboxes.append([xmin, ymin, xmax, ymax])
        
        points = []
        for k in range(6):
            kp_x = int((detections[i, 4 + k*2    ] * img.shape[1]).item())
            kp_y = int((detections[i, 4 + k*2 + 1] * img.shape[0]).item())
            points.append((kp_x, kp_y))
        landmarks.append(points)
        
    return bboxes, landmarks


def array2detections(raw_boxes, raw_scores, input_size, anchors, clipping_thresh = 100.0, score_thresh = 0.5):
    """
    The output of the neural network is a tensor of shape (b, 896, 16)
    containing the bounding box regressor predictions, as well as a tensor 
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) tensors, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    assert len(raw_boxes.shape) == 3
    assert raw_boxes.shape[1] == 896 # num_anchors
    assert raw_boxes.shape[2] == 16 # num_coords

    assert len(raw_scores.shape) == 3
    assert raw_scores.shape[1] == 896 # num_anchors
    assert raw_scores.shape[2] == 1 # num_classes
    assert raw_boxes.shape[0] == raw_scores.shape[0]
        
    detection_boxes = decode_boxes(raw_boxes, anchors, input_size)
        
    raw_scores = np.clip(raw_scores, -clipping_thresh, clipping_thresh)
    detection_scores = np.squeeze(sigmoid(raw_scores), axis=-1)
    #detection_scores = np.squeeze(raw_scores, axis=-1)
        
    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_boxes.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
        output_detections.append(np.concatenate((boxes, scores), axis=-1))

    return output_detections


def nms(detections, min_suppression_thresh=0.3):
    """
    The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.
        
    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0: return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    remaining = detections[:, 16].argsort()[::-1]
    
    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other 
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_thresh
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]
            total_score = scores.sum()
            weighted = np.sum(coordinates * scores, axis=0) / total_score
            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections