

def postprocess_boxes(result_list, img_shape, model_width, model_height):
    scalex = img_shape[0] / model_width
    scaley = img_shape[1] / model_height
    
    all_detections = []
    for j in range(result_list[1][0][0]):
        bbox = []
        x1 = result_list[0][0][0 * result_list[1][0][0] + j] * scalex
        y1 = result_list[0][0][1 * result_list[1][0][0] + j] * scalex
        x2 = result_list[0][0][2 * result_list[1][0][0] + j] * scalex
        y2 = result_list[0][0][3 * result_list[1][0][0] + j] * scalex
        scores = float(result_list[0][0][4 * result_list[1][0][0] + j])
        classes = result_list[0][0][5 * result_list[1][0][0] + j]
        all_detections.append([x1, y1, x2, y2, scores, classes])
        
    return all_detections