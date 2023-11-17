import acl
import cv2
import numpy as np
from src.util import padRightDownCorner
from src.postprocessing import getOutputs, calculatePeaks, findConnections, findSubset, drawHumans

def get_sizes(model_desc):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    print("model input size", input_size)
    for i in range(input_size):
        print("input ", i)
        print("model input dims", acl.mdl.get_input_dims(model_desc, i))
        print("model input datatype", acl.mdl.get_input_data_type(model_desc, i))
        model_input_height, model_input_width = acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3]
    print("=" * 50)
    print("model output size", output_size)
    for i in range(output_size):
            print("output ", i)
            print("model output dims", acl.mdl.get_output_dims(model_desc, i))
            print("model output datatype", acl.mdl.get_output_data_type(model_desc, i))
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:3]
    print("=" * 50)
    print("[Model] class Model init resource stage success")
    return model_input_height,model_input_width


def preprocessing(img_path,model_desc):
    model_input_height, model_input_width = get_sizes(model_desc)
    imageToTest = cv2.resize(img_path, (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 128)
    img_resized = cv2.resize(imageToTest_padded, (model_input_width, model_input_height))[:, :, ::-1]
    img_resized = img_resized.astype(np.float32)
    return img_resized,pad


def postprocessing(result_list,img_org_bgr,data,pad):
    # Seperate outputs as heatmap and Part Affinity Fields
    heatmap, paf = getOutputs(result_list,img_org_bgr,data,pad)
    # Calculate peaks
    all_peaks = calculatePeaks(heatmap)
    # Find connections 
    connection_all, special_k = findConnections(paf, all_peaks, img_org_bgr)
    # Find subset value for visualizing outputs
    subset, candidate= findSubset(connection_all, all_peaks, special_k)
    # Draw outputs
    canvas = drawHumans(subset,img_org_bgr,candidate)
    return canvas