import cv2
import numpy as np
from skimage import io


def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    # resize and get dimensions
    before_h, before_w, _ = img.shape
    img = cv2.resize(img, dsize=(square_size[0], square_size[1]), interpolation = interpolation)
    after_h, after_w, chanel = img.shape

    # magnify image size
    target_size = mag_ratio * max(after_h, after_w)

    # set original image size
    if target_size > square_size[0]:
        target_size = square_size[0]
    
    ratio = target_size / max(after_h, after_w)    

    target_h, target_w = int(after_h * ratio), int(after_w * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, chanel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc

    # calculate aspect for each dimension
    ratio_w = 1 / (square_size[0] / before_w)
    ratio_h =  1 / (square_size[1] / before_h)
    
    return resized, ratio_h, ratio_w


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    
    return img