import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplf


# RGB:
_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
    1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
    0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.857, 0.857, 0.857, 1.000,
    1.000, 1.000
]).astype(np.float32).reshape(-1, 3)


# fmt: on
def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask
        assert m.shape[1] != 2, m.shape
        assert m.shape == (height, width), m.shape
        self._mask = m.astype("uint8")

    @property
    def mask(self):
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(
                    self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    @staticmethod
    def mask_to_polygons(mask):
        mask = np.ascontiguousarray(
            mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP,
                               cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def area(self):
        return self.mask.sum()


def draw_sem_seg(ax, segments, img_shape, default_font_size):
    for seg in segments:
        
        mask = GenericMask(seg, *img_shape)
        color = random_color(rgb=True, maximum=1)
        for segment in mask.polygons:
            segment = segment.reshape(-1, 2)
            polygon = mpl.patches.Polygon(
                segment,
                fill=True,
                facecolor=mplc.to_rgb(color) + (0.5,),
                edgecolor=mplc.to_rgb(color) + (1,),
                linewidth=max(default_font_size // 15, 1),
            )
            ax.add_patch(polygon)


def draw_segment(segments, org_img):
    rgb_img = org_img[:, :, ::-1]
    segments = np.array(segments)
    img_height, img_width = rgb_img.shape[0], rgb_img.shape[1]
    fig = mplf.Figure((img_width / 100, img_height / 100), frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.imshow(rgb_img,
              extent=(0, img_width, img_height, 0),
              interpolation="nearest")

    default_font_size = max(np.sqrt(img_width * img_height) // 90, 10)
    img_shape = (img_height, img_width)
    draw_sem_seg(ax, segments, img_shape, default_font_size)

    return fig


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # 


def draw_label(org_img, mask_pred):
    #get img shape
    img_h, img_w = org_img.shape[:2]

    segments = list()
    mask_pred = softmax(mask_pred)
    mask_pred = mask_pred > 0.5
    mask_pred = mask_pred * 255
    mask_pred = mask_pred.astype("uint8")

    for i in range(1,21):

        if np.any(mask_pred[i][...] > 200):
            mask_pred_ = mask_pred[i, :, :]
            
            im_mask = cv2.resize(mask_pred_, (img_w, img_h))
            segments.append(im_mask)
        
    return draw_segment(segments, org_img)
    