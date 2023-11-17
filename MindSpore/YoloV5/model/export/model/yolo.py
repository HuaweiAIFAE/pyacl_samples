import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from model.backbone import YOLOv5Backbone, Conv, BottleneckCSP

from utils.config import config as default_config

class YOLO(nn.Cell):
    def __init__(self, backbone, shape):
        super(YOLO, self).__init__()
        self.backbone = backbone
        self.config = default_config
        self.config.out_channel = (self.config.num_classes + 5) * 3

        self.conv1 = Conv(shape[5], shape[4], k=1, s=1)
        self.CSP5 = BottleneckCSP(shape[5], shape[4], n=1*shape[6], shortcut=False)
        self.conv2 = Conv(shape[4], shape[3], k=1, s=1)
        self.CSP6 = BottleneckCSP(shape[4], shape[3], n=1*shape[6], shortcut=False)
        self.conv3 = Conv(shape[3], shape[3], k=3, s=2)
        self.CSP7 = BottleneckCSP(shape[4], shape[4], n=1*shape[6], shortcut=False)
        self.conv4 = Conv(shape[4], shape[4], k=3, s=2)
        self.CSP8 = BottleneckCSP(shape[5], shape[5], n=1*shape[6], shortcut=False)
        self.back_block1 = YoloBlock(shape[3], self.config.out_channel)
        self.back_block2 = YoloBlock(shape[4], self.config.out_channel)
        self.back_block3 = YoloBlock(shape[5], self.config.out_channel)

        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """
        input_shape of x is (batch_size, 3, h, w)
        feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        """
        img_height = x.shape[2] * 2
        img_width = x.shape[3] * 2

        feature_map1, feature_map2, feature_map3 = self.backbone(x)

        c1 = self.conv1(feature_map3)
        ups1 = ops.ResizeNearestNeighbor((img_height // 16, img_width // 16))(c1)
        c2 = self.concat((ups1, feature_map2))
        c3 = self.CSP5(c2)
        c4 = self.conv2(c3)
        ups2 = ops.ResizeNearestNeighbor((img_height // 8, img_width // 8))(c4)
        c5 = self.concat((ups2, feature_map1))
        # out
        c6 = self.CSP6(c5)
        c7 = self.conv3(c6)

        c8 = self.concat((c7, c4))
        # out
        c9 = self.CSP7(c8)
        c10 = self.conv4(c9)
        c11 = self.concat((c10, c1))
        # out
        c12 = self.CSP8(c11)
        small_object_output = self.back_block1(c6)
        medium_object_output = self.back_block2(c9)
        big_object_output = self.back_block3(c12)
        return small_object_output, medium_object_output, big_object_output


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv5.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(12, 255)

    """
    def __init__(self, in_channels, out_channels):
        super(YoloBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        """construct method"""

        out = self.conv(x)
        return out

class DetectionBlock(nn.Cell):
    """
     YOLOv5 detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: config, Configuration instance.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32)
     """

    def __init__(self, scale, config=default_config, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
            self.scale_x_y = 1.2
            self.offset_x_y = 0.1
        elif scale == 'm':
            idx = (3, 4, 5)
            self.scale_x_y = 1.1
            self.offset_x_y = 0.05
        elif scale == 'l':
            idx = (6, 7, 8)
            self.scale_x_y = 1.05
            self.offset_x_y = 0.025
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4+1+self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        self.pow = ops.Pow()
        self.transpose = ops.Transpose()
        self.exp = ops.Exp()
        self.conf_training = is_training

    def construct(self, x, input_shape):
        """construct method"""
        num_batch = x.shape[0]
        grid_size = x.shape[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = self.reshape(x, (num_batch,
                                      self.num_anchors_per_scale,
                                      self.num_attrib,
                                      grid_size[0],
                                      grid_size[1]))
        prediction = self.transpose(prediction, (0, 3, 4, 1, 2))

        grid_x = ms.numpy.arange(grid_size[1])
        grid_y = ms.numpy.arange(grid_size[0])
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.scale_x_y * self.sigmoid(box_xy) - self.offset_x_y + grid) / \
                 ops.cast(ops.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = self.exp(box_wh) * self.anchors / input_shape

        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.conf_training:
            return prediction, box_xy, box_wh
        return self.concat((box_xy, box_wh, box_confidence, box_probs))


class YOLOV5(nn.Cell):
    """
    YOLOV5 network.

    Args:
        is_training: Bool. Whether train or not.

    Returns:
        Cell, cell instance of YOLOV5 neural network.

    Examples:
        YOLOV5s(True)
    """

    def __init__(self, is_training, version=0):
        super(YOLOV5, self).__init__()
        self.config = default_config

        # YOLOv5 network
        self.shape = self.config.input_shape[version]
        self.feature_map = YOLO(backbone=YOLOv5Backbone(shape=self.shape), shape=self.shape)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training)
        self.detect_2 = DetectionBlock('m', is_training=is_training)
        self.detect_3 = DetectionBlock('s', is_training=is_training)

    def construct(self, x, input_shape):
        small_object_output, medium_object_output, big_object_output = self.feature_map(x)
        output_big = self.detect_1(big_object_output, input_shape)
        output_me = self.detect_2(medium_object_output, input_shape)
        output_small = self.detect_3(small_object_output, input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


class YOLOV5s_Infer(nn.Cell):
    """
    YOLOV5 Infer.
    """

    def __init__(self, input_shape, version=0):
        super(YOLOV5s_Infer, self).__init__()
        self.network = YOLOV5(is_training=False, version=version)
        self.input_shape = input_shape

    def construct(self, x):
        return self.network(x, self.input_shape)
