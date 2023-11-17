import cv2
import math
import numpy as np


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def slope(self, other):
        return np.rad2deg(np.arctan2(-(self.y - other.y), self.x - other.x))
    
    def __repr__(self):
        return "x:{:2d}, y:{:2d}".format(self.x, self.y)


class BBox:

    def __init__(self, box_coord, offset_x, offset_y, image, text='', keep_line_height=False):
        points = _get_sorted_points(box_coord)
        self.box=box_coord
        self.points = points
        self.x_left = min(points, key=lambda p: p.x).x - offset_x  # left of box
        self.y_top = min(points, key=lambda p: p.y).y - offset_y  # top of box
        self.x_right = max(points, key=lambda p: p.x).x + offset_x  # right of box
        self.y_bottom = max(points, key=lambda p: p.y).y + offset_y  # bottom of box
        self.height = self.y_bottom - self.y_top
        self.width = self.x_right - self.x_left
        if not keep_line_height:
            self.line_height = self.height
        self.offset_x = offset_x  # offset to enlarge the box to capture context better
        self.offset_y = offset_y  # offset to enlarge the box to capture context better
        self.image = image
        self.text = text
        self.slope = points[2].slope(points[3])

    def size(self):
        return self.width, self.height

    def get_points(self):
        return self.box 
        
    def get_text(self):
        return self.text

    def get_crop(self, offset_x, offset_y):
        """
        Takes opencv image and returns cropped bounding box
        """
        cropped = self.image[get_plus(self.y_top - offset_y):get_plus(self.y_bottom + offset_y),
                                           get_plus(self.x_left - offset_x):get_plus(self.x_right + offset_x)]
        
        # if box is a vertical text, rotate the box (accepted vertical if ratio of edges is smaller than .7
        if (self.width + 2*offset_x) / (self.height + 2*offset_y) < .7:
            print("#"*50)
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        return cropped
    
    def is_sameline(self, other):
        """
        Checks if two boxes are in the same line. Two boxes are in the same line if
        both box's middle point is in the same horizontal area of other box
        """
        self_y_middle = self.y_top + self.height / 2
        other_y_middle = other.y_top + other.height / 2
        if self.y_bottom > other_y_middle > self.y_top and other.y_bottom > self_y_middle > other.y_top:
            return True
        return False

    def is_horizontally_intersecting(self, other):
        """
        Check if two boxes intersect if boxes are top aligned
        :return:
        """
        if (self.x_left < other.x_left < self.x_right) or (other.x_left < self.x_left < other.x_right):
            if self.is_sameline(other):
                return True
        return False

    def merge_intersecting_box(self, other):
        """
        Merge two boxes if they are in the same line, mistakenly detected as two boxes and form a word together
        Difference from merge_box is the check on text similarities of two boxes
        """
        if self.text[-1] == other.text[0] and not self.text[-1].isdigit():
            self.text += other.text[1:]
        elif self.text[-2:] == other.text[:2] and not self.text[-2:].isdigit():
            self.text += other.text[2:]
        elif self.text[-3:] == other.text[:3] and not self.text[-3:].isdigit():
            self.text += other.text[3:]
            
        self.x_left = self.x_left if self.x_left < other.x_left else other.x_left
        self.y_top = self.y_top if self.y_top < other.y_top else other.y_top
        self.x_right = self.x_right if self.x_right > other.x_right else other.x_right
        self.y_bottom = self.y_bottom if self.y_bottom > other.y_bottom else other.y_bottom
        self.height = self.y_bottom - self.y_top
        self.width = self.x_right - self.x_left
        self.line_height = self.height
        
    def __str__(self):
        return "X: {}-{}, Y: {}-{}, H-W-LH: {}-{}-{}, text: {}".format(
            self.x_left, self.x_right, self.y_top, self.y_bottom, self.height, self.width, self.line_height, self.text)
    
    def __repr__(self):
        return "X: {}-{}, Y: {}-{}, H-W-LH: {}-{}-{}, text: {}".format(
            self.x_left, self.x_right, self.y_top, self.y_bottom, self.height, self.width, self.line_height, self.text)
    
    
def get_plus(points):
        return max(0, points)
    
    
def compare_alldims(this, other):
    """
    Compares two bounding boxes: one box is less than other if:
        it is above the other box positionally with some threshold
          or if they are horizontally aligned:
        box on left is less than other box
    :param this:
    :param other:
    :return:
    """
    topalign_percentage = .5
    top_alignment_threshold = min(this.line_height, other.line_height) * topalign_percentage
    if this.is_sameline(other) or abs(this.y_top - other.y_top) < top_alignment_threshold:
        return this.x_left - other.x_left
    return this.y_top - other.y_top


def compare_titledims(this, other):
    """
    Compares two bounding boxes: one box is less than other if:
        it has a smaller line height, if line heights are close (less that 10 percent of bigger one)
            box that is below other one is LESS THAN other
    :param this:
    :param other:
    """
    line_height_percentage = .1
    line_height_threshold = max(this.line_height, other.line_height) * line_height_percentage
    if abs(this.line_height - other.line_height) > line_height_threshold:
        return this.line_height - other.line_height
    return other.y_top - this.y_top


def _get_sorted_points(box_coords):
    """
    Return a list of 4 Points (representing 4 corners of rectangle) in clockwise order starting from top left corner
    :param box_coords:
    :return:
    """
    # coordinates of points are given in clockwise order
    points = [Point(box_coords[0], box_coords[1]), Point(box_coords[2], box_coords[3]),
              Point(box_coords[4], box_coords[5]), Point(box_coords[6], box_coords[7])]
    # sort points wrt x coords to find 2 corners on the left
    x_sorted_idx = sorted(range(len(points)), key=lambda k: points[k].x)
    left_corners = [points[i] for i in x_sorted_idx[:2]]
    # the corner point with smaller y coord is the top left point of rectangle
    y_sorted_left_corners_idx = sorted(range(len(left_corners)), key=lambda k: left_corners[k].y)
    top_left_idx = x_sorted_idx[y_sorted_left_corners_idx[0]]

    # return points in clockwise order starting from top left corner
    points = [points[i % len(points)] for i in range(top_left_idx, top_left_idx+len(points))]
    return points
