import cv2
import numpy as np


class ElemPatterns:
    def __init__(self, basic_width, patterns, is_resize = True):
        self.basic_width = basic_width
        if basic_width is not None and is_resize:
            patterns = [cv2.resize(pattern, (basic_width, basic_width), interpolation= cv2.INTER_LINEAR) for pattern in patterns]


        self.patterns = patterns

    def get_pattern(self, label, sub_label = None):
        return self.patterns[label].copy()


def get_circle_pattern(basic_width, outer_round_ratio = 1, inner_round_ratio = 0.5, is_inner_white = True):
    center = [(basic_width - 1)/2, (basic_width - 1)/2]
    if is_inner_white:
        pattern = np.ones((basic_width, basic_width))
    else:
        pattern = np.zeros((basic_width, basic_width))

    radius = (basic_width + 1)/2 * outer_round_ratio
    radius_inner = radius * inner_round_ratio
    draw_solid_circle(pattern, center, radius, not is_inner_white)
    draw_solid_circle(pattern, center, radius_inner, is_inner_white)

    return pattern

def get_solid_circle_pattern(basic_width, outer_round_ratio = 1,is_inner_white = True):
    center = [(basic_width - 1)/2, (basic_width - 1)/2]
    if is_inner_white:
        pattern = np.zeros((basic_width, basic_width))
    else:
        pattern = np.ones((basic_width, basic_width))

    radius = (basic_width + 1)/2 * outer_round_ratio
    draw_solid_circle(pattern, center, radius, is_inner_white)

    return pattern

def get_square_pattern(basic_width, outer_radius, inner_radius, is_inner_white = True):
    center = [(basic_width - 1)/2, (basic_width - 1)/2]
    if is_inner_white:
        pattern = np.ones((basic_width, basic_width))
    else:
        pattern = np.zeros((basic_width, basic_width))

    draw_solid_square(pattern, center, outer_radius, not is_inner_white)
    draw_solid_square(pattern, center, inner_radius, is_inner_white)


    return pattern


def get_cross_pattern(basic_width, outer_space_width, border_width = 0, is_inner_white = False):
    center = [(basic_width - 1)/2, (basic_width - 1)/2]
    pattern = np.zeros((basic_width, basic_width))
    if is_inner_white:
        pattern = np.ones((basic_width, basic_width))
        color = 0
    else:
        pattern = np.zeros((basic_width, basic_width))
        color = 1

    osw = int(outer_space_width)
    # outer space
    pattern[:osw, :osw] = color
    pattern[:osw, -osw:] = color
    pattern[-osw:, :osw] = color
    pattern[-osw:, -osw:] = color


    # border
    if border_width >0:
        bd = int(border_width)
        pattern[:bd,:] = color
        pattern[-bd:,:] = color
        pattern[:, :bd] = color
        pattern[:, -bd:] = color

    return pattern

def draw_solid_square(image, center, radius, is_white):
    if is_white:
        color = 1
    else:
        color = 0


    image[int(center[1] - radius):int(center[1] + radius+1), int(center[0] - radius):int(center[0] + radius+1)] = color

def draw_solid_circle(image, center, radius, is_white):
    if is_white:
        color = 1
    else:
        color = 0

    thickness = -1
    cv2.circle(image, (int(center[0]), int(center[1])), int(radius), color, thickness) 

if __name__ == "__main__":
    basic_width = 501
    pattern = get_circle_pattern(basic_width, inner_round_ratio= 0.5)
    cv2.imshow('circle pattern', pattern)

    pattern = get_solid_circle_pattern(basic_width, outer_round_ratio= 0.8)
    cv2.imshow('solid circle pattern', pattern)

    outer_space_width = int((basic_width-1)/3)
    pattern = get_cross_pattern(basic_width, outer_space_width=outer_space_width, is_inner_white = True)
    cv2.imshow('cross pattern', pattern)


    outer_radius = basic_width//2
    inner_radius = basic_width//4
    pattern = get_square_pattern(basic_width, outer_radius, inner_radius)
    cv2.imshow('square pattern', pattern)
    cv2.waitKey(0)
    