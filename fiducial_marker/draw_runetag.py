import cv2
import math
import numpy as np
def draw_runetag(unit_runetag, scale_in_pixel, is_white_background = True):
    xylim = unit_runetag.get_xylim()    
    x_and_y_with_labels = unit_runetag.get_keypoints_with_labels()
    radius_list = unit_runetag.get_circle_radius()


    w = int(scale_in_pixel * (xylim[1] - xylim[0]))
    image = np.zeros([w,w], dtype= np.float32)
    center = w/2 -0.5


    for radius, kpt_with_id in zip(radius_list, x_and_y_with_labels):
        if  kpt_with_id[2] == 0: continue
        x,y = kpt_with_id[:2]
        add_circle(image, x * scale_in_pixel+ center, y * scale_in_pixel+  center, radius * scale_in_pixel)

    if is_white_background:
        image = 1-image
    return image

def add_circle(image, x, y, radius, is_white = True):
    n_sigma = 4
    tl = [int(x - radius - 1), int(y - radius - 1)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x +radius +1), int(y +radius + 1)]
    map_h, map_w = image.shape
    br[0] = min(br[0], map_w)
    br[1] = min(br[1], map_h)

    if is_white:
        color = 1
    else:
        color = 0

    for map_y in range(tl[1], br[1]):
        for map_x in range(tl[0], br[0]):
            d2 = (map_x  - x) * (map_x - x) + \
                (map_y - y) * (map_y - y)

            d = math.sqrt(d2)
            if d < radius:
                image[map_y, map_x] = color
            # elif d < radius +2:
            #     image[map_y, map_x] = max(radius+1-d,0)
            
def draw_corner_patterns(image, scale_in_pixel, x_and_y_with_labels, pattern_ratio = 0.2, is_white_background= True):
    radius = pattern_ratio/2
    center = image.shape[1]/2 -0.5, image.shape[0]/2 -0.5
    for kpt_with_id in x_and_y_with_labels:
        x,y = kpt_with_id[:2]
        label = kpt_with_id[2]

        if label ==0:
            add_circle(image, x * scale_in_pixel+ center[0], y * scale_in_pixel+  center[1], radius * scale_in_pixel, is_white=not is_white_background)
        else:
            add_circle(image, x * scale_in_pixel+ center[0], y * scale_in_pixel+  center[1], radius * scale_in_pixel, is_white=not is_white_background)
            add_circle(image, x * scale_in_pixel+ center[0], y * scale_in_pixel+  center[1], radius *0.5 *  scale_in_pixel, is_white=is_white_background)


if __name__ == "__main__":
    from fiducial_marker.unit_runetag import *
    import random
    nb_test = 500
    scale_in_pixel = 400
    is_white_background = True
    pattern_ratio_all = [0.1, 0.2, 0.4]
    for _ in range(nb_test):
        kpt_labels = generate_random_binary_ids()
        unit_runetag = UnitRunetag(kpt_labels)
        is_white_background = random.randint(0,2) %2 ==0

        image = draw_runetag(unit_runetag, scale_in_pixel, is_white_background = is_white_background)

        pattern_ratio = pattern_ratio_all[random.randint(0, len(pattern_ratio_all)-1)]
        x_and_y_with_labels = unit_runetag.get_ordered_corners_with_labels()
        draw_corner_patterns(image, scale_in_pixel, x_and_y_with_labels, is_white_background = is_white_background, pattern_ratio=pattern_ratio)
        cv2.imshow('runetag', image)
        c = cv2.waitKey(0)

        if c == 27 or c ==ord('q'): break

