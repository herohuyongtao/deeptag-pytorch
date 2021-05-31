import cv2
import numpy as np


def rotate_image(image, main_idx = 0):
    # Perform the counter clockwise rotation holding at the center
    if main_idx ==1:

        # 90 degrees
        # M = cv2.getRotationMatrix2D(center, angle90, scale)
        # image = cv2.warpAffine(image, M, (h, w))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif main_idx == 2:
        # 180 degrees
        # M = cv2.getRotationMatrix2D(center, angle180, scale)
        # image = cv2.warpAffine(image, M, (w, h))
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif main_idx ==3:
        # 270 degrees
        # M = cv2.getRotationMatrix2D(center, angle270, scale)
        # image = cv2.warpAffine(image, M, (h, w))
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def combine_text_with_pattern(pattern, text_image, area_ratio = 0.5, is_text_white = False):
    if text_image is None or text_image.size ==0: return
    
    text_size = max(text_image.shape[:2])
    pattern_size = max(pattern.shape[:2])
    scale = area_ratio*pattern_size/text_size
    text_image = cv2.resize(text_image, None, fx= scale, fy= scale)
    h, w = text_image.shape[:2]
    ph, pw = pattern.shape[:2]
    pos = int(round((pw-w)/2)), int(round((ph-h)/2))
    region = pattern[pos[1]:pos[1]+h, pos[0]:pos[0]+w]
    if is_text_white:
        region[:,:] = np.maximum(region * (1-text_image), text_image)
    else:
        region[:,:] = np.maximum(region * text_image, text_image)

    # region[:,:] = np.maximum(region[:,:], max_val)

def put_text_on_tight_rectangle(text, font = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, is_text_white = False, rotate_main_idx = 0):
    if text == '': return np.zeros((0,0))
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]    
    image = np.zeros((int(textsize[1]*1.4), int(textsize[0]*1.2)))
    color = (1,1,1)
    cv2.putText(image, text, (int(textsize[0]*0.1),int(textsize[1] * 1.2)),font, fontScale, color, thickness = thickness)
    image = rotate_image(image, main_idx = rotate_main_idx)
    image = tight_crop(image)
    if not is_text_white:
        image = 1-image

    return image

def tight_crop(image):
    # find tight boundary
    sum_list_x = np.sum(image, axis = 0)
    sum_list_y = np.sum(image, axis = 1)
    left = 0
    right = len(sum_list_x)- 1
    top = 0
    bottom = len(sum_list_y)- 1

    flag1 = True
    flag2 = True
    while left < right and (flag1 or flag2):
        if sum_list_x[left] > 0:
            flag1 = False
        else:
            left +=1

        if sum_list_x[right] > 0:
            flag2 = False
        else:
            right -=1 


    flag1 = True
    flag2 = True
    while top < bottom and (flag1 or flag2):
        if sum_list_y[top] > 0:
            flag1 = False
        else:
            top +=1

        if sum_list_y[bottom] > 0:
            flag2 = False
        else:
            bottom -=1 

    image = image[top:bottom+1, left:right+1]

    return image



if __name__ == "__main__":
    from fiducial_marker.elem_patterns import get_solid_circle_pattern
    text = 'AB'
    text_image = put_text_on_tight_rectangle(text, fontScale = 8, thickness= 40, rotate_main_idx= 3)
    print('text image size:', text_image.shape)
    cv2.imshow(text, text_image)

    basic_width = 501
    pattern = get_solid_circle_pattern(basic_width, outer_round_ratio= 0.8)
    combine_text_with_pattern(pattern, text_image, area_ratio = 0.5)
    cv2.imshow('solid circle pattern', pattern)

    cv2.waitKey(0)


    text = 'AB'
    text_image = put_text_on_tight_rectangle(text, fontScale = 8, thickness= 40, rotate_main_idx= 3, is_text_white=True)
    print('text image size:', text_image.shape)
    cv2.imshow(text, text_image)

    basic_width = 501
    pattern = get_solid_circle_pattern(basic_width, outer_round_ratio= 0.8, is_inner_white= False)
    combine_text_with_pattern(pattern, text_image, area_ratio = 0.5)
    cv2.imshow('solid circle pattern', pattern)

    cv2.waitKey(0)