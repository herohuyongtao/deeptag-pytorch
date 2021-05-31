import cv2
import numpy as np
from fiducial_marker.draw_text import put_text_on_tight_rectangle, combine_text_with_pattern
import random
class RandomTextGenerator:
    def __init__(self, list_len = 100,candidates = None, max_char_num = 1, empty_text_ratio = 0.5, area_ratio_range = [0.2, 0.6], is_random_rotate = True):
        if candidates is None:candidates = [chr(ord('A')+ii) for ii in range(26)]
        self.candidates = candidates
        self.empty_text_ratio =empty_text_ratio
        self.max_char_num = max_char_num
        self.area_ratio_range =area_ratio_range
        self.is_random_rotate =is_random_rotate
        self.list_len = list_len

    def get_random_text_list(self, empty_text_ratio=None):
        candidates = self.candidates
        if empty_text_ratio is None:
            empty_text_ratio =self.empty_text_ratio
        max_char_num =self.max_char_num
        list_len = self.list_len

        text_list = []
        for _ in range(list_len):
            text = ''
            if random.random() > empty_text_ratio:
    
                char_num  = random.randint(1, max_char_num)            
                for _ in range(char_num):
                    text +=candidates[random.randint(0,len(candidates)-1)]

            
            text_list.append(text)
        return text_list

    def get_random_text_param_list(self):
        '''
            list of (fontScale, thickness_ratio, rotate_main_idx, area_ratio)
        '''
        list_len = self.list_len
        area_ratio_range = self.area_ratio_range
        is_random_rotate =self.is_random_rotate

        text_param_list = []
        for _ in range(list_len):  
            fontScale =    8       
            
            thickness_ratio = random.randint(2, 6)
            if is_random_rotate:
                rotate_main_idx = random.randint(0,3)
            else:
                rotate_main_idx = 0
            area_ratio = area_ratio_range[0] + random.random() * (area_ratio_range[1] - area_ratio_range[0])
            text_param = (fontScale, thickness_ratio, rotate_main_idx, area_ratio)
            text_param_list.append(text_param)
        return text_param_list

def draw_text_on_pattern(image, text_list, unit_chessboard_tag, basic_width, font = cv2.FONT_HERSHEY_SIMPLEX, border_sizes = 0, valid_labels = [], is_text_white = False, text_param_list = (8, 4, 0, 0.5)):
    ''' 
        image:
            marker image
        text_list:
            list of text to be shown
        unit_chessboard_tag:
            a chessboard tag without actual size
        basis width:
            width of each element
        border_sizes:
            size of borders, [top, bottom, left, right]
        is_white_border: 
            is border white color
        text_param_list:
            list of (fontScale, thickness_ratio, rotate_main_idx, area_ratio)
    '''


    if type(border_sizes) is not list:
        border_sizes = [border_sizes] * 4

    if type(text_param_list) is not list:
        text_param_list = [text_param_list]* len(text_list)


    border_sizes = [int(round(bd)) for bd in border_sizes[:4]]
    fine_grid_size = unit_chessboard_tag.get_fine_grid_size()
    top, bottom, left, right = border_sizes


    # draw text on patterns
    bw = basic_width
    max_elem_label = unit_chessboard_tag.get_max_elem_label()

    if type(is_text_white) is not list:
        is_text_white = [is_text_white]* (max_elem_label+1)
    x_y_with_labels = unit_chessboard_tag.get_elements_with_labels() 
    text_idx = 0
    for ii, x_y_with_label in enumerate(x_y_with_labels):
        x, y, label = x_y_with_label[:3] # idx and label of region
        label = int(label)

        if label not in valid_labels:
            continue

        if text_idx >= len(text_list) or text_idx >= len(text_param_list):
            break

        xx, yy = x * bw +left, y * bw + top # left up corner of the retion
        region = image[yy:yy + bw, xx:xx + bw] 
        
        fontScale, thickness_ratio, rotate_main_idx, area_ratio = text_param_list[text_idx] 
        text = text_list[text_idx]
        text_image =  put_text_on_tight_rectangle(text, font= font, fontScale=fontScale, thickness=fontScale *thickness_ratio, is_text_white= is_text_white[label], rotate_main_idx = rotate_main_idx)     
        region = image[yy:yy + bw, xx:xx + bw]
        combine_text_with_pattern(region, text_image, area_ratio, is_text_white =is_text_white[label])      
        text_idx +=1 




def draw_chessboard_tag(unit_chessboard_tag, basic_width, border_sizes = 0, is_white_border = True, elem_patterns = None):
    '''
        unit_chessboard_tag:
            a chessboard tag without actual size
        basis width:
            width of each element
        border_sizes:
            size of borders, [top, bottom, left, right]
        is_white_border: 
            is border white color
        elem_patters: 
            patter image for each elem_label; for binary markers, there should be at least two patters; is elem_patterns is None, the default pattern is black/white for 0/1
    '''

    if type(border_sizes) is not list:
        border_sizes = [border_sizes] * 4
    
    border_sizes = [int(round(bd)) for bd in border_sizes[:4]]
    fine_grid_size = unit_chessboard_tag.get_fine_grid_size()
    top, bottom, left, right = border_sizes
    h = fine_grid_size * basic_width + top + bottom
    w = fine_grid_size * basic_width + left + right
    image = np.zeros([h, w])


    # draw borders
    if is_white_border:
        # draw white border
        border_color = 1
        image[:top, :] = border_color
        image[-bottom:, :] = border_color
        image[:, :left] = border_color
        image[:, -right:] = border_color

        bg_color = 0

    else:
        # already black
        border_color = 0
        bg_color = 1
    

    # draw patterns
    bw = basic_width
    max_elem_label = unit_chessboard_tag.get_max_elem_label()
    x_y_with_labels = unit_chessboard_tag.get_elements_with_labels() 

    default_colors = [0, 1, bg_color]

    for ii, x_y_with_label in enumerate(x_y_with_labels):
        x, y, label = x_y_with_label[:3] # idx and label of region
        label = int(label)
        xx, yy = x * bw +left, y * bw + top # left up corner of the retion
        region = image[yy:yy + bw, xx:xx + bw] 
        try:
            # try to get patterns
            pattern = elem_patterns.get_pattern(label)
            # if pattern.shape[0]!= bw or pattern.shape[1]!= bw:
            #     pattern = cv2.resize(pattern, (bw, bw), interpolation= cv2.INTER_LINEAR)
        except:
            pattern = default_colors[min(label, len(default_colors)-1)]
        # get a square region
        region[:,:] = pattern
    

    return image

if __name__ == "__main__":
    from fiducial_marker.aruco_dict import get_all_aruco_dict_flags, get_aruco_info
    from fiducial_marker.unit_arucotag import UnitArucoTag
    import random
    from fiducial_marker.elem_patterns import *


    IS_RANDOM_BORDER = False
    basic_width = 51
    border_sizes = 51
    is_white_border = True
    is_inner_white =True

    pattern_bw = 501
    # two_patterns = [
    #     get_cross_pattern(pattern_bw, pattern_bw//3, border_width= pattern_bw//10, is_inner_white=True), 
    #     get_solid_circle_pattern(pattern_bw, outer_round_ratio= 0.8), get_square_pattern(pattern_bw, pattern_bw//2, pattern_bw//4),get_circle_pattern(pattern_bw, outer_round_ratio= 0.8, inner_round_ratio = 0.5)
    #     ] 

    elem_pattern_images = [get_square_pattern(pattern_bw, pattern_bw//2, pattern_bw//6, is_inner_white= is_inner_white),         
            get_solid_circle_pattern(pattern_bw, outer_round_ratio= 0.8, is_inner_white= not is_inner_white), 
            get_cross_pattern(pattern_bw, int(pattern_bw * 2/5), border_width= pattern_bw//10, is_inner_white=is_inner_white), 
            get_square_pattern(pattern_bw, pattern_bw//2, pattern_bw//4, is_inner_white= not is_inner_white),      
            get_circle_pattern(pattern_bw, outer_round_ratio= 0.5, inner_round_ratio = 0.3, is_inner_white = is_inner_white), 
            get_solid_circle_pattern(pattern_bw, outer_round_ratio= 0.4, is_inner_white= not is_inner_white),
            get_square_pattern(pattern_bw, pattern_bw//2, pattern_bw//4, is_inner_white= is_inner_white),  

            ] 

    elem_patterns = ElemPatterns(basic_width, elem_pattern_images)
    text_list = list('HELLO WORLD DEEP TAG')

    dict_flags, dict_sizes = get_all_aruco_dict_flags()
    for k in dict_flags.keys():
        print(k)
        grid_size = dict_sizes[k]['grid_size']
        dict_size = dict_sizes[k]['dict_size']
        tag_id = random.randint(0, dict_size-1)
        arucoBits = get_aruco_info(dict_flags[k], tag_id)

        binary_ids = arucoBits.astype(np.uint8).ravel().tolist()
        print(binary_ids)

        unit_arucotag = UnitArucoTag(grid_size, binary_ids, max_label = 1 , max_elem_label= 1)
        unit_arucotag_4cls = UnitArucoTag(grid_size, binary_ids, max_label = 1 , max_elem_label= 3)

        if IS_RANDOM_BORDER:
            border_sizes = [random.randint(3, basic_width) for _ in range(4)]
        image = draw_chessboard_tag(unit_arucotag, basic_width, border_sizes= border_sizes, is_white_border = is_white_border, elem_patterns = None)

        image_with_patterns = draw_chessboard_tag(unit_arucotag, basic_width, border_sizes= border_sizes, is_white_border = is_white_border, elem_patterns = elem_patterns)

        image_4cls_with_patterns = draw_chessboard_tag(unit_arucotag_4cls, basic_width, border_sizes= border_sizes, is_white_border = is_white_border, elem_patterns = elem_patterns)

        image_with_patterns_with_text = image_with_patterns.copy()
        draw_text_on_pattern(image_with_patterns_with_text,text_list, unit_arucotag, basic_width, border_sizes= border_sizes, valid_labels = [1], is_text_white= is_inner_white)

        cv2.imshow('arucotag', image)
        cv2.imshow('arucotag_with_patterns', image_with_patterns)
        cv2.imshow('arucotag_4cls_with_patterns', image_4cls_with_patterns)
        cv2.imshow('image_with_patterns_with_text', image_with_patterns_with_text)
        c = cv2.waitKey(0)
        if c == 27 or c == ord('q'):
            break