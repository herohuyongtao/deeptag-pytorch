from util.match_and_warp import iterative_match_and_warp, circle_match_and_warp
from util.homo_transform import get_homo_from_corners_to_corners
import numpy as np
class FineGridMatcher:
    def __init__(self, ordered_corners_in_crop, ordered_corners_in_crop_set = [], cameraMatrix= None, distCoeffs = None, max_warp_try = 3):
        self.ordered_corners_in_crop = ordered_corners_in_crop
        self.ordered_corners_in_crop_set = ordered_corners_in_crop_set # other possible corners
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.H_list = get_H_list_for_corners_in_crop(ordered_corners_in_crop, ordered_corners_in_crop_set)
        self.max_warp_try = max_warp_try

    def match_fine_grid(self, unordered_points, H_crop, unit_corners, unit_points, is_circle_match = False, num_circles = 1):
        '''
            match unordered_points with each possible NxN grid
        '''
        ordered_corners_in_crop =self.ordered_corners_in_crop
        ordered_corners_in_crop_set= self.ordered_corners_in_crop_set
        cameraMatrix =self.cameraMatrix
        distCoeffs =self.distCoeffs
        H_list = self.H_list
        max_warp_try = self.max_warp_try

        # H_list = eye(3)
        
        if not is_circle_match:
            ordered_points = iterative_match_and_warp(unordered_points, unit_points, unit_corners, ordered_corners_in_crop,  distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H_crop, H_list_for_cpts_in_crop= H_list, max_warp_try= max_warp_try )
        else:
            ordered_points = circle_match_and_warp(unordered_points, unit_points, unit_corners, ordered_corners_in_crop,  distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H_crop, H_list_for_cpts_in_crop= H_list, num_circles = num_circles)

        return ordered_points




def get_H_list_for_corners_in_crop(ordered_corners, ordered_corners_set):
    H_list = [np.eye(3)]

    # scale one point
    for ordered_corners_scaled in ordered_corners_set:
        for main_idx in range(4):
            ordered_corners_new = ordered_corners.copy()
            ordered_corners_new[main_idx] = ordered_corners_scaled[main_idx].copy()
            H = get_homo_from_corners_to_corners(ordered_corners, ordered_corners_new)
            H_list.append(H)

    # scale one side
    for ordered_corners_scaled in ordered_corners_set:
        for main_idx in range(4):
            ordered_corners_new = ordered_corners.copy()
            if (main_idx)%2 ==0:
                ordered_corners_new[main_idx-1] = [ordered_corners_scaled[main_idx-1][0], ordered_corners[main_idx-1][1]]
                ordered_corners_new[main_idx]  = [ordered_corners_scaled[main_idx][0], ordered_corners[main_idx][1]]
            else:
                ordered_corners_new[main_idx-1] = [ordered_corners[main_idx-1][0], ordered_corners_scaled[main_idx-1][1] ]
                ordered_corners_new[main_idx]  = [ordered_corners[main_idx][0], ordered_corners_scaled[main_idx][1]]
            H = get_homo_from_corners_to_corners(ordered_corners, ordered_corners_new)
            H_list.append(H)

    return H_list