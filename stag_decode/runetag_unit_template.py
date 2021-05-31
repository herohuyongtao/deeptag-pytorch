from fiducial_marker.unit_runetag import UnitRunetag
import math
from util.distorted_homo_transform import controlpoints_to_keypoints_in_crop_with_homo
import numpy as np
from util.homo_transform import warpPerspectivePts
from fiducial_marker.unit_chessboard_tag import get_tag_id_decimal
from stag_decode.pose_estimator import PoseSolver
from util.util import select_valid_vals
class UnitTagTemplate:
    def __init__(self, tag_type = 'runetag', num_circles = 3, is_print = False):
        '''
            type: runetag/...
        '''
        # runetag
        UnitTagClass = UnitRunetag
        is_need_main_idx = False
        kpt_start_idx = 0 
        step_elem_num = 1


        self.tag_type = tag_type
        self.UnitTagClass = UnitTagClass
        self.unit_tags_dict = get_unit_tags(UnitTagClass)
        self.is_need_main_idx = is_need_main_idx
        self.kpt_start_idx = kpt_start_idx
        self.step_elem_num = step_elem_num
        self.num_circles = num_circles
        self.is_print = is_print

    def get_pose_solver_dict(self, cameraMatrix, distCoeffs):
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num
        tag_type = self.tag_type


        pose_solver_dict = {}
        for unit_tag in unit_tags_dict.values():
            pose_solver = PoseSolver(unit_tag, step_elem_num, cameraMatrix = cameraMatrix, distCoeffs=distCoeffs, tag_type = tag_type)
            num_grid_points = len(pose_solver.fine_grid_points_anno)
            pose_solver_dict[num_grid_points] = pose_solver

        return pose_solver_dict

            

    

    def get_step_elem_num(self):
        return self.step_elem_num


    def match_fine_grid(self, unordered_points, H, fine_grid_matcher_list):
        '''
            match unordered_points with each unit_tag template
        '''
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num
        n_list = list(unit_tags_dict.keys())
        # num_points_sqrt = int(round(math.sqrt(len(unordered_points))))
        num_points = len(unordered_points)
        unordered_points_num = len(unordered_points)
        num_circles = self.num_circles

        if type(fine_grid_matcher_list) != list:
            fine_grid_matcher_list = [fine_grid_matcher_list]
    

        n_offsets = list(range(-10, 10))

        max_match_ratio = 0
        best_ordered_points = []
        for n_offset in n_offsets:
            n = num_points + n_offset
            if n not in n_list: continue
            unit_tag = unit_tags_dict[n]
            unit_corners_labels = unit_tag.get_ordered_corners_with_labels()
            unit_points_labels = unit_tag.get_keypoints_with_labels()
            unit_corners = [kpt[:2] for kpt in unit_corners_labels]
            unit_points = [kpt[:2] for kpt in unit_points_labels]

            for fine_grid_matcher in fine_grid_matcher_list:
                ordered_points = fine_grid_matcher.match_fine_grid(unordered_points,  H, unit_corners, unit_points, is_circle_match = True, num_circles = num_circles)

                # match_ratio, count, total_count = check_match_ratio(ordered_points, unordered_points_num)
                match_ratio, count, total_count = check_match_ratio(ordered_points, len(unit_points))
                if match_ratio> max_match_ratio: 
                    max_match_ratio = match_ratio
                    best_ordered_points = ordered_points
                    if self.is_print:
                        print( '%dx%d, match_ratio: %d/%d(%d) = %.2f'% (n//num_circles,num_circles,count,total_count,unordered_points_num, count/total_count))
                if abs(count -total_count)<3: break                
            if count == total_count: break

        return max_match_ratio, best_ordered_points

    def update_corners_in_image(self, ordered_points, H_crop, valid_flag_list = None, cameraMatrix = None, distCoeffs = None):
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num
        # n = int(round(math.sqrt(len(ordered_points))))
        n = len(ordered_points)

        unit_tag = unit_tags_dict[n]
        ordered_points = [kpt[:2] for kpt in ordered_points]
        unit_corners_labels = unit_tag.get_ordered_corners_with_labels()
        unit_points_labels = unit_tag.get_keypoints_with_labels()
        unit_corners = [kpt[:2] for kpt in unit_corners_labels]
        unit_points = [kpt[:2] for kpt in unit_points_labels]

        unit_points = select_valid_vals(unit_points, valid_flag_list)
        ordered_points = select_valid_vals(ordered_points, valid_flag_list)

        corners_in_crop_updated = controlpoints_to_keypoints_in_crop_with_homo(unit_points, ordered_points, unit_corners,  distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H_crop)
        inv_H_crop = np.linalg.inv(H_crop)
        corners_in_image_updated = warpPerspectivePts(inv_H_crop, corners_in_crop_updated)
        return corners_in_image_updated

    def get_unit_tag(self, ordered_points):
        unit_tags_dict = self.unit_tags_dict
        n = len(ordered_points)
        unit_tag = unit_tags_dict[n]
        return unit_tag

    def get_main_idx(self, label_scores_all, check_code_len = 3):
        num_circles = self.num_circles
        num_slot_in_circle = len(label_scores_all)//num_circles
        codes = []
        for ii in range(num_slot_in_circle):
            bids = label_scores_all[ii*num_circles:(ii+1)*num_circles]
            codes.append(get_tag_id_decimal(bids[::-1]) )

        max_val = 0
        main_idx = 0
        for ii in range(num_slot_in_circle):
            total = 0
            for jj in range(check_code_len):
                total +=codes[(ii+jj)%num_slot_in_circle]
            if total > max_val:
                main_idx = ii
                max_val = total

        return main_idx, -1


    def reorder_points_with_main_idx(self, fine_grid_points, main_idx):
        if main_idx ==0:
            return fine_grid_points, fine_grid_points

        num_circles = self.num_circles
        num_slot_in_circle = len(fine_grid_points)//num_circles

        idx_list = list(range(main_idx, num_slot_in_circle)) + list(range(main_idx))
        fine_grid_points_new = []
        for ii in idx_list:
            for jj in range(num_circles):
                fine_grid_points_new.append(fine_grid_points[ii*num_circles + jj])

        return fine_grid_points_new, fine_grid_points_new

    def get_tag_id_decimal(self, label_scores_all, max_len = 10):
        return get_tag_id_decimal(label_scores_all[:max_len])


        



def check_match_ratio(ordered_points, unordered_points_num):   
    # if len(ordered_points) ==0: return False

    num_detected_kpts = len(ordered_points)

    # count valid points
    count = num_detected_kpts
    for pt in ordered_points:
        if len(pt)>2 and pt[2]<0:
            count -=1

    total_count = max(unordered_points_num, num_detected_kpts)
    match_ratio = count/total_count

    return match_ratio, count, total_count





def get_unit_tags(UnitTagClass, num_slots = 43, num_layers = 3):
    '''
        get a list of unit_tag of different grid size
    '''
    unit_tags_dict = {}
    binary_ids = [ii%2 for ii in range(num_slots*num_layers)]
    unit_tag = UnitTagClass(binary_ids)
    n = num_slots * num_layers
    unit_tags_dict[n] = unit_tag

    return unit_tags_dict