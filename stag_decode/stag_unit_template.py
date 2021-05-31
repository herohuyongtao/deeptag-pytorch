from fiducial_marker.unit_arucotag import UnitArucoTag
from fiducial_marker.unit_topotag import UnitTopoTag
import math
from util.distorted_homo_transform import controlpoints_to_keypoints_in_crop_with_homo
import numpy as np
from util.homo_transform import warpPerspectivePts
from fiducial_marker.unit_chessboard_tag import get_tag_id_decimal
from stag_decode.pose_estimator import PoseSolver
from util.util import select_valid_vals
class UnitTagTemplate:
    def __init__(self, tag_type = 'topotag', grid_size_list = [3,4,5], step_elem_num = None, is_print = False):
        '''
            type: topotag/arucotag
        '''
        if tag_type == 'topotag':
            UnitTagClass = UnitTopoTag
            is_need_main_idx = True
            if step_elem_num is None:
                step_elem_num = 2
            kpt_start_idx = 2
        else:
            # arucotag
            UnitTagClass = UnitArucoTag
            is_need_main_idx = False
            step_elem_num = 1
            kpt_start_idx = 0
        self.tag_type = tag_type
        self.UnitTagClass = UnitTagClass
        self.unit_tags_dict = get_unit_tags(UnitTagClass, grid_size_list, step_elem_num)
        self.is_need_main_idx = is_need_main_idx
        self.step_elem_num = step_elem_num
        self.kpt_start_idx = kpt_start_idx
        self.is_print = is_print

    def get_pose_solver_dict(self, cameraMatrix, distCoeffs):
        '''
            pose_solver for different NxN
        '''
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num
        pose_solver_dict = {}
        for unit_tag in unit_tags_dict.values():
            pose_solver = PoseSolver(unit_tag, step_elem_num, cameraMatrix = cameraMatrix, distCoeffs=distCoeffs)
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
        num_points_sqrt = int(round(math.sqrt(len(unordered_points))))
        unordered_points_num = len(unordered_points)

        if type(fine_grid_matcher_list) != list:
            fine_grid_matcher_list = [fine_grid_matcher_list]
    


        # match NxN grid
        n_offsets = [0, 1, -1]

        max_match_ratio = 0
        stop_flag = False
        best_ordered_points = []
        for n_offset in n_offsets:
            n = num_points_sqrt + n_offset
            if n not in n_list: continue
            unit_tag = unit_tags_dict[n]
            unit_corners = unit_tag.get_ordered_corners()
            unit_points = unit_tag.get_fine_grid_points(is_center = True, step_elem_num= step_elem_num)

            possible_match_ratio = min(len(unordered_points), len(unit_points))/ max(len(unordered_points), len(unit_points))
            if max_match_ratio > possible_match_ratio:
                continue


            for fine_grid_matcher in fine_grid_matcher_list:
                ordered_points = fine_grid_matcher.match_fine_grid(unordered_points,  H, unit_corners, unit_points)
                match_ratio, count, total_count = check_match_ratio(ordered_points, unordered_points_num)
                if match_ratio> max_match_ratio: 
                    max_match_ratio = match_ratio
                    best_ordered_points = ordered_points
                if self.is_print:
                    print( '%dx%d, match_ratio: %d/%d = %.2f'% (n,n,count,total_count,count/total_count))
                if abs(count -total_count)<3: 
                    stop_flag = count == total_count
                    break                
            if stop_flag: break

        return max_match_ratio, best_ordered_points

    def update_corners_in_image(self, ordered_points, H_crop, valid_flag_list = None, cameraMatrix = None, distCoeffs = None):
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num
        n = int(round(math.sqrt(len(ordered_points))))

        unit_tag = unit_tags_dict[n]
        unit_corners = unit_tag.get_ordered_corners()
        unit_points = unit_tag.get_fine_grid_points(is_center = True, step_elem_num= step_elem_num)

        unit_points = select_valid_vals(unit_points, valid_flag_list)
        ordered_points = select_valid_vals(ordered_points, valid_flag_list)

        corners_in_crop_updated = controlpoints_to_keypoints_in_crop_with_homo(unit_points, ordered_points, unit_corners,  distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H_crop)
        inv_H_crop = np.linalg.inv(H_crop)
        corners_in_image_updated = warpPerspectivePts(inv_H_crop, corners_in_crop_updated)
        return corners_in_image_updated

    def get_unit_tag(self, ordered_points):
        unit_tags_dict = self.unit_tags_dict
        n = int(round(math.sqrt(len(ordered_points))))
        unit_tag = unit_tags_dict[n]
        return unit_tag

    def get_main_idx(self, label_scores_all):
        '''
            Get main rotational direction with the minimal decimal tag id.
            But TopoTag has its own definition.
            
        '''


        is_need_main_idx = self.is_need_main_idx
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num

        


        n = int(round(math.sqrt(len(label_scores_all))))
        unit_tag = unit_tags_dict[n]
        kpt_start_idx = self.kpt_start_idx

        # rotate 4 directions, get scores and ids
        decimal_ids = []
        mean_scores = []
        for main_idx in range(4):
            x_and_y_with_labels = unit_tag.get_keypoints_with_labels(main_idx= main_idx, step_elem_num=step_elem_num)
            rotated_idx_list = [pt[1]//step_elem_num*n + pt[0]//step_elem_num for pt in x_and_y_with_labels]
            rotated_label_scores = [label_scores_all[idx] for idx in rotated_idx_list]
            rotated_max_label_scores = []
            rotated_binary_ids = []
            for vals in rotated_label_scores:
                if type(vals) is list:
                    # score
                    if vals[0]> vals[1]: rotated_binary_ids.append(0)
                    else: rotated_binary_ids.append(1)
                    rotated_max_label_scores.append(max(0, max(vals[:2])))
                else:
                    # bid
                    bid = vals
                    rotated_binary_ids.append(min(max(0,bid),1))
                    rotated_max_label_scores.append(int(bid>=0 and bid<=1))


            rotated_decimal_id =get_tag_id_decimal(rotated_binary_ids[kpt_start_idx:])
            decimal_ids.append(rotated_decimal_id)
            mean_scores.append(sum(rotated_max_label_scores))


        if is_need_main_idx:
            # rotate to get max score
            decimal_id = None
            max_mean_score = -1
            for curr_idx in range(4):
                if decimal_id is None or max_mean_score < mean_scores[curr_idx]:
                    main_idx = curr_idx
                    max_mean_score = mean_scores[curr_idx]
                    decimal_id = decimal_ids[curr_idx]

        else:
            # rotate to get min decimal id
            decimal_id = None
            max_mean_score = -1
            for curr_idx in range(4):
                if decimal_id is None or decimal_id > decimal_ids[curr_idx]:
                    main_idx = curr_idx
                    max_mean_score = mean_scores[curr_idx]
                    decimal_id = decimal_ids[curr_idx]

            decimal_id = -1 # clear decimal id
        return main_idx, decimal_id

    def reorder_points_with_main_idx(self, fine_grid_points, main_idx):
        '''
            rotate with main_idx
            point_type: 'fine_grid_points'
        '''
        kpt_start_idx = self.kpt_start_idx
        # is_need_main_idx = self.is_need_main_idx
        unit_tags_dict = self.unit_tags_dict
        step_elem_num = self.step_elem_num
        n = int(round(math.sqrt(len(fine_grid_points))))
        unit_tag = unit_tags_dict[n]


        x_and_y_with_labels = unit_tag.get_fine_grid_points(main_idx= main_idx, step_elem_num=step_elem_num)
        rotated_idx_list = [pt[1]//step_elem_num*n + pt[0]//step_elem_num for pt in x_and_y_with_labels]
        fine_grid_points_rotated = [fine_grid_points[idx] for idx in rotated_idx_list]



        x_and_y_with_labels = unit_tag.get_keypoints_with_labels(main_idx= main_idx, step_elem_num=step_elem_num)
        rotated_idx_list = [pt[1]//step_elem_num*n + pt[0]//step_elem_num for pt in x_and_y_with_labels]
        keypoints_rotated = [fine_grid_points[idx] for idx in rotated_idx_list[kpt_start_idx:]]
        return fine_grid_points_rotated, keypoints_rotated


        

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






def get_unit_tags(UnitTagClass, grid_size_list, step_elem_num):
    '''
        get a list of unit_tag of different grid size
    '''
    unit_tags_dict = {}
    for grid_size in grid_size_list:
        binary_ids = [ii%2 for ii in range(grid_size*grid_size)]
        unit_tag = UnitTagClass(grid_size, binary_ids)
        n = unit_tag.get_fine_grid_size(step_elem_num= step_elem_num)
        unit_tags_dict[n] = unit_tag

    return unit_tags_dict