import math
import numpy as np
from numpy.lib.function_base import hamming
class MarkerDict:
    def __init__(self, codebook, unit_tag_template, hamming_dist = 4, max_check_count = 5000):

        self.codebook = codebook
        self.unit_tag_template = unit_tag_template
        self.rotation_set = self.get_rotation_set() 
        self.hamming_dist = hamming_dist
        self.max_check_count = max_check_count

    def get_main_idx(self, binary_ids):        
 
        main_idx = 0
        decimal_id = -1
        if len(self.codebook) > 0:
            # hash
            binary_ids_rotated_set = []
            for ii, idx_list in enumerate(self.rotation_set):
                binary_ids_rotated = [binary_ids[idx] for idx in idx_list]
                binary_ids_rotated_set.append(np.array(binary_ids_rotated))
                binary_ids_rotated = tuple(binary_ids_rotated)
                if binary_ids_rotated in self.codebook:
                    main_idx = ii
                    decimal_id = self.codebook[binary_ids_rotated]
                    break

            # hamming distance
            if decimal_id < 0 and self.hamming_dist >0:
                check_count = 0
                min_hamming_dist = self.hamming_dist
                for k in self.codebook.keys():
                    
                    # check for each key
                    cand = np.array(k).ravel()
                    for ii, binary_ids_rotated in enumerate(binary_ids_rotated_set):
                        check_count += 1
                        # check for each rotation
                        dist = check_hamming_dist_1d(cand, binary_ids_rotated)
                        if dist<= min_hamming_dist:
                            main_idx = ii
                            decimal_id = self.codebook[k]
                            min_hamming_dist = dist
                            if min_hamming_dist <=1: 
                                break

                    if min_hamming_dist <=1 or check_count > self.max_check_count:
                        break
                

        return main_idx, decimal_id
    
    def get_rotation_set(self):
        unit_tag_template = self.unit_tag_template
        codebook = self.codebook
        if len(codebook) ==0: return []

        num_keypoints = len(list(codebook.keys())[0])         
        if unit_tag_template.tag_type == 'runetag': 
            num_rotations = num_keypoints//unit_tag_template.num_circles
            
        else:
            num_rotations = 4           
            num_keypoints = int((math.sqrt(num_keypoints) + 2)**2)

        rotation_set = []
        idx_list_all = list(range(num_keypoints))

        for main_idx in range(num_rotations):
            _, kpt_idx_list_rotated = unit_tag_template.reorder_points_with_main_idx(idx_list_all, main_idx)
            rotation_set.append(kpt_idx_list_rotated)
        return rotation_set

def check_hamming_dist_1d(x, y):
    # difference of length
    hdist = abs(len(x) - len(y))
    
    # difference of labels
    num = min(len(x), len(y))
    if num > 0:
        hdist += sum(x[:num] != y[:num])
    return hdist
