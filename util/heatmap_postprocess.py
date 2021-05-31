from operator import itemgetter

import cv2
import math
import numpy as np
import time

def suppress_and_average_keypoints_with_vals(keypoints, scores, sigma = 2, vals = None, min_cos= 0.9, is_norm_val = True, vals2 = None, is_norm_val2= False, is_fast_version = True):
    '''
        clustering the keypoints and take an average in each group 
    '''

    if type(sigma) != list:
        sigma = [sigma] * len(keypoints)


    # grouping with keypoints[i][0], keypoints[i][1] and vals[i]
    # keep keypoints[i][2:] with the highest scores[i] in group
    groups = []
    groups_info = []
    group_idx_dict = [-1] * len(keypoints)
    keypoints_np = np.float32(keypoints)
    for i in range(len(keypoints)):

        # if keypoints[i] is not in a group, then create a new group
        group_idx = group_idx_dict[i]
        if group_idx<0:
            group_idx = len(groups)
            groups.append([i])   
            groups_info.append((scores[i], keypoints[i][2:]))     
            group_idx_dict[i] = group_idx  

        elif is_fast_version:
            continue


        # compare to each ungroupeed keypoints[j]
        if len(keypoints) == i + 1:
            # last keypoints
            break


        dist = np.sqrt((keypoints_np[i,0]- keypoints_np[i+1:,0])** 2 + (keypoints_np[i,1] - keypoints_np[i+1:, 1])** 2)
        for j in range(i+1, len(keypoints)):
            if group_idx_dict[j]>=0: continue

            # dist_check = math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 + (keypoints[i][1] - keypoints[j][1]) ** 2)
            # print(dist_check, dist[j-i-1])

            # check keypoint distance
            if dist[j-i-1] < min(sigma[i], sigma[j]):

                # check vals with cos distance
                if vals is not None and get_cos_dist(vals[i], vals[j])< min_cos:
                    continue

                groups[group_idx].append(j)
                group_idx_dict[j] = group_idx

                # keep the info with highest score
                if scores[j] > groups_info[group_idx][0]:
                    groups_info[group_idx] = (scores[j], keypoints[j][2:])


    # take an average inside each group
    keypoints_new = []
    scores_new = []        
    vals_new = []
    vals2_new = []
    for group, group_info in zip(groups, groups_info):
        x_in_group = np.float32([keypoints[idx][0] for idx in group])
        y_in_group = np.float32([keypoints[idx][1] for idx in group])
        score_in_group = np.float32([scores[idx] for idx in group])
        score_in_group /= np.sum(score_in_group)
        kpt = [float(np.sum(x_in_group*score_in_group)), float(np.sum(y_in_group*score_in_group))]
        kpt += group_info[1]
        score = group_info[0]

        if vals is not None:
            vals_in_group =np.float32([vals[idx] for idx in group])
            vv = np.sum(np.float32(vals_in_group) * np.reshape(score_in_group, (-1, 1)), axis=0).ravel()
            if is_norm_val:
                vv /= max(norm3d(vv), 1e-20)
            vals_new.append(vv.tolist())

        if vals2 is not None:
            vals2_in_group =np.float32([vals2[idx] for idx in group])
            vv = np.sum(np.float32(vals2_in_group) * np.reshape(score_in_group, (-1, 1)), axis=0).ravel()
            if is_norm_val2:
                vv /= max(norm3d(vv), 1e-20)
            vals2_new.append(vv.tolist())
        

        keypoints_new.append(kpt)
        scores_new.append(score)

    return keypoints_new, scores_new, vals_new, vals2_new, groups

def suppress_and_average_keypoints(keypoints, scores, sigma = 2, is_fast_version = True):
    '''
        clustering the keypoints and take an average in each group 
    '''
    keypoints_new, scores_new, _, _, groups = suppress_and_average_keypoints_with_vals(keypoints, scores, sigma = sigma, is_fast_version = is_fast_version)
    return keypoints_new, scores_new, groups


def get_cos_dist(vec1, vec2):
    vec1 = np.float32(vec1)
    vec2 = np.float32(vec2)
    dist_cos = np.sum(vec1*vec2)/max(norm3d(vec1)* norm3d(vec2), 1e-20)
    dist_cos = min(max(-1,dist_cos),1)
    return dist_cos

def norm3d(xyz):
    return np.sqrt(np.sum(np.float32(xyz)**2))

