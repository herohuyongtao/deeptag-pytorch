import cv2
import numpy as np
from util.distorted_homo_transform import controlpoints_to_keypoints_in_crop_with_homo
from util.homo_transform import warpPerspectivePts

def circle_match_and_warp(kpts_with_ids_in_crop, ordered_kpts_gt, cpts_gt, cpts_in_crop,  distCoeffs= None, cameraMatrix = None, H = None, max_warp_try = 10, H_list_for_cpts_in_crop = [np.eye(3)], num_circles = 1):
    '''
        kpts_with_ids_in_crop: kpts with ids detected from heatmaps
        ordered_kpts_gt: unit ordered kpts
        cpts_gt: unit control points, such as unit corners
        cpts_in_crop: control points in crop, unit corners in crop
        H: mat for cropping
        

    '''


    # iterative warp and match template with keypoints
    kpts_with_ids =   kpts_with_ids_in_crop
    # if H is None:
    #     H = np.eye(3)

    max_matched_num = 0
    H_curr = None
    for H_init in H_list_for_cpts_in_crop:
        cpts_in_crop_updated = warpPerspectivePts(H_init, cpts_in_crop)
        kpts_cand = controlpoints_to_keypoints_in_crop_with_homo(cpts_gt, cpts_in_crop_updated, ordered_kpts_gt, distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H)
        # max_match_dist = 5
        max_match_dist = float(norm3d(np.float32(kpts_cand[0][:2])-np.float32(kpts_cand[1][:2]))) * 0.5
        # max_match_dist_refine = max_match_dist* (2/3)
        # print('max_match_dist:', max_match_dist)


        # get translation 
        H_translate = np.eye(3)
        kpts_mean = np.mean(np.float32([kpt[:2] for kpt in kpts_with_ids]), axis = 0)
        kpts_cand_mean = np.mean(np.float32([kpt[:2] for kpt in kpts_cand]), axis = 0)
        H_translate[:2,2] = kpts_mean -kpts_cand_mean

        # first try
        H_new, match_flags_cand, match_flags, match_ids, mean_dist = match_and_warp(kpts_cand, kpts_with_ids, max_match_dist, H = H_translate)

        # # second try if necessary
        # if H_new is None:
        #     dist_step = 2
        #     H_new, match_flags_cand, match_flags, match_ids, mean_dist = match_and_warp(kpts_cand, kpts_with_ids, max_match_dist * dist_step, H= np.eye(3))

        if H_new is not None and sum(match_flags_cand) > max_matched_num:
            max_match_dist_refine = min(mean_dist * 1.01, max_match_dist)
            H_curr = np.matmul(H_new, H_init)
            # match_flags_cand_curr = match_flags_cand        
            # match_ids_curr = match_ids
            max_matched_num = sum(match_flags_cand)

        if max_matched_num > len(ordered_kpts_gt) - 5: break

    ordered_kpts_with_ids = []
   

    # iterate till small error
    if H_curr is not None: 
        # warp template with latest H
        # ordered_kpt_candidates_warp = warpPerspectivePts(H_curr, ordered_kpt_candidates) 
        cpts_in_crop_updated = warpPerspectivePts(H_curr, cpts_in_crop)
        ordered_kpt_candidates_warp = controlpoints_to_keypoints_in_crop_with_homo(cpts_gt, cpts_in_crop_updated, ordered_kpts_gt, distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H)


        match_flags_cand_curr, match_ids_curr = align_multiple_circles(match_flags_cand, match_ids, kpts_with_ids, kpts_cand, num_circles)

    
        for ii, (flag, mid, kpt_can) in enumerate(zip(match_flags_cand_curr, match_ids_curr, ordered_kpt_candidates_warp)):
            if flag:
                kpt_with_id = kpts_with_ids[mid].copy()
            else:
                bid = -1
                kpt_with_id = kpt_can + [bid]

            ordered_kpts_with_ids.append(kpt_with_id)

    

    return ordered_kpts_with_ids


def align_multiple_circles(match_flags_cand, match_ids, kpts_with_ids, kpts_cand, num_circles):
    # unmatch_keypoints
    kpts_dict = {kid:True for kid in range(len(kpts_with_ids))}


    for mid in match_ids:
        if mid>=0:
            del kpts_dict[mid]

    # add points for each circle
    match_flags_cand_circle_set = []
    match_ids_circle_set = []
    for ic in range(num_circles):
        dists = []
        # mean dist of nearby points        

        for ii in range(ic, len(kpts_cand)-num_circles, num_circles):
            mid1 = match_ids[ii]
            mid2 = match_ids[ii+num_circles]
            if mid1 <0 or mid2 <0: continue                
            dist = norm3d(np.float32(kpts_with_ids[mid1][:2])- np.float32(kpts_with_ids[mid2][:2]))
            dists.append(dist)

        mean_dist = float(np.median(dists))


        # add points
        match_flags_cand_circle = []
        match_ids_circle = []
        for ii in range(ic-num_circles, len(kpts_cand)-num_circles, num_circles):
            mid1 = match_ids[ii]
            mid2 = match_ids[ii+num_circles]
            if match_flags_cand[ii]:
                match_flags_cand_circle.append(match_flags_cand[ii])
                match_ids_circle.append(mid1)
            flag = False
            if mid1 <0 or mid2 <0: continue

            for jj in kpts_dict.keys():
                d = norm3d(np.float32(kpts_with_ids[mid1][:2])- np.float32(kpts_with_ids[mid2][:2]))
                d1 = norm3d(np.float32(kpts_with_ids[mid1][:2])- np.float32(kpts_with_ids[jj][:2]))
                d2 = norm3d(np.float32(kpts_with_ids[mid2][:2])- np.float32(kpts_with_ids[jj][:2]))
                # if d > mean_dist * 1.8 and d1 < mean_dist * 1.5 and d2 < mean_dist * 1.5:
                if d > mean_dist * 1.5 and d1 < d * 0.8 and d2 < d * 0.8 :
                # if d > (d1 + d2) * 0.6:
                    tmp_key = jj
                    flag = True
                    break

            if flag:                     
                match_flags_cand_circle.append(True)
                match_ids_circle.append(tmp_key)

                del kpts_dict[tmp_key]




        match_flags_cand_circle_set.append(match_flags_cand_circle)
        match_ids_circle_set.append(match_ids_circle)
            # if ii+num_circles >= len(kpts_cand)-num_circles and match_flags_cand[ii+num_circles]:
            #     match_flags_cand_circle.append(match_flags_cand[ii+num_circles])
            #     match_ids_circle.append(mid1)


    ref_ic = num_circles//2
    for ic in range(num_circles):
        if len(match_flags_cand_circle_set[ref_ic]) != len(kpts_cand)//num_circles and len(match_flags_cand_circle_set[ic]) == len(kpts_cand)//num_circles:
            ref_ic = ic
            break

    # fix each circle, align with previous circle or the next circle
    match_flags_cand_new = [False] * len(kpts_cand)
    match_ids_new = [-1] * len(kpts_cand)
    ic_list = [ref_ic] + list(range(ref_ic, -1, -1)) + list(range(ref_ic+1, num_circles))
    for ic in ic_list:
        match_flags_cand_circle = match_flags_cand_circle_set[ic]
        match_ids_circle = match_ids_circle_set[ic]
        if len(match_flags_cand_circle) < len(kpts_cand)// num_circles:
            match_flags_cand_circle += [False] * (len(kpts_cand)// num_circles - len(match_flags_cand_circle))
            match_ids_circle += [-1] * (len(kpts_cand)// num_circles - len(match_ids_circle))


        
        if ic == ref_ic:
            best_offset = 1
            match_flags_cand_circle_aligned = match_flags_cand_circle[best_offset:] + match_flags_cand_circle[:best_offset]
            match_ids_circle_aligned = match_ids_circle[best_offset:] + match_ids_circle[:best_offset]
        else:     
            # align with previous circle                   
            # get the first valid point in the previous circle
            if ic > ref_ic:
                prev_ic = ic -1
            else:
                prev_ic = ic +1

            for ii in range(len(kpts_cand)//num_circles):
                if match_flags_cand_new[ii * num_circles + prev_ic]:
                    mid = match_ids_new[ii * num_circles + prev_ic]
                    idx = ii
                    break


            # align the first point
            best_offset = None
            min_dist = None
            for offset in range(-4, 4):
                if not match_flags_cand_circle[idx + offset]: continue

                mid2 = match_ids_circle[idx + offset]
                d = norm3d(np.float32(kpts_with_ids[mid][:2])- np.float32(kpts_with_ids[mid2][:2]))
                if best_offset is None or d < min_dist:
                    best_offset = offset
                    min_dist = d

            if best_offset is not None:
                match_flags_cand_circle = match_flags_cand_circle[best_offset:] + match_flags_cand_circle[:best_offset]
                match_ids_circle = match_ids_circle[best_offset:] + match_ids_circle[:best_offset]


            # align from the second point
            match_flags_cand_circle_prev = [match_flags_cand_new[ii] for ii in range(prev_ic, len(kpts_cand), num_circles)]
            match_ids_circle_prev = [match_ids_new[ii] for ii in range(prev_ic, len(kpts_cand), num_circles)]       
            match_flags_cand_circle_aligned = []
            match_ids_circle_aligned = []
            # align from the second point            
            len1 = len(match_flags_cand_circle_prev)
            len2 = len(match_flags_cand_circle)
            MAX_OFFSET = 5 + abs(len2- len1)
            for ii in range(len(kpts_cand)//num_circles):
                min_dist1 = None 
                min_dist2 = None                  
                best_offset = None
                prev_offset = 0
                if match_ids_circle_prev[ii] >=0:
                    mid1 = match_ids_circle_prev[ii]
                    min_dist = None
                    # find the nearest
                    for xx in range(-MAX_OFFSET + prev_offset, prev_offset + MAX_OFFSET+1):
                        mid2 = match_ids_circle[(ii+xx)%len2]
                        if mid2 <0: continue                

                        
                        d = norm3d(np.float32(kpts_with_ids[mid1][:2])- np.float32(kpts_with_ids[mid2][:2]))


                        # check valid
                        is_valid_flag = True
                        for yy in range(-MAX_OFFSET, MAX_OFFSET+1):
                            if yy == 0: continue
                            mid3 = match_ids_circle_prev[(ii+yy)%len1]
                            d2 = norm3d(np.float32(kpts_with_ids[mid3][:2])- np.float32(kpts_with_ids[mid2][:2]))
                            if d > d2:
                                is_valid_flag = False 
                                break

                        if (min_dist is None or min_dist > d) and is_valid_flag:
                            min_dist = d
                            best_offset = xx


                # append with valid match
                if best_offset is not None:
                    mid = match_ids_circle[(ii+best_offset)%len2]
                    flag = True
                    prev_offset = best_offset
                else:
                    mid = -1
                    flag = False
                match_flags_cand_circle_aligned.append(flag)
                match_ids_circle_aligned.append(mid)


        # replace 
        for ii in range(ic, len(kpts_cand), num_circles):
            match_flags_cand_new[ii] = match_flags_cand_circle_aligned[ii//num_circles]
            match_ids_new[ii] = match_ids_circle_aligned[ii//num_circles]

    # num_points_in_circle = len(kpts_cand)//num_circles
    # idx_list = [0] * num_circles
    # for ii in range(num_points_in_circle):



    return match_flags_cand_new, match_ids_new

def iterative_match_and_warp(kpts_with_ids_in_crop, ordered_kpts_gt, cpts_gt, cpts_in_crop,  distCoeffs= None, cameraMatrix = None, H = None, max_warp_try = 10, H_list_for_cpts_in_crop = [np.eye(3)]):
    '''
        kpts_with_ids_in_crop: kpts with ids detected from heatmaps
        ordered_kpts_gt: unit ordered kpts
        cpts_gt: unit control points, such as unit corners
        cpts_in_crop: control points in crop, unit corners in crop
        H: mat for cropping
        

    '''


    # iterative warp and match template with keypoints
    kpts_with_ids =   kpts_with_ids_in_crop
    # if H is None:
    #     H = np.eye(3)

    max_matched_num = 0
    H_curr = None
    for H_init in H_list_for_cpts_in_crop:
        cpts_in_crop_updated = warpPerspectivePts(H_init, cpts_in_crop)
        kpts_cand = controlpoints_to_keypoints_in_crop_with_homo(cpts_gt, cpts_in_crop_updated, ordered_kpts_gt, distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H)
        # max_match_dist = 5
        max_match_dist = float(norm3d(np.float32(kpts_cand[0][:2])-np.float32(kpts_cand[1][:2]))) * 0.5
        # max_match_dist_refine = max_match_dist* (2/3)
        # print('max_match_dist:', max_match_dist)


        # get translation 
        H_translate = np.eye(3)
        kpts_mean = np.mean(np.float32([kpt[:2] for kpt in kpts_with_ids]), axis = 0)
        kpts_cand_mean = np.mean(np.float32([kpt[:2] for kpt in kpts_cand]), axis = 0)
        H_translate[:2,2] = kpts_mean -kpts_cand_mean

        # first try
        H_new, match_flags_cand, match_flags, match_ids, mean_dist = match_and_warp(kpts_cand, kpts_with_ids, max_match_dist, H = H_translate)

        # # second try if necessary
        # if H_new is None:
        #     dist_step = 2
        #     H_new, match_flags_cand, match_flags, match_ids, mean_dist = match_and_warp(kpts_cand, kpts_with_ids, max_match_dist * dist_step, H= np.eye(3))

        if H_new is not None and sum(match_flags_cand) > max_matched_num:
            max_match_dist_refine = min(mean_dist * 1.01, max_match_dist)
            H_curr = np.matmul(H_new, H_init)
            match_flags_cand_curr = match_flags_cand        
            match_ids_curr = match_ids
            max_matched_num = sum(match_flags_cand)

        if max_matched_num > len(ordered_kpts_gt) - 5: break

    ordered_kpts_with_ids = []
   

    # iterate till small error
    if H_curr is not None:    
        # H_curr = H_new
        # match_flags_cand_curr = match_flags_cand        
        # match_ids_curr = match_ids
        max_match_dist_tmp = max_match_dist_refine # tight dist
        # print('max_match_dist: %.2f'% max_match_dist_tmp)
        for i_try in range(max_warp_try):
            cpts_in_crop_updated = warpPerspectivePts(H_curr, cpts_in_crop)
            kpts_cand = controlpoints_to_keypoints_in_crop_with_homo(cpts_gt, cpts_in_crop_updated, ordered_kpts_gt, distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H)
            if len(kpts_cand) == 0: break

            num_match = sum(match_flags_cand)
            H_new, match_flags_cand, match_flags, match_ids, mean_dist = match_and_warp(kpts_cand, kpts_with_ids, max_match_dist_tmp, H= np.eye(3))  
            num_match_curr = sum(match_flags_cand)
            
            # print('match num (max_dist=%.2f): %d/%d=>%d/%d'% (max_match_dist_tmp, num_match, len(match_flags_cand), num_match_curr, len(match_flags_cand)))
            if (num_match_curr <num_match and i_try > 0) or H_new is None:                
                break

            H_curr = H_new
            match_flags_cand_curr = match_flags_cand
            match_ids_curr = match_ids

            if num_match_curr > len(match_flags_cand)-3 and num_match_curr == num_match:  
                break

            if num_match_curr == len(ordered_kpts_gt): break

            # max_match_dist_tmp = mean_dist + 0.01

        
        # warp template with latest H
        # ordered_kpt_candidates_warp = warpPerspectivePts(H_curr, ordered_kpt_candidates) 
        cpts_in_crop_updated = warpPerspectivePts(H_curr, cpts_in_crop)
        ordered_kpt_candidates_warp = controlpoints_to_keypoints_in_crop_with_homo(cpts_gt, cpts_in_crop_updated, ordered_kpts_gt, distCoeffs= distCoeffs, cameraMatrix = cameraMatrix, H = H)

    
        for ii, (flag, mid, kpt_can) in enumerate(zip(match_flags_cand_curr, match_ids_curr, ordered_kpt_candidates_warp)):
            if flag:
                kpt_with_id = kpts_with_ids[mid].copy()
            else:
                bid = -1
                kpt_with_id = kpt_can + [bid]

            ordered_kpts_with_ids.append(kpt_with_id)

    

    return ordered_kpts_with_ids


def match_and_warp(kpts_cand, kpts_with_ids, max_match_dist, H = None):
    if H is None: H = np.eye(3)

    match_flags_cand = [False] * len(kpts_cand)
    match_flags = [False] * len(kpts_with_ids)
    match_ids = [-1] * len(kpts_cand)

    kpts_cand_warp = warpPerspectivePts(H, kpts_cand)
    dist_list = []
    for ii, kpt_tem in enumerate(kpts_cand_warp):
        if match_flags_cand[ii]: continue
        for jj, kpt in enumerate(kpts_with_ids):
            if match_flags[jj]: continue
            dist = norm3d(np.float32(kpt[:2])- np.float32(kpt_tem[:2]))
            if dist < max_match_dist:
                match_flags_cand[ii] = True
                match_flags[jj] = True
                match_ids[ii] = jj
                dist_list.append(dist)
                break

    matched_kpts = []
    matched_kpts_cand = []
    for mid, flag, kpt_tem in zip(match_ids, match_flags_cand, kpts_cand_warp):
        if flag:
            matched_kpts.append(kpts_with_ids[mid][:2])
            matched_kpts_cand.append(kpt_tem[:2])



    if len(matched_kpts) <4 or len(matched_kpts_cand) <4:
        H_new = None
        mean_dist = max_match_dist
    else:
        H_est1, _ = cv2.findHomography(np.float32(matched_kpts_cand), np.float32(matched_kpts))
        if H_est1 is not None:
            H_new = np.mat(H_est1)* np.mat(H)
            H_new = np.array(H_new)
        else:
            H_new = None

        mean_dist = max(dist_list)
    return H_new, match_flags_cand, match_flags, match_ids, mean_dist 

def norm3d(xyz):
    return np.sqrt(np.sum(np.float32(xyz)**2))