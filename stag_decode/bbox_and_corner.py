import numpy as np
from util.vec_orientation import decode_orientation_vecs
from util.heatmap_postprocess import suppress_and_average_keypoints_with_vals
from util.homo_transform import get_homo_from_corners_to_corners, warpPerspectivePts

# from util.visualization import draw_circles, draw_boxes
def detect_center_and_corners(features_set_list, min_center_score, min_corner_score):
    '''
        Detect and clustering bounding boxes and unordered corners from the CNN output
    '''

    # get accurate corners
    corner_pos_cand, corner_scores_all_cand, corner_labels_cand, corner_vals_cand = detect_keypoints_with_hms(features_set_list['grid_pred'], min_score=min_corner_score)
    if len(corner_scores_all_cand)>0:
        corner_scores_cand = np.max(np.float32(corner_scores_all_cand), axis = 1).tolist()
    else:
        corner_scores_cand = []

    corners_with_ids_cand = [c+[l] for c, l in zip(corner_pos_cand, corner_labels_cand)]
    corner_directions_cand =decode_orientation_vecs(corner_vals_cand)
    corners_with_ids, corner_scores, corner_directions, _,  _ = suppress_and_average_keypoints_with_vals(corners_with_ids_cand, corner_scores_cand, sigma = 8, vals = corner_directions_cand, min_cos= 0.9, is_norm_val = True)


    
    # get bboxes
    tag_box_cand, tag_scores_all_cand, tag_labels_cand, tag_vals_cand = detect_keypoints_with_hms(features_set_list['ssd_pred'], min_score=min_center_score)
    tag_anchors_cand  = vals_to_local_anchors(tag_box_cand, tag_vals_cand)
    if len(tag_scores_all_cand) >0:
        tag_scores_cand = np.max(np.float32(tag_scores_all_cand), axis = 1).tolist()
    else:
        tag_scores_cand = []
        
    centers_with_labels_cand = [c[-2:] + [l] for c, l in zip(tag_anchors_cand, tag_labels_cand)]
    centers_vals_cand = [b+c[:-2] for c, b in zip(tag_anchors_cand, tag_box_cand)]
    center_sigmas = [((box[2]-box[0])+(box[3]-box[1]))/6 for box in tag_box_cand]

    centers_with_ids, center_scores, boxes, centers_vals,  _ = suppress_and_average_keypoints_with_vals(centers_with_labels_cand, tag_scores_cand, sigma = center_sigmas, vals = None, min_cos= 0.9, is_norm_val = False, vals2=centers_vals_cand)
    boxes = [val[:4] for val in centers_vals]
    anchors_in_boxes = [np.float32(val[4:]).reshape((-1,2)).tolist() for val in centers_vals]


    image_res = {'scale': 1,
                'centers_with_ids': centers_with_ids,
            'center_scores': center_scores,
            'corners_with_ids': corners_with_ids,
            'corner_scores': corner_scores, 
            'corner_directions':corner_directions,
            'boxes': boxes,
            'anchors_in_boxes': anchors_in_boxes,
    }


    return image_res

def detect_keypoints_with_hms(pred_res, min_score = 0.2):
    '''
        Thresholding CNN outputs
    '''

    confidences_pred_np, kpts_or_boxes_pred_np, vals_pred = pred_res

    # detect keypoint candidates
    mask = np.max(confidences_pred_np[:, 1:], axis = 1) > min_score
    kpts_or_boxes_cand = kpts_or_boxes_pred_np[mask, :].tolist()
    scores_cand = confidences_pred_np[mask, 1:].tolist()
    if len(scores_cand) >0:
        labels_cand = np.argmax(scores_cand, axis = 1).tolist()
    else:
        labels_cand = []
    vals_cand = vals_pred[mask, :].tolist()

    return kpts_or_boxes_cand, scores_cand, labels_cand, vals_cand


def vals_to_local_anchors(boxes, vals):
    '''
        From CNN output values to an ROI
    '''

    if len(vals) == 0: return []
    boxes_np = np.float32(boxes)
    vals_np = np.float32(vals)
    anchors_np = np.zeros(vals_np.shape, dtype=np.float32)

    for ii in range(anchors_np.shape[1]):
        xy = ii%2
        anchors_np[:, ii] = vals_np[:, ii] * (boxes_np[:, xy+2]-boxes_np[:, xy]) + (boxes_np[:, xy]+boxes_np[:, xy+2])*0.5
    # anchors_np = np.reshape(anchors_np, (anchors_np.shape[0], -1, 2))
    anchors = anchors_np.tolist()

    return anchors
    

def center_and_corners_to_tags(image_res, is_allow_no_corner_refine = True, is_with_corner_refine = True):
    '''
        Refine ROI with accurate unordered corners
    '''

    boxes = image_res['boxes']
    centers_with_ids = image_res['centers_with_ids']
    corners_with_ids = image_res['corners_with_ids']
    corner_directions = image_res['corner_directions']
    anchors_in_boxes = image_res['anchors_in_boxes']
    center_scores  = image_res['center_scores']

    # link corner and center, then refine
    center_to_corner_links = []
    # valid_center_flags = [True] * len(centers_with_ids)
    tag_res = []
    for ii in range(len(centers_with_ids)):
        bbox = boxes[ii]
        center_pos = centers_with_ids[ii][:2]
        tag_id= centers_with_ids[ii][2]
        anchors = anchors_in_boxes[ii]
        center_score = center_scores[ii]
        linked_corner_idx_list, cos_dist_list, rel_pos_list = assign_corners_to_one_center(center_pos, bbox, corners_with_ids, corner_directions)
        corners_selected = [corners_with_ids[idx][:2] for idx in linked_corner_idx_list]
        corner_ids_selected = [corners_with_ids[idx][2] for idx in linked_corner_idx_list]
        center_to_corner_links.append(corners_selected)

        valid_flag = False
        if len(anchors)>2 and len(linked_corner_idx_list) >= 1 and is_with_corner_refine:

            # refine the corner points for each tag candidate
            best_ids, guess_points = sort_corners_in_a_tag(center_pos, anchors, corners_selected)
            ordered_corners, center_link_score, corner_scores = guess_ordered_corners_and_scores(corners_selected,  
                                                                    cos_dist_list, 
                                                                    rel_pos_list, 
                                                                    best_ids, 
                                                                    guess_points )


            # reorder corners have different class labels
            main_idx = 0
            max_label = 0
            for curr_orient_idx in range(len(best_ids)):
                best_id = best_ids[curr_orient_idx]
                if best_id >=0 and corner_ids_selected[best_id] > max_label:
                    max_label = corner_ids_selected[best_id]
                    # main_idx is down-right
                    main_idx = (curr_orient_idx + 2)% len(best_ids)
            ordered_corners = ordered_corners[main_idx:] + ordered_corners[:main_idx]

            if sum([best_id>=0 for best_id in best_ids]) <len(anchors):
                # with guess points
                tag_id = -1

            valid_flag = True

        elif not is_with_corner_refine or (is_allow_no_corner_refine and len(linked_corner_idx_list) ==0):
            # without refinement
            ordered_corners = anchors
            corner_scores = [0] * len(anchors)
            center_link_score = 0
            main_idx = 0
            valid_flag = True

        if valid_flag:
            # store results if valid
            tag_detect_info = {            
                'center_pos': center_pos,
                'tag_id': tag_id,
                'center_score': center_score,
                'center_link_score': center_link_score,
                'ordered_corners': ordered_corners,
                'corner_scores': corner_scores,
                'corner_anchors': anchors,
                'main_idx': main_idx
                }
            tag_res.append(tag_detect_info)



    
    return tag_res, center_to_corner_links 
def guess_ordered_corners_and_scores(corners_selected, cos_dist_list, rel_pos_list, best_ids, guess_points ):
    '''
        Refinement or pratial refinement (if not every corner can be refine)
    '''


    ordered_corners = []
    corner_scores = []
    for guess_point, best_id in zip(guess_points, best_ids):
        if best_id >=0:
            # get accurate corner
            corner = corners_selected[best_id]
            score = cos_dist_list[best_id] * (1/max(1,rel_pos_list[best_id])) 

        else:
            # get guessed corner
            corner = guess_point
            score = 0

        ordered_corners.append(corner)
        corner_scores.append(score)


    # check if exists
    center_score = sum(corner_scores) /len(corner_scores)           

    return ordered_corners, center_score, corner_scores

def sort_corners_in_a_tag(center_pos, anchors, corners_selected, min_unit_dist_in_tag = 0.8):       
    '''
        Match anchors with corners_selected
    '''

    # convert to unit space
    unit_ref_points = [[0, 0], [1,0], [1,1], [0,1], [0.5, 0.5]]
    H = get_homo_from_corners_to_corners(anchors+[center_pos], unit_ref_points)    
    unit_corners_cand = warpPerspectivePts(H, corners_selected)

    # convert to log-polar, and tilt, and convert back to unit space
    polar_corners_cand = xy_to_polar(unit_corners_cand, unit_ref_points[-1])
    polar_corners_cand_tilted, tilt_degree = tilt_polar_corners(polar_corners_cand)
    unit_corners_cand_tilted = polar_to_xy(polar_corners_cand_tilted, unit_ref_points[-1])






    # get dist from candidate points to anchors, in unit space
    num_cand = len(corners_selected)
    best_ids = []
    match_flags = [False] * num_cand
    for ii in range(4):
        min_dist = min_unit_dist_in_tag
        best_id = -1
        ref_pt  = np.float32(unit_ref_points[ii])
        for jj in range(num_cand):
            if match_flags[jj]: continue
            pt = np.float32(unit_corners_cand_tilted[jj])
            dist = norm3d(ref_pt-pt)
            if dist<min_dist:
                best_id = jj
                min_dist = dist
        if best_id >=0:
            match_flags[best_id] = True
        best_ids.append(best_id)



    # guess_points
    polar_ref_points = xy_to_polar(unit_ref_points[:4], unit_ref_points[-1])
    x_corrections = []
    h_corrections = []
    # for ii, match_flag in enumerate(match_flags):
    #     if not match_flags: continue
    #     if ii >= len(polar_ref_points): break
    for ii in range(len(best_ids)):
        if best_ids[ii] <0: continue
        polar_ref_point =polar_ref_points[ii]
        matched_point = polar_corners_cand[best_ids[ii]]
        x_correct = matched_point[0]/polar_ref_point[0]
        h_correct = matched_point[1] - polar_ref_point[1]
        x_corrections.append(x_correct)
        h_corrections.append(h_correct)

    if len(x_corrections) ==0:
        guess_points = anchors
    else:
        x_correct_mean = np.mean(x_corrections)
        h_correct_mean = np.mean(h_corrections)
        # h_correct_mean = -tilt_degree
        H_inv = np.linalg.inv(H)
        polar_unit_guess_points = [[pt[0] *  x_correct_mean, pt[1] +  h_correct_mean] for pt in polar_ref_points]
        unit_guess_points = polar_to_xy(polar_unit_guess_points, unit_ref_points[-1])
        guess_points = warpPerspectivePts(H_inv, unit_guess_points)

        # # visualize to debug
        # image_vis = np.zeros((500,500,3))
        # colors= [(255,0,0)]
        # visualize_unit_points(image_vis, unit_ref_points[:4], colors)
        # colors= [(0,0,255)]
        # visualize_unit_points(image_vis, unit_corners_cand, colors, required_idx = False)
        # colors= [(0,255,255)]
        # visualize_unit_points(image_vis, unit_corners_cand_tilted, colors, required_idx = False)
        # colors= [(255,255,0)]
        # visualize_unit_points(image_vis, unit_guess_points, colors, required_idx = True)
        # cv2.waitKey(0)

    return best_ids, guess_points

def tilt_polar_corners(polar_corners):
    '''
        Automatically tilt the points
    '''
    polar_corners_np = np.float32(polar_corners)

    # tilt a little
    h_ori = polar_corners_np[:,1]
    h = h_ori +180+45

    tilt_degree_cand = []
    rot_tries = [0, -10, 10]

    for rot_try in rot_tries:
        h_rot = np.round((h+rot_try)/90) *90-h
        sorted_idx =  np.argsort(np.abs(h_rot))[:4]
        if len(sorted_idx) >=4:
            tilt_degree_try = np.mean(h_rot[sorted_idx])
        else:
            tilt_degree_try = h_rot[0]
        tilt_degree_cand.append(tilt_degree_try)

    max_idx = np.argmax(np.abs(tilt_degree_cand) * (np.abs(tilt_degree_cand)<45))
    tilt_degree = tilt_degree_cand[max_idx]

    polar_corners_np[:,1] += tilt_degree

    return polar_corners_np.tolist(), tilt_degree

def xy_to_polar(points, center):
    points_centered = np.float32(points) - np.float32([center])
    x = np.sqrt(np.sum(points_centered **2, axis = 1))
    h = np.arctan2(points_centered[:,1], points_centered[:,0])/np.pi * 180
    polar_points = np.concatenate([x[:,np.newaxis],h[:,np.newaxis]], axis=1).tolist()
    return polar_points

def polar_to_xy(polar_points, center):
    polar_points_np = np.float32(polar_points)
    polar_points_np[:,1] *=np.pi/180
    x = np.cos(polar_points_np[:,1]) * polar_points_np[:,0]
    y = np.sin(polar_points_np[:,1])* polar_points_np[:,0]
    points = np.concatenate([x[:,np.newaxis],y[:,np.newaxis]], axis=1)
    points += np.float32(center).reshape([-1,2])
    return points.tolist()

# def visualize_unit_points(image_vis, unit_points, colors, scale =0.3, radius = 5, required_idx = False):
#     h,w = image_vis.shape[:2]

#     vis_points = [[(pt[0] -0.5) * w*scale + w//2, (pt[1]-0.5) * h*scale + h//2] for pt in unit_points]
#     draw_circles(image_vis, vis_points, colors, radius= radius, required_idx= required_idx)
#     cv2.imshow('corners_in_unit_space', image_vis)
    

def assign_corners_to_one_center(center_pos, bbox, corners, vecs, cos_thresh = 0.6, rel_dist_thresh = 1.5):
    '''
        assign valid corners to a center/bbox:
            1) corners that point to the center
            2) corners that are inside the bbox

    '''
    if len(corners) ==0: return [],[],[]
    corners_np = np.float32(corners)[:,:2]
    vecs_np = np.float32(vecs)
    center_pos_np = np.float32(center_pos[:2])


    # get cos dist
    vecs_rel_np = corners_np - center_pos_np.reshape((-1,2))
    vec_len = np.sqrt(np.sum(vecs_np **2, axis = 1))
    vec_rel_len = np.sqrt(np.sum(vecs_rel_np **2, axis = 1))

    cos_dist = np.sum(vecs_rel_np*vecs_np, axis = 1)/np.maximum(vec_len * vec_rel_len, 1e-20)

    # get score for position
    four_axis_np = np.float32([[bbox[0] - center_pos[0], 0],
                                    [0, bbox[1] - center_pos[1]],
                                    [bbox[2] - center_pos[0], 0],
                                    [0, bbox[3] - center_pos[1]]])
    four_axis_len_np = np.abs(np.sum(four_axis_np, axis = 1))

    rel_dist = np.zeros((len(corners),),dtype=np.float32)

    for ii in range(4):
        rel_dist_along_axis = np.sum(vecs_rel_np*four_axis_np[ii, :].reshape([-1, 2]), axis = 1)
        rel_dist_along_axis = rel_dist_along_axis/(four_axis_len_np[ii]*four_axis_len_np[ii])
        rel_dist = np.maximum(rel_dist, rel_dist_along_axis)

    # select valid links
    idx_list = np.int32(list(range(len(corners))))
    mask = (cos_dist > max(cos_thresh, np.max(cos_dist)*0.85)) & (rel_dist < rel_dist_thresh) # np.max(cos_dist)*0.85?
    idx_list_selected = idx_list[mask].tolist()
    cos_dist_selected = cos_dist[mask].tolist()
    rel_dist_selected = rel_dist[mask].tolist()
    return idx_list_selected, cos_dist_selected, rel_dist_selected

def norm3d(xyz):
    return np.sqrt(np.sum(xyz**2))