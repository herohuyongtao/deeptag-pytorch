from util.heatmap_postprocess import suppress_and_average_keypoints
import math
import numpy as np


def fill_empty_ids(ordered_kpts_with_ids, kpts_with_ids_cand):
    for kpt1 in ordered_kpts_with_ids:
        if kpt1[2] <0:
            min_dist = 10000
            for kpt2 in kpts_with_ids_cand:                
                dist = math.sqrt((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2) 
                if dist < min_dist:
                    min_dist = dist
                    kpt1[2] = kpt2[2]

    return ordered_kpts_with_ids




def detect_keypoints_with_hms(confidences_pred_np, grid_pred_np, sigma3_nms = 5, min_score = 0.3, min_trust_ratio = 0.6):
    # detect keypoint candidates

    mask = np.max(confidences_pred_np[:, 1:], axis = 1) > min_score
    kpts_cand = grid_pred_np[mask, :]
    scores_cand_all = confidences_pred_np[mask, 1:]

    trust_ratio = np.sum(np.sum(scores_cand_all, axis = 1) > 0.5)/scores_cand_all.shape[0]
    trust_flag = trust_ratio >min_trust_ratio

    # trust_flag = True
    if trust_flag:

        labels = np.argmax(scores_cand_all, axis = 1)
        kpts_with_ids_cand = [kpt + [label] for kpt, label in zip(kpts_cand.tolist(), labels.tolist())]
        scores_cand_all = scores_cand_all.tolist()
        scores_cand = [vals[kpts_with_id[2]] for kpts_with_id, vals in zip(kpts_with_ids_cand, scores_cand_all)]     



        kpts_with_ids, scores,_ = suppress_and_average_keypoints(kpts_with_ids_cand, scores_cand, sigma = sigma3_nms)
    else:
        kpts_with_ids, scores = [], []
    return kpts_with_ids, scores