
from torch import conv_transpose1d
from util.homo_transform import warpPerspectivePts
from stag_decode.fine_grid_matcher import FineGridMatcher
from stag_decode.keypoints import *
import cv2
import time

class StagDecoder:
    def __init__(self, predictor, unit_tag_template, marker_dict, cameraMatrix= None, distCoeffs = None, sigma_keypoints = 5, min_grid_match_ratio = 0.8, batch_size = 8, border_ratio = 0.15, image_rect_size = (256,256), iter_num = 1):

        self.predictor = predictor
        self.features = None
        self.sigma_keypoints = sigma_keypoints
        self.cameraMatrix = cameraMatrix
        self.distCoeffs =distCoeffs
        self.unit_tag_template = unit_tag_template
        self.min_grid_match_ratio = min_grid_match_ratio
        self.batch_size = batch_size
        self.image_rect_size = image_rect_size
        self.roi_in_crop = get_crop_ordered_corners(image_rect_size[0], image_rect_size[1], border_ratio)
        self.iter_num = iter_num
        self.marker_dict = marker_dict



    def detect_tags(self, image, rois):

        batch_size = self.batch_size
        iter_num = self.iter_num
        self.cnn_timing_list = []
        self.prediction_timing_list = []
        self.postprocess_timing_list = []

        rois_curr = rois
        valid_ids = None

        # iteration by updating ROIs
        decoded_tags = [{'is_valid': False} for _ in range(len(rois))]
        t0 = time.time()
        for iter_i in range(iter_num):

            

            print('>iter #%d'% iter_i) 
            crops, H_list, valid_ids = rectify_crops(image, rois_curr, self.roi_in_crop, self.image_rect_size, ids = valid_ids)

            # # visualize crops
            # for ii, crop in enumerate(crops):
            #     cv2.imshow('crop_%d_%d'%(iter_i, ii), crop)
            # cv2.waitKey(0)

            # CNN prediction
            features_pred_list = []
            batch_id = 0             
            while batch_id * batch_size < len(crops):
                t1 = time.time()
                features_pred_list += self.predictor.predict(crops[batch_id *batch_size:(batch_id+1)*batch_size])
                self.cnn_timing_list.append(self.predictor.cnn_timing)
                self.prediction_timing_list.append(time.time()-t1)
                batch_id +=1
                
            # decode from CNN outputs
            for features_pred, H, roi_idx in zip(features_pred_list, H_list, valid_ids):
                # print('--ROI #%d--'% (roi_idx))
                t2 = time.time()
                decoded_tag, roi_updated = self.decode_one_tag(features_pred, H)
                decoded_tag['image_size_crop'] =  self.image_rect_size
                rois_curr[roi_idx] = roi_updated # update roi
                decoded_tags[roi_idx] = decoded_tag # update result
                self.postprocess_timing_list.append(time.time()-t2)
                # print('Valid tag.' if decoded_tag['is_valid'] else 'Invalid.')

        self.total_timing = time.time() -t0
        return decoded_tags

    def decode_one_tag(self, features_pred, H):
        cameraMatrix = self.cameraMatrix
        distCoeffs = self.distCoeffs
        unit_tag_template = self.unit_tag_template
        sigma_keypoints = self.sigma_keypoints
        min_grid_match_ratio = self.min_grid_match_ratio
        marker_dict = self.marker_dict

        confidences_pred_np, grid_pred_np = features_pred['grid_pred'][:2]
        corners_pred_np = features_pred['corners_pred'][1]

        ordered_corners = [corners_pred_np.tolist()[idx] for idx in [0,1,3,2]]

        # detect fine_grid_points from cnn outputs
        fine_grid_points_with_ids_candidates, scores_candidates = detect_keypoints_with_hms(confidences_pred_np, grid_pred_np, sigma3_nms = sigma_keypoints)




        # match fine_grid_points with unit_tag_templates
        fine_grid_matcher = FineGridMatcher(ordered_corners, ordered_corners_in_crop_set = [], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        match_ratio, ordered_fine_grid_points_ids = unit_tag_template.match_fine_grid(fine_grid_points_with_ids_candidates, H, [fine_grid_matcher])




        # refine and get ids
        roi_updated = None
        if match_ratio > min_grid_match_ratio:
            # get id if missing
            ordered_fine_grid_points_ids = fill_empty_ids(ordered_fine_grid_points_ids, fine_grid_points_with_ids_candidates)

            # update roi
            roi_updated = unit_tag_template.update_corners_in_image(ordered_fine_grid_points_ids, H, valid_flag_list= None, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)

            # get main_idx and reorder
            # _, ordered_kpts_with_ids = unit_tag_template.reorder_points_with_main_idx(ordered_fine_grid_points_ids, 0)
            main_idx, decimal_id = marker_dict.get_main_idx([kpt[2] for kpt in ordered_fine_grid_points_ids])
            if decimal_id < 0:
                main_idx, decimal_id = unit_tag_template.get_main_idx([kpt[2] for kpt in ordered_fine_grid_points_ids])
            ordered_fine_grid_points_ids, ordered_kpts_with_ids = unit_tag_template.reorder_points_with_main_idx(ordered_fine_grid_points_ids, main_idx)
            binary_ids = [kpt[2] for kpt in ordered_kpts_with_ids]

            # get score
            tag_score = sum(scores_candidates)/len(scores_candidates)
            
            # get unit_tag
            unit_tag = unit_tag_template.get_unit_tag(ordered_fine_grid_points_ids)

            # unwarp keypoints
            H_inv = np.linalg.inv(H)
            ordered_fine_grid_points_in_image = warpPerspectivePts(H_inv, [kpt[:2] for kpt in ordered_fine_grid_points_ids])





            # save results
            decoded_tag = {'is_valid': True, 
                        'H_crop': H,
                        'tag_id': decimal_id,
                        'binary_ids': binary_ids, 
                        'keypoints_with_ids': ordered_fine_grid_points_ids,
                        'keypoints_in_images': ordered_fine_grid_points_in_image,
                        'score': tag_score,
                        'unit_tag': unit_tag,
                        'main_idx': main_idx, 
                        'rotation_idx': 0
                        }

        else:

            decoded_tag = {'is_valid': False}
            roi_updated = None

   

        return decoded_tag, roi_updated


def rectify_crops(image, rois, roi_in_crop, image_rect_size, ids = None):
    
    crops = []
    H_list = []
    valid_ids = []
    th, tw = image_rect_size
    for ii, roi in enumerate(rois):
        if roi is not None:
            H, _ = cv2.findHomography(np.float32(roi), np.float32(roi_in_crop))
            image_rect = cv2.warpPerspective(image, H, (tw,th))         
            crops.append(image_rect)   
            H_list.append(H)  
            if ids is None:
                valid_ids.append(ii)
            else:
                valid_ids.append(ids[ii])
    return crops, H_list, valid_ids 



def get_crop_ordered_corners(h, w, border_ratio):
    # border_ratio is in [0,0.3]
    bw = max(h,w) * border_ratio
    ordered_corners_in_crop = [[bw-0.5, bw-0.5], [w-bw-0.5, bw-0.5], [w-bw-0.5, h-bw-0.5], [bw-0.5, h-bw-0.5] ]
    return ordered_corners_in_crop