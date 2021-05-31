from stag_decode.bbox_and_corner import *
import time

class StagDetector:
    def __init__(self, predictor, min_center_score = 0.3, min_corner_score = 0.3, is_allow_no_corner_refine = True, is_with_corner_refine = True):


        self.image_in = None
        self.tag_res = []
        self.predictor = predictor
        self.min_center_score = min_center_score
        self.min_corner_score = min_corner_score
        self.is_allow_no_corner_refine = is_allow_no_corner_refine
        self.is_with_corner_refine = is_with_corner_refine


    def detect_rois(self, image_in, scale = 1):

        min_center_score = self.min_center_score
        min_corner_score = self.min_corner_score
        is_allow_no_corner_refine = self.is_allow_no_corner_refine
        is_with_corner_refine = self.is_with_corner_refine
        self.cnn_timing = -1
        self.total_timing = -1
        self.predition_timing = -1

        t0 = time.time() # timing
        
        #  pred features       
        features_pred = self.predictor.predict(image_in, scale)[0]
        self.cnn_timing = self.predictor.cnn_timing
        self.predition_timing = time.time() - t0

        # sparse unordered corners, bbox with 5 anchors (center + 4 ordered corners)
        image_res = detect_center_and_corners(features_pred, min_center_score = min_center_score, min_corner_score = min_corner_score)
        

        # center_to_corner_links and tags (4 ordered corners)
        tag_res, center_to_corner_links = center_and_corners_to_tags(image_res, is_allow_no_corner_refine=is_allow_no_corner_refine, is_with_corner_refine = is_with_corner_refine)
        self.total_timing = time.time() -t0 # timing

        # save results
        self.image_res =image_res
        self.tag_res = tag_res 
        self.center_to_corner_links = center_to_corner_links
        self.features_pred = features_pred


        return tag_res