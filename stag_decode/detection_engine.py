from stag_decode.marker_dict import MarkerDict
import numpy as np
from stag_decode.stag_roi_detector import StagDetector
from stag_decode.stag_decoder import StagDecoder
from stag_decode.stag_unit_template import UnitTagTemplate
from stag_decode.runetag_unit_template import UnitTagTemplate as UnitRuneTagTemplate
from network.predictors import DecoderPredictor, DetectorPredictor
from stag_decode.marker_dict import MarkerDict
from stag_decode.visualization import visualize_rt, visual_keypoints, vis_tag_rois, vis_center_and_corners
import cv2

class DetectionEngine:
    def __init__(self, model_detector, model_decoder, device, tag_type, grid_size_cand_list, 
                    stg2_iter_num = 1, border_ratio = 0.15, detector_stride_list = [8,16,32,64], min_center_score = 0.2, min_corner_score = 0.3, batch_size_stg2 = 4,
                    cameraMatrix= None, distCoeffs = None, tag_real_size_in_meter_dict = {}, step_elem_num = None, tag_anno_dict = {},  image_rect_size = None, 
                    codebook = {}, hamming_dist = 0):
        '''
            ##params:
                model_detector: Model of Stage-1;
                model_decoder: Model of Stage-2;
                device: GPU/CPU device for models;
                tag_type: topotag / arucotag / runetag;
                grid_size_cand_list: [4] for topotag, [4,5,6,7] for arucotag, [] for runetag;
                min_center_score: minimal score for boxes in Stage-1;
                min_corner_score: minimal score for corners in Stage-1 ;
                cameraMatrix: camera matrix, [[800, 0, w/2], [800, 0, h/2], [0,0,1]];
                distCoeffs: distortion coefficients, [0,0,0,0,0,0,0,0];
                codebook: marker library,
                hamming_dist: maximal hamming distance when checking the marker library, 
                batch_size_stg2: batch size for Stage-2.
        '''

        # tag_type='topotag', grid_size_cand_list= [3,4,5]
        # tag_type='arucotag', grid_size_cand_list= [4,5,6,7]
        if cameraMatrix is None or len(cameraMatrix) == 0:
            cameraMatrix =  [[600, 0, 640], [0, 600, 400], [0, 0, 1]]
        if distCoeffs is None or len(distCoeffs) == 0:
            distCoeffs = [0] * 8

        cameraMatrix = np.float32(cameraMatrix).reshape(3,3)
        distCoeffs = np.float32(distCoeffs)

        if image_rect_size is None:
            image_rect_size = (256, 256) if tag_type != 'runetag' else (512, 512)
        




        if tag_type == 'runetag':
            unit_tag_template = UnitRuneTagTemplate(tag_type=tag_type)
        else:
            unit_tag_template = UnitTagTemplate(tag_type=tag_type, grid_size_list= grid_size_cand_list, step_elem_num=step_elem_num)

        marker_dict  = MarkerDict(codebook, unit_tag_template, hamming_dist= hamming_dist)

        # detector
        detector_predictor = DetectorPredictor(model_detector, device, stride_list = detector_stride_list)
        self.stag_detector = StagDetector(detector_predictor, min_center_score = min_center_score, min_corner_score = min_corner_score, is_with_corner_refine = True)

        # decoder
        sigma_keypoints = 5 if tag_type != 'runetag' else 8 
        decoder_predictor = DecoderPredictor(model_decoder, device, image_rect_size)
        self.stag_decoder = StagDecoder(decoder_predictor, unit_tag_template, marker_dict, border_ratio = border_ratio, iter_num= stg2_iter_num, cameraMatrix= cameraMatrix, distCoeffs = distCoeffs, image_rect_size = image_rect_size, sigma_keypoints= sigma_keypoints, batch_size= batch_size_stg2)


        # self.step_elem_num = unit_tag_template.get_step_elem_num()
        self.decoded_tags_history = [] 
        self.pose_solver_dict = unit_tag_template.get_pose_solver_dict(cameraMatrix= cameraMatrix, distCoeffs = distCoeffs)
        self.tag_anno_dict = tag_anno_dict
        self.tag_real_size_in_meter_dict = tag_real_size_in_meter_dict
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.valid_count = 0
        self.tag_type = tag_type
        self.codebook = codebook


      


    def process(self, image, detect_scale = None):
        stag_detector= self.stag_detector
        stag_decoder = self.stag_decoder

        if detect_scale is None:
            h, w = image.shape[:2]
            detect_scale = min(640, min(h, w))/ min(h, w)

        if type(detect_scale) == list:
            detect_scales = detect_scale
        else:
            detect_scales = [detect_scale]


        # stage-1
        self.image = image
        print('>>>>>>>Stage-1<<<<<<<')
        rois_info = stag_detector.detect_rois(image, detect_scales[0] )
        self.bbox_corner_info = stag_detector.image_res

        # stage-1 print
        print('%d ROIs'%len(rois_info))

 
        # stage-2
        print('>>>>>>>Stage-2<<<<<<<')
        rois = [roi_info['ordered_corners'] for roi_info in rois_info]
        self.rois_info = rois_info        
        decoded_tags = stag_decoder.detect_tags(image, rois.copy())

        # stage-2 print
        print('Valid ROIs:', end=' ')
        for ii, decoded_tag in enumerate(decoded_tags):
            if decoded_tag['is_valid']: print('%d'% ii, end=', ')
        print('')

        # estimate pose
        for ii, decoded_tag in enumerate(decoded_tags):
            if not decoded_tag['is_valid']: continue
            roi = rois[ii]
            pose_result = self.estimate_pose(decoded_tag, roi)

            for k in pose_result:
                decoded_tag[k] = pose_result[k] 

        self.decoded_tags = decoded_tags
        return decoded_tags


    def print_timming(self):

        roi_num = len(self.decoded_tags)
        valid_tag_num = sum([decoded_tag['is_valid'] for decoded_tag in self.decoded_tags])
        batch_size_stg2 = self.stag_decoder.batch_size
        avg_predition_time = sum(self.stag_decoder.prediction_timing_list)/max(len(self.decoded_tags),1)
        avg_propocess_time = sum(self.stag_decoder.postprocess_timing_list)/max(valid_tag_num,1)

        print('------timing (sec.)------')
        print('Stage-1 :                   ', 
        '[CNN %.4f] %.4f'%(self.stag_detector.predition_timing, self.stag_detector.total_timing))
        print('Stage-2 (1 marker):         ', 
        '[CNN %.4f] %.4f'%(avg_predition_time, avg_predition_time + avg_propocess_time))
        print('Stage-2 (%d rois, %d markers):' %(roi_num, valid_tag_num),
        '[CNN %.4f] %.4f'%(sum(self.stag_decoder.prediction_timing_list), self.stag_decoder.total_timing))


    def estimate_pose(self, decoded_tag, roi):
        tag_real_size_in_meter_dict = self.tag_real_size_in_meter_dict
        pose_solver_dict = self.pose_solver_dict
        tag_anno_dict = self.tag_anno_dict

        fine_grid_points = decoded_tag['keypoints_in_images']
        tag_id_decimal = decoded_tag['tag_id']
         

        # get 6DoF pose
        num_fine_grid_points =len(fine_grid_points) 

        if num_fine_grid_points in pose_solver_dict:
            pose_solver = pose_solver_dict[num_fine_grid_points]
            # check marker size
            if (num_fine_grid_points, tag_id_decimal)  in  tag_real_size_in_meter_dict:
                tag_real_size_in_meter = tag_real_size_in_meter_dict[(num_fine_grid_points, tag_id_decimal) ] 
            elif -1 in tag_real_size_in_meter_dict:
                tag_real_size_in_meter = tag_real_size_in_meter_dict[-1]
            else:
                tag_real_size_in_meter = pose_solver.tag_real_size_in_meter_default

            # check customized annotation
            if (num_fine_grid_points, tag_id_decimal) in tag_anno_dict:
                tag_kpts_anno = tag_anno_dict[(num_fine_grid_points, tag_id_decimal) ]
            else:
                tag_kpts_anno = None

            # pose from stg2                                   
            res_code, rvecs, tvecs = pose_solver.keypoints_to_pose(fine_grid_points, kpts_valid_flags=None, tag_real_size_in_meter=tag_real_size_in_meter, tag_kpts_anno=tag_kpts_anno)

            
            # pose from stg1
            if pose_solver.tag_type == 'runetag':
                main_idx = 0
            else:
                main_idx = decoded_tag['main_idx']
            ordered_corners_stg1_rotated = roi[main_idx:] + roi[:main_idx]
            tag_kpts_anno_corners = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5,0], [-0.5, 0.5, 0]]
            res_code_stg1, rvecs_stg1, tvecs_stg1 = pose_solver.keypoints_to_pose(ordered_corners_stg1_rotated, kpts_valid_flags=None,tag_real_size_in_meter=tag_real_size_in_meter, tag_kpts_anno=tag_kpts_anno_corners)
            
            
        else:
            res_code, rvecs, tvecs =  None, None, None
            res_code_stg1, rvecs_stg1, tvecs_stg1 =  None, None, None
            tag_real_size_in_meter = 0

        pose_result = {}
        pose_result['rvecs'] = rvecs 
        pose_result['tvecs'] = tvecs 
        pose_result['tag_real_size_in_meter'] = tag_real_size_in_meter 
        pose_result['rvecs_stg1'] = rvecs_stg1 
        pose_result['tvecs_stg1'] = tvecs_stg1 

        return pose_result

    def visualize(self, is_pause = True):

        image = self.image
        time_to_wait = 0 if is_pause else 1
         # draw rois
        image_out_rois = vis_tag_rois(image, self.rois_info)
        image_out_bbox_corner = vis_center_and_corners(image, self.bbox_corner_info)

        # draw keypoints and pose
        image_out_pose = image.copy()
        image_rect_list = []

        decoded_tags = self.decoded_tags
        
        for decoded_tag in decoded_tags:
            if not decoded_tag['is_valid']: continue

            tag_id_decimal = decoded_tag['tag_id']
            rotate_idx = 0



            
            visualize_rt(image_out_pose, decoded_tag['rvecs'], decoded_tag['tvecs'], self.cameraMatrix, self.distCoeffs, decoded_tag['tag_real_size_in_meter'], tag_id_decimal = tag_id_decimal, tid_text_pos=None, score=None, is_draw_cube =False, rotate_idx=rotate_idx, text_color= None)

            image_rect = visual_keypoints(image, decoded_tag, tag_id_decimal)
            image_rect_list.append(image_rect)

        image_out_kpts = cv2.hconcat(image_rect_list)
        if image_out_kpts is None:
            image_out_kpts = np.zeros([300, 300, 3],dtype=np.uint8)

        cv2.imshow('image with pose', image_out_pose)
        cv2.imshow('Boxes and corners (Stage-1)', image_out_bbox_corner)
        cv2.imshow('ROIs (Stage-1)', image_out_rois)
        cv2.imshow('Keypoints (Stage-2)', image_out_kpts)
        c = cv2.waitKey(time_to_wait)

        return c