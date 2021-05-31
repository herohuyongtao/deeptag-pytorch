import cv2
import numpy as np
from util.homo_transform import  warpPerspectivePts
from util.util import select_valid_vals

class PoseSolver:
    def __init__(self, unit_tag, step_elem_num, cameraMatrix=None, distCoeffs=None, tag_real_size_in_meter_default= 0.05, is_ransac_solvepnp= False, tag_type = None):
        self.cameraMatrix = np.array(cameraMatrix)
        self.distCoeffs =np.array(distCoeffs)
        self.is_ransac_solvepnp = is_ransac_solvepnp
        self.tag_real_size_in_meter_default =tag_real_size_in_meter_default
        if tag_type == 'runetag':
            fine_grid_points_anno = get_runetag_keypoints_anno(unit_tag)
        else:
            # default unit tag
            fine_grid_points_anno = get_fine_grid_points_anno(unit_tag, step_elem_num)
        self.fine_grid_points_anno = fine_grid_points_anno
        self.tag_type = tag_type

        
   

    def keypoints_to_pose(self, kpts_in_image, tag_kpts_anno = None, tag_real_size_in_meter = None, kpts_valid_flags = None):
        cameraMatrix = self.cameraMatrix
        distCoeffs = self.distCoeffs

        if tag_real_size_in_meter is None:
            tag_real_size_in_meter = self.tag_real_size_in_meter_default

        if tag_kpts_anno is None:
            tag_kpts_anno = self.fine_grid_points_anno

        kpts =  [kpt[:2] for kpt in kpts_in_image]

        if len(tag_kpts_anno)>0 and len(tag_kpts_anno[0]) ==2:
            tag_kpts_anno = [kpt_anno + [0] for kpt_anno in tag_kpts_anno]

        # estimate R|T with solvePNP
        tag_kpts_anno_valid = select_valid_vals(tag_kpts_anno, kpts_valid_flags)
        kpts_valid = select_valid_vals(kpts, kpts_valid_flags)

        if len(kpts_valid) ==4:
            res_code, rvecs, tvecs = cv2.solvePnP(np.array(tag_kpts_anno_valid) * tag_real_size_in_meter, np.array(kpts_valid), cameraMatrix, distCoeffs ,flags= cv2.SOLVEPNP_IPPE)
        else:
            if not self.is_ransac_solvepnp:
                res_code, rvecs, tvecs = cv2.solvePnP(np.array(tag_kpts_anno_valid) * tag_real_size_in_meter, np.array(kpts_valid), cameraMatrix, distCoeffs ,flags= cv2.SOLVEPNP_ITERATIVE)
            else:
                # ransac
                res_code, rvecs, tvecs, _ = cv2.solvePnPRansac(tag_kpts_anno_valid* tag_real_size_in_meter, np.array(kpts_valid), cameraMatrix, distCoeffs)
    
        self.tag_pose = res_code, rvecs, tvecs
        return res_code, rvecs, tvecs



def get_fine_grid_points_anno(unit_tag, step_elem_num, is_inverse_x = True):
    unit_points = unit_tag.get_fine_grid_points(is_center = True, step_elem_num= step_elem_num)
    w = unit_tag.get_fine_grid_size(step_elem_num= 1)
    x, y = unit_tag.get_center_pos()

    H1 = np.float32([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    H2 = np.float32([[1/w, 0, 0], [0, 1/w, 0], [0, 0, 1]])
    H = np.matmul(H2, H1)
    fine_grid_points_anno = warpPerspectivePts(H, unit_points)
    if is_inverse_x:
        fine_grid_points_anno = [[-pt[0]] + pt[1:] for pt in fine_grid_points_anno]

    return fine_grid_points_anno

def get_runetag_keypoints_anno(unit_runetag, is_inverse_x = True):
    x_and_y_with_labels = unit_runetag.get_keypoints_with_labels()
    unit_points = [kpt[:2] for kpt in x_and_y_with_labels]
    x, y = unit_runetag.get_center_pos()

    H1 = np.float32([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    H2 = np.float32([[1/2, 0, 0], [0, 1/2, 0], [0, 0, 1]])
    H = np.matmul(H2, H1)
    fine_grid_points_anno = warpPerspectivePts(H, unit_points)
    if is_inverse_x:
        fine_grid_points_anno = [[-pt[0]] + pt[1:] for pt in fine_grid_points_anno]

    return fine_grid_points_anno