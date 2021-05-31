from util.homo_transform import warpPerspectivePts
import cv2
import numpy as np

def controlpoints_to_keypoints_in_crop_with_homo(controlpoints_gt, controlpoints_in_crop, keypoints_gt, distCoeffs= None, cameraMatrix = None, H = None):
    keypoints_in_crop = []
    controlpoints_gt = [kpt[:2] for kpt in controlpoints_gt]
    controlpoints_in_crop= [kpt[:2] for kpt in controlpoints_in_crop]
    keypoints_gt= [kpt[:2] for kpt in keypoints_gt]
    if distCoeffs is None or cameraMatrix is None or H is None:
        # warp in crop

        H_gt2undist, status  = cv2.findHomography(np.float32(controlpoints_gt), np.float32(controlpoints_in_crop))
        if H_gt2undist is not None:
            keypoints_in_crop = warpPerspectivePts(H_gt2undist, keypoints_gt)
    else:
        # undistort and warp
        cameraMatrix = np.float32(cameraMatrix)
        distCoeffs = np.float32(distCoeffs)
        controlpoints_undistorted =  from_crop_to_undistorted(controlpoints_in_crop, H, cameraMatrix, distCoeffs) 
        H_gt2undist, status  = cv2.findHomography(np.float32(controlpoints_gt), np.float32(controlpoints_undistorted))
        if H_gt2undist is not None:
            keypoints_undistorted = warpPerspectivePts(H_gt2undist, keypoints_gt)
            keypoints_in_crop = from_undistorted_to_crop(keypoints_undistorted, H, cameraMatrix, distCoeffs)

    return keypoints_in_crop

def from_crop_to_undistorted(points_in_crop, H, cameraMatrix, distCoeffs):
    points_in_crop = [kpt[:2] for kpt in points_in_crop]
    H_inv = np.linalg.inv(np.array(H))
    points_in_image = warpPerspectivePts(H_inv, points_in_crop, image_scale=1)
    if distCoeffs is None or cameraMatrix is None:
        points_in_image_undistorted = points_in_image
    else:
        points_in_image_undistorted = cv2.undistortPoints(np.array(points_in_image), cameraMatrix, distCoeffs) 
        points_in_image_undistorted = points_in_image_undistorted.reshape((-1,2))

        
    return points_in_image_undistorted

def from_undistorted_to_crop(points_undistorted, H, cameraMatrix, distCoeffs):
    points_undistorted = [kpt[:2] for kpt in points_undistorted]
    if distCoeffs is None or cameraMatrix is None:
        points_in_crop = warpPerspectivePts(H, points_undistorted, image_scale=1)
    else:
        points_undistorted = np.float32(points_undistorted)
        points_undistorted_homo = cv2.convertPointsToHomogeneous(points_undistorted)
        points_in_image,_ = cv2.projectPoints(points_undistorted_homo, np.float32([0,0,0]), np.float32([0,0,0]),cameraMatrix, distCoeffs)
        points_in_image = points_in_image.reshape((-1,2))
        points_in_image = points_in_image.tolist()
        points_in_crop = warpPerspectivePts(H, points_in_image, image_scale=1)

        
    return points_in_crop