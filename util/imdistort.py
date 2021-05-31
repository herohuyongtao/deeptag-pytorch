import numpy as np
import cv2


def distort_image_and_keypoints(image, keypoints, h, w, k1):
    '''
        Distort image and keypoints 
        with cv2.undistort and cv2.undistortPoints
    '''

    cameraMatrix =[[ 900, 0., w//2],[ 0., 900, h//2],[0., 0., 1. ]]
    distCoeffs = [0.0]*8
    distCoeffs[0] = -k1


    cameraMatrix = np.array(cameraMatrix)
    distCoeffs = np.array(distCoeffs)
    

    if image is None:
        image_new = None
    else:
        image_new = cv2.undistort(image, cameraMatrix, distCoeffs, newCameraMatrix= cameraMatrix)
    if keypoints is None:
        keypoints_new  = None
    elif len(keypoints) ==0:
        keypoints_new = []
    else:
        kpts_np = np.float32(keypoints)
        kpts_np_new = cv2.undistortPoints(kpts_np, cameraMatrix, distCoeffs, R = np.eye(3), P= cameraMatrix )
        keypoints_new = np.reshape(kpts_np_new, [kpts_np_new.shape[0], kpts_np_new.shape[2]]).tolist()
    return image_new, keypoints_new


