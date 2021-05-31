import numpy as np
import math
import cv2

def warpPerspectivePts_np(H, pts_np, image_scale = 1):
    '''
        pts_np: Nx2
    '''

    n = pts_np.shape[0]
    ones = np.ones([n, 1])
    pts_np_homo = np.concatenate([pts_np, ones], axis=1)
    pts_np_warp = H.dot(pts_np_homo.T)
    pts_np_warp = np.float32(np.transpose(pts_np_warp))
    pts_np_warp[:,0] =pts_np_warp[:,0] /pts_np_warp[:,2] 
    pts_np_warp[:,1] =pts_np_warp[:,1] /pts_np_warp[:,2] 
    pts_np_warp = pts_np_warp[:,:2]
    pts_np_warp = pts_np_warp/image_scale
    return pts_np_warp

def warpPerspectivePts(H, pts, image_scale = 1):
    pts_np = np.float32(pts)
    pts_np_warp = warpPerspectivePts_np(H, pts_np, image_scale)
    pts_warp = pts_np_warp.tolist()
    return pts_warp


def warpPerspectivePts_with_vals(H, pts, image_scale = 1):
    pts_np = np.float32([pt[:2] for pt in pts])
    pts_np_warp = warpPerspectivePts_np(H, pts_np, image_scale)
    pts_warp = [[pt[0]/image_scale, pt[1]/image_scale]+ pt0[2:] for pt0, pt in zip(pts, pts_np_warp.tolist())]
    return pts_warp

def get_translate_mat(x,y):
    T = np.float32([[1,0,x], [0,1,y], [0,0,1]])
    return T

def get_scale_translate_mat(x,y, sx=1, sy = 1):
    H = np.float32([[sx,0,x], [0,sy,y], [0,0,1]])
    return H

def get_homo_from_image_to_corners(h,w, corners_dst):
    corners_src = np.float32([[0,0], [w,0], [w,h], [0,h]])-0.5
    H = get_homo_from_corners_to_corners(corners_src, corners_dst)
    return H

def get_homo_from_corners_to_corners(corners_src, corners_dst):
    H, _ = cv2.findHomography(np.float32(corners_src), np.float32(corners_dst))
    return H



def get_homography_matrix(theta_init_idx, p1, p2, sx, sy, theta_angle,  h, w):
    '''
    h, w: size of image
    perform transformations:
        rotate: theta_init_idx = 0, 1, 2, 3 ... for 0, 90, 180, 270 degress
        projective: p1, p2
        affine: sx, sy
        rotate: theta_angle 0~360 degree
        recenter and estimate new h, w

        
    return:
        H: combined transformation
        nh, nw: new size of the image


    reference:
    https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4 
    '''
    nw, nh = w, h

    # Euclidean, translate to center, rotate, tranlate back
    # theta = math.pi * theta_init_idx/180
    theta = math.pi * 0.5 * theta_init_idx
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cx = w/2
    cy = h/2
    pz = int(max(w, h) * 0.25)
    H_m1 = [[1, 0, -cx],
        [0, 1, -cy],
        [0,0,1]]
    H_rot= [[cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0,0,1]]
    H_m2 = [[1, 0, cx+pz],
        [0, 1, cy+pz],
        [0,0,1]]
    He = np.mat(H_m2)*np.mat(H_rot)*np.mat(H_m1)





    # Affine
    Ha = [[1, sy, 0],
        [sx, 1, 0],
        [0,0,1]]

    # Projective

    Hp = [[1,0,0],
          [0,1,0],
          [p1,p2,1]]


    H_a_p =  np.mat(Ha)*np.mat(Hp) * He
    H_a_p /= H_a_p[2,2]


    # four corners
    pts = [[0,0], [0, h], [w, 0], [w, h]]
    pts_warp = warpPerspectivePts(H_a_p, pts)
    x_min = min([pt[0] for pt in pts_warp])
    y_min = min([pt[1] for pt in pts_warp])
    x_max = max([pt[0] for pt in pts_warp])
    y_max = max([pt[1] for pt in pts_warp])

    # update center point
    w2, h2 = x_max-x_min, y_max-y_min
    


    # padding
    pz = int(max(w2, h2) * 0.1)
    # H_pad = [[1, 0, pz],
    #     [0, 1, pz],
    #     [0,0,1]]



    # Euclidean, translate to center, rotate, tranlate back
    theta2 = math.pi * theta_angle/180
    cos_theta = math.cos(theta2)
    sin_theta = math.sin(theta2)
    cx = (x_max+x_min)/2
    cy = (y_max+y_min)/2
    H_m1 = [[1, 0, -cx],
        [0, 1, -cy],
        [0,0,1]]
    H_rot= [[cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0,0,1]]
    H_m2 = [[1, 0, cx+pz],
        [0, 1, cy+pz],
        [0,0,1]]
    He2 = np.mat(H_m2)*np.mat(H_rot)*np.mat(H_m1)

    # combine
    H = He2 * H_a_p

    # size of the image
    pts_warp = warpPerspectivePts(H, pts)
    x_min = min([pt[0] for pt in pts_warp])
    y_min = min([pt[1] for pt in pts_warp])
    x_max = max([pt[0] for pt in pts_warp])
    y_max = max([pt[1] for pt in pts_warp])

    nw, nh = x_max-x_min+2, y_max-y_min+2

    # translate

    H_c = [[1, 0, -x_min+1],
        [0, 1, -y_min+1],
        [0,0,1]]

    H = np.mat(H_c)*H
    


    return H, nw, nh 


