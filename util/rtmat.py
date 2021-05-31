from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import cv2
def rmat2quat(rmat):
    # q_array = [x, y, z, w]
    rmat = np.reshape(np.float32(rmat), (3,3))
    try:    
        r = R.from_matrix(rmat) # scipy == 1.5.2
    except:
        r = R.from_dcm(rmat) # scipy == 1.3.1

    q_array = r.as_quat()
    return q_array

def quart2rmat(q_array):
    # q_array = [x, y, z, w]

    r = R.from_quat(q_array)    
    try:    
        r = np.array(r.as_matrix()) # scipy == 1.5.2
    except:
        r = np.array(r.as_dcm()) # scipy == 1.3.1

    return r


def diff_rtmat(rtmat, rtmat_ref):
    rmat, tvec = decompose_rtmat(rtmat)
    rmat_ref, tvec_ref = decompose_rtmat(rtmat_ref)


    rmat_inv = np.linalg.inv(rmat) 
    # differece
    diff_rmat = np.dot(rmat_inv, rmat_ref)
    diff_tvec = tvec_ref - tvec

    # angle from quaterion
    diff_rmat_quar = rmat2quat(diff_rmat)
    angle = math.acos(diff_rmat_quar[-1]) * 2
    if angle > math.pi:
        angle = 2* math.pi - angle
    angle_degree = angle / math.pi *180
    # distance with L2 Norm
    distance = float(math.sqrt(np.sum(diff_tvec**2)))


    return distance, angle_degree, diff_rmat, diff_tvec

def compose_rtmat(rmat, tvec):
    rmat = np.reshape(np.float32(rmat), (3,3))
    tvec = np.reshape(np.float32(tvec), (3,1))
    rtmat = np.ones( (4,4), dtype = np.float32)
    rtmat[:3,:3] = rmat
    rtmat[:3, 3] = tvec.ravel()
    rtmat[3,3] = 1
    return rtmat


def decompose_rtmat(rtmat):
    rtmat = np.reshape(np.float32(rtmat), (4,4))
    rmat = rtmat[:3,:3]
    tvec = rtmat[:3, 3]
    return rmat, tvec

def rtmat2rtvecs(rtmat):
    rmat, tvecs = decompose_rtmat(rtmat)
    rvecs,_ = cv2.Rodrigues(rmat)
    return rvecs, tvecs

def rtvecs2rtmat(rvecs, tvecs):
    rmat,_ = cv2.Rodrigues(rvecs)
    rtmat = compose_rtmat(rmat, tvecs)
    return rtmat

def rmat_rotate_xyz(angle, axis_id):
    rmat = np.eye(3)
    if axis_id == 0:
        # x
        rmat[1,1] = math.cos(angle)
        rmat[2,1] = math.sin(angle)
        rmat[2,2] = math.cos(angle)
        rmat[1,2] = -math.sin(angle)

    elif axis_id == 1:
        # y
        rmat[0,0] = math.cos(-angle)
        rmat[2,0] = -math.sin(-angle)
        rmat[2,2] = math.cos(-angle)
        rmat[0,2] = math.sin(-angle)

    elif axis_id == 2:
        # z
        rmat[0,0] = math.cos(angle)
        rmat[1,0] = math.sin(angle)
        rmat[1,1] = math.cos(angle)
        rmat[0,1] = -math.sin(angle)
    return rmat


def avg_quaternion_markley(q_array_list):
    '''
        % by Tolga Birdal
        % Q is an Mx4 matrix of quaternions. Qavg is the average quaternion
        % Based on 
        % Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. 
        % "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, 
        % no. 4 (2007): 1193-1197.
    '''
    
    A= 0
    for q_array in q_array_list:
        q_array = np.array(q_array)[:,np.newaxis]
        if q_array[-1, :] <0:
            q_array = -q_array
        q_array_t = np.transpose(q_array)
        A += np.matmul(q_array, q_array_t) 



    A /= max(1, len(q_array_list))

    w, v = np.linalg.eig(A)
    max_idx = int(np.argmax(w))
    q_array_avg = v[:,max_idx]
    if q_array_avg[-1] <0:
        q_array_avg = -q_array_avg
    return q_array_avg
    
def avg_rtmat_markley(rtmat_list):
    tvec_list = []
    q_array_list = []
    for rtmat in rtmat_list:
        rmat, tvec = decompose_rtmat(rtmat)
        q_array_list.append(rmat2quat(rmat))
        tvec_list.append(tvec)

    q_array_avg = avg_quaternion_markley(q_array_list)
    tvec_avg = np.mean(np.array(tvec_list),axis = 0)
    rmat_avg = quart2rmat(q_array_avg)
    rtmat_avg = compose_rtmat(rmat_avg, tvec_avg)
    return rtmat_avg

def avg_std_rtmat_list(rtmat_list):
    rtmat_avg = avg_rtmat_markley(rtmat_list)
    degrees =[]
    dists = []
    for rtmat in rtmat_list:
        distance, angle_degree, diff_rmat, diff_tvec = diff_rtmat(rtmat, rtmat_avg)
        dists.append(distance)
        degrees.append(angle_degree)
    degree_std = float(np.std(degrees))
    degree_mean = float(np.average(degrees))
    dist_std = float(np.std(dists))
    dist_mean = float(np.average(dists))
    return dist_mean, dist_std, degree_mean, degree_std, rtmat_avg

if __name__ == '__main__':

    theta = math.pi/4
    rmat = quart2rmat([0,0,math.sin(theta/2),math.cos(theta/2)])
    rmat_ref = [[1,0,0], [0,1,0], [0,0,1]]
    tvec = [0,0,2]
    tvec_ref = [1,0,1]




    q_array = rmat2quat(rmat_ref)
    rmat_new = quart2rmat(q_array)

    print(rmat_ref)
    print(q_array)
    print(rmat_new)

    rtmat = compose_rtmat(rmat, tvec)
    rtmat_ref = compose_rtmat(rmat_ref, tvec_ref)

    distance, angle, diff_rmat, diff_tvec = diff_rtmat(rtmat, rtmat_ref)
    print('distance, angle, diff_rmat, diff_tvec:')
    print(distance, angle, diff_rmat, diff_tvec )


    q_array = rmat2quat(rmat)
    q_array_ref = rmat2quat(rmat_ref)

    q_array_avg = avg_quaternion_markley([q_array, q_array])
    rmat_avg = quart2rmat(q_array_avg)
    print('Qavg:', q_array_avg)
    print('rmat_avg:', rmat_avg)


    rtmat_avg = avg_rtmat_markley([rtmat, rtmat_ref])
    print('rtmat_avg:', rtmat_avg)


    dist_mean, dist_std, degree_mean, degree_std, rtmat_avg = avg_std_rtmat_list([rtmat]*8 + [rtmat_ref]*7)
    print('dist_mean, dist_std, degree_mean, degree_std, rtmat_avg')
    print(dist_mean, dist_std, degree_mean, degree_std, rtmat_avg)