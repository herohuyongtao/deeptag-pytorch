import numpy as np

NUM_QUAN_ORIENTATION =4
def get_quantized_orientations_to_center(keypoints, center_pos):
    '''
        get vecs from keypoints to center, and encode vecs as soft_bins
    '''

    ch = NUM_QUAN_ORIENTATION
    vec_ref_x = np.float32([1.0, 0])
    vec_ref_y = np.float32([0, 1.0])


    # find center
    center_pos_np = np.float32(center_pos[:2])

    soft_bins_list = []
    for ii, kpt in enumerate(keypoints): 
        vec = np.float32(kpt[:2]) - center_pos_np
        soft_bins = vec_to_soft_bins(vec)

        soft_bins_list.append(soft_bins)
    return soft_bins_list

def decode_orientation_vecs(soft_bins_set):
    '''
        convert the soft bins back to vecs
    '''
    corner_directions = []
    for soft_bins in soft_bins_set:
        vec = soft_bins_to_vec(soft_bins)
        corner_directions.append(vec)
    return corner_directions

def vec_to_soft_bins(vec, soft_bins_sum = 1e-20):
    '''
        encode a vec as 4 values
    '''

    vec = np.float32(vec)
    vec_norm = norm3d(vec)
    if vec_norm < 1e-20:            
        soft_bins = [0.5,0.5,0.5,0.5]  
    else:
        vec = vec/vec_norm
        xyweight = (vec + 1) * 0.5      

        soft_bins = [xyweight[0], xyweight[1], 1-xyweight[0], 1-xyweight[1]]
    return soft_bins



def soft_bins_to_vec(soft_bins, soft_bins_sum = 1e-20):
    '''
        convert 4 values to a normalized vector
    '''
    # soft_bins = np.float32(soft_bins)
    # soft_bins = soft_bins/max(1e-20, np.sum(soft_bins))
    soft_bins = [min(max(0, b),1) for b in soft_bins]
    soft_bins_sum  = min(max(0, soft_bins_sum),1)
    xx = soft_bins[0] / max(soft_bins[0] +  soft_bins[2], soft_bins_sum)
    yy = soft_bins[1] / max(soft_bins[1] +  soft_bins[3], soft_bins_sum)
    xx = (xx-0.5)*2
    yy = (yy-0.5)*2

    vec_norm = max(norm3d([xx,yy]),1e-20)

    vec = [xx/vec_norm, yy/vec_norm]


    return vec

def norm3d(xyz):
    return np.sqrt(np.sum(np.float32(xyz)**2))