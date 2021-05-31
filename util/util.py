import cv2
import numpy as np
import random

def box_overlap(box, prev_boxes):


    for b in prev_boxes:
        x_min, y_min, x_max, y_max = max(box[0], b[0]), max(box[1], b[1]),min(box[2], b[2]),min(box[3], b[3])
        if x_max>=x_min and y_max >=y_min: 
            return True
        
    return False


def pt_in_box(pt, box):
    x_min, y_min, x_max, y_max = box
    return pt[0] >= x_min and pt[0] <= x_max and pt[1] >= y_min and  pt[1] <=y_max
        


def translate_points(pts, x, y):
    return [[pt[0] + x, pt[1] + y] for pt in pts]

def translate_points_with_vals(pts, x, y):
    return [[pt[0] + x, pt[1] + y]+list(pt)[2:] for pt in pts]


def get_degree(vec_in_local_system, vec_ref_in_cam_system, rvec):
    vec_in_local_system_np = np.reshape(np.float32(vec_in_local_system), (3,1))
    vec_ref_in_cam_system = np.reshape(np.float32(vec_ref_in_cam_system), (3,1))
    rmat, _ = cv2.Rodrigues(np.float32(rvec))
    vec_in_cam_system = np.matmul(rmat, vec_in_local_system_np)
    cos_val = dotproduct(vec_in_cam_system, vec_ref_in_cam_system)
    cos_val /= max(norm3d(vec_in_cam_system, vec_in_cam_system), 1e-20)
    cos_val /= max(norm3d(vec_ref_in_cam_system, vec_ref_in_cam_system), 1e-20)

    degree = float(np.arccos(cos_val)/np.pi * 180)
    return degree

def dotproduct(vec1, vec2):
    vec1 = np.float32(vec1)
    vec2 = np.float32(vec2)
    return np.sum(vec1*vec2)

def norm3d(vec1, vec2):
    return np.sqrt(dotproduct(vec1, vec2))


def pull_valid_start_pos(gw, gh, area_list, boxes_prev, max_try = 200):    
    for _ in range(max_try):
        for (x1,y1,x2,y2) in area_list:
            if x2-gw+1 < x1 or y2-gh+1 < y1: continue
            x = random.randint(x1, x2-gw+1)
            y = random.randint(y1, y2-gh+1)
            box = [x, y, x+gw-1, y+gh-1]
            if not box_overlap(box, boxes_prev):
                return [x, y], True

    return [0,0], False

def select_valid_vals(vals, flags = None):
    if flags is None: return vals
    vals_selected = []
    for flag, val in zip(flags, vals):
        if flag:
            vals_selected.append(val)
    return vals_selected

if __name__ == "__main__":
    import random
    nb_test = 10
    
    area_list = [[0,0,511,511]]

    for _ in range(nb_test):
        boxes_prev = []
        image = np.zeros((512,512))
        for _ in range(10):
            gw = random.randint(30, 300)
            # gh = random.randint(1, 2)* gw
            gh = gw
            start_pos, is_valid_pos  = pull_valid_start_pos(gw, gh, area_list, boxes_prev)
            if is_valid_pos:
                x, y = start_pos
                box = [x, y, x+gw-1, y+gh-1]
                boxes_prev.append(box)
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (1,1,1))

        cv2.imshow('bbox', image)
        c = cv2.waitKey(0)
        if c == 27 or c == ord('q'): break