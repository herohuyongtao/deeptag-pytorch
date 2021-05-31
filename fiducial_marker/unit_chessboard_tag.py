import enum
from util.homo_transform import warpPerspectivePts_with_vals, warpPerspectivePts

'''
A general chessboard-like tag without actual size
'''

class UnitChessboardTag:
    def __init__(self, n, elem_labels, kpt_labels, mask_labels, max_label = 1, max_elem_label = 1):
        '''
            n: the chessboard size is n * n
            elem_labels: size of n*n, present the color of each cell
            kpt_labels: size of n*n, present the kpt id of each cell; if kpt id > max_label, it represent background.
            mask_labels: size of n*n, present the mask label of each cell; if label > max_label, it represent background.
        '''

        self.fine_grid_size = n
        self.elem_labels = elem_labels
        self.max_elem_label = max_elem_label
        self.kpt_labels = kpt_labels
        self.mask_labels = mask_labels
        self.max_label = max_label

    def get_fine_grid_size(self, step_elem_num = 1):
        if step_elem_num ==1: return self.fine_grid_size
        return len(list(range(0, self.fine_grid_size, step_elem_num)))
        

    def get_elements_with_labels(self, main_idx = 0, is_center = False, H = None):
        labels = self.elem_labels
        max_label = self.max_elem_label
        fine_grid_size = self.fine_grid_size
        x_and_y_with_labels = get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = labels, max_label = max_label, H = H)
        return x_and_y_with_labels

    def get_keypoints_with_labels(self, main_idx = 0, is_center = False, step_elem_num = 1, H = None):
        labels = self.kpt_labels
        max_label = self.max_label
        fine_grid_size = self.fine_grid_size
        x_and_y_with_labels = get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = labels, max_label = max_label, step_elem_num = step_elem_num, H = H)
        return x_and_y_with_labels


    def get_mask_with_labels(self, main_idx = 0, is_center = False, H = None):
        labels = self.mask_labels
        max_label = self.max_label
        fine_grid_size = self.fine_grid_size
        x_and_y_with_labels = get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = labels, max_label = max_label, H = H)
        return x_and_y_with_labels

    def get_corner_fine_grid_points(self, main_idx = 0, step_elem_num = 1, H = None):
        labels = None
        is_center = False
        max_label = -1
        fine_grid_size = self.fine_grid_size + 1
        x_and_y_with_labels = get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = labels, max_label = max_label, step_elem_num = step_elem_num, H = H)
        return x_and_y_with_labels   
            
    def get_fine_grid_points(self, main_idx = 0, is_center = False, step_elem_num = 1, H = None):
        labels = None
        max_label = -1
        fine_grid_size = self.fine_grid_size
        x_and_y_with_labels = get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = labels, max_label = max_label, step_elem_num = step_elem_num, H = H)
        return x_and_y_with_labels



    def get_fine_grid_points_with_labels(self, main_idx = 0, is_center = False, step_elem_num = 1, H = None):
        # labels = None
        # max_label = -1
        labels = self.elem_labels
        max_label = self.max_elem_label
        fine_grid_size = self.fine_grid_size
        x_and_y_with_labels = get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = labels, max_label = max_label, step_elem_num = step_elem_num, H = H)
        return x_and_y_with_labels

    def get_ordered_corners(self, main_idx = 0, H = None):
        n = self.fine_grid_size
        corners = [[0,0], [n, 0],  [n, n],  [0, n]]
        ordered_corners = corners[main_idx:] + corners[:main_idx]
        if H is not None:
            ordered_corners = warpPerspectivePts(H, ordered_corners)
        return ordered_corners
    def get_center_pos(self, H = None):
        n = self.fine_grid_size
        center_pos = [n/2, n/2]
        if H is not None:
            center_pos = warpPerspectivePts(H, [center_pos])[0]
        return center_pos

    def get_max_label(self):
        return self.max_label

    def get_max_elem_label(self):
        return self.max_elem_label

def get_x_and_y_with_labels(fine_grid_size, is_center, main_idx, labels = None, max_label = -1, step_elem_num = 1, H = None):
    '''
        is_center: is cell center or not
    '''

    if is_center:
        offset = 0.5
    else:
        offset = 0
    # get x_y list with main_idx =0
    default_x_y_list = []
    for y in range(fine_grid_size):
        for x in range(fine_grid_size):
            default_x_y_list.append([x + offset, y + offset])


    # re-ordered x_y list and filtered with labels
    ordered_idx_list = get_ordered_idx_list(fine_grid_size, main_idx = 0, step_elem_num = step_elem_num)
    ordered_idx_list_rotated = get_ordered_idx_list(fine_grid_size, main_idx = main_idx, step_elem_num = step_elem_num)
    x_and_y_with_labels = []
    for idx, idx_rotated in zip(ordered_idx_list, ordered_idx_list_rotated):
        x_y = default_x_y_list[idx_rotated]
        if labels is not None and len(labels) > idx and max_label>=0:
            # filter the points with lobels
            if labels[idx] > max_label:
                continue
            else:
                x_y += [labels[idx]]

        x_and_y_with_labels.append(x_y)

    if H is not None:        
        x_and_y_with_labels =  warpPerspectivePts_with_vals(H, x_and_y_with_labels)
    
    return x_and_y_with_labels
    
    

                


def get_ordered_idx_list(grid_size, main_idx = 0, step_elem_num = 1):
    # get ordered template
    idx_list = []
    for ii in range(0, grid_size, step_elem_num):
        for jj in range(0, grid_size, step_elem_num):           
            if main_idx ==0:
                idx = ii * grid_size + jj
            elif main_idx ==1:
                idx = jj * grid_size + (grid_size - ii -1)
            elif main_idx ==2:
                idx = (grid_size - ii - 1)*grid_size + (grid_size - jj -1)
            elif main_idx ==3:
                idx = (grid_size - jj -1)*grid_size + ii
                
            idx_list.append(idx)
    return idx_list

def rotate_binary_ids(binary_ids, grid_size, main_idx):
    idx_list = get_ordered_idx_list(grid_size, main_idx)
    binary_ids_rotated = [binary_ids[idx] for idx in idx_list]
    return binary_ids_rotated

def get_tag_id_decimal(binary_ids, start_idx = 0, base_num = 2):
    num = 0
    for bid in binary_ids[start_idx:]:
        num *=base_num
        num +=bid
    return num

def decimal_to_binary_ids(tag_id_decimal, grid_size, start_idx = 0, base_num = 2):
    binary_ids_len = grid_size * grid_size -2
    binary_ids_rev = []
    while tag_id_decimal > 0:
        binary_ids_rev.append(tag_id_decimal%2)
        tag_id_decimal //=base_num


    binary_ids_rev += [0]* (binary_ids_len-len(binary_ids_rev))
    binary_ids = binary_ids_rev[::-1]
    return binary_ids

def gen_rand_binary_ids(grid_size, max_label=1, start_idx = 0):
    import random
    block_num = grid_size*grid_size-start_idx
    binary_ids = [random.randint(0,max_label) for _ in range(block_num)]
    return binary_ids


def check_hamming_dist(binary_ids, binary_ids_ref, start_idx = 0,start_idx_ref = 0):
    count = 0
    for bid, bid_ref in zip(binary_ids[start_idx:], binary_ids_ref[start_idx_ref:]):
        if bid != bid_ref: count +=1

    ham_dist = count + abs((len(binary_ids)- start_idx)-(len(binary_ids_ref)- start_idx_ref))
    return ham_dist


if __name__ == '__main__':
    grid_size = 13
    for main_idx in range(4):
        idx_list = get_ordered_idx_list(grid_size, main_idx, step_elem_num = 2)
        print(idx_list)

    import random
    binary_ids = [random.randint(0,1) for _ in range(9)]
    binary_ids_ref = [random.randint(0,1) for _ in range(7)]
    print(binary_ids)
    print(binary_ids_ref)
    ham_dist = check_hamming_dist(binary_ids, binary_ids_ref, start_idx=2, start_idx_ref=0)
    print('ham_dist', ham_dist)