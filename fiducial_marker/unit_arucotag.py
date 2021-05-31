import numpy as np
from fiducial_marker.unit_chessboard_tag import UnitChessboardTag
import random



'''
A chessboard-like arucotag without actual size
'''

class UnitArucoTag(UnitChessboardTag):
    def __init__(self, grid_size, binary_ids, max_label=1, max_elem_label = 1):
        '''
            grid_size: grid of keypoints 3 or 4 or 5
            binary_ids: not include two baseline points, list of length grid_size*grid_size-2
        '''

        elem_border_label = 0
        elem_corner_label = 0
        if max_elem_label - max_label> 0:
            elem_corner_label = max_label + 1
        if max_elem_label - max_label> 1:
            elem_border_label = max_label + 2

        elem_labels, kpt_labels, mask_labels = gen_aruco_labels(grid_size, binary_ids, elem_border_label = elem_border_label, elem_corner_label = elem_corner_label)
        n = grid_size +2
        self.binary_ids = binary_ids
        self.grid_size = grid_size
        super().__init__(n, elem_labels, kpt_labels, mask_labels, max_label, max_elem_label)
    def get_binary_ids(self):
        return self.binary_ids

    def get_grid_size(self):
        return self.grid_size


def gen_aruco_labels(grid_size, binary_ids, max_label = 1, elem_border_label = 0, elem_corner_label = 0):
    n = grid_size + 2
    n_sq = n*n
    elem_labels = [elem_border_label] * n_sq
    kpt_labels = [max_label+1] * n_sq
    mask_labels = [max_label+1] * n_sq

    elem_labels[0] = elem_corner_label
    elem_labels[n-1]= elem_corner_label
    elem_labels[-n] = elem_corner_label
    elem_labels[-1]= elem_corner_label

    for yy in range(grid_size):
        for xx in range(grid_size):
            bid = binary_ids[yy * grid_size + xx]
            idx =(yy+1) * n + xx + 1
            # fill kpt label
            kpt_labels[idx] = bid
            # fill the image
            elem_labels[idx] = bid
            # fill mask label
            mask_labels[idx] = bid

    return elem_labels, kpt_labels, mask_labels




if __name__ == '__main__':
    

    def print_x_y_with_labels(x_y_with_labels, n, what_to_print = 'all'):
        idx =0
        for ii in range(n):            
            pts = []
            while idx < len(x_y_with_labels):
                pt = x_y_with_labels[idx]
                if pt[1] == ii:
                    pts.append(pt)
                    idx+=1
                else:
                    break

            if what_to_print == 'all':
                bws = pts
            elif what_to_print == 'label':
                bws = [pt[2] for pt in pts]
            elif what_to_print == 'x_y':
                bw = [pt[:2] for pt in pts]
            print(bws)

    def print_labels(x_y_with_labels, n):
        print_x_y_with_labels(x_y_with_labels, n, what_to_print = 'label')

    def print_x_y(x_y_with_labels, n):
        print_x_y_with_labels(x_y_with_labels, n, what_to_print = 'x_y')

    from fiducial_marker.unit_chessboard_tag import gen_rand_binary_ids
    grid_size = 4

    # normal topotag

    binary_ids = gen_rand_binary_ids(grid_size)
    unit_arucotag = UnitArucoTag(grid_size, binary_ids)
    n = unit_arucotag.get_fine_grid_size()
    # print image
    x_y_with_labels = unit_arucotag.get_elements_with_labels()    
    print('image:')
    print_labels(x_y_with_labels, n)

    # print kpts
    x_y_with_labels = unit_arucotag.get_keypoints_with_labels()
    print('kpts:')
    print_x_y_with_labels(x_y_with_labels, n)


    # print masks
    x_y_with_labels = unit_arucotag.get_mask_with_labels()
    print('masks:')
    print_x_y_with_labels(x_y_with_labels, n)


    # topotag with more labels    
    binary_ids = gen_rand_binary_ids(grid_size, max_label=2)
    unit_arucotag = UnitArucoTag(grid_size, binary_ids, max_label = 2 , max_elem_label= 4)
    n = unit_arucotag.get_fine_grid_size()
    # print image
    x_y_with_labels = unit_arucotag.get_elements_with_labels()    
    print('image:')
    print_labels(x_y_with_labels, n)