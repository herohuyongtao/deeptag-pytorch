import numpy as np
from fiducial_marker.unit_chessboard_tag import UnitChessboardTag

'''
A chessboard-like topotag without actual size
'''

class UnitTopoTag(UnitChessboardTag):
    def __init__(self, grid_size, binary_ids, max_label=1, max_elem_label = 1):
        '''
            grid_size: grid of keypoints 3 or 4 or 5
            binary_ids: not include two baseline points, list of length grid_size*grid_size-2
        '''

        elem_bg_label = max(max_label, max_elem_label)
        elem_block_label = 0
        elem_labels, kpt_labels, mask_labels = gen_topotag_labels(grid_size, binary_ids, max_label, elem_bg_label = elem_bg_label, elem_block_label = elem_block_label)
        n = grid_size * 4 + 1
        self.binary_ids = binary_ids
        self.grid_size = grid_size
        super().__init__(n, elem_labels, kpt_labels, mask_labels, max_label, max_elem_label)
    def get_binary_ids(self):
        return self.binary_ids

    def get_grid_size(self):
        return self.grid_size


def gen_topotag_labels(grid_size, binary_ids, max_label = 1, elem_bg_label = 0, elem_block_label =0):
    bz = 3 # size of black block
    n = grid_size*(bz+1) + 1
    n_sq = n*n
    elem_labels = [elem_bg_label] * n_sq
    kpt_labels = [max_label+1] * n_sq
    mask_labels = [max_label+1] * n_sq
    baseline_pt_label = max_label
    # two labels of baseline points
    binary_ids_all = [baseline_pt_label]* 2 + binary_ids
    
    spos = [2, 2]
    for yy in range(grid_size):        
        for xx in range(grid_size):            
            # label of kpt            
            bid = binary_ids_all[yy * grid_size + xx]

            # position of kpt
            y = spos[1] + (bz +1) * yy 
            x = spos[0] + (bz +1) * xx             
            if yy == 0 and xx ==1:
                x -= int((bz+1)/2)
            pos = [x, y]
            
            
            idx =y * n + x
            # fill kpt label
            kpt_labels[idx] = bid
            # fill the image
            elem_labels[idx] = bid

            
            for ii in [-1, 0, 1]:
                for jj in [-1, 0, 1]:
                    idx1 = (y + ii)  * n + (x + jj)
                    # fill mask label
                    mask_labels[idx1] = bid
                    # fill the block in image, except for the center
                    if ii !=0 or jj!=0:
                        elem_labels[idx1] = elem_block_label



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

    binary_ids = gen_rand_binary_ids(grid_size, start_idx = 2)
    unit_topotag = UnitTopoTag(grid_size, binary_ids)
    n = unit_topotag.get_fine_grid_size()
    # print image
    x_y_with_labels = unit_topotag.get_elements_with_labels()    
    print('image:')
    print_labels(x_y_with_labels, n)

    # print kpts
    x_y_with_labels = unit_topotag.get_keypoints_with_labels()
    print('kpts:')
    print_x_y_with_labels(x_y_with_labels, n)


    # print masks
    x_y_with_labels = unit_topotag.get_mask_with_labels()
    print('masks:')
    print_x_y_with_labels(x_y_with_labels, n)


    # topotag with more labels    
    binary_ids = gen_rand_binary_ids(grid_size, max_label=2, start_idx = 2)
    unit_topotag = UnitTopoTag(grid_size, binary_ids, max_label = 2 , max_elem_label= 3)
    n = unit_topotag.get_fine_grid_size()
    # print image
    x_y_with_labels = unit_topotag.get_elements_with_labels()    
    print('image:')
    print_labels(x_y_with_labels, n)