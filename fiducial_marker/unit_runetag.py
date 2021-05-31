import cv2
import numpy as np
import random
from util.homo_transform import warpPerspectivePts_with_vals, warpPerspectivePts
class UnitRunetag:
    def __init__(self, kpt_labels, radius_ratio = 17.8, gap_factor = 1.3):


        # parameters for whole
        num_layers = 3
        ellysize = 1/radius_ratio        
        alpha = ellysize * 2 * gap_factor
        num_slots_for_layer = int(np.floor(2 * np.pi/alpha))
        alpha = 2 * np.pi / num_slots_for_layer

        self.gap_factor = gap_factor 
        self.radius_ratio = radius_ratio
        self.alpha = alpha
        self.num_layers = num_layers
        self.num_slots_for_layer = num_slots_for_layer
        self.ellysize = ellysize

        # locations of marker points
        markpoints = []
        for i in range(num_slots_for_layer):
            markpoints_in_slot = []
            for j in range(num_layers):
                angle = alpha * i
                radius = (num_layers + j + 1)/(2* num_layers)
                markpoint = MarkPoint(radius * np.cos(angle), radius * np.sin(angle), angle, ellysize*radius)
                markpoints.append(markpoint)
        self.markpoints = markpoints

        # 0 or 1        
        self.kpt_labels = kpt_labels

        # range
        self.xylim = [-1.2, 1.2]


    def get_keypoints_with_labels(self, H = None):
        x_and_y_with_labels = [[markpoint.x , markpoint.y , label] for markpoint, label in zip(self.markpoints, self.kpt_labels)]
        if H is not None:
            x_and_y_with_labels = warpPerspectivePts_with_vals(H, x_and_y_with_labels)
        return x_and_y_with_labels

    def get_ordered_corners_with_labels(self, main_idx = 0, H = None):
        corners = [[-1.0,-1.0,0], [1.0, -1.0, 0],  [1.0, 1.0, 1],  [-1.0, 1.0, 0]]
        x_and_y_with_labels = corners[main_idx:] + corners[:main_idx]
        if H is not None:
            x_and_y_with_labels = warpPerspectivePts_with_vals(H, x_and_y_with_labels)
        return x_and_y_with_labels

    def get_circle_radius(self):
        radius_list = [markpoint.ellyradius for markpoint in self.markpoints]
        return radius_list

    def get_center_pos(self, H = None):
        center_pos = [0.0, 0.0]
        if H is not None:
            center_pos = warpPerspectivePts(H, [center_pos])[0]
        return center_pos
    
    def get_xylim(self):
        return self.xylim

class MarkPoint:
    def __init__(self, x, y, angle, ellyradius):
        self.x = x
        self.y = y
        self.angle = angle
        self.ellyradius = ellyradius


def slot_codes_to_binary_ids(codes, num_layers = 3, is_outer_first = False):

    sub_codes = {}
    for ii in range(2**num_layers-1):
        sub_code = []
        num = ii +1
        for _ in range(num_layers):
            sub_code.append(num % 2)
            num //=2
        if is_outer_first:
            # from outer to inner 
            sub_code = sub_code[::-1]
        sub_codes[ii] = sub_code

    # from inside to outside
    binary_ids = []
    for code in codes:
        for ii in range(num_layers):
            binary_ids.append(sub_codes[code][ii])

    return binary_ids

def generate_random_binary_ids(num_slots = 43, num_layers = 3):
    codes =  [random.randint(0, 2**num_layers -2 +2**(num_layers-1) )%(2**num_layers-1) for _ in range(num_slots)] # fewer dots
    # codes =  [random.randint(0, 2**num_layers -2) for _ in range(num_slots)]
    binary_ids = slot_codes_to_binary_ids(codes)
    return binary_ids

if __name__ == '__main__':

    import matplotlib.pyplot as plt 
    

    kpt_labels = generate_random_binary_ids()
    unit_runetag = UnitRunetag(kpt_labels)
 
    
    # print('codes:', codes)

    figure, axes = plt.subplots() 
    # cc = plt.Circle(( 0.5 , 0.5 ), 0.4 , alpha=0.1) 

    for label, markpoint in zip(unit_runetag.kpt_labels, unit_runetag.markpoints):
        if  label == 0: continue
        cc = plt.Circle(( markpoint.x , -markpoint.y ), markpoint.ellyradius, alpha=0.5) 
        axes.add_artist( cc ) 


    axes.set_aspect( 1 ) 
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.title( 'RenuTag with Colored Circles' ) 
    plt.show()