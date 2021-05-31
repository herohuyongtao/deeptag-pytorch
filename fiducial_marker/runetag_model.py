from fiducial_marker.unit_runetag import * 


def write_runetag_model(filename, unit_runetag, marker_size, tag_id, markerName='R129'):
    # originally, the dots are arranged counter-clockwise
    # in our setting, the dots are arranged clockwise
    # both are from inner circle to outer circle

    num_layers = unit_runetag.num_layers
    num_slots_for_layer = unit_runetag.num_slots_for_layer
    binary_ids = unit_runetag.kpt_labels
    x_and_y_with_labels = unit_runetag.get_keypoints_with_labels()
    radius_big = marker_size/2.0
    with open(filename, 'w') as fp:
        # basic info
        fp.write('RUNE_direct')
        fp.writelines('\n')
        fp.writelines(markerName)
        fp.writelines('\n')
        fp.writelines(str(radius_big))
        fp.writelines('\n')
        fp.writelines('mm')
        fp.writelines('\n')
        fp.writelines(str(num_layers * num_slots_for_layer))
        fp.writelines('\n')
        fp.writelines(str(unit_runetag.radius_ratio))
        fp.writelines('\n')
        fp.writelines(str(num_layers))
        fp.writelines('\n')
        fp.writelines(str(unit_runetag.gap_factor))
        fp.writelines('\n')
        fp.writelines('-1')
        fp.writelines('\n')
        fp.writelines(str(tag_id))
        # fp.writelines('\n')


        # counter-clockwise, from outer circle to inner circle
        idx_list = list(range(num_slots_for_layer))
        idx_list =idx_list[:1] + idx_list[1:][::-1]
        for ii in idx_list:
            for jj in range(num_layers):
                fp.writelines('\n')
                idx = ii* num_layers + jj
                bid = binary_ids[idx]
                fp.writelines(str(bid))

                if bid ==1:
                    cx = x_and_y_with_labels[idx][0] * radius_big
                    cy = x_and_y_with_labels[idx][1] * radius_big
                    radius = unit_runetag.markpoints[idx].ellyradius * radius_big
                    k = cx*cx+cy*cy-radius*radius

   
                    #
                    #  Circle quadratic form
                    #   1   0   -cx
                    #   0   1   -cy
                    # -cx  -cy  cx^2 + cy^2 - r^2
                    #
                    circle_data = []
                    circle_data += [1.0, 0.0, -cx]
                    circle_data += [0.0, 1.0, -cy]
                    circle_data += [-cx, -cy, k]

                    for val in circle_data:
                        fp.writelines(' '+ str(round(val,6)))

        




def read_runetag_codebook(filename, num_layers = 3, is_counter_clockwise = True, is_outer_first = True):
    # originally, the dots are arranged counter-clockwise, from outer circle to inner circle
    # in our setting, the dots are arranged clockwise, from inner circle to outer circle
    codebook = {}
    with open(filename, 'r') as fp:
        line = fp.readline()
        nb_codes = int(line.split()[0])
        for _ in range(nb_codes):
            line = fp.readline()
            if line is None: 
                break
            nums = [int(s) for s in line.split()]   
            tag_id = nums[0]
            num_slots_for_layer = nums[1]
            if is_counter_clockwise:
                codes_reordered = nums[3:] + [nums[2]]
                codes_reordered = codes_reordered[::-1]
            else:
                codes_reordered = nums[2:]
            binary_ids = slot_codes_to_binary_ids(codes_reordered, is_outer_first= is_outer_first)
            codebook[tag_id] = {}
            codebook[tag_id]['num_layers'] = num_layers
            codebook[tag_id]['num_slots_for_layer'] = num_slots_for_layer
            codebook[tag_id]['binary_ids'] = binary_ids

    return codebook

if __name__=='__main__':
    from fiducial_marker.unit_runetag import UnitRunetag
    codebook_filename = 'codebook/runetag_codebook.txt'
    marker_size = 45 # mm
    tag_code_txt = 'runetag_r129_107.txt'
     
    codebook = read_runetag_codebook(codebook_filename)
    tag_id = 107
    print(codebook[tag_id])
    binary_ids = codebook[tag_id]['binary_ids']


    unit_runetag = UnitRunetag(binary_ids)
    write_runetag_model(tag_code_txt, unit_runetag, marker_size, tag_id)