def load_marker_codebook(filename, tag_type):
    
    if tag_type == 'topotag': 
        return {}
    elif tag_type == 'runetag': 
        from fiducial_marker.runetag_model import read_runetag_codebook 
        codebook = read_runetag_codebook(filename)
        rotate_idx_dict = {tuple(codebook[k]['binary_ids']): k for k in codebook}
    else:
        with open(filename, 'r') as f:
            rotate_idx_dict = {}
            lines = f.readlines()
            for line in lines:
                nums = [int(val.strip()) for val in line.split(',')]
                rotate_idx_dict[tuple(nums[1:])] = nums[0]

    return rotate_idx_dict