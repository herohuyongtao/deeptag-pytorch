from network.decoder_net import DecoderNet as decoder
from network.detector_net import DetectorNet as detector
import torch
import os
# import time
def load_deeptag_models(tag_family, device = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    TAG_TYPE = 1
    if tag_family == 'aruco' or tag_family == 'apriltag' or tag_family == 'artookkitplus':
        TAG_TYPE = 1
    elif tag_family == 'topotag':
        TAG_TYPE = 0
    elif tag_family == 'apriltagxo' or tag_family == 'apriltagxa' :
        TAG_TYPE = 3
    elif tag_family == 'runetag':
        TAG_TYPE = 4

    checkpoint_dir = 'models'
    checkpoint_dir_2stg = 'models'

    
    num_classes_keypoints =2
    if TAG_TYPE ==0:
        # model param
        model_filename = 'topotag_roi_detector.pth'
        model_filename_2stg = 'topotag_decoder.pth'

        tag_type='topotag'
        # grid_size_cand_list= [3,4,5]
        grid_size_cand_list= [4]
        num_classes = 4

    
    elif TAG_TYPE ==1:
        # model param
        model_filename = 'arucotag_roi_detector.pth'
        model_filename_2stg = 'arucotag_decoder.pth'

        tag_type='arucotag'
        grid_size_cand_list= [4,5,6,7]
        # grid_size_cand_list= [6]
        num_classes = 3


    elif TAG_TYPE ==3:
        # model param
        model_filename = 'arucotag_xab_roi_detector.pth'
        model_filename_2stg = 'arucotag_xab_decoder.pth'

        tag_type='arucotag'
        # grid_size_cand_list= [4,5,6,7]
        grid_size_cand_list= [6]
        num_classes = 3

    elif TAG_TYPE ==4:
        # model param
        model_filename = 'runetag_roi_detector.pth'
        model_filename_2stg = 'runetag_decoder_512x.pth'

        tag_type='runetag'
        grid_size_cand_list= []
        num_classes = 3
        num_classes_keypoints = 3



    

    print('===========> loading model <===========')
    num_masks = 2
    num_channels = 128
    num_channels_refiner = 32
    

    model_detector = detector(num_channels=num_channels, num_masks=num_masks, num_classes = 2, num_classes_keypoints = num_classes_keypoints)
    state_dict = torch.load(os.path.join(checkpoint_dir, model_filename))
    model_detector.load_state_dict(state_dict)
    model_detector.to(device)
    #print(model_detector)

    model_decoder = decoder(num_channels=num_channels_refiner, num_masks=num_masks, num_classes = num_classes)
    state_dict_2stg = torch.load(os.path.join(checkpoint_dir_2stg, model_filename_2stg))
    model_decoder.load_state_dict(state_dict_2stg)
    model_decoder.to(device)
    #print(model_decoder)  


    return model_detector, model_decoder, device, tag_type, grid_size_cand_list