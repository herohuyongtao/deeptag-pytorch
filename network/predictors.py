import torch
import numpy as np
import torch.nn.functional as F
from util.grid_prior import generate_grid_priors, convert_locations_to_kpts 
from util.box_utils import get_box_specs, generate_ssd_priors, center_form_to_corner_form, convert_locations_to_boxes
import cv2
import time

class DetectorPredictor:
    def __init__(self, model, device, stride_list = [8,16,32,64], center_variance= 0.1, size_variance = 0.2):
        model.eval()

        self.device = device
        self.model = model
        self.stride_list = stride_list
        self.priors = None
        self.center_variance = center_variance
        self.size_variance = size_variance

    def get_grid_and_ssd_priors(self, image_size):
        stride_list = self.stride_list
        device =self.device

        grid_priors_list =  [generate_grid_priors(image_size, stride, clamp = False).to(device).unsqueeze(0) for stride in stride_list]
        box_specs = get_box_specs(image_size,stride_list)
        ssd_priors = generate_ssd_priors(box_specs, image_size).to(device).unsqueeze(0)
        ssd_priors_corner_form = center_form_to_corner_form(ssd_priors)

        priors= {}
        priors['image_size']  =image_size
        priors['ssd_priors'] =ssd_priors
        priors['box_specs'] = box_specs
        priors['grid_priors_list'] =grid_priors_list
        priors['ssd_priors_corner_form'] = ssd_priors_corner_form

        return priors


    def predict(self, images, scale):
        device = self.device
        center_variance = self.center_variance
        size_variance = self.size_variance

        if type(images) is not list:
            images = [images]


        images = [crop_or_pad(image_in, scale, stride = max(self.stride_list)) for image_in in images]


        if self.priors is None or self.priors['image_size'][0]!= images[0].shape[0] or self.priors['image_size'][1] != images[0].shape[1]:
            self.priors = self.get_grid_and_ssd_priors(images[0].shape[:2])

         # stage-1 prediction
        h,w = self.priors['image_size']

        image_torch = np_to_torch(images, device, std = 255)


        t0 = time.time()
        

        grid_confidences_pred_np, grid_vals_pred_np, grid_kpts_pred_np, ssd_confidences_pred, ssd_vals_pred, ssd_boxes_pred = pred_detector_features(self.model, image_torch, self.priors, w, h, scale, center_variance, size_variance)

        self.cnn_timing = time.time()-t0

        ssd_vals_pred_np = ssd_vals_pred.cpu().data.numpy()
        ssd_confidences_pred_np = ssd_confidences_pred.cpu().data.numpy()
        ssd_boxes_pred_np = ssd_boxes_pred.cpu().data.numpy()



        features_pred = []
        for ii in range(len(images)):
            features = { 'grid_pred':[grid_confidences_pred_np[ii,...], grid_kpts_pred_np[ii,...], grid_vals_pred_np[ii,...]],
                'ssd_pred':[ssd_confidences_pred_np[ii,...], ssd_boxes_pred_np[ii,...], ssd_vals_pred_np[ii,...]]}
            features_pred.append(features)


        return features_pred    



class DecoderPredictor:
    def __init__(self, model, device, image_rect_size, center_variance = 0.05, stride = 8):
        model.eval()

        self.center_variance = center_variance
        self.grid_priors = generate_grid_priors(image_rect_size, stride).to(device).unsqueeze(0)
        self.corner_priors = generate_grid_priors(image_rect_size, image_rect_size[0]//2).to(device).unsqueeze(0)
        self.device = device
        self.model = model
        self.image_rect_size = image_rect_size
        

    def predict(self, images):
        
        center_variance = self.center_variance
        grid_priors =self.grid_priors
        device = self.device
        corner_priors = self.corner_priors

        if type(images) is not list:
            images = [images]
        

        # stage-2 prediction
        th, tw = images[0].shape[:2]
        imgs_torch_2stg = np_to_torch(images, device)
        
        t0 = time.time()
        
        confidences_pred, corners_pred, grid_pred = pred_decoder_features(self.model, imgs_torch_2stg, corner_priors, grid_priors, tw, th, center_variance)
        
        self.cnn_timing = time.time()-t0
        
        confidences_pred_np = confidences_pred.cpu().data.numpy()
        corners_pred_np = corners_pred.cpu().data.numpy()
        grid_pred_np = grid_pred.cpu().data.numpy()

        features_pred = []
        for ii in range(len(images)):
            features = { 'grid_pred':[confidences_pred_np[ii,...], grid_pred_np[ii,...], None],
            'corners_pred':[None, corners_pred_np[ii,...], None]}
            features_pred.append(features)

        return features_pred

def pred_detector_features(model, image_torch, priors, w, h, scale, center_variance, size_variance):
    stages_output = model(image_torch)
        
    init_masks_pred, grid_confidences_pred, grid_locations_pred, grid_vals_pred = stages_output[0:4]

    # corners
    grid_priors = priors['grid_priors_list'][0]
    
    grid_confidences_pred = reshape_features(grid_confidences_pred)
    grid_locations_pred = reshape_features(grid_locations_pred)
    grid_vals_pred = reshape_features(grid_vals_pred)
    grid_confidences_pred = F.softmax(grid_confidences_pred, dim=2)
    
    grid_kpts_pred= convert_locations_to_kpts(grid_locations_pred, grid_priors, center_variance=center_variance)
    grid_kpts_pred[...,0] *=w/scale
    grid_kpts_pred[...,1] *=h/scale

    
    grid_confidences_pred_np = grid_confidences_pred.cpu().data.numpy()
    grid_vals_pred_np = grid_vals_pred.cpu().data.numpy()
    grid_kpts_pred_np = grid_kpts_pred.cpu().data.numpy()



    # center and bbox
    ssd_confidences_pred, ssd_locations_pred, ssd_vals_pred = stages_output[4:7]
    

    ssd_confidences_pred =reshape_features(ssd_confidences_pred)
    ssd_locations_pred =reshape_features(ssd_locations_pred)
    ssd_vals_pred =reshape_features(ssd_vals_pred)

    ssd_confidences_pred = F.softmax(ssd_confidences_pred, dim=2)

    ssd_boxes_pred = convert_locations_to_boxes(
        ssd_locations_pred, priors['ssd_priors'], center_variance, size_variance
    )
    ssd_boxes_pred = center_form_to_corner_form(ssd_boxes_pred)
    ssd_boxes_pred[...,::2] *= w/scale
    ssd_boxes_pred[...,1::2] *= h/scale

    return grid_confidences_pred_np, grid_vals_pred_np, grid_kpts_pred_np, ssd_confidences_pred, ssd_vals_pred, ssd_boxes_pred


def pred_decoder_features(model, imgs_torch_2stg, corner_priors, grid_priors, tw, th, center_variance):

    stages_output_2stg = model(imgs_torch_2stg)    
        

    init_masks_pred, corner_locations_pred, confidences_pred, locations_pred = stages_output_2stg
    

    corner_locations_pred = reshape_features(corner_locations_pred)
    confidences_pred = reshape_features(confidences_pred)
    locations_pred = reshape_features(locations_pred)
    confidences_pred = F.softmax(confidences_pred, dim=2)

    corners_pred = convert_locations_to_kpts(corner_locations_pred, corner_priors, center_variance=center_variance)
    grid_pred= convert_locations_to_kpts(locations_pred, grid_priors, center_variance=center_variance)
    corners_pred[...,0] *=tw
    corners_pred[...,1] *=th
    grid_pred[...,0] *=tw
    grid_pred[...,1] *=th


    return confidences_pred, corners_pred, grid_pred

def np_to_torch(images, device, std = 255.0):
    images_torch = [np_to_torch_one(np_array, device, std) for np_array in images]
    if len(images_torch) ==1: return images_torch[0]
    return torch.cat(images_torch, dim=0)

    # images = [np.transpose(np.float32(np_array) / std, (2,0,1))[np.newaxis,:,:,:] for np_array in images]
    # images = np.concatenate(images, axis=0)
    # return torch.from_numpy(images).to(device) 

def np_to_torch_one(np_array, device, std = 255.0):
    return torch.from_numpy(np.transpose(np.float32(np_array) / std, (2,0,1))[np.newaxis,:,:,:]).to(device)


def features_from_torch(features):
    return np.transpose(features.squeeze().cpu().data.numpy(), (1, 2, 0))

def data_from_torch(data):
    return data.squeeze().cpu().data.numpy()


def reshape_features(features):
    if len(features.shape)==4:
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(features.size(0), -1, features.size(3))

    return features

def crop_or_pad(image_ori, scale = 1, stride = 64):
    # resize
    if scale ==1:
        image = image_ori.copy()
    else:
        image = cv2.resize(image_ori, None, fx =scale, fy =scale)

    # crop or pad
    h,w,ch = image.shape
    nh = h//stride * stride
    nw = w//stride * stride
    s = max(h,w)
    image_crop_or_pad= image[:nh,:nw,:] 

    return image_crop_or_pad