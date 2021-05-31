import itertools
import math
# import numpy as np
import torch


def generate_grid_priors(image_size, stride, clamp=True, image_size_train = (768,768)):
    feature_map_size = image_size[0]//stride, image_size[1]//stride
    scale = image_size[0] / stride, image_size[1]/stride
    h, w = feature_map_size    
    priors = []
    for j in range(feature_map_size[0]):
        for i in range(feature_map_size[1]):
            x_center = (i + 0.5) / scale[1]
            y_center = (j + 0.5) / scale[0]
            # x_center = ((i + 0.5) * stride - 0.5)
            # y_center = ((j + 0.5) * stride - 0.5)
            # x_center /= image_size[1]
            # y_center /= image_size[0]
            priors.append([
                x_center,
                y_center
            ])

    # priors = np.array(priors, dtype=np.float32)
    priors = torch.tensor(priors)
    if clamp:
        # np.clip(priors, 0.0, 1.0, out=priors)
        torch.clamp(priors, 0.0, 1.0, out=priors)

        
    return priors


def convert_locations_to_kpts(locations, priors, center_variance=0.05):
    return locations * center_variance + priors[..., :2]



def convert_kpts_to_locations(kpts, priors,  center_variance=0.05):
    return (kpts - priors)/center_variance

def dist_of(pts0, pts1, eps = 1e-5):
    dist = torch.sqrt((pts0[...,0]-pts1[...,0])**2 +  (pts0[...,1]-pts1[...,1])**2)
    return dist

def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  dist_threshold = 0.05):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 2): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 2): corner form priors
    Returns:
        boxes (num_priors, 2): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    dists = dist_of(gt_boxes.unsqueeze(0)*1000, corner_form_priors.unsqueeze(1)*1000)
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = dists.min(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = dists.min(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior > dist_threshold*1000] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels





