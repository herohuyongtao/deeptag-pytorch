import torch
from torch import nn
import torch.nn.functional as F
from network.conv import conv, conv_dw, conv_dw_no_bn
import math

INTER_CHANNEL_NUM = 512
INTER_CHANNEL_NUM_BLOCK = 512
INTER_CHANNEL_NUM_REFINE =128

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x




class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features





class InitialStage(nn.Module):
    def __init__(self, num_channels, num_masks):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.masks = nn.Sequential(
            conv(num_channels, INTER_CHANNEL_NUM_BLOCK, kernel_size=1, padding=0, bn=False),
            conv(INTER_CHANNEL_NUM_BLOCK, num_masks, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        masks = self.masks(trunk_features)
        return [masks]






        

class DetectorNet(nn.Module):
    def __init__(self, num_channels=64, num_masks=2, num_classes = 2, num_scales = 4, num_box_versions = 6, num_vals = 10, num_classes_keypoints = 2):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  INTER_CHANNEL_NUM//16, stride=2, bias=False),
            conv_dw( INTER_CHANNEL_NUM//16,  INTER_CHANNEL_NUM//8),
            conv_dw( INTER_CHANNEL_NUM//8, INTER_CHANNEL_NUM//4, stride=2),
            conv_dw(INTER_CHANNEL_NUM//4, INTER_CHANNEL_NUM//4),
            conv_dw(INTER_CHANNEL_NUM//4, INTER_CHANNEL_NUM//2, stride=2),
            conv_dw(INTER_CHANNEL_NUM//2, INTER_CHANNEL_NUM//2),
            conv_dw(INTER_CHANNEL_NUM//2, INTER_CHANNEL_NUM),  # conv4_2
            conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM, dilation=2, padding=2),
            conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM),
            conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM),
            conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM),
            conv_dw(INTER_CHANNEL_NUM, INTER_CHANNEL_NUM)   # conv5_5
        )
        self.cpm = Cpm(INTER_CHANNEL_NUM, num_channels)  
        self.initial_stage = InitialStage(num_channels, num_masks)
        self.align = conv(num_channels + num_masks, num_channels, kernel_size=1, padding=0, bn=False)   
        self.keypoint_regressor = KeypointRegressor(num_channels=num_channels, num_classes=num_classes_keypoints)    
        self.ssd_regressor = SSDRegressor(num_channels = num_channels, num_classes=num_classes, num_box_versions=num_box_versions, num_vals = num_vals)


        
    def forward(self, x):

        model_features = self.model(x)
        backbone_features = self.cpm(model_features)
        init_masks = self.initial_stage(backbone_features)[0]
        x1 = self.align(torch.cat([init_masks, backbone_features], dim = 1))
        confidence, location, orientation = self.keypoint_regressor(x1)
        ssd_confidences, ssd_locations, ssd_vals = self.ssd_regressor(x1)
        return init_masks, confidence, location, orientation, ssd_confidences, ssd_locations, ssd_vals


class KeypointRegressor(nn.Module):
    def __init__(self, num_channels=64, num_classes = 3):
        super().__init__()
        self.extras = nn.Sequential(
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
        )
        self.regression_header = conv(num_channels, 2, bn= False, relu=False)
        self.classification_header = conv(num_channels, num_classes, bn= False, relu=False)
        self.orientation_header = conv(num_channels, 4, bn= False, relu=False)
    def forward(self, x):
        x1 = self.extras(x)
        confidence = self.classification_header(x1)
        location =self.regression_header(x1)
        orientation = self.orientation_header(x1)
        return confidence, location, orientation


class SSDRegressor(nn.Module):
    def __init__(self, num_channels=64, num_classes = 2, num_scales = 4, num_box_versions = 6, num_vals = 10):
        super().__init__()
        self.extras = nn.Sequential(
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
        )
        self.stride_features = nn.ModuleList()
        for _ in range(num_scales -1):
            stride_layers = nn.Sequential(
                        nn.Conv2d(in_channels=num_channels, out_channels=num_channels//2, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=num_channels//2, out_channels=num_channels, kernel_size=3, stride=2, padding=1),
                        nn.ReLU()
                    )
            self.stride_features.append(stride_layers)
    

        self.regression_headers = nn.ModuleList()
        for _ in range(num_scales):
            self.regression_headers.append(SSDRegressionHeaders(num_channels = num_channels, num_classes=num_classes, num_box_versions=num_box_versions, num_vals = num_vals))

    def forward(self, x):
        num_scales = len(self.regression_headers)


        confidences, locations, vals = [],  [], []
        x1 = self.extras(x)
        for ii in range(num_scales):
            if ii>0:
                x1 = self.stride_features[ii-1](x1)

            c, l, v = self.regression_headers[ii](x1)
            confidences.append(c)
            locations.append(l)
            vals.append(v)
        
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        vals = torch.cat(vals, 1)
        return confidences, locations, vals

class SSDRegressionHeaders(nn.Module):
    def __init__(self, num_channels=64, num_classes = 2, num_box_versions = 6, num_vals = 10):
        super().__init__()
        self.regression_header = nn.Conv2d(in_channels=num_channels, out_channels=num_box_versions * 4, kernel_size=3, padding=1)
        self.classification_headers = nn.Conv2d(in_channels=num_channels, out_channels=num_box_versions * num_classes, kernel_size=3, padding=1)
        self.val_headers = nn.Conv2d(in_channels=num_channels, out_channels=num_box_versions * num_vals, kernel_size=3, padding=1)
        self.num_box_versions = num_box_versions

    def forward(self, x):
        num_box_versions = self.num_box_versions
        locations = self.regression_header(x)
        confidences = self.classification_headers(x)
        vals = self.val_headers(x)

        locations = reshape_features(locations, num_box_versions)
        confidences = reshape_features(confidences, num_box_versions)
        vals = reshape_features(vals, num_box_versions)


        return confidences, locations, vals



def reshape_features(features, num_box_versions = 1):
    if len(features.shape)==4:
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(features.size(0), -1, features.size(3)//num_box_versions)

    return features