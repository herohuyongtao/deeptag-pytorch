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






        

class DecoderNet(nn.Module):
    def __init__(self, num_channels=32, num_masks=2, num_classes = 3):
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
        self.conner_regressor = CornerRegressor(num_channels)
        self.keypoint_regressor = KeypointRegressor(num_channels=num_channels, num_classes=num_classes)


    


        
    def forward(self, x):

        model_features = self.model(x)
        backbone_features = self.cpm(model_features)
        init_masks = self.initial_stage(backbone_features)[0]
        x1 = self.align(torch.cat([init_masks, backbone_features], dim = 1))
        corners = self.conner_regressor(x1)
        confidence, location = self.keypoint_regressor(x1)
        return init_masks, corners, confidence, location


class KeypointRegressor(nn.Module):
    def __init__(self, num_channels=32, num_classes = 3):
        super().__init__()
        self.extras = nn.Sequential(
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
            conv(num_channels, num_channels),
        )
        self.regression_header = conv(num_channels, 2, bn= False, relu=False)
        self.classification_header = conv(num_channels, num_classes, bn= False, relu=False)
    def forward(self, x):
        x1 = self.extras(x)
        confidence = self.classification_header(x1)
        location =self.regression_header(x1)
        return confidence, location



class CornerRegressor(nn.Module):
    def __init__(self, num_channels=32, feature_map_shape = [32, 32]):
        super().__init__()

        nb_layers = int(math.log2(feature_map_shape[0]))-1
        self.stride_features = nn.Sequential(
            *[conv(num_channels, num_channels, stride=2) for _ in  range(nb_layers)]
        )

        self.regressor = conv(num_channels, 2, kernel_size=1, padding=0, bn= False, relu=False)   


        self.feature_map_shape = feature_map_shape


    def forward(self, x):
        feature_map_shape = self.feature_map_shape
        # linear_feature_num_mul = self.linear_feature_num_mul

        if feature_map_shape[-2] != x.shape[-2] or feature_map_shape[-1] != x.shape[-1]:
            x = F.interpolate(x, size=feature_map_shape, mode='bilinear', align_corners=False)
        

        x2 = self.stride_features(x)
        corners = self.regressor(x2)

        return corners

