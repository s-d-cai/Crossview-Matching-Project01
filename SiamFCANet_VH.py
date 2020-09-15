# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:59:49 2020

@author: dage
"""

import math

import torch
import torch.nn as nn
from torchvision.models import ResNet
from torch.autograd import Variable 
import torch.nn.functional as F
#from graphviz import Digraph


################### local context awaring module (a kind of attention)
class ContextAware(nn.Module):
    ### 512 for basic and 2048 for battleneck
    def __init__(self, channel=512):
        super(ContextAware, self).__init__()
        ### provides 3 different size kernel for processing local features
        #self.pool_down = nn.AvgPool2d(kernel_size=(3,3), stride=(2,2), padding=0)
        self.conv_local1 = nn.Conv2d(in_channels=channel, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_local2 = nn.Conv2d(in_channels=channel, out_channels=128, kernel_size=5, stride=1, padding=2)
        ### if feat-map's W*H > 7*7
        #self.conv_local3 = nn.Conv2d(in_channels=channel, out_channels=96, kernel_size=7, stride=1, padding=3)
        #self.conv_rw = nn.Conv2d(in_channels=352, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        ### if feat-map's W*H = 7*7
        self.conv_rw = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.w_prob = nn.Sigmoid()
        #self.w_nonlinear = nn.ReLU(inplace=True) ### apply relu to substitute sigmoid
        
        self.repeatSize = channel

    def forward(self, x):
        ### local processing
        x_b1 = self.conv_local1(x)
        x_b1 = self.relu(x_b1)
        x_b2 = self.conv_local2(x)
        x_b2 = self.relu(x_b2)
        ## if feat-map's W*H > 7*7
        #x_b3 = self.conv_local3(x)
        #x_b3 = self.relu(x_b3)
        ### concat
        ## if feat-map's W*H > 7*7
        #y = [x_b1, x_b2, x_b3]
        ## if feat-map's W*H = 7*7
        y = [x_b1, x_b2]
        
        y = torch.cat(y, 1)
        ### reweighting
        y = self.conv_rw(y)
        y = self.w_prob(y)
        #y = self.w_nonlinear(y)
        ### repeat the channel
        y = y.repeat(1,self.repeatSize,1,1)
        
        return x * y


###################
        
###****** FCAM: Feature Context-Based Attention Module ******###
# using multi-scale receptive fields for analyzing contextual info, inspired by CRN and CBAM #
class MulFCAM(nn.Module):
    def __init__(self, channel, rate_reduct=16):
        super(MulFCAM, self).__init__()
        ### for channel attention ###
        ###
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        ###
        self.conv_down = nn.Conv2d(channel, channel // rate_reduct, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_up = nn.Conv2d(channel // rate_reduct, channel, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
               
       ### for spatial attention ###
        self.conv_local1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_local2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv_local3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.conv_rw = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.repeatSize = channel
        
    def forward(self, x):
        y_a = self.avg_pool(x)
        y_m = self.max_pool(x)
        
        y_a = self.conv_down(y_a)
        y_m = self.conv_down(y_m)
        
        y_a = self.relu(y_a)
        y_m = self.relu(y_m)
        
        y_a = self.conv_up(y_a)
        y_m = self.conv_up(y_m)
        
        y = y_a + y_m
        
        y = self.sigmoid(y)
        
        x = x*y
        
        ###
        ### local processing
        y_a = x.mean(1,keepdim=True) # avgpool along channels
        y_m, _ = x.max(1,keepdim=True) # maxpool along channels
        ### concat
        y = [y_a, y_m]
        y = torch.cat(y, 1)
        
        y_b1 = self.conv_local1(y)
        y_b1 = self.relu(y_b1)
        y_b2 = self.conv_local2(y)
        y_b2 = self.relu(y_b2)
        y_b3 = self.conv_local3(y)
        y_b3 = self.relu(y_b3)
        ### concat
        y = [y_b1, y_b2, y_b3]
        y = torch.cat(y, 1)
        y = self.conv_rw(y)
        
        y = self.sigmoid(y)
        
        ### repeat the channel
        y = y.repeat(1,self.repeatSize,1,1)
        
        return x*y

############################# Basic module ############################### 
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


### add CBAM ###
class FCAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(FCAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.FCAM = MulFCAM(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.FCAM(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#####################################################

########################
########################## mul CBAM model ###########################
class SiamFCANet_Base(nn.Module):

    def __init__(self, block, layers, num_classes=365):
        self.inplanes = 64
        self.inplanes_p = 64
        super(SiamFCANet_Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_p = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_p = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.relu_p = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_p = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer1_p = self._make_layer_p(block, 64, layers[0])
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2_p = self._make_layer_p(block, 128, layers[1], stride=2)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3_p = self._make_layer_p(block, 256, layers[2], stride=2)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer4_p = self._make_layer_p(block, 512, layers[3], stride=2)
        
        ### context awaring module
        self.rw_emb = ContextAware(512)
        self.rw_emb_p = ContextAware(512)
        
        self.featVec = nn.Linear(512*7*7, 1024)
        self.featVec_p = nn.Linear(512*7*7, 1024)      
        
        ### initialize 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #n = m.weight.shape[0] * m.weight.shape[1]
                nn.init.xavier_normal_(m.weight, gain=0.1)
                m.bias.data.zero_()
        
        ### layers for orientation regression
        #self.angle_fc8 = nn.Linear(2*4096, 100)
        self.angle_fc8 = nn.Linear(2*1024, 100)
        nn.init.xavier_normal_(self.angle_fc8.weight,  gain=2.0)
        #nn.init.constant_(self.angle_fc1.weight, 0.1)
        self.angle_fc9 = nn.Linear(100, 2)
        nn.init.xavier_normal_(self.angle_fc9.weight, gain=2.0)
        #nn.init.constant_(self.angle_fc2.weight, 0.1)
        
        self.angle_relu = nn.ReLU(inplace=True)
        #self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_layer_p(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_p != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_p, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers_p = []
        layers_p.append(block(self.inplanes_p, planes, stride, downsample))
        self.inplanes_p = planes * block.expansion
        for i in range(1, blocks):
            layers_p.append(block(self.inplanes_p, planes))

        return nn.Sequential(*layers_p)
    
    def l2_norm(self,Input):
        input_size = Input.size()
        Buffer = torch.pow(Input, 2)

        Normp = torch.sum(Buffer, 1).add_(1e-10)
        Norm = torch.sqrt(Normp)

        _output = torch.div(Input, Norm.view(-1, 1).expand_as(Input))

        output = _output.view(input_size)

        return output
    
    ###### forward streetview
    def forward_SV(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        ### do the context reweighting
        x = self.rw_emb(x)  ### size = W,H,C
        ###
        x = x.view(x.size(0), -1) ### size = W*H*C
        x = self.featVec(x)
        ### for 3 fc layer net
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        ### fetch the output of fc7
        angleVecHalf = x
        ###########
        
        x = F.normalize(x, p=2, dim=1)
        
        #x = self.featvec(x)
        
        ### embedding a l2-norm layer to form a normalized feature vectors
        #features = self.l2_norm(x)
        
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #alpha=10.0
        #alpha = 1.0
        #features = features*alpha
        
        ### return features
        #return features, angleVecHalf
        return x, angleVecHalf
    
    ###### forward overhead
    def forward_OH(self, x):
        x = self.conv1_p(x)
        x = self.bn1_p(x)
        x = self.relu_p(x)
        x = self.maxpool_p(x)

        x = self.layer1_p(x)
        x = self.layer2_p(x)
        x = self.layer3_p(x)
        x = self.layer4_p(x)
        
        ### do the context reweighting
        x = self.rw_emb_p(x)  ### size = W,H,C
        ###
        x = x.view(x.size(0), -1) ### size = W*H*C
        x = self.featVec_p(x)
        ### for 3 fc layer net
        #x = self.fc1_p(x)
        #x = self.relu_p(x)
        #x = self.fc2_p(x)
        #x = self.relu_p(x)
        ### fetch the output of fc7
        angleVecHalf = x
        ###########
        
        x = F.normalize(x, p=2, dim=1)
        
        #x = self.featvec_p(x)
        
        ### embedding a l2-norm layer to form a normalized feature vectors
        #features = self.l2_norm(x)
        
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #alpha=10.0
        #alpha = 1.0
        #features = features*alpha
        
        ### return features
        #return features, angleVecHalf
        return x, angleVecHalf
    
    
    ### forward
    def forward_Tri(self, x_A, x_P, x_N):
        features_A, vecHalfA = self.forward_SV(x_A)
        features_P, vecHalfP = self.forward_OH(x_P)
        features_N, _ = self.forward_OH(x_N)
        
        ### for angle regression
        angle_fc7 = [vecHalfA, vecHalfP]
        angle_fc7 = torch.cat(angle_fc7, 1)
        angle_fc7 = self.angle_relu(angle_fc7)
        #angle_ORvec = 0.0
        
        angle_ORvec = self.angle_fc8(angle_fc7)
        angle_ORvec = self.angle_relu(angle_ORvec)
        angle_ORvec = self.angle_fc9(angle_ORvec) ### vec length for each instance is 2 (sin(angle) and cos(angle))
        #################
        
        angle_ORvec = F.normalize(angle_ORvec, p=2, dim=1)
        
        return features_A, features_P, features_N, angle_ORvec
        
    ### forward siam
    def forward(self, x_grd, x_sat):
        global_grd, vecHalf_grd = self.forward_SV(x_grd)
        global_sat, vecHalf_sat = self.forward_OH(x_sat)
        
        ### for angle regression
        angle_fc7 = [vecHalf_grd, vecHalf_sat]
        angle_fc7 = torch.cat(angle_fc7, 1)
        angle_fc7 = self.angle_relu(angle_fc7)
        #angle_ORvec = 0.0
        
        #angle_ORvec = self.angle_fc8(angle_fc7) # for 4096 length vec
        angle_ORvec = self.angle_fc8(angle_fc7)
        angle_ORvec = self.angle_relu(angle_ORvec)
        angle_ORvec = self.angle_fc9(angle_ORvec) ### vec length for each instance is 2 (sin(angle) and cos(angle))
        #################
        
        angle_ORvec = F.normalize(angle_ORvec, p=2, dim=1)
        
        return global_grd, global_sat, angle_ORvec
        #################
        


#### return OR net split version
######### basic block ##############
def SiamFCANet18_VH( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = SiamFCANet_Base(FCAMBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def SiamFCANet34_VH( **kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = SiamFCANet_Base(FCAMBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

############################
    

