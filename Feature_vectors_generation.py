#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Michael (S. Cai)
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from SiamFCANet import SiamFCANet18_CVUSA, SiamFCANet34_CVUSA

from input_data import InputData

import numpy as np
import skimage
from skimage import io, transform
import cv2
import os

from PIL import Image 

import random
from numpy.random import randint as randint
from numpy.random import uniform as uniform

### in python2 list type data need copy.copy() method to realize .copy() as in numpy array
import copy

########################
torch.backends.cudnn.benchmark = True # use cudnn
########################

# scan the files
def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if len(FileNames) > 0:
        for fn in FileNames:
            if len(FlagStr) > 0:
                if IsSubString(FlagStr, fn):
                    fullfilename = os.path.join(FindPath, fn)
                    FileList.append(fullfilename)
            else:
                fullfilename = os.path.join(FindPath, fn)
                FileList.append(fullfilename)

    if len(FileList) > 0:
        FileList.sort()

    return FileList


def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag   

################# triplet data preparing ####################

class Triplet_ImageData(Dataset):
    ###label_list 0 1  A means Anchor and P means positive
    def __init__(self, root_path, grd_list, sat_list): 


        self.image_names_grd = grd_list
        self.image_names_sat = sat_list

        self.up_root = root_path
        ######
       
        
    def __len__(self):

        return len(self.image_names_grd)

    def __getitem__(self, idx):

        ### for anchor
        data_names_grd = os.path.join(self.up_root, self.image_names_grd[idx])
        image_grd = Image.open(data_names_grd)
        
        ### adjust with torchvision
        trans_img_G = transforms.ToTensor()(image_grd)
        # torchvision is R G B and opencv is B G R
        trans_img_G[0] = trans_img_G[0]*255.0 - 123.6  # Red
        trans_img_G[1] = trans_img_G[1]*255.0 - 116.779  # Green
        trans_img_G[2] = trans_img_G[2]*255.0 - 103.939  # Blue
        
        ######################################
        
        ###### for positive  this is the most technical part for feeding data
        data_names_sat = os.path.join(self.up_root, self.image_names_sat[idx])
        image_sat = Image.open(data_names_sat)
        
        ## randomly create an angle
        #angle = np.random.randint(-180,180) # full rotate
        angle = np.random.randint(0, 4) * 90 # 4 angle view with interval=90
        
        rand_crop = random.randint(1, 748)
        if rand_crop > 512:
            start = int((750 - rand_crop) / 2)
            box=(start,start,start + rand_crop,start + rand_crop)
            image_sat = image_sat.crop(box)
            
        
        trans_img_S = transforms.Resize([512,512], interpolation=Image.ANTIALIAS)(image_sat)
        trans_img_S = trans_img_S.rotate(angle, resample=Image.BICUBIC)
        trans_img_S = transforms.ToTensor()(trans_img_S)
        trans_img_S[0] = trans_img_S[0]*255.0 
        trans_img_S[1] = trans_img_S[1]*255.0
        trans_img_S[2] = trans_img_S[2]*255.0
        
        ### adjust with skimage
        ## needs to be changed into float64 and also change the turns of axis
        ## to pass the Perspective transformation, initially

        trans_img_S[0] = trans_img_S[0] - 123.6  # Red
        trans_img_S[1] = trans_img_S[1] - 116.779  # Green
        trans_img_S[2] = trans_img_S[2] - 103.939  # Blue
        
        
        ### angle vector
        angle2radian = (np.pi/180.0)
        ### np.sin and np.cos are base on radian rather than angle
        ### use the pair of sin and cos is because sin and cos couple can determine an certain angle
        angle_tensor = torch.Tensor([np.sin(angle*angle2radian), np.cos(angle*angle2radian)])
        
        ########################################
    
        return trans_img_G, trans_img_S, angle_tensor, angle

########################

##### for testing #####

class ImageDataForExam(Dataset):
    ###label_list 0 1  A means Anchor and P means positive
    def __init__(self, grd_list, sat_list): 

        
        self.image_names_grd = grd_list
        self.image_names_sat = sat_list
        
        ######
    
    def __len__(self):

        return len(self.image_names_grd)

    def __getitem__(self, idx):

        ### for query data
        data_names_grd = os.path.join('', self.image_names_grd[idx])
        image_grd = Image.open(data_names_grd)
        
        ### adjust with torchvision
        trans_img_G = transforms.ToTensor()(image_grd)
        # torchvision is R G B and opencv is B G R
        trans_img_G[0] = trans_img_G[0]*255.0 - 123.6  # Red
        trans_img_G[1] = trans_img_G[1]*255.0 - 116.779  # Green
        trans_img_G[2] = trans_img_G[2]*255.0 - 103.939  # Blue
        
        ######################################
        
        ###### for examing data
        data_names_sat = os.path.join('', self.image_names_sat[idx])
        image_sat = Image.open(data_names_sat)
        
        ### adjust with torchvisison
        trans_img_S = transforms.Resize([512,512], interpolation=Image.ANTIALIAS)(image_sat)
        trans_img_S = transforms.ToTensor()(trans_img_S)
        
        trans_img_S[0] = trans_img_S[0]*255.0 - 123.6  # Red
        trans_img_S[1] = trans_img_S[1]*255.0 - 116.779  # Green
        trans_img_S[2] = trans_img_S[2]*255.0 - 103.939  # Blue
        
        ########################################
        
        return trans_img_G, trans_img_S

#################

### load data
data = InputData()
trainList = data.id_list
trainIdxList = data.id_idx_list
testList = data.id_test_list
testIdxList = data.id_test_idx_list
    
#######################
up_root = 'dataset/'


### vectors restoring path
save_path = 'vectors/'

###########################

mini_batch = 8

########################### Feature Extraction ############################
### feature vectors generation
def FeatVecGen(net_test, model_name):
    ### net evaluation state

    net_test.eval()
    
    filenames_query = []
    filenames_examing = []

    for rawTestList in testList:
        info_query = up_root + rawTestList[1]
        filenames_query.append(info_query)
        info_examing = up_root + rawTestList[0]
        filenames_examing.append(info_examing)
    
    
    my_data = ImageDataForExam(filenames_query, filenames_examing)
                                     
    mini_batch = 8
    testloader = DataLoader(my_data, batch_size=mini_batch, shuffle=False, num_workers=8)
    
    N_data = len(filenames_query)
    vec_len = 4096
    
    ### N_data % mini_batch 
    nail = N_data % mini_batch
    ### N_data // mini_batch 
    max_i = N_data // mini_batch
    ### creat a space for restoring features
    query_vec = np.zeros([N_data,vec_len], dtype=np.float32)
    examing_vec = np.zeros([N_data,vec_len], dtype=np.float32)
    
    ### feature extraction
    for i, data in enumerate(testloader, 0):
        data_query, data_examing = data
        data_query, data_examing = Variable(data_query).cuda(), Variable(data_examing).cuda()
        
        outputs_query, _ = net_test.forward_SV(data_query)
        outputs_examing, _ = net_test.forward_OH(data_examing)
        
        ###### feature vectors feeding
        if(i<max_i):
            m = mini_batch*i
            n = mini_batch*(i+1)
            query_vec[m:n] = outputs_query.data.cpu().numpy()
            examing_vec[m:n] = outputs_examing.data.cpu().numpy()
        else:
            m = mini_batch*i
            n = mini_batch*i + nail
            query_vec[m:n] = outputs_query.data.cpu().numpy()
            examing_vec[m:n] = outputs_examing.data.cpu().numpy()
        
        if(i % 8 == 0):
            print(i)
    
    path = 'vectors/'
    np.save(path + model_name + '_query.npy', query_vec)
    np.save(path + model_name + '_ref.npy', examing_vec)
    print('vec produce done')

##########################

### Siam-FCANet 18 ###

model_name = 'SFCANet18'

net = SiamFCANet18_CVUSA()
net.cuda()

weight_path = 'weights/FCANET18/'
net.load_state_dict(torch.load(weight_path+'SFCANet_18.pth'))

FeatVecGen(net, model_name)

###

### Siam-FCANet 34 ###

model_name = 'SFCANet34'

net = SiamFCANet18_CVUSA()
net.cuda()

weight_path = 'weights/FCANET34/'
net.load_state_dict(torch.load(weight_path+'SFCANet_34.pth'))

FeatVecGen(net, model_name)

###
