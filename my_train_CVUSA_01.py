#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Michael (S. Cai)
"""

###################### adding angle regression ##################
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from SiamFCANet import SiamFCANet18_CVUSA, SiamFCANet34_CVUSA

from losses_for_training import SoftMargin_TriLoss_OR_UnNorm
from losses_for_training import HER_TriLoss_OR_UnNorm


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
   

################# triplet data preparation ####################

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

#########################

##### for test #####

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
        
        ###### for exam data
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
    
########################################
up_root = 'dataset/CVUSA/'

####################### model assignment #######################

#################

net_pre = SiamFCANet18_CVUSA() ### pre model 

weight_path = 'weights/FCANET18/'
net_pre.load_state_dict(torch.load(weight_path+'Model_best.pth'))

net_pre_dict = net_pre.state_dict() ### 
#################

net = SiamFCANet18_CVUSA()
net.cuda()

net_dict = net.state_dict()

### update net.state_dict() for fine-tune
net_pre_dict = {k: v for k, v in net_pre_dict.items() if k in net_dict}
net_dict.update(net_pre_dict)
net.load_state_dict(net_dict)

save_path = 'weights/FCANET18/'
###########################

mini_batch = 12

### a space to restore loss value
loss_vec = np.zeros(600000, dtype=np.float32)

########################### ranking test (partial test set) ############################
def RankTest(net_test, best_rank_result):
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
        
        ###### feature vector feeding
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
            
        print(i)
    
    print('Vec Completed')
        
    ### load vectors
    query_vectors = query_vec
    
    repo_vectors = examing_vec
    
    N_data = query_vectors.shape[0]
    
    ### metrics for evaluation (top 1%)
    length = int(N_data * 0.01) + 1
    
    ### top 1
    #length = 1
        
    ### top 10
    #length = 10
    
    ###
    correct_num = 0
    error_num = 0
    ### ranking and comparing
    for i in range(0,N_data):
        ### query vector inflation, N_data times
        #query_tile = np.tile(query_vectors[i], (N_data,1))
        ### comparing
        diff_QE = query_vectors[i] - repo_vectors
        ### distance norm,  N_data normed dist
        dist_QE = np.linalg.norm(diff_QE, 2, 1)
        ### distance ranking whether true match is in the dist_rank
        dist_rank = np.argsort(dist_QE)[0:length]
        if(i in dist_rank):
            correct_num = correct_num+1
        else:
            error_num = error_num+1
            
        if(i % 100 == 0):
            print(i)
        
    ### recall at top 1%
    result = np.float(correct_num)/np.float(N_data)
    print('result: ', result)
    
    net_test.train()
    
    ### restore the best model weights
    if(result > best_rank_result):
        torch.save(net_test.state_dict(), save_path + 'Model_best.pth')
        np.save(save_path + 'best_result.npy', result)
    
    return result
##########################
    
###### learning criterion assignment #######

criterion = HER_TriLoss_OR_UnNorm()


bestRankResult = 0.1 #

# loop over the dataset
N_loop = 100
for epoch in range(0,N_loop):
    
    ### save model

    if(epoch>0):
        compNum = epoch % 5
        print('taking snapshot ...')
        torch.save(net.state_dict(), save_path + 'model_' + np.str(compNum) + '.pth')
        #np.save(save_path + 'loss_vec' + np.str(epoch) + 'epoch.npy', loss_vec)
    

    ### ranking test
    if(epoch>50):
        current = RankTest(net, bestRankResult)
        if(current>bestRankResult):
            bestRankResult = current
            
    
    if(epoch==0):
        theta1 = 1.0
        theta2 = 0.5
    else:
        theta1 = 1.0
        theta2 = 0.5
    

    net.train()
    
    ###
    base_lr = 0.00005
    base_lr = base_lr*( (1.0-float(epoch)/float(N_loop))**(1.0) )
    
    ### 
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()
    
    ################# txt file and input data preparation ###############
    ### load txt  and shuffle
    train_data_shuffled = random.sample(trainList, len(trainList))
    
    grd_list_shuffled = []
    sat_list_shuffled = []
    for rawList in train_data_shuffled:
        grd_list_shuffled.append(rawList[1])
        sat_list_shuffled.append(rawList[0])
    
    ############ Exemplar list will be shuffled in training ############
    num_files = mini_batch * (len(grd_list_shuffled) // mini_batch)
    
    ### number to control length of data
    repNum = 1
    
    num_files = repNum*num_files
    
    filenames_grd = grd_list_shuffled[0:num_files]
    filenames_sat = sat_list_shuffled[0:num_files]
    
    ### triplet ImageData
    triplet_data = Triplet_ImageData(up_root, filenames_grd, filenames_sat)
    
    ### feeding A and P into train loader
    trainloader = DataLoader(triplet_data, batch_size=mini_batch, shuffle=False, num_workers=8)
    
    for Loop, TripletData in enumerate(trainloader, 0):  
        # get the inputs
        inputs_grd, inputs_sat, angleLabel, angle_sight = TripletData
        
        # wrap them in Variable
        inputs_grd, inputs_sat, angleLabel = Variable(inputs_grd).cuda(), Variable(inputs_sat).cuda(), Variable(angleLabel).cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        ### anchor is streetview, positive and negative are overhead views
        ##
        grd_global, sat_global, anglePred = net.forward(inputs_grd, inputs_sat)
        
        ###
        margin = 0.15
        margin = 0.15*torch.mean(( grd_global.pow(2).sum(1) + sat_global.pow(2).sum(1) )/2.0)
        marginCal = np.float32(margin.detach().cpu().numpy())
        
        loss, pos_dist, nega_dist = criterion(sat_global, grd_global, marginCal, angleLabel, anglePred, theta1, theta2)
        
        
        loss.backward()
        
        #!!! .step() weight = weight - learning_rate * gradient
        optimizer.step()
        
        ###### 
        IterTimes = epoch * (num_files // mini_batch) + Loop #the iteration times
        
        # print statistics 
        running_loss = np.float(loss.data)
        p_dist = pos_dist.detach().cpu().numpy()
        n_dist = nega_dist.detach().cpu().numpy()
        
        ### record the loss
        loss_vec[IterTimes] = running_loss
        
        if Loop % 10 == 9:   # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch, Loop, running_loss))
            print('positive distance 01: ', p_dist)
            print('negative distance 01: ', n_dist)
            

print('Finished Training')

### test after training
current = RankTest(net, bestRankResult)
if(current>bestRankResult):
    bestRankResult = current
    
