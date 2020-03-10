#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Michael (S. Cai)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np



class WeightedSoftMarginLoss(nn.Module):
    """
    CVM
    """
    ### the value of margin is given according to the facenet
    def __init__(self, loss_weight=10.0):
        super(WeightedSoftMarginLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, sat_global, grd_global, mini_batch, batch_hard_count=0):
        dist_array = 2 - 2 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diagonal(dist_array)
        
        if(batch_hard_count==0):
            pair_n = mini_batch*(mini_batch - 1.0)
            
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))) / pair_n
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))) / pair_n
            
            loss = (loss_g2s + loss_s2g) / 2.0     
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
            triplet_dist_g2s = triplet_dist_g2s - torch.diag(torch.diagonal(triplet_dist_g2s))
            top_k_g2s, _ = torch.topk((triplet_dist_g2s.t()), batch_hard_count)
            loss_g2s = torch.mean(top_k_g2s)
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
            triplet_dist_s2g = triplet_dist_s2g - torch.diag(torch.diagonal(triplet_dist_s2g))
            top_k_s2g, _ = torch.topk(triplet_dist_s2g, batch_hard_count)
            loss_s2g = torch.mean(top_k_s2g)
            
            loss = (loss_g2s + loss_s2g) / 2.0
            #loss = loss_g2s
            
            pos_dist_avg = pos_dist.mean()
            nega_dist_avg = dist_array.mean()
        
            return loss, pos_dist_avg, nega_dist_avg.sum()   
    

### OR version
class WeightedSoftMarginLossOR(nn.Module):
    """
    CVM complemented with orientation regression 
    """
    ### the value of margin is given according to the facenet
    def __init__(self, loss_weight=10.0):
        super(WeightedSoftMarginLossOR, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, sat_global, grd_global, mini_batch, batch_hard_count, angle_label, angle_pred):
        dist_array = 2 - 2 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diagonal(dist_array)
        
        if(batch_hard_count==0):
            pair_n = mini_batch*(mini_batch - 1.0)
            
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight)) / pair_n
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            loss_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight)) / pair_n
            
            #loss = (loss_g2s + loss_s2g) / 2.0     
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
            top_k_g2s, _ = torch.topk(torch.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = torch.mean(top_k_g2s)
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
            top_k_s2g, _ = torch.topk(triplet_dist_s2g, batch_hard_count)
            loss_s2g = torch.mean(top_k_s2g)
            
            #loss = (loss_g2s + loss_s2g) / 2.0
        
        ### angle regression
        dist_OR = (angle_pred - angle_label).pow(2).sum(1)
        # ground view as anchor
        loss_OR_g = dist_OR.repeat(dist_array.size()[0],1) # the way of repeat and t() according to pos_dist 
        loss_OR_g = loss_OR_g / pair_n
        
        # satellite view as anchor
        loss_OR_s = dist_OR.repeat(dist_array.size()[0],1).t() # the way of repeat and t() according to pos_dist 
        loss_OR_s = loss_OR_s / pair_n
        
        
        #loss_OR = (loss_OR_g + loss_OR_s) / 2.0
        
        # loss combine
        theta1 = 10.0
        theta2 = 5.0
        loss_g2s = theta1*loss_g2s + theta2*loss_OR_g # ground as anchor
        loss_s2g = theta1*loss_s2g + theta2*loss_OR_s # satellite as anchor
        
        loss_merge = torch.sum(loss_g2s + loss_s2g) / 2.0
        
        pos_dist_avg = pos_dist.mean()
        nega_dist_avg = (dist_array - torch.diag(pos_dist)) / pair_n

        return loss_merge, pos_dist_avg, nega_dist_avg.sum()  
    
    
    
### for un-normalized features, OR version
class SoftMargin_TriLoss_OR_UnNorm(nn.Module):
    """
    SoftMargin_TriLoss_OR
    """
    ### init
    def __init__(self, margin=10.0):
        super(SoftMargin_TriLoss_OR_UnNorm, self).__init__()
        #self.margin = 20.25*margin
        
    def forward(self, sat_global, grd_global, marginCal, angle_label, angle_pred, theta1, theta2):
        
        #self.margin = marginCal
        
        distance_negative = torch.autograd.Variable(torch.zeros(grd_global.shape[0],grd_global.shape[0])).cuda()
        for l in range(grd_global.size()[0]):
            distance_negative[l] = ( grd_global[l].repeat(grd_global.shape[0],1) - sat_global ).pow(2).sum(1)
        
        distance_positive = torch.diagonal(distance_negative)
        
        ### loss computing
        losses = torch.log(1.0 + torch.exp(distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative))

        ### orientation regression loss (angle_label = [sin(angle), cos(angle)])
        
        losses_OR = (angle_pred - angle_label).pow(2).sum(1)
        losses_OR = losses_OR.repeat(grd_global.size()[0],1).t()
        losses_OR = losses_OR
        
        #loss combining
        losses_merge = theta1*losses + theta2*losses_OR
        
        ###### exp based loss

        return losses_merge.mean(), distance_positive.mean(), distance_negative.mean() 

    
# for un-normalized features
class HER_TriLoss_OR_UnNorm(nn.Module):
    """
    HER_TriLoss_OR_UnNorm (Hard Exemplar Reweighting Triplet Loss)
    """
    ### init
    def __init__(self, margin=10.0):
        super(HER_TriLoss_OR_UnNorm, self).__init__()
        #self.margin = 20.25*margin
        
    def forward(self, sat_global, grd_global, marginCal, angle_label, angle_pred, theta1, theta2):
        
        self.margin = marginCal
        
        distance_negative = torch.autograd.Variable(torch.zeros(grd_global.shape[0],grd_global.shape[0])).cuda()
        for l in range(grd_global.size()[0]):
            distance_negative[l] = ( grd_global[l].repeat(grd_global.shape[0],1) - sat_global ).pow(2).sum(1)
        
        distance_positive = torch.diagonal(distance_negative)
        
        ### distance rectification factor - beta
        beta = self.margin/2.0
        
        ### rectified distance for computing weight mask
        dist_rec = distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative + beta
        
        # dist_clamp = torch.clamp(dist_clamp, max=30.0) # max=30.0, for preventing overflow which leads to inf or nan
    
        p = 1.0/(1.0 + torch.exp( dist_rec ))
        
        ### weight mask generating 
        w_mask = F.relu(-torch.log2(p + 0.00000001))
        
        ### weight mask pruning
        w_low = -np.log2(1.0/(1.0 + np.exp(  -1.0*self.margin + beta ) + 0.00000001) )
        w_high = -np.log2(1.0/(1.0 + np.exp(  -0.0*self.margin + beta ) + 0.00000001) )
        
        w_mask[w_mask<w_low] = 0.1/grd_global.shape[0] # pruning over simple data
        
        w_mask[w_mask>w_high] = w_high # pruning over extreme hard data
        
        
        # diagonal elements need to be neglected (set to zero)
        w_mask = w_mask - torch.diag(torch.diagonal(w_mask))
        
        # scaling fector - alpha (recommended setting is 1.0)
        alpha = 1.0
        
        # main loss computing
        losses = w_mask * torch.log(1.0 + torch.exp( alpha*(distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative)))
                
        ### orientation regression loss (angle_label = [sin(angle), cos(angle)])
        
        losses_OR = (angle_pred - angle_label).pow(2).sum(1)
        losses_OR = losses_OR.repeat(grd_global.size()[0],1).t()

        # OR loss computing
        losses_OR = w_mask * losses_OR 
        
        #loss combining, as a recommendation - theta1 : theta2 = 2 : 1 (here theta2 can be a number in {1,2,3,...,10})
        losses_merge = theta1*losses + theta2*losses_OR 
        
        ###### exp based loss

        return losses_merge.mean(), distance_positive.mean(), distance_negative.mean()
    
#############################

    



