#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable



class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True, add_batch_norm=False, is_training=True, do_dim_reduction=True):
        """
        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.is_training = is_training
        self.do_dim_reduction = do_dim_reduction
        
        self.cluster_bn = nn.BatchNorm1d(cluster_size) #batch normalize for clusters
        
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size))
        nn.init.normal_(self.cluster_weights, mean=0.0, std=1.0 / math.sqrt(feature_size))
        
        self.cluster_biases = nn.Parameter(torch.randn(cluster_size))
        nn.init.normal_(self.cluster_biases, mean=0.0, std=1.0 / math.sqrt(feature_size))
        
        self.cluster_weights2 = nn.Parameter(torch.randn(1,feature_size, cluster_size))
        nn.init.normal_(self.cluster_weights2, mean=0.0, std=1.0 / math.sqrt(feature_size))
        
        self.hidden1_weights = nn.Parameter(torch.randn(self.cluster_size*self.feature_size, self.output_dim))
        nn.init.normal_(self.hidden1_weights, mean=0.0, std=1.0 / math.sqrt(self.cluster_size))
        
        self.hidden1_biases = nn.Parameter(torch.randn(output_dim))
        nn.init.constant_(self.hidden1_biases, 0.1)
        
        self.gating_weights = nn.Parameter(torch.randn(output_dim, output_dim))
        nn.init.normal_(self.gating_weights, mean=0.0, std=1.0 / math.sqrt(output_dim))
        
        self.gating_biases = nn.Parameter(torch.randn(output_dim))
        nn.init.normal_(self.gating_biases, mean=0.0, std=1.0 / math.sqrt(output_dim))
        
        self.gating_bn = nn.BatchNorm1d(output_dim)
        self.gating_bn.weight.data.fill_(1)
        self.gating_bn.bias.data.zero_()
        
    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """
        
        #input_dim = input_layer.shape[1] # extract the dimension of input layer
        
        gates = torch.matmul(input_layer, self.gating_weights)
        
        if(self.add_batch_norm):
            gates = self.gating_bn(gates)
        else:
            gates = gates + self.gating_biases
            
        gates = F.sigmoid(gates)
        
        activation_gates = input_layer*gates
        
        return activation_gates
            
        
    def forward(self, x):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 
        
        '
         ;l mqae'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        
        x = F.normalize(x, p=2, dim=1) # do L2_normalization
        reshaped_input = x.view(-1,self.feature_size)
        
        ### activate with conv
        activation = torch.matmul(reshaped_input, self.cluster_weights)
        
        ### with
        if(self.add_batch_norm):
            ### initialize
            self.cluster_bn.weight.data.fill_(1)
            self.cluster_bn.bias.data.zero_()
            activation = self.cluster_bn(activation)
        else:
            activation += self.cluster_biases
            
        activation = F.softmax(activation, dim=1) # dim=1 is aim to clusters
        activation = activation.view(-1, self.max_samples, self.cluster_size)
        
        a_sum = torch.sum(activation,dim=-2,keepdim=True)
        
        a = a_sum * self.cluster_weights2
        
        activation = activation.permute(0,2,1)
        
        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)
        
        vlad = torch.matmul(activation,reshaped_input)
        vlad = vlad.permute(0,2,1)
        vlad = vlad-a
        ### intra-clusters norm
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.view(-1, self.cluster_size*self.feature_size)
        ### inter-clusters norm
        vlad = F.normalize(vlad, p=2, dim=1)
        
        if(self.do_dim_reduction):
            vlad = torch.matmul(vlad, self.hidden1_weights)
            vlad += self.hidden1_biases
        
        ### 
        #if(self.gating):
        #    vlad = self.context_gating(vlad)
        
        return vlad

### residual NetRVLAD
class NetRVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True, add_batch_norm=False, is_training=True, do_dim_reduction=True, input_norm=True):
        """
        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """
        super(NetRVLAD, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.is_training = is_training
        self.do_dim_reduction = do_dim_reduction
        self.input_norm = input_norm
        
        self.cluster_bn = nn.BatchNorm1d(cluster_size) #batch normalize for clusters
        
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size))
        nn.init.normal_(self.cluster_weights, mean=0.0, std=1.0 / math.sqrt(feature_size))
        
        self.cluster_biases = nn.Parameter(torch.randn(cluster_size))
        nn.init.normal_(self.cluster_biases, mean=0.0, std=1.0 / math.sqrt(feature_size))
        
        self.hidden1_weights = nn.Parameter(torch.randn(self.cluster_size*self.feature_size, self.output_dim))
        nn.init.normal_(self.hidden1_weights, mean=0.0, std=1.0 / math.sqrt(self.cluster_size))
        
        self.hidden1_biases = nn.Parameter(torch.randn(output_dim))
        nn.init.constant_(self.hidden1_biases, 0.1)
        
        self.gating_weights = nn.Parameter(torch.randn(output_dim, output_dim))
        nn.init.normal_(self.gating_weights, mean=0.0, std=1.0 / math.sqrt(output_dim))
        
        self.gating_biases = nn.Parameter(torch.randn(output_dim))
        nn.init.normal_(self.gating_biases, mean=0.0, std=1.0 / math.sqrt(output_dim))
        
        self.gating_bn = nn.BatchNorm1d(output_dim)
        self.gating_bn.weight.data.fill_(1)
        self.gating_bn.bias.data.zero_()
        
    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """
        
        #input_dim = input_layer.shape[1] # extract the dimension of input layer
        
        gates = torch.matmul(input_layer, self.gating_weights)
        
        if(self.add_batch_norm):
            gates = self.gating_bn(gates)
        else:
            gates = gates + self.gating_biases
            
        gates = F.sigmoid(gates)
        
        activation_gates = input_layer*gates
        
        return activation_gates
            
        
    def forward(self, x):
        """Forward pass of a NetRVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = input.view(-1, feature_size)

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        # if do l2-norm in the begining
        if(self.input_norm):
            x = F.normalize(x, p=2, dim=1) # do L2_normalization
        
        reshaped_input = x.view(-1,self.feature_size)
        
        ### activate with conv
        activation = torch.matmul(reshaped_input, self.cluster_weights)
        
        ### with
        if(self.add_batch_norm):
            ### initialize
            self.cluster_bn.weight.data.fill_(1)
            self.cluster_bn.bias.data.zero_()
            activation = self.cluster_bn(activation)
        else:
            activation += self.cluster_biases
            
        activation = F.softmax(activation, dim=1) # dim=1 is aim to clusters
        activation = activation.view(-1, self.max_samples, self.cluster_size)
        
        #a_sum = torch.sum(activation,dim=-2,keepdim=True)
        
        #a = a_sum * self.cluster_weights2
        
        activation = activation.permute(0,2,1)
        
        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)
        
        vlad = torch.matmul(activation,reshaped_input)
        vlad = vlad.permute(0,2,1)
        #vlad = vlad-a
        ### intra-clusters norm
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.view(-1, self.cluster_size*self.feature_size)
        ### inter-clusters norm
        vlad = F.normalize(vlad, p=2, dim=1)
        
        if(self.do_dim_reduction):
            vlad = torch.matmul(vlad, self.hidden1_weights)
            vlad += self.hidden1_biases
        
        ### 
        if(self.gating):
            vlad = self.context_gating(vlad)
        
        return vlad
##############################   