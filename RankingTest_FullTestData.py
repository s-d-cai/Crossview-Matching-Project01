#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Michael (S. Cai)
"""


import numpy as np

### vectors import
query_vecs_path = 'vectors/SFCANet18_query.npy'
query_vectors = np.load(query_vecs_path)

exam_vecs_path = 'vectors/SFCANet18_ref.npy'
repo_vectors = np.load(exam_vecs_path)


N_data = query_vectors.shape[0]
### evaluation metrics
# top 1%
length = (N_data // 100) + 1

# top-1
#length = 1

# top-10
#length = 10

###
correct_num = 0
error_num = 0
### ranking test
for i in range(0,N_data):
    ### comparing
    diff_QE = query_vectors[i] - repo_vectors
    ### distance normalization
    dist_QE = np.linalg.norm(diff_QE, 2, 1)
    ### distance ranking and telling whether the true match is in the dist_rank
    dist_rank = np.argsort(dist_QE)[0:length]
    if(i in dist_rank):
        correct_num = correct_num+1
    else:
        error_num = error_num+1
        
    if(i % 50 == 0):
        print(i)
        
        
### show result of recall at the current evaluation metric
result = np.float(correct_num)/np.float(N_data)
print('result: ', result)
        