# -*- coding: utf-8 -*-
"""
@author: Michael (S. Cai)
"""

import numpy as np

### vectors import
query_vecs_path = 'vectors/SFCANet18_query.npy'
query_vec = np.load(query_vecs_path)

exam_vecs_path = 'vectors/SFCANet18_ref.npy'
examing_vec = np.load(exam_vecs_path)
        

N_data = query_vec.shape[0]

### keep vector's lentgh
length = int(N_data * 0.01) + 1

###
correct_num = 0
error_num = 0


dist_E = (examing_vec**2).sum(1).reshape(N_data,1)
dist_Q = (query_vec**2).sum(1).reshape(N_data,1).T

dist_array = dist_E + dist_Q - 2 * np.matmul(examing_vec, query_vec.T)


### ranking and comparing
for i in range(0, N_data):
    
    gt_dist = dist_array[i, i]
    prediction = np.sum(dist_array[:, i] < gt_dist)
    if(prediction < length):
        correct_num = correct_num+1
    else:
        error_num = error_num+1
        
    if(i % 1000 == 0):
        print(i)
    
### recall at top 1%
result = np.float(correct_num)/np.float(N_data)
print('result: ', result)

