# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:13:46 2017

@author: mchehreg
"""
import numpy as np
import random 
import sys

def max_correlation(my_graph, my_K, my_itr_num):

    my_N = np.size(my_graph,0)

    best_Obj = -sys.float_info.max 
    best_Sol = np.zeros((my_N,), dtype = int)

    for itr in range(0,my_itr_num):
  
        cur_Sol = np.zeros((my_N,), dtype = int) 
        
        for i in range(0,my_N):
            cur_Sol[i] = random.randint(0,my_K-1)

        # calc total cost
        cur_Obj = 0.0
        for i in range(0,my_N):
            for j in range(0,i):
                if cur_Sol[i] == cur_Sol[j]:
                    cur_Obj = cur_Obj + my_graph[i,j]


        old_Obj = cur_Obj - 1

        while cur_Obj-old_Obj > sys.float_info.epsilon:
            old_Obj = cur_Obj

            order = list(range(0,my_N))
            random.shuffle(order)
            
            for i in range(0,my_N):
                cur_Ind = order[i]
                temp_Objs = np.zeros((my_K,), dtype = float)
                
                for j in range(0,my_N): 
                    if j != cur_Ind:
                        temp_Objs[cur_Sol[j]] = temp_Objs[cur_Sol[j]] + my_graph[cur_Ind,j]


                sep_Obj = temp_Objs[cur_Sol[cur_Ind]]
                temp_Objs[cur_Sol[cur_Ind]] = cur_Obj
                
                for k in range(0,my_K):
                    if k != cur_Sol[cur_Ind]:
                        temp_Objs[k] = cur_Obj - sep_Obj + temp_Objs[k]
                                             
                        
                temp_max = np.argmax(temp_Objs)
                cur_Sol[cur_Ind] = temp_max
                cur_Obj = temp_Objs[temp_max]

        if itr == 0 or cur_Obj > best_Obj:
            best_Sol = np.array(cur_Sol)
            best_Obj = cur_Obj
            
    return best_Sol, best_Obj

def max_correlation_dynamic_K(my_graph, my_K, my_itr_num, rand_state):
    my_N = np.size(my_graph, 0)
    #print("SIZE: ", my_N)
    K_dyn = np.minimum(my_K, my_N)
    #print("NUM CLUSTERS: ", K_dyn)

    best_Obj = -sys.float_info.max 
    best_Sol = np.zeros((my_N,), dtype=int)

    N_iter = 15

    for itr in range(0,my_itr_num):
        cur_Sol = np.zeros((my_N,), dtype=int) 
        
        for i in range(0,my_N):
            cur_Sol[i] = rand_state.randint(0, K_dyn)
            
        # to gaurantee non-empty clusters
        temp_indices = rand_state.choice(range(0, my_N), K_dyn, replace=False)
        for k in range(0,K_dyn):
            cur_Sol[temp_indices[k]] = k

        cur_Obj = 0.0
        for k in range(0, K_dyn):
            inds = np.where(cur_Sol == k)[0]
            lower_triangle_indices = np.tril_indices(len(inds), -1) 
            cur_Obj += np.sum(my_graph[np.ix_(inds, inds)][lower_triangle_indices])

        old_Obj = cur_Obj - 1.0
        #print("ENTERED ALG: ", itr)

        while True:
            if (cur_Obj-old_Obj) <= sys.float_info.epsilon or N_iter <= 0:
                break
            N_iter -= 1
            old_Obj = cur_Obj
            indices = np.arange(0, my_N)
            rand_state.shuffle(indices)
            for i in indices:
                temp_Objs = np.zeros(K_dyn)
                for k in range(0, K_dyn):
                    inds = np.where(cur_Sol == k)[0]
                    inds = inds[inds != i]
                    temp_Objs[k] = np.sum(my_graph[i, inds])

                if np.max(temp_Objs) < 0.0:
                    # cerate a new cluster
                        cur_Obj = cur_Obj - temp_Objs[cur_Sol[i]]
                        cur_Sol[i] = K_dyn
                        K_dyn = K_dyn + 1
                else:
                    sep_Obj = temp_Objs[cur_Sol[i]]
                    temp_Objs[cur_Sol[i]] = cur_Obj
                    for k in range(0,K_dyn):
                        if k != cur_Sol[i]:
                            temp_Objs[k] = cur_Obj - sep_Obj + temp_Objs[k]

                    temp_old_cluster = cur_Sol[i]
                    temp_max = np.argmax(temp_Objs)
                    cur_Sol[i] = temp_max
                    cur_Obj = temp_Objs[temp_max]
                        
                    # check the empy cluster, shinke if necessary
                    K_dyn_temp = len(np.unique(cur_Sol))
                    if K_dyn_temp < K_dyn:
                        for j in range(0,my_N):
                            if cur_Sol[j] > temp_old_cluster:
                                cur_Sol[j] = cur_Sol[j] - 1
                        K_dyn = K_dyn - 1

        if itr == 0 or cur_Obj > best_Obj:
            best_Sol = np.array(cur_Sol)
            best_Obj = cur_Obj
            
    return best_Sol, best_Obj

if __name__ == "__main__":

    sim = -np.ones((9,9))
    sim[0:3,0:3] = +1
    sim[3:6,3:6] = +1
    sim[6:9,6:9] = +1
    
    lables = np.array([0,0,0,1,1,1,2,2,2])

    sol1 = max_correlation(sim, 3, 5)
    sol2 = max_correlation_dynamic_K(sim, 1, 5)
