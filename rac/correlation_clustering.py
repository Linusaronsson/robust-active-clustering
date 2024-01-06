# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:13:46 2017

@author: mchehreg
"""
import numpy as np
import random 
import sys
from scipy.special import softmax
from scipy import sparse

def max_correlation(my_graph, my_K, my_itr_num):

    my_N = np.size(my_graph,0)

    best_Obj = -sys.float_info.max 
    best_Sol = np.zeros((my_N,), dtype = int)
        
    N_iter = 30

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

        while True:
            if (cur_Obj-old_Obj) <= sys.float_info.epsilon or N_iter <= 0:
                break
            N_iter -= 1
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

def fast_max_correlation(graph, K, iterations):
    N = graph.shape[0]  # Number of nodes
    best_obj = -sys.float_info.max  # Initialize best objective to lowest possible float
    best_solution = np.zeros((N,), dtype=int)  # Initialize best solution as an array of zeros

    for _ in range(iterations):
        # Initial solution
        current_solution = np.random.randint(0, K, size=N)
        
        # Calculate initial objective value
        current_obj = np.sum(graph * (current_solution[:, None] == current_solution))

        while True:
            improved = False
            for i in np.random.permutation(N):
                current_cluster = current_solution[i]
                # Calculate the objective for the current configuration
                cluster_obj = np.sum(graph[i] * (current_solution == current_cluster))
                best_change = 0
                best_cluster = current_cluster
                
                # Try moving node i to a different cluster and calculate new objective
                for new_cluster in range(K):
                    if new_cluster == current_cluster:
                        continue  # Skip if it's the same class
                    # Calculate the objective if i were in the new class
                    new_cluster_obj = np.sum(graph[i] * (current_solution == new_cluster))
                    # Calculate the change in objective
                    change = new_cluster_obj - cluster_obj
                    # If the change is positive and better than the best_change, update best_change
                    if change > best_change:
                        best_change = change
                        best_cluster = new_cluster
                
                # If best_change is positive, update the solution
                if best_change > 0:
                    # Adjust current_obj by subtracting contribution from old cluster and adding contribution to new cluster
                    current_obj += best_change - cluster_obj
                    current_solution[i] = best_cluster
                    improved = True
            
            # If no improvement is found, break the while loop
            if not improved:
                break
        
        # Update best solution
        if current_obj > best_obj:
            best_obj = current_obj
            best_solution = current_solution.copy()

    return best_solution, best_obj

def max_correlation_dynamic_K(my_graph, my_K, my_itr_num):
    my_N = np.size(my_graph, 0)
    #print("SIZE: ", my_N)
    K_dyn = np.minimum(my_K, my_N)
    #print("NUM CLUSTERS: ", K_dyn)

    best_Obj = -sys.float_info.max 
    best_Sol = np.zeros((my_N,), dtype=int)

    N_iter = 30

    for itr in range(0,my_itr_num):
        cur_Sol = np.zeros((my_N,), dtype=int) 
        
        for i in range(0,my_N):
            cur_Sol[i] = np.random.randint(0, K_dyn)
            
        # to gaurantee non-empty clusters
        temp_indices = np.random.choice(range(0, my_N), K_dyn, replace=False)
        for k in range(0,K_dyn):
            cur_Sol[temp_indices[k]] = k

        cur_Obj = 0.0
        for k in range(0, K_dyn):
            inds = np.where(cur_Sol == k)[0]
            lower_triangle_indices = np.tril_indices(len(inds), -1) 
            cur_Obj += np.sum(my_graph[np.ix_(inds, inds)][lower_triangle_indices])

        old_Obj = cur_Obj - 1.0
        #print("ENTERED ALG: ", itr)

        for _ in range(30): 
            if (cur_Obj-old_Obj) <= sys.float_info.epsilon:
                break
            N_iter -= 1
            old_Obj = cur_Obj
            indices = np.arange(0, my_N)
            np.random.shuffle(indices)
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

from sklearn.metrics import adjusted_rand_score

def mean_field_clustering(S, K, betas, true_labels=None, max_iter=100, tol=1e-6, noise_level=0.0, predicted_labels=None, q=None, h=None):
    #np.fill_diagonal(S, 0)
    N = S.shape[0]

    if q is None:
        if predicted_labels is None:
            predicted_labels, _ = max_correlation_dynamic_K(S, K, 5)
        beta = betas[0]

        K = len(np.unique(predicted_labels))
        h = np.zeros((N, K))

        for k in range(K):
            cluster_indices = np.where(predicted_labels == k)[0]
            for i in range(N):
                h[i, k] = S[i, cluster_indices].sum()
        
        #beta = 50
        q = softmax(beta*h, axis=1)
    else:
        q = np.copy(q)
        h = np.copy(h)

    #print("INITIAL Q: ", q)

    #n_level = 0.3
    #noise = n_level * (np.random.rand(N, K) - 0.5)
    #q += noise
    #q = np.maximum(q, 0)  # Ensure q stays non-negative
    #q /= np.sum(q, axis=1, keepdims=True)  # Re-normalize q

    #if is_sparse and not sparse.issparse(S):
        #S = sparse.csr_matrix(S)
    
    #max_iter = 1000
    #betas = [1]
    #tol = 1e-10
    #old_diff = np.inf
    for beta in betas:
        for iteration in range(max_iter):
            h = -S.dot(q)
            #h = -np.dot(S, q)
            q_new = softmax(beta*-h, axis=1)
            #print("--------")
            
            #current_solution = np.argmax(q_new, axis=1)
            #current_ari = adjusted_rand_score(current_solution, predicted_labels)
            #current_ari2 = adjusted_rand_score(current_solution, true_labels)
            #current_ari3 = adjusted_rand_score(predicted_labels, true_labels)
            # Check for convergence
            diff = np.linalg.norm(q_new - q)
            #print("iteration: ", iteration, " diff: ", diff, " beta: ", beta, " ari: ", current_ari, "mf: ", current_ari2, "local search: ", current_ari3)
            #if np.abs(diff - old_diff) < tol:
            if diff < tol:
                print(f'Converged after {iteration} iterations')
                break

            #old_diff = diff
            q = q_new

            # Inject noise
            #noise = noise_level * (np.random.rand(N, K) - 0.5)
            #q += noise
            #q = np.maximum(q, 0)  # Ensure q stays non-negative
            #q /= np.sum(q, axis=1, keepdims=True)  # Re-normalize q
    return np.argmax(q, axis=1), q, h

if __name__ == "__main__":

    sim = -np.ones((9,9))
    sim[0:3,0:3] = +1
    sim[3:6,3:6] = +1
    sim[6:9,6:9] = +1
    
    lables = np.array([0,0,0,1,1,1,2,2,2])

    sol1 = max_correlation(sim, 3, 5)
    sol2 = max_correlation_dynamic_K(sim, 1, 5)
