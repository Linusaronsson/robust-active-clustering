import numpy as np
import random 
import sys
from scipy.special import softmax
from scipy import sparse

def max_correlation(S, K, num_iterations):

    N = np.size(S, 0)

    best_objective = -sys.float_info.max 
    best_solution = np.zeros((N,), dtype = int)
        
    N_iter = 30

    for itr in range(0, num_iterations):
        current_solution = np.zeros((N,), dtype = int) 
        for i in range(0,N):
            current_solution[i] = random.randint(0,K-1)

        # calc total cost
        current_objective = 0.0
        for i in range(0,N):
            for j in range(0,i):
                if current_solution[i] == current_solution[j]:
                    current_objective = current_objective + S[i,j]

        old_objective = current_objective - 1

        while True:
            if (current_objective-old_objective) <= sys.float_info.epsilon or N_iter <= 0:
                break
            N_iter -= 1
            old_objective = current_objective
            order = list(range(0,N))
            random.shuffle(order)
            
            for i in range(0,N):
                cur_ind = order[i]
                temp_objects = np.zeros((K,), dtype = float)
                
                for j in range(0,N): 
                    if j != cur_ind:
                        temp_objects[current_solution[j]] = temp_objects[current_solution[j]] + S[cur_ind,j]

                sep_Obj = temp_objects[current_solution[cur_ind]]
                temp_objects[current_solution[cur_ind]] = current_objective
                
                for k in range(0,K):
                    if k != current_solution[cur_ind]:
                        temp_objects[k] = current_objective - sep_Obj + temp_objects[k]
                                             
                        
                temp_max = np.argmax(temp_objects)
                current_solution[cur_ind] = temp_max
                current_objective = temp_objects[temp_max]

        if itr == 0 or current_objective > best_objective:
            best_solution = np.array(current_solution)
            best_objective = current_objective
            
    return best_solution, best_objective

def fast_max_correlation(S, K, iterations):
    N = S.shape[0]  # Number of nodes
    best_objective = -sys.float_info.max  # Initialize best objective to lowest possible float
    best_solution = np.zeros((N,), dtype=int)  # Initialize best solution as an array of zeros

    for _ in range(iterations):
        # Initial solution
        current_solution = np.random.randint(0, K, size=N)
        
        # Calculate initial objective value
        current_obj = np.sum(S * (current_solution[:, None] == current_solution))

        while True:
            improved = False
            for i in np.random.permutation(N):
                current_cluster = current_solution[i]
                # Calculate the objective for the current configuration
                cluster_obj = np.sum(S[i] * (current_solution == current_cluster))
                best_change = 0
                best_cluster = current_cluster
                
                # Try moving node i to a different cluster and calculate new objective
                for new_cluster in range(K):
                    if new_cluster == current_cluster:
                        continue  # Skip if it's the same class
                    # Calculate the objective if i were in the new class
                    new_cluster_obj = np.sum(S[i] * (current_solution == new_cluster))
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
        if current_obj > best_objective:
            best_objective = current_obj
            best_solution = current_solution.copy()

    return best_solution, best_objective

def max_correlation_dynamic_K(S, K, num_iterations):
    N = np.size(S, 0)
    #print("SIZE: ", N)
    K_dyn = np.minimum(K, N)
    #print("NUM CLUSTERS: ", K_dyn)

    best_objective = -sys.float_info.max 
    best_solution = np.zeros((N,), dtype=int)

    N_iter = 30

    for itr in range(0,num_iterations):
        current_solution = np.zeros((N,), dtype=int) 
        
        for i in range(0,N):
            current_solution[i] = np.random.randint(0, K_dyn)
            
        # to gaurantee non-empty clusters
        temp_indices = np.random.choice(range(0, N), K_dyn, replace=False)
        for k in range(0,K_dyn):
            current_solution[temp_indices[k]] = k

        current_objective = 0.0
        for k in range(0, K_dyn):
            inds = np.where(current_solution == k)[0]
            lower_triangle_indices = np.tril_indices(len(inds), -1) 
            current_objective += np.sum(S[np.ix_(inds, inds)][lower_triangle_indices])

        old_objective = current_objective - 1.0

        for _ in range(30): 
            if (current_objective-old_objective) <= sys.float_info.epsilon:
                break
            N_iter -= 1
            old_objective = current_objective
            indices = np.arange(0, N)
            np.random.shuffle(indices)
            for i in indices:
                temp_objects = np.zeros(K_dyn)
                for k in range(0, K_dyn):
                    inds = np.where(current_solution == k)[0]
                    inds = inds[inds != i]
                    temp_objects[k] = np.sum(S[i, inds])

                if np.max(temp_objects) < 0.0:
                    # cerate a new cluster
                        current_objective = current_objective - temp_objects[current_solution[i]]
                        current_solution[i] = K_dyn
                        K_dyn = K_dyn + 1
                else:
                    sep_Obj = temp_objects[current_solution[i]]
                    temp_objects[current_solution[i]] = current_objective
                    for k in range(0,K_dyn):
                        if k != current_solution[i]:
                            temp_objects[k] = current_objective - sep_Obj + temp_objects[k]

                    temp_old_cluster = current_solution[i]
                    temp_max = np.argmax(temp_objects)
                    current_solution[i] = temp_max
                    current_objective = temp_objects[temp_max]
                        
                    # check the empy cluster, shinke if necessary
                    K_dyn_temp = len(np.unique(current_solution))
                    if K_dyn_temp < K_dyn:
                        for j in range(0,N):
                            if current_solution[j] > temp_old_cluster:
                                current_solution[j] = current_solution[j] - 1
                        K_dyn = K_dyn - 1

        if itr == 0 or current_objective > best_objective:
            best_solution = np.array(current_solution)
            best_objective = current_objective
            
    return best_solution, best_objective

from sklearn.metrics import adjusted_rand_score

def mean_field_clustering(S, K, betas, max_iter=100, tol=1e-6, noise=0, reinit=False, predicted_labels=None, q=None, h=None):
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
        
        q = softmax(beta*h, axis=1)
    else:
        q = np.copy(q)
        h = np.copy(h)

    if noise > 0:
        q += noise*np.random.randn(*q.shape)
        q /= np.sum(q, axis=1)[:, None]

    if reinit:
        q = np.random.dirichlet(np.ones(K), N)

    for beta in betas:
        for iteration in range(max_iter):
            h = -S.dot(q)
            q_new = softmax(beta*-h, axis=1)
            
            # Check for convergence
            #diff = np.linalg.norm(q_new - q)
            #if diff < tol:
                #print(f'Converged after {iteration} iterations')
                #break

            q = q_new

    return np.argmax(q, axis=1), q, h

if __name__ == "__main__":

    sim = -np.ones((9,9))
    sim[0:3,0:3] = +1
    sim[3:6,3:6] = +1
    sim[6:9,6:9] = +1
    
    lables = np.array([0,0,0,1,1,1,2,2,2])

    sol1 = max_correlation(sim, 3, 5)
    sol2 = max_correlation_dynamic_K(sim, 1, 5)
