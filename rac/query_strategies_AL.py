from typing import Any
import numpy as np 
from itertools import combinations, product
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy as scipy_entropy
from scipy import sparse
from rac.correlation_clustering import mean_field_clustering
from rac.correlation_clustering import max_correlation, fast_max_correlation, max_correlation_dynamic_K, mean_field_clustering
import scipy

class QueryStrategyAL:
    def __init__(self, al):
        self.al = al

    def select_batch(self, acq_fn, batch_size):
        if acq_fn == "uniform":
            self.info_matrix = np.random.rand(len(self.al.Y_pool))
        elif acq_fn == "entropy":
            self.info_matrix = self.compute_entropy()
        elif acq_fn == "cc_entropy":
            self.info_matrix = self.compute_cc_entropy()
        else:
            raise ValueError("Invalid acquisition function: {}".format(acq_fn))

        return self.select_objects(batch_size, self.info_matrix, acq_noise=self.al.acq_noise)

    def select_objects(self, batch_size, I, acq_noise=False):
        informative_scores = I
        if acq_noise:
            num_pairs = len(informative_scores)
            #informative_scores += np.abs(np.min(informative_scores))
            #informative_scores[informative_scores < 0] = 0
            #print("max: ", np.max(informative_scores))
            #print("min: ", np.min(informative_scores))
            #informative_scores = np.abs(informative_scores)
            if self.al.use_power:
                informative_scores = np.log(informative_scores)
            #print("max log: ", np.max(informative_scores))
            #print("min log: ", np.min(informative_scores))
            power_beta = 1
            informative_scores = informative_scores + scipy.stats.gumbel_r.rvs(loc=0, scale=1/power_beta, size=num_pairs, random_state=None)
        else:
            unique_diffs = np.diff(np.unique(informative_scores))
            if unique_diffs.size > 0:
                noise_level = np.abs(np.min(unique_diffs)) / 10
            else:
                noise_level = 1e-10
            informative_scores = informative_scores + np.random.uniform(-noise_level, noise_level, informative_scores.shape)

        top_B_indices = np.argpartition(informative_scores, -batch_size)[-batch_size:]
        return top_B_indices

    def compute_entropy(self):
        probs = self.al.model.predict_proba(self.al.X_pool)
        return scipy_entropy(probs.T)

    def compute_cc_entropy(self):
        # Probabilities for X_train (one-hot encode Y_train)
        prob_all = scipy_softmax(20*self.al.queried_labels, axis=1)
        #prob_train = np.zeros((self.al.Y_train.size, self.al.Y.max()+1))
        #prob_train[np.arange(self.al.Y_train.size), self.al.Y_train] = 1

        # Predict probabilities for X_pool
        prob_pool = self.al.model.predict_proba(self.al.X_pool)

        unqueried_indices = np.where(np.sum(self.al.queried_labels, axis=1) == 0)[0]
        prob_all[unqueried_indices] = prob_pool[unqueried_indices]

        # Initialize similarity matrix
        N = prob_all.shape[0]
        S = np.zeros((N, N))

        # Calculate expected similarity
        for i in range(N):
            for j in range(N):
                if i != j:
                    P_S_ij_plus_1 = np.sum(prob_all[i, :] * prob_all[j, :])
                    E_S_ij_plus_1 = P_S_ij_plus_1
                    E_S_ij_minus_1 = E_S_ij_plus_1 - 1
                    E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
                    S[i, j] = E_S_ij

        # Ensure diagonal is zero
        np.fill_diagonal(S, 0)

        self.num_clusters = np.unique(self.al.Y_train).size
        #self.clustering_solution, _ = max_correlation_dynamic_K(S, self.num_clusters, 5)
        self.clustering_solution, _ = fast_max_correlation(S, self.num_clusters, 5)
        self.num_clusters = np.unique(self.clustering_solution).size
        clust_sol, q, h = mean_field_clustering(
            S=S, K=self.num_clusters, betas=[self.al.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.clustering_solution
        )

        #print("HERE: ", self.num_clusters)
        #print(np.max(self.al.Y))
        #print(q.shape)


        pool_qs = q[len(self.al.Y_train):]
        I = scipy_entropy(pool_qs, axis=1) 
        return I


            
    
