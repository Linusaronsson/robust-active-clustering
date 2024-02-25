from typing import Any
import numpy as np 
from itertools import combinations, product
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy as scipy_entropy
from scipy import sparse
from rac.correlation_clustering import mean_field_clustering
from rac.correlation_clustering import max_correlation, max_correlation_dynamic_K, mean_field_clustering
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
        return self.al.pool_indices[top_B_indices]

    def compute_entropy(self):
        probs = self.al.model.predict_proba(self.al.X_pool)
        return scipy_entropy(probs.T)

    def compute_cc_entropy(self):
        print("ASDIASJD")
        proba_pool = self.al.model.predict_proba(self.al.X_pool)
        entropy_pool = np.array([scipy_entropy(proba) for proba in proba_pool])
        entropy_pool_normalized = (entropy_pool - entropy_pool.min()) / (entropy_pool.max() - entropy_pool.min())
        Y_pool = np.argmax(proba_pool, axis=1)
        Y_all = np.concatenate([self.al.Y_train, Y_pool])
        entropy_all = np.concatenate([np.zeros(len(self.al.Y_train)), entropy_pool_normalized])
        N = len(Y_all)
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                base_similarity = 1 if Y_all[i] == Y_all[j] else -1
                # Adjust the similarity less if both are certain (low entropy), more if uncertain (high entropy)
                entropy_adjustment = 1 - max(entropy_all[i], entropy_all[j])  # Use max to consider the most uncertain in the pair
                adjusted_similarity = base_similarity * entropy_adjustment
                S[i, j] = S[j, i] = adjusted_similarity
        np.fill_diagonal(S, 0)

        self.num_clusters = np.unique(self.al.Y_train).size
        self.clustering_solution, _ = max_correlation_dynamic_K(S, self.num_clusters, 5)
        self.num_clusters = np.unique(self.clustering_solution).size
        clust_sol, q, h = mean_field_clustering(
            S=S, K=self.num_clusters, betas=[self.al.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.clustering_solution
        )


        pool_qs = q[len(self.al.Y_train):]
        I = scipy_entropy(pool_qs, axis=1) 
        return I


            
    
