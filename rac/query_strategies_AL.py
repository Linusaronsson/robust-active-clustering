from typing import Any
import numpy as np 
from itertools import combinations, product
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy as scipy_entropy
from scipy import sparse

from rac.correlation_clustering import mean_field_clustering
from rac.correlation_clustering import max_correlation, fast_max_correlation, max_correlation_dynamic_K, mean_field_clustering
from rac.active_learning_strategies import BADGE, EntropySampling, CoreSet, BALDDropout, EntropySamplingDropout
from rac.utils.utils import LabeledToUnlabeledDataset, CustomDataset

import torch

import scipy
import matplotlib.pyplot as plt

class QueryStrategyAL:
    def __init__(self, al):
        self.al = al

    def select_batch(self, acq_fn, batch_size):
        if acq_fn == "uniform":
            self.info_matrix = np.random.rand(len(self.al.Y_pool))
            self.info_matrix[self.al.queried_indices] = 0
        elif acq_fn == "entropy":
            self.info_matrix = self.compute_entropy()
        elif acq_fn == "cc_entropy":
            self.info_matrix = self.compute_cc_entropy()
        elif acq_fn == "entropy_mc":
            self.info_matrix = self.compute_entropy_mc()
        elif acq_fn == "bald":
            self.info_matrix = self.compute_bald()
        elif acq_fn == "cc_entropy_mc":
            self.info_matrix = self.compute_cc_entropy_mc()
        elif acq_fn == "coreset":
            #self.info_matrix = self.compute_coreset()
            X_pool_no_train = self.al.X_pool[self.al.unqueried_indices]
            Y_pool_no_train = self.al.Y_pool_queried[self.al.unqueried_indices]
            pool_dataset = CustomDataset(X_pool_no_train, Y_pool_no_train, transform=self.al.test_transform)
            train_dataset = CustomDataset(self.al.X_train, self.al.Y_train, transform=self.al.test_transform)
            es = CoreSet(train_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
            idxs = es.select(batch_size)
            return self.al.unqueried_indices[idxs]
        elif acq_fn == "badge":
            #self.info_matrix = self.compute_badge()
            #X_pool_no_train = self.X_pool[self.al.unqueried_indices]
            #Y_pool_no_train = self.Y_pool_queried[self.al.unqueried_indices]
            #pool_dataset = CustomDataset(X_pool_no_train, Y_pool_no_train, transform=self.test_transform)
            #train_dataset = CustomDataset(self.X_train, self.Y_train, transform=self.test_transform)
            X_pool_no_train = self.al.X_pool[self.al.unqueried_indices]
            Y_pool_no_train = self.al.Y_pool_queried[self.al.unqueried_indices]
            pool_dataset = CustomDataset(X_pool_no_train, Y_pool_no_train, transform=self.al.test_transform)
            train_dataset = CustomDataset(self.al.X_train, self.al.Y_train, transform=self.al.test_transform)
            es = BADGE(train_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
            idxs = es.select(batch_size)
            return self.al.unqueried_indices[idxs]
        else:
            raise ValueError("Invalid acquisition function: {}".format(acq_fn))

        if not self.al.allow_requery:
            query_counts = np.sum(self.al.queried_labels, axis=1)    
            self.info_matrix[self.al.queried_indices] = -10000 * query_counts[self.al.queried_indices]
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
        #dataset = CustomDataset(self.X_train, self.Y_train, transform=self.transform)
        #test_dataset = CustomDataset(self.X_test, self.Y_test, transform=self.test_transform)

        #@@@@@@@@@
        #pool_dataset = CustomDataset(self.al.X_pool, self.al.Y_pool_queried, transform=self.al.test_transform)
        #es = EntropySampling(pool_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
        #I = es.acquire_scores(LabeledToUnlabeledDataset(pool_dataset))
        #@@@@@@@@@@

        #max_indices = np.argmax(self.al.queried_labels, axis=1)
        #prob_all = np.zeros(self.al.queried_labels.shape)
        #prob_all[np.arange(len(max_indices)), max_indices] = 1
        #prob_train = np.zeros((self.al.Y_train.size, self.al.Y.max()+1))
        #prob_train[np.arange(self.al.Y_train.size), self.al.Y_train] = 1

        #Predict probabilities for X_pool
        prob_pool = self.al.model.predict_proba(self.al.X_pool)

        #prob_all[self.al.unqueried_indices] = prob_pool[self.al.unqueried_indices]
        I = scipy_entropy(prob_pool, axis=1)
        return I
        #return I.cpu()

    def compute_cc_entropy(self):
        print("SELECTING")
        # Probabilities for X_train (one-hot encode Y_train)
        #prob_all = scipy_softmax(100000*self.al.queried_labels, axis=1)
        #max_indices = np.argmax(self.al.queried_labels, axis=1)
        #prob_all = np.zeros(self.al.queried_labels.shape)
        #prob_all[np.arange(len(max_indices)), max_indices] = 1
        #prob_train = np.zeros((self.al.Y_train.size, self.al.Y.max()+1))
        #prob_train[np.arange(self.al.Y_train.size), self.al.Y_train] = 1

        # Predict probabilities for X_pool
        #pool_dataset = CustomDataset(self.al.X_pool, self.al.Y_pool_queried, transform=self.al.test_transform)
        #es = EntropySampling(pool_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
        #prob_pool = es.predict_prob(LabeledToUnlabeledDataset(pool_dataset))
        #prob_pool = prob_pool.cpu().numpy()

        prob_pool = self.al.model.predict_proba(self.al.X_pool)
        prob_all = prob_pool

        #prob_all[self.al.unqueried_indices] = prob_pool[self.al.unqueried_indices]

        print("hello")

        # Initialize similarity matrix
        N = prob_all.shape[0]
        S = np.zeros((N, N))
        print(S.shape)

        #queried_indices_mask = np.zeros(N, dtype=bool)
        #queried_indices_mask[self.al.queried_indices] = True

        # Case when sim_init == "t1"
#        if self.al.sim_init == "t1":
#            queried_matrix = np.outer(queried_indices_mask, queried_indices_mask)
#            equality_matrix = self.al.Y_pool_queried[:, None] == self.al.Y_pool_queried
#            # Apply queried_indices_mask to filter out unqueried indices
#            S = np.where(queried_matrix & equality_matrix, 1, -1)
#            # Set to 0 where either i or j is not in queried_indices
#            S = np.where(queried_matrix, S, 0)
#        else:
#            # Vectorize the 'else' block calculations
#            P_S_ij_plus_1 = np.dot(prob_all, prob_all.T)
#            E_S_ij_plus_1 = P_S_ij_plus_1
#            E_S_ij_minus_1 = E_S_ij_plus_1 - 1
#            E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
#            S = E_S_ij

        #Calculate expected similarity
        for i in range(N):
            for j in range(0, i):
                if i != j:
                    if self.al.sim_init_qs == "t1":
                        if i in self.al.queried_indices and j in self.al.queried_indices:
                            S[i, j] = 1 if self.al.Y_pool_queried[i] == self.al.Y_pool_queried[j] else -1
                            S[j, i] = S[i, j]
                        else:
                            S[i, j] = 0
                            S[j, i] = 0
                    else:
                        P_S_ij_plus_1 = np.sum(prob_all[i, :] * prob_all[j, :])
                        E_S_ij_plus_1 = P_S_ij_plus_1
                        E_S_ij_minus_1 = E_S_ij_plus_1 - 1
                        E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
                        S[i, j] = E_S_ij
                        S[j, i] = S[i, j]

        print("hello2")
        # Ensure diagonal is zero
        np.fill_diagonal(S, 0)

        self.num_clusters = np.unique(self.al.Y_train).size

        print("here1")
        if self.al.dynamic_K:
            self.clustering_solution, _ = max_correlation_dynamic_K(S, self.num_clusters, 5)
        else:
            self.clustering_solution, _ = max_correlation(S, self.num_clusters, 5)
        print("here2")
        self.num_clusters = np.unique(self.clustering_solution).size
        clust_sol, q, h = mean_field_clustering(
            S=S, K=self.num_clusters, betas=[self.al.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.clustering_solution
        )
        print("here3")

        #print("HERE: ", self.num_clusters)
        #print(np.max(self.al.Y))
        #print(q.shape)

        #p = prob_pool
        ## Calculate entropy for each data point in p and q
        #entropy_p = scipy_entropy(p, axis=1)
        #entropy_q = scipy_entropy(q, axis=1)

        ## Rank data points by entropy
        #rank_p = np.argsort(entropy_p)
        ##rank_q = np.argsort(entropy_q)
        #N_large = len(rank_p)
        ## Visualization for large number of data points
        ## Adjust visualization for large number of data points to display a subset of data point numbers on the x-axis
        #plt.figure(figsize=(12, 7))
        #plt.plot(entropy_p[rank_p], label='Entropy of p', marker='', linestyle='-', linewidth=1)
        #plt.plot(entropy_q[rank_p], label='Entropy of q', marker='', linestyle='-', linewidth=1)
        #plt.title("Iteration " + str(self.al.ii))
        #plt.xlabel('Data Point')
        #plt.ylabel('Entropy')
        #plt.legend()
        #plt.grid(True)

        ## Choose a subset of data point numbers for the x-axis
        #x_ticks = np.arange(0, N_large, N_large // 10)  # Display every 10th of the total number of data points
        #plt.xticks(x_ticks)

        #file_path = "plots/entropy_comparison + " + str(self.al.ii) + ".png"
        #plt.savefig(file_path)

        I = scipy_entropy(q, axis=1) 
        return I
    
    def compute_entropy_mc(self):
        pool_dataset = CustomDataset(self.al.X_pool, self.al.Y_pool_queried, transform=self.al.test_transform)
        es = EntropySamplingDropout(pool_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
        I = es.acquire_scores(LabeledToUnlabeledDataset(pool_dataset))
        return I.cpu()

    def compute_bald(self):
        pool_dataset = CustomDataset(self.al.X_pool, self.al.Y_pool_queried, transform=self.al.test_transform)
        es = BALDDropout(pool_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
        I = es.acquire_scores(LabeledToUnlabeledDataset(pool_dataset))
        return I.cpu()

    def compute_cc_entropy_mc(self):
                # Probabilities for X_train (one-hot encode Y_train)
        #prob_all = scipy_softmax(100000*self.al.queried_labels, axis=1)
        max_indices = np.argmax(self.al.queried_labels, axis=1)
        prob_all = np.zeros(self.al.queried_labels.shape)
        prob_all[np.arange(len(max_indices)), max_indices] = 1
        #prob_train = np.zeros((self.al.Y_train.size, self.al.Y.max()+1))
        #prob_train[np.arange(self.al.Y_train.size), self.al.Y_train] = 1

        # Predict probabilities for X_pool
        #prob_pool = self.al.model.predict_proba(self.al.X_pool)
        pool_dataset = CustomDataset(self.al.X_pool, self.al.Y_pool_queried, transform=self.al.test_transform)
        es = EntropySamplingDropout(pool_dataset, LabeledToUnlabeledDataset(pool_dataset), self.al.model, self.al.n_classes)
        #I = es.acquire_scores(pool_dataset)
        prob_pool = es.predict_prob_dropout(LabeledToUnlabeledDataset(pool_dataset), 10)
        prob_pool = prob_pool.cpu().numpy()
        #log_probs = torch.log(probs)
        #U = -(probs*log_probs).sum(1)

        prob_all[self.al.unqueried_indices] = prob_pool[self.al.unqueried_indices]

        # Initialize similarity matrix
        N = prob_all.shape[0]
        S = np.zeros((N, N))

        # Calculate expected similarity
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.al.sim_init == "t1":
                        if i in self.al.queried_indices and j in self.al.queried_indices:
                            S[i, j] = 1 if self.al.Y_pool_queried[i] == self.al.Y_pool_queried[j] else -1
                            S[j, i] = S[i, j]
                        else:
                            S[i, j] = 0
                            S[j, i] = 0
                    else:
                        P_S_ij_plus_1 = np.sum(prob_all[i, :] * prob_all[j, :])
                        E_S_ij_plus_1 = P_S_ij_plus_1
                        E_S_ij_minus_1 = E_S_ij_plus_1 - 1
                        E_S_ij = P_S_ij_plus_1 * E_S_ij_plus_1 + (1 - P_S_ij_plus_1) * E_S_ij_minus_1
                        S[i, j] = E_S_ij
                        S[j, i] = S[i, j]

        # Ensure diagonal is zero
        np.fill_diagonal(S, 0)

        self.num_clusters = np.unique(self.al.Y_train).size

        if self.al.dynamic_K:
            self.clustering_solution, _ = max_correlation_dynamic_K(S, self.num_clusters, 5)
        else:
            self.clustering_solution, _ = max_correlation(S, self.num_clusters, 5)
        self.num_clusters = np.unique(self.clustering_solution).size
        clust_sol, q, h = mean_field_clustering(
            S=S, K=self.num_clusters, betas=[self.al.mean_field_beta], max_iter=100, tol=1e-10, 
            predicted_labels=self.clustering_solution
        )

        #print("HERE: ", self.num_clusters)
        #print(np.max(self.al.Y))
        #print(q.shape)

        #p = prob_pool
        ## Calculate entropy for each data point in p and q
        #entropy_p = scipy_entropy(p, axis=1)
        #entropy_q = scipy_entropy(q, axis=1)

        ## Rank data points by entropy
        #rank_p = np.argsort(entropy_p)
        ##rank_q = np.argsort(entropy_q)
        #N_large = len(rank_p)
        ## Visualization for large number of data points
        ## Adjust visualization for large number of data points to display a subset of data point numbers on the x-axis
        #plt.figure(figsize=(12, 7))
        #plt.plot(entropy_p[rank_p], label='Entropy of p', marker='', linestyle='-', linewidth=1)
        #plt.plot(entropy_q[rank_p], label='Entropy of q', marker='', linestyle='-', linewidth=1)
        #plt.title("Iteration " + str(self.al.ii))
        #plt.xlabel('Data Point')
        #plt.ylabel('Entropy')
        #plt.legend()
        #plt.grid(True)

        ## Choose a subset of data point numbers for the x-axis
        #x_ticks = np.arange(0, N_large, N_large // 10)  # Display every 10th of the total number of data points
        #plt.xticks(x_ticks)

        #file_path = "plots/entropy_comparison + " + str(self.al.ii) + ".png"
        #plt.savefig(file_path)

        I = scipy_entropy(q, axis=1) 
        return I

    
