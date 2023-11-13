import time
import math

#from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, MPCKMeans, COPKMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, f1_score
import itertools
import numpy as np
#import cupy as np
from scipy.spatial import distance
#from sklearn.cluster import KMeans
#from scipy.stats import entropy as scipy_entropy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from torchvision import transforms
from scipy.stats import entropy as scipy_entropy
from scipy.special import softmax

from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier

from rac.pred_models import ACCNet, CustomTensorDataset

from rac.experiment_data import ExperimentData
from rac.correlation_clustering import max_correlation, max_correlation_dynamic_K, mean_field_clustering
from rac.query_strategies import QueryStrategy

import warnings
warnings.filterwarnings("once") 


class ActiveClustering:
    def __init__(self, X, Y, repeat_id, initial_clustering_solution=None, **kwargs):
        for key, value in kwargs.items():
            self.__dict__.update(kwargs[key])

        self.X, self.Y = X, Y
        self.repeat_id = repeat_id
        self.initial_clustering_solution = initial_clustering_solution
        self.ac_data = ExperimentData(Y, repeat_id, **kwargs)
        self.qs = QueryStrategy(self)

        self.clustering = None
        self.clustering_solution = None
        self.N = len(self.Y)
        self.n_edges = (self.N*(self.N-1))/2
        #self.queried_edges = set()

        if self.num_feedback < 1:
            self.query_size = math.ceil(self.n_edges * self.num_feedback)
        else:
            self.query_size = self.num_feedback

        #self.random = np.random.RandomState(self.repeat_id+self.seed+317421)
        np.random.seed(self.repeat_id+self.seed+317421)
        self.input_dim = self.X.shape[1]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = None

        if hasattr(self, "tau"):
            if self.tau == -1:
                self.tau = np.inf
            else:
                self.tau = self.tau

        if not hasattr(self, "regular_noise"):
            self.regular_noise = True

        if not hasattr(self, "clustering_alg"):
            self.clustering_alg = "CC"

        if not hasattr(self, "predict_sims"):
            self.predict_sims = False
        
        if not hasattr(self, "infer_sims"):
            self.infer_sims = False

        if not hasattr(self, "infer_sims2"):
            self.infer_sims2 = False

        if not hasattr(self, "infer_sims3"):
            self.infer_sims3 = False

        if not hasattr(self, "binary_noise"):
            self.binary_noise = False

        if not hasattr(self, "init_noise_level"):
            self.init_noise_level = 0

    def run_AL_procedure(self):
        self.start_time = time.time()
        self.initialize_ac_procedure()
        self.store_experiment_data(initial=True)
        perfect_rand_count = 0
        early_stopping_count = 0
        old_rand = adjusted_rand_score(self.Y, self.clustering_solution)
        total_queries = 0

        if self.acq_fn in ["QECC"]:
            stopping_criteria = self.query_size * 250
        if self.acq_fn in ["nCOBRAS", "COBRAS"]:
            stopping_criteria = self.query_size * 200
        else:
            stopping_criteria = self.n_edges

        #early_stopping = (self.n_edges/self.query_size)*1
        early_stopping = (self.n_edges/self.query_size)*0.55
        #ii = 0

        ii = 1
        while total_queries < stopping_criteria: 
        #while True:
            self.start = time.time()
            ii += 1
            if self.acq_fn == "QECC":
                self.clustering = self.QECC_heur(self.query_size * ii)
                self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
                self.clustering_solution = self.clustering_solution_from_clustering(self.clustering)
            elif self.acq_fn == "nCOBRAS":
                #if (ii * self.num_feedback * 100) % (0.006 * 100) == 0: 
                if (ii % 1) == 0: 
                    noise_level_cobras = (self.noise_level/2) + (self.persistent_noise_level/2)
                    if self.noise_level == 0 and self.persistent_noise_level == 0:
                        noise_level_cobras = 0.05
                    try:
                        print("SIZE: ", self.query_size * ii)
                        print("noise prob: ", noise_level_cobras)
                        noisy_querier = ProbabilisticNoisyQuerier(None, self.Y, (self.noise_level/2), self.query_size * ii, random_seed=self.repeat_id+self.seed+3892, sim_matrix=self.ground_truth_pairwise_similarities_noisy)
                        clusterer = COBRAS(correct_noise=True, noise_probability=noise_level_cobras, certainty_threshold=0.95, minimum_approximation_order=2, maximum_approximation_order=3, seed=self.repeat_id+self.seed+4873)
                        all_clusters, runtimes, *_ = clusterer.fit(self.X, -1, None, noisy_querier)
                        print("DONE????")
                    except Exception:
                        print("nCOBRAS failed")
                        break
                    print("DID IT FAIL???")
                    # only store the first two return values
                    self.clustering_solution = all_clusters[-1]
                    self.clustering_solution = np.array(self.clustering_solution, dtype=np.uint32)
                    self.clustering, num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
                    self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
            elif self.acq_fn == "COBRAS":
                #if (ii * self.num_feedback * 100) % (0.01 * 100) == 0: 
                if (ii % 10) == 0: 
                    try:
                        noisy_querier = ProbabilisticNoisyQuerier(None, self.Y, (self.noise_level/2), self.query_size * ii, random_seed=self.repeat_id+self.seed+3892, sim_matrix=self.ground_truth_pairwise_similarities_noisy)
                        clusterer = COBRAS(correct_noise=False, seed=self.repeat_id+self.seed+4873)
                        all_clusters, runtimes, *_ = clusterer.fit(self.X, -1, None, noisy_querier)
                    except Exception:
                        print("COBRAS failed")
                        break
                    # only store the first two return values
                    self.clustering_solution = all_clusters[-1]
                    self.clustering_solution = np.array(self.clustering_solution, dtype=np.uint32)
                    self.clustering, num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
                    self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
            else:
                self.edges = self.qs.select_batch(
                    acq_fn=self.acq_fn,
                    batch_size=self.query_size
                )

                #if len(objects) == 1:
                    #raise ValueError("Singleton cluster in run_AL_procedure(...)")

                if len(self.edges) != self.query_size:
                    raise ValueError("Num queried {} not equal to query size {}!!".format(len(self.edges[0]), self.query_size))
                
                for ind1, ind2 in self.edges:
                    self.update_similarity(ind1, ind2)

                if self.predict_sims:
                    self.predict_similarities()

                if self.infer_sims:
                    self.transitive_closure()

                time_nn = time.time()
                self.update_clustering() 
                print("time_nn: ", time.time() - time_nn)

                for i in range(self.N):
                    for j in range(i):
                        if self.violates_clustering(i, j):
                            self.violations[i, j] = np.abs(self.pairwise_similarities[i, j])
                            self.violations[j, i] = np.abs(self.pairwise_similarities[j, i])
                        else:
                            self.violations[i, j] = 0
                            self.violations[j, i] = 0

            self.store_experiment_data()
            total_queries += self.query_size

            self.total_time_elapsed = time.time() - self.start_time

            #num_hours = self.total_time_elapsed / 3600
            #if num_hours > 60:
            #    break

            if self.verbose:
                print("iteration: ", ii)
                print("prop_queried: ", total_queries/self.n_edges)
                print("feedback_freq max: ", np.max(self.feedback_freq))
                print("rand score: ", adjusted_rand_score(self.Y, self.clustering_solution))
                print("time: ", time.time()-self.start)
                if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                    print("num queries: ", len(self.edges))
                print("num clusters: ", self.num_clusters)
                #print("-----------------")
                
            #ii += 1
            current_rand = adjusted_rand_score(self.Y, self.clustering_solution)

            #if current_rand > 0.98:
                #break
            
            if current_rand == old_rand:
                early_stopping_count += 1
            else:
                early_stopping_count = 0
            
            if current_rand >= 0.99:
                perfect_rand_count += 1
            else:
                perfect_rand_count = 0
            
            old_rand = current_rand

            if early_stopping_count > early_stopping or perfect_rand_count > 3:
                break
        
        if self.save_matrix_data:
            self.ac_data.feedback_freq = self.feedback_freq
            self.ac_data.pairwise_similarities = self.pairwise_similarities
        return self.ac_data

    def QECC_heur(self, num_queries):
        nodes = set(np.arange(0, self.N).tolist())
        vertices = nodes.copy()
        query_budget = num_queries
        clusters = []
        while len(vertices) > 1 and query_budget > (len(vertices) - 1):
            unique_pairs = list(itertools.combinations(vertices, 2))
            np.random.shuffle(unique_pairs)
            for (ind1, ind2) in unique_pairs:
                query = self.get_similarity(ind1, ind2)
                query_budget -= 1
                if query >= 0:
                    break

            cluster = {ind1, ind2}
            for nd in vertices - {ind1, ind2}:
                if self.get_similarity(ind2, nd) > 0:
                    cluster.add(nd)
            clusters.append(list(cluster))
            query_budget = query_budget - len(vertices) + 2
            vertices = vertices - cluster
        for nd in vertices:
            clusters.append([nd])
        return clusters

    def QECC(self, num_queries):
        nodes = set(np.arange(0, self.N).tolist())
        vertices = nodes.copy()
        query_budget = num_queries
        clusters = []
        while len(vertices) > 1 and query_budget > (len(vertices) - 1):
            ind1 = np.random.choice(len(vertices), 1, replace=False)[0]
            cluster = {ind1}
            for nd in vertices - {ind1}:
                if self.get_similarity(ind1, nd) > 0:
                    cluster.add(nd)
            clusters.append(list(cluster))
            query_budget = query_budget - len(vertices) + 1
            vertices = vertices - cluster
        for nd in vertices:
            clusters.append([nd])
        return clusters

    def store_experiment_data(self, initial=False):
        time_now = time.time() 
        if self.dataset == "synthetic":
            lower_triangle_indices = np.tril_indices(self.N, -1)
            estimated_sims = self.pairwise_similarities[lower_triangle_indices]
            estimated_sims_binary = estimated_sims.copy()
            estimated_sims_binary[np.where(estimated_sims >= 0)] = 1
            estimated_sims_binary[np.where(estimated_sims < 0)] = -1
            true_sims = self.ground_truth_pairwise_similarities[lower_triangle_indices]

            self.ac_data.accuracy.append(accuracy_score(true_sims, estimated_sims_binary))
            self.ac_data.precision.append(precision_score(true_sims, estimated_sims_binary, pos_label=1))
            self.ac_data.recall.append(recall_score(true_sims, estimated_sims_binary, pos_label=1))
            self.ac_data.precision_neg.append(precision_score(true_sims, estimated_sims_binary, pos_label=-1))
            self.ac_data.recall_neg.append(recall_score(true_sims, estimated_sims_binary, pos_label=-1))
            self.ac_data.f1_score.append(f1_score(true_sims, estimated_sims_binary, pos_label=1, average="binary"))
            self.ac_data.f1_score_weighted.append(f1_score(true_sims, estimated_sims_binary, pos_label=1, average="weighted"))

            estimated_sims_binary = self.sim_matrix_from_clustering(self.clustering)[lower_triangle_indices]
            estimated_sims_binary[np.where(estimated_sims_binary >= 0)] = 1
            estimated_sims_binary[np.where(estimated_sims_binary < 0)] = -1

            self.ac_data.accuracy_clustering.append(accuracy_score(true_sims, estimated_sims_binary))
            self.ac_data.precision_clustering.append(precision_score(true_sims, estimated_sims_binary, pos_label=1))
            self.ac_data.recall_clustering.append(recall_score(true_sims, estimated_sims_binary, pos_label=1))
            self.ac_data.precision_neg_clustering.append(precision_score(true_sims, estimated_sims_binary, pos_label=-1))
            self.ac_data.recall_neg_clustering.append(recall_score(true_sims, estimated_sims_binary, pos_label=-1))
            self.ac_data.f1_score_clustering.append(f1_score(true_sims, estimated_sims_binary, pos_label=1, average="binary"))
            self.ac_data.f1_score_weighted_clustering.append(f1_score(true_sims, estimated_sims_binary, pos_label=1, average="weighted"))

        self.ac_data.rand.append(adjusted_rand_score(self.Y, self.clustering_solution))
        self.ac_data.ami.append(adjusted_mutual_info_score(self.Y, self.clustering_solution))
        self.ac_data.v_measure.append(v_measure_score(self.Y, self.clustering_solution))
        self.ac_data.num_clusters.append(self.num_clusters)
        
        if initial:
            self.ac_data.Y = self.Y
            self.ac_data.num_queries.append(0)
            self.ac_data.time.append(0.0)
            num_pos = 0
            num_neg = 0
            num_pos_ground_truth = 0
            num_neg_ground_truth = 0
            self.ac_data.num_pos.append(num_pos)
            self.ac_data.num_neg.append(num_neg)
            self.ac_data.num_pos_ground_truth.append(num_pos_ground_truth)
            self.ac_data.num_neg_ground_truth.append(num_neg_ground_truth)
        else:
            self.ac_data.num_queries.append(self.query_size) 
            print("TIME AFTER RUN: ", time_now - self.start)
            self.ac_data.time.append(time_now - self.start)
            if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                inds = self.edges[:, 0], self.edges[:, 1]
                num_pos = np.sum(self.pairwise_similarities[inds] >= 0, axis=0)
                num_neg = np.sum(self.pairwise_similarities[inds] < 0, axis=0)
                num_pos_ground_truth = np.sum(self.ground_truth_pairwise_similarities[inds] >= 0, axis=0) 
                num_neg_ground_truth = np.sum(self.pairwise_similarities[inds] < 0, axis=0)
                self.ac_data.num_pos.append(num_pos)
                self.ac_data.num_neg.append(num_neg)
                self.ac_data.num_pos_ground_truth.append(num_pos_ground_truth)
                self.ac_data.num_neg_ground_truth.append(num_neg_ground_truth)

        time_after_init = time.time() - time_now
        print("TIME AFTER INIT: ", time_after_init)

    def in_same_cluster(self, o1, o2):
        return self.clustering_solution[o1] == self.clustering_solution[o2]

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] >= 0) or \
                (self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] < 0)

    def initialize_ac_procedure(self):
        self.construct_ground_truth_sim_matrix()
        self.construct_initial_sim_matrix()

        if self.init_noise_level > 0 and self.sim_init_type == "custom":
            total_noisy_edges = int(self.n_edges * self.init_noise_level)
            edges, objects = self.qs.select_batch("unif", "pairs", total_noisy_edges)

            for ind1, ind2 in edges:
                self.pairwise_similarities[ind1, ind2] *= -1
                self.pairwise_similarities[ind2, ind1] *= -1

        self.init_persistent_noise()
        self.feedback_freq = np.zeros((self.N, self.N)) + 1
        self.violations = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i):
                if self.violates_clustering(i, j):
                    self.violations[i, j] = np.abs(self.pairwise_similarities[i, j])
                    self.violations[j, i] = np.abs(self.pairwise_similarities[j, i])

    def construct_ground_truth_sim_matrix(self):
        num_clusters = np.max(self.Y) + 1
        self.num_clusters_ground_truth = num_clusters
        self.ground_truth_clustering = [[] for _ in range(num_clusters)]

        for i in range(len(self.Y)):
            self.ground_truth_clustering[self.Y[i]].append(i)

        self.ground_truth_pairwise_similarities = -1*np.ones((self.N, self.N))
        #self.ground_truth_pairwise_similarities = np.zeros((self.N, self.N))
        for cind in self.ground_truth_clustering:
            self.ground_truth_pairwise_similarities[np.ix_(cind, cind)] = 1
        self.ground_truth_pairwise_similarities_noisy = self.ground_truth_pairwise_similarities.copy()

    def init_persistent_noise(self):
        self.edges_persistent = []
        self.total_flips = int(self.n_edges * self.persistent_noise_level)
        if self.persistent_noise_level == 0:
            return
        edges, objects = self.qs.select_batch("unif", "pairs", self.total_flips)

        for ind1, ind2 in edges:
            if self.binary_noise:
                if np.random.uniform(-1.0, 1.0) > 0:
                    self.ground_truth_pairwise_similarities_noisy[ind1, ind2] *= -1
                    self.ground_truth_pairwise_similarities_noisy[ind2, ind1] *= -1
            else:
                noisy_val = np.random.uniform(0.15, 0.5)
                if np.random.uniform(-1.0, 1.0) > 0:
                    self.ground_truth_pairwise_similarities_noisy[ind1, ind2] = -noisy_val
                    self.ground_truth_pairwise_similarities_noisy[ind2, ind1] = -noisy_val
                else:
                    self.ground_truth_pairwise_similarities_noisy[ind1, ind2] = noisy_val
                    self.ground_truth_pairwise_similarities_noisy[ind2, ind1] = noisy_val


    def clustering_from_clustering_solution(self, clustering_solution):
        num_clusters = np.max(clustering_solution) + 1
        clustering = [[] for _ in range(num_clusters)]
        for i in range(len(clustering_solution)):
            clustering[clustering_solution[i]].append(i)
        return clustering, num_clusters

    def find_cluster(self, d, cluster_indices):
        i = 0
        for ds in cluster_indices:
            if d in ds:
                return i
            i += 1
        print("FIND CLUSTER UNREACHABLE")
        return -1

    def clustering_solution_from_clustering(self, clustering):
        clustering_solution = np.zeros(self.N, dtype=np.uint32)
        for u in range(self.N):
            clustering_solution[u] = self.find_cluster(u, clustering)
        return clustering_solution

    def sim_matrix_from_clustering(self, clustering):
        pairwise_similarities = -self.sim_init*np.ones((self.N, self.N))
        for cind in clustering:
            pairwise_similarities[np.ix_(cind, cind)] = self.sim_init
        return pairwise_similarities

    def construct_initial_sim_matrix(self):
        if self.acq_fn in ["QECC", "COBRAS", "nCOBRAS"]:
            self.sim_init_type = "random_clustering"

        if self.sim_init_type == "zeros":
            self.pairwise_similarities = np.zeros((self.N, self.N))
            self.num_clusters = 1
            self.update_clustering() 
        elif self.sim_init_type == "uniform_random":
            self.pairwise_similarities = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.num_clusters = 1
            self.update_clustering() 
        elif self.sim_init_type == "uniform_random2":
            self.pairwise_similarities = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.pairwise_similarities[np.where(self.pairwise_similarities >= 0)] = self.sim_init
            self.pairwise_similarities[np.where(self.pairwise_similarities < 0)] = -self.sim_init
            self.num_clusters = 1
            self.update_clustering() 
        elif self.sim_init_type == "uniform_random_clustering":
            self.pairwise_similarities = np.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.num_clusters = 1
            self.update_clustering() 
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        elif self.sim_init_type == "inverse_dist":
            D = distance.cdist(self.X, self.X, 'euclidean')
            sim_matrix = np.max(D) - D + np.min(D)
            self.pairwise_similarities = self.sim_init * (2 * sim_matrix - np.max(sim_matrix) -  np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
            np.fill_diagonal(self.pairwise_similarities, 0.0)
            self.num_clusters = 1
            self.update_clustering() 
        elif self.sim_init_type == "custom":
            self.clustering_solution = np.array(self.initial_clustering_solution)
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
            self.update_clustering() 
        elif self.sim_init_type == "random_clustering":
            objects = np.arange(self.N)
            np.random.shuffle(objects)
            k = self.K_init
            cluster_size = int(len(objects) / k)
            cluster_sizes = [cluster_size for _ in range(k)]
            split = np.split(objects, np.cumsum(cluster_sizes))
            clustering = [clust.tolist() for clust in split]
            if len(split) > k:
                clustering[0] += clustering[-1]
                del clustering[-1]
            i = 0
            self.clustering_solution = np.zeros(self.N, dtype=np.uint32)
            for clust in clustering:
                self.clustering_solution[clust] = i
                i += 1
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
            self.update_clustering() 
        else:
            raise ValueError("Invalid sim init type in construct_initial_sim_matrix(...)")
        np.fill_diagonal(self.pairwise_similarities, 0.0)

    # Input: list of objects to update clustering w.r.t. (subset of all objects)
    def update_clustering(self):
        if self.clustering_alg == "CC":
            self.clustering_solution, _ = max_correlation_dynamic_K(self.pairwise_similarities, self.num_clusters, 5)
            self.clustering = self.clustering_from_clustering_solution(self.clustering_solution)[0]
            self.num_clusters = len(self.clustering)
        elif self.clustering_alg == "MPCKMeans":
            num_classes = len(np.unique(self.Y))
            clusterer = MPCKMeans(n_clusters=num_classes, max_iter=10, w=1)
            lower_triangle_indices = np.tril_indices(self.N, -1)
            ind_pos = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] >= 0))[0]
            ind_neg = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] < 0))[0]
            ind1_pos, ind2_pos = lower_triangle_indices[0][ind_pos], lower_triangle_indices[1][ind_pos]
            ind1_neg, ind2_neg = lower_triangle_indices[0][ind_neg], lower_triangle_indices[1][ind_neg]
            ml = [(i1, i2) for i1, i2 in zip(ind1_pos, ind2_pos)]
            cl = [(i1, i2) for i1, i2 in zip(ind1_neg, ind2_neg)]
            clusterer.fit(X=self.X, y=self.Y, ml=ml, cl=cl)
            self.num_clusters = num_classes
            self.clustering_solution = clusterer.labels_
            self.clustering = self.clustering_from_clustering_solution(self.clustering_solution)[0]
        elif self.clustering_alg == "mean_field":
            self.clustering_solution, self.q, self.h = mean_field_clustering(self.pairwise_similarities, self.num_clusters, betas=[self.mean_field_beta], max_iter=100, tol=1e-10, noise_level=0.0) 
            self.clustering = self.clustering_from_clustering_solution(self.clustering_solution)[0]
            self.num_clusters = len(self.clustering)
        else:
            raise ValueError("Invalid clustering algorithm in update_clustering(...)")
    
    def update_similarity(self, ind1, ind2, custom_query=None, update_freq=True):
        #if update_freq:
        self.feedback_freq[ind1, ind2] += 1
        self.feedback_freq[ind2, ind1] += 1

        feedback_frequency = self.feedback_freq[ind1, ind2]
        similarity = self.pairwise_similarities[ind1, ind2]

        if custom_query is not None:
            query = custom_query
        elif np.random.rand() <= self.noise_level:
            if self.binary_noise:
                if np.random.uniform(-1.0, 1.0) > 0:
                    query = -self.ground_truth_pairwise_similarities_noisy[ind1, ind2]
                else:
                    query = self.ground_truth_pairwise_similarities_noisy[ind1, ind2]
            else:
                #if self.regular_noise:
                    #query = np.random.uniform(-1.0, 1.0)
                #else:
                noisy_val = np.random.uniform(0.15, 0.5)
                if np.random.uniform(-1.0, 1.0) > 0:
                    query = -noisy_val
                else:
                    query = noisy_val
        else:
            query = self.ground_truth_pairwise_similarities_noisy[ind1, ind2]

        if self.running_avg:
            self.pairwise_similarities[ind1, ind2] = ((feedback_frequency-1) * similarity + query)/(feedback_frequency)
            self.pairwise_similarities[ind2, ind1] = ((feedback_frequency-1) * similarity + query)/(feedback_frequency)
        else:
            self.pairwise_similarities[ind1, ind2] = query
            self.pairwise_similarities[ind2, ind1] = query

    def get_similarity(self, ind1, ind2):
        if np.random.rand() <= self.noise_level:
            return np.random.uniform(-1.0, 1.0)
        else:
            return self.ground_truth_pairwise_similarities_noisy[ind1, ind2]

    def predict_similarities(self): 
        #input_dim = self.X.shape[1]
        self.retrain_net = False

        print("HERE", self.X.shape)
        self.X = np.float32(self.X)
        if self.retrain_net or self.net is None:
            self.net = ACCNet(base_net=self.base_net, siamese=self.siamese, input_dim=self.X.shape[1], p=0.0).to(self.device)

        lower_triangle_indices = np.tril_indices(self.N, -1) # -1 gives lower triangle without diagonal (0 includes diagonal)
        #cond1 = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] > 0.5) & (self.edges_predicted[lower_triangle_indices] == False))[0]
        #cond2 = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] < -0.5) & (self.edges_predicted[lower_triangle_indices] == False))[0]
        cond_pos = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] > 0.5))[0]
        cond_neg = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] < -0.5))[0]

        print("cond_pos len ", len(cond_pos))
        print("cond_neg len ", len(cond_neg))
        if len(cond_pos) < 10 or len(cond_neg) < 10:
            return

        print("predicting sims...")
        print("cond_pos shape ", cond_pos.shape)
        print("cond_neg shape ", cond_neg.shape)

        #ind_pos = self.random.choice(cond_pos, np.min([len(cond_pos), len(cond_neg), 10000]))
        #ind_neg = self.random.choice(cond_neg, len(ind_pos))
        #print(len(indices1), len(indices2))

        ind_pos = np.random.choice(cond_pos, np.min([len(cond_pos), 4000]), replace=False)
        ind_neg = np.random.choice(cond_neg, np.min([len(cond_neg), 4000]), replace=False)

        if len(ind_pos) < len(ind_neg):
            indices = np.concatenate([ind_neg, ind_pos])
        else:
            indices = np.concatenate([ind_pos, ind_neg])
        ind1, ind2 = lower_triangle_indices[0][indices], lower_triangle_indices[1][indices]
        print("HERE: ", self.X.shape)
        x1 = self.X[ind1]
        x2 = self.X[ind2]
        #dataset = np.concatenate((x1, x2), axis=1)
        labels = self.pairwise_similarities[ind1, ind2]
        lab1 = np.where(labels >= 0)
        lab2 = np.where(labels < 0)
        labels[lab1] = 1.0
        labels[lab2] = 0.0
        #print("Dataset shape: ", dataset.shape)
        print("x1 shape: ", x1.shape)
        print("x2 shape: ", x2.shape)

        cifar_training_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        if self.base_net == "three_layer_net" or self.base_net == "two_layer_net":
            train_dataset = CustomTensorDataset(x1, x2, torch.Tensor(labels), train=True, transform=None)
        else:
            train_dataset = CustomTensorDataset(x1, x2, torch.Tensor(labels), train=True, transform=cifar_training_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
        #criterion = nn.MSELoss()
        #criterion = nn.SmoothL1Loss()
        if self.criterion == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif self.criterion == "mse":
            criterion = nn.MSELoss()
        elif self.criterion == "smoothl1":
            criterion = nn.SmoothL1Loss()
        elif self.criterion == "contrastive":
            #criterion = ContrastiveLoss()
            pass
        else:
            raise ValueError("Invalid criterion")
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0005, momentum=0.9)
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=0.0001)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=0.0001)
        print("training...")
        print(len(train_dataset))
        self.net.train()
        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            step = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                x1, x2, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(x1, x2)
                outputs = outputs.reshape((outputs.shape[0]))
                #labels = labels.reshape((labels.shape[0], 1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                step += 1

            print("loss: ", running_loss/step)
            step = 0
            running_loss = 0.0

        print("predicting")
        num_preds = self.query_size * 25
        edges, objects = self.qs.select_batch("freq", "pairs", num_preds)
        ind1, ind2 = edges[:, 0], edges[:, 1]

        dat1 = self.X[ind1]
        dat2 = self.X[ind2]
        
        cifar_test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        if self.base_net == "three_layer_net" or self.base_net == "two_layer_net":
            dat_all = CustomTensorDataset(dat1, dat2, transform=None)
        else:
            dat_all = CustomTensorDataset(dat1, dat2, transform=cifar_test_transform)

        test_loader = torch.utils.data.DataLoader(dat_all, shuffle=False, batch_size=1024)
        preds = []
        self.net.eval()
        for i, data in enumerate(test_loader, 0):
            input1, input2 = data[0].to(self.device), data[1].to(self.device)
            pred = self.net(input1, input2)
            #pred = torch.clip(pred, min=-1, max=1)
            pred = nn.Sigmoid()(pred)
            preds.extend(pred[:, 0].tolist())
        print("NUM PREDS: ", num_preds)
        countt = 0
        
        pred_binary = []
        true_binary = []
        pred_binary_ent = []
        true_binary_ent = []
        for i1, i2, pred in zip(ind1, ind2, preds):
            prob = [1-pred, pred]
            entropy = scipy_entropy(prob)
            #print("ENTROPY: ", entropy)
            if pred >= 0.5:
                pred_binary.append(1)
            else:
                pred_binary.append(0)
            true_binary.append(self.ground_truth_pairwise_similarities[i1, i2])

            if entropy > 0.0001:
                continue
            countt += 1

            #pred = (pred - 0.5) * 2
            if pred >= 0.5:
                pred = 0.25
                pred_binary_ent.append(1)
            else:
                pred_binary_ent.append(0)
                pred = -0.25

            #self.pairwise_similarities[i1, i2] = pred
            #self.pairwise_similarities[i2, i1] = pred
            true_binary_ent.append(self.ground_truth_pairwise_similarities[i1, i2])
            self.update_similarity(i1, i2, custom_query=pred, update_freq=False)
        
        pred_binary = np.array(pred_binary)
        true_binary = np.array(true_binary)
        true_binary[np.where(true_binary >= 0)[0]] = 1
        true_binary[np.where(true_binary < 0)[0]] = 0

        pred_binary_ent = np.array(pred_binary_ent)
        true_binary_ent = np.array(true_binary_ent)
        true_binary_ent[np.where(true_binary_ent >= 0)[0]] = 1
        true_binary_ent[np.where(true_binary_ent < 0)[0]] = 0

        if len(true_binary) > 0:
            print("Accuracy: ", accuracy_score(true_binary, pred_binary))
            print("Recall: ", recall_score(true_binary, pred_binary))
            print("Precision: ", precision_score(true_binary, pred_binary))

        if len(true_binary_ent) > 0:
            print("Accuracy ent: ", accuracy_score(true_binary_ent, pred_binary_ent))
            print("Recall ent: ", recall_score(true_binary_ent, pred_binary_ent))
            print("Precision ent: ", precision_score(true_binary_ent, pred_binary_ent))
        print("COUNTT: ", countt)

    def transitive_closure(self):
        print("INFERRING @@@@")
        n = self.pairwise_similarities.shape[0]
        ml = []  # Must-link pairs
        cl = []  # Cannot-link pairs

        # Extracting must-link and cannot-link pairs from the similarity matrix
        for i in range(n):
            for j in range(i):
                if self.pairwise_similarities[i][j] > 0.3:
                    ml.append((i, j))
                    ml.append((j, i))
                elif self.pairwise_similarities[i][j] < -0.3:
                    cl.append((i, j))
                    cl.append((j, i))

        # Function to add both directions of a link in a graph
        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        # Initialize graphs for must-link and cannot-link
        ml_graph = {i: set() for i in range(n)}
        cl_graph = {i: set() for i in range(n)}

        # Add must-link pairs to the graph
        for (i, j) in ml:
            add_both(ml_graph, i, j)

        # Depth-first search to find connected components in must-link graph
        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)
        
        #print(ml_graph)

        # Find transitive closure of must-link graph
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        #print(ml_graph)

        # Update cannot-link graph based on must-link information
        for (i, j) in cl:
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)

        # Check for inconsistencies
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('Inconsistent constraints between %d and %d' % (i, j))

        # Update the similarity matrix based on transitive closure
        for i in range(n):
            for j in range(n):
                if i != j:
                    if j in ml_graph[i]:
                        self.pairwise_similarities[i][j] = 1
                        self.pairwise_similarities[j][i] = 1
                    elif j in cl_graph[i]:
                        self.pairwise_similarities[i][j] = -1
                        self.pairwise_similarities[j][i] = -1




    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] >= 0) or \
                (self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] < 0)
                        
            