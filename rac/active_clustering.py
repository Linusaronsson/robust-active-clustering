import time
import math

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
from scipy.stats import entropy as scipy_entropy

from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier

from rac.utils.models import TwoLayerNet, ThreeLayerNet
from rac.experiment_data import ExperimentData
from rac.correlation_clustering import max_correlation, max_correlation_dynamic_K
from rac.query_strategies import QueryStrategy

import warnings
warnings.filterwarnings("once") 

#self.net = ACCNet(TwoLayerNet(input_dim, 512, 1024), TwoLayerNet(input_dim, 512, 1024)).to(self.device)
class ACCNet(nn.Module):
    def __init__(self, net1, net2):
        super(ACCNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.combined_net1 = ThreeLayerNet(1024, 256, 2048, 516)
        self.combined_net2 = ThreeLayerNet(256, 1, 128, 32)

    def forward(self, X):
        if X.shape[1] != 2:
            raise ValueError("WRONG INPUT DIM")
        out1 = self.net1(X[:, 0, :])
        out2 = self.net2(X[:, 1, :])
        combined = torch.cat((out1, out2), 1)
        res1 = F.relu(self.combined_net1(combined))
        #return torch.clip(self.combined_net(combined), min=-1, max=1)
        return self.combined_net2(res1)

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

        if self.num_feedback < 1:
            self.query_size = math.ceil(self.n_edges * self.num_feedback)
        else:
            self.query_size = self.num_feedback

        self.random = np.random.RandomState(self.repeat_id+self.seed+317421)
        self.input_dim = self.X.shape[1]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = None

        if hasattr(self, "tau"):
            if self.tau == -1:
                self.tau = np.inf
            else:
                self.tau = self.tau

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
            stopping_criteria = self.query_size * 200
        if self.acq_fn in ["nCOBRAS", "COBRAS"]:
            stopping_criteria = self.query_size * 120
        else:
            stopping_criteria = self.n_edges

        early_stopping = (self.n_edges/self.query_size)*1
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
                self.edges, objects = self.qs.select_batch(
                    acq_fn=self.acq_fn,
                    local_regions=self.local_regions,
                    batch_size=self.query_size)

                #if len(objects) == 1:
                    #raise ValueError("Singleton cluster in run_AL_procedure(...)")

                if len(self.edges) != self.query_size:
                    raise ValueError("Num queried {} not equal to query size {}!!".format(len(self.edges[0]), self.query_size))
                
                for ind1, ind2 in self.edges:
                    self.update_similarity(ind1, ind2)

                if self.predict_sims:
                    self.predict_similarities()

                if self.infer_sims:
                    self.infer_similarities()

                if self.infer_sims2:
                    num_inferred = np.inf
                    while num_inferred > 0:
                        num_inferred = self.infer_similarities2()

                if self.infer_sims3:
                    self.infer_similarities3()

                time_nn = time.time()
                if self.force_global_update:
                    self.update_clustering(np.arange(self.N).tolist()) 
                else:
                    self.update_clustering(objects) 
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
                #print("iteration: ", ii)
                print("prop_queried: ", total_queries/self.n_edges)
                print("feedback_freq max: ", np.max(self.feedback_freq))
                print("rand score: ", adjusted_rand_score(self.Y, self.clustering_solution))
                #clustering_saved = self.clustering.copy()
                #self.update_clustering(np.arange(self.N).tolist()) 
                #print("rand score global: ", adjusted_rand_score(self.Y, self.clustering_solution))
                #print("cost global ", self.compute_clustering_cost(self.pairwise_similarities, self.clustering))
                #self.clustering = clustering_saved
                #self.update_clustering(objects) 
                #print("rand score local: ", adjusted_rand_score(self.Y, self.clustering_solution))
                #print("cost local", self.compute_clustering_cost(self.pairwise_similarities, self.clustering))
                print("time: ", time.time()-self.start)
                
                if self.acq_fn not in ["QECC", "COBRAS", "nCOBRAS"]:
                    print("num queries: ", len(self.edges[0]))
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
            
            if current_rand >= 1.0:
                perfect_rand_count += 1
            else:
                perfect_rand_count = 0
            
            old_rand = current_rand

            if early_stopping_count > early_stopping or perfect_rand_count > 5:
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
            self.random.shuffle(unique_pairs)
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
            ind1 = self.random.choice(len(vertices), 1, replace=False)[0]
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
                if self.random.uniform(-1.0, 1.0) > 0:
                    self.ground_truth_pairwise_similarities_noisy[ind1, ind2] *= -1
                    self.ground_truth_pairwise_similarities_noisy[ind2, ind1] *= -1
            else:
                noisy_val = self.random.uniform(0.15, 0.5)
                if self.random.uniform(-1.0, 1.0) > 0:
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

        if self.sim_init_type == "uniform_random":
            self.clustering = [np.arange(self.N).tolist()]
            self.pairwise_similarities = self.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.update_clustering(np.arange(self.N).tolist()) 
        elif self.sim_init_type == "uniform_random_clustering":
            self.clustering = [np.arange(self.N).tolist()]
            self.pairwise_similarities = self.random.uniform(
                low=-self.sim_init,
                high=self.sim_init, 
                size=(self.N, self.N)
            )
            self.update_clustering(np.arange(self.N).tolist()) 
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        elif self.sim_init_type == "inverse_dist":
            D = distance.cdist(self.X, self.X, 'euclidean')
            sim_matrix = np.max(D) - D + np.min(D)
            self.pairwise_similarities = self.sim_init * (2 * sim_matrix - np.max(sim_matrix) -  np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
            np.fill_diagonal(self.pairwise_similarities, 0.0)
            self.clustering = [np.arange(self.N).tolist()]
            self.update_clustering(np.arange(self.N).tolist()) 
        elif self.sim_init_type == "custom":
            self.clustering_solution = np.array(self.initial_clustering_solution)
            self.clustering, self.num_clusters = self.clustering_from_clustering_solution(self.clustering_solution)
            self.pairwise_similarities = self.sim_matrix_from_clustering(self.clustering)
        elif self.sim_init_type == "random_clustering":
            objects = np.arange(self.N)
            self.random.shuffle(objects)
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
        else:
            raise ValueError("Invalid sim init type in construct_initial_sim_matrix(...)")
        np.fill_diagonal(self.pairwise_similarities, 0.0)

    # Input: list of objects to update clustering w.r.t. (subset of all objects)
    def update_clustering(self, objects):
        objects = np.unique(objects)
        clusters = [] 
        i = 0
        for cluster_objs in self.clustering:
            if not set(cluster_objs).isdisjoint(objects):
                clusters.append(i)
            i += 1
        n_clusters = len(clusters)

        temp_graph = self.pairwise_similarities[np.ix_(objects, objects)]
        new_clustering, _ = max_correlation_dynamic_K(temp_graph, n_clusters, 3, self.random)

        self.clustering = np.delete(np.array(self.clustering, dtype=object), clusters, axis=0).tolist()

        # integrate clustering solution
        n_new_clusters = np.max(new_clustering) + 1
        for i in range(n_new_clusters):
            new_cluster = objects[np.where(new_clustering == i)].tolist()
            self.clustering.append(new_cluster)
        
        self.clustering_solution = np.zeros(self.N, dtype=np.uint32)
        for k in range(len(self.clustering)):
            self.clustering_solution[self.clustering[k]] = k
        #self.num_clusters = np.max(self.clustering_solution) + 1
        self.num_clusters = len(self.clustering)
    
    def update_similarity(self, ind1, ind2, custom_query=None, update_freq=True):
        #if update_freq:
        self.feedback_freq[ind1, ind2] += 1
        self.feedback_freq[ind2, ind1] += 1

        feedback = self.feedback_freq[ind1, ind2]
        similarity = self.pairwise_similarities[ind1, ind2]

        if custom_query is not None:
            query = custom_query
        elif self.random.rand() <= self.noise_level:
            if self.binary_noise:
                if self.random.uniform(-1.0, 1.0) > 0:
                    query = -self.ground_truth_pairwise_similarities_noisy[ind1, ind2]
                else:
                    query = self.ground_truth_pairwise_similarities_noisy[ind1, ind2]
            else:
                noisy_val = self.random.uniform(0.15, 0.5)
                if self.random.uniform(-1.0, 1.0) > 0:
                    query = -noisy_val
                else:
                    query = noisy_val

                #query = self.random.uniform(-1.0, 1.0)
        else:
            query = self.ground_truth_pairwise_similarities_noisy[ind1, ind2]

        self.pairwise_similarities[ind1, ind2] = ((feedback-1) * similarity + query)/(feedback)
        self.pairwise_similarities[ind2, ind1] = ((feedback-1) * similarity + query)/(feedback)
        #if self.violates_clustering(ind1, ind2):
        #    self.violations[ind1, ind2] = np.abs(self.pairwise_similarities[ind1, ind2])
        #    self.violations[ind2, ind1] = np.abs(self.pairwise_similarities[ind2, ind1])
        #else:
        #    self.violations[ind1, ind2] = 0.0
        #    self.violations[ind2, ind1] = 0.0

    def get_similarity(self, ind1, ind2):
        if self.random.rand() <= self.noise_level:
            return self.random.uniform(-1.0, 1.0)
        else:
            return self.ground_truth_pairwise_similarities_noisy[ind1, ind2]

    def predict_similarities(self): 
        input_dim = self.X.shape[1]
        self.retrain_net = False

        if self.retrain_net or self.net is None:
            self.net = ACCNet(TwoLayerNet(input_dim, 512, 1024), TwoLayerNet(input_dim, 512, 1024)).to(self.device)

        lower_triangle_indices = np.tril_indices(self.N, -1) # -1 gives lower triangle without diagonal (0 includes diagonal)
        #cond1 = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] > 0.5) & (self.edges_predicted[lower_triangle_indices] == False))[0]
        #cond2 = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] < -0.5) & (self.edges_predicted[lower_triangle_indices] == False))[0]
        cond1 = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] > 0.5))[0]
        cond2 = np.where((self.feedback_freq[lower_triangle_indices] > 1) & (self.pairwise_similarities[lower_triangle_indices] < -0.5))[0]
        print("cond1 len ", len(cond1))
        print("cond2 len ", len(cond2))
        if len(cond1) < 50 or len(cond2) < 50:
            return

        print("predicting sims...")
        print("cond1 shape ", cond1.shape)
        print("cond2 shape ", cond2.shape)

        indices1 = self.random.choice(cond1, np.min([len(cond1), len(cond2), 2000]))
        indices2 = self.random.choice(cond2, len(indices1))
        #indices1 = cond1
        #indices2 = cond2
        indices = np.concatenate([indices1, indices2])
        ind1, ind2 = lower_triangle_indices[0][indices], lower_triangle_indices[1][indices]
        x1 = self.X[ind1].reshape((len(ind1), 1, input_dim))
        x2 = self.X[ind2].reshape((len(ind2), 1, input_dim))
        dataset = np.concatenate((x1, x2), axis=1)
        labels = self.pairwise_similarities[ind1, ind2]
        lab1 = np.where(labels >= 0)
        lab2 = np.where(labels < 0)
        labels[lab1] = 1.0
        labels[lab2] = 0.0
        print("Dataset shape: ", dataset.shape)
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset), torch.Tensor(labels))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        #criterion = nn.MSELoss()
        #criterion = nn.SmoothL1Loss()
        criterion = nn.BCEWithLogitsLoss()
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0005, momentum=0.9)
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=0.0001)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        print("training...")
        print(len(train_dataset))
        for epoch in range(25):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                outputs = outputs.reshape((outputs.shape[0]))
                #labels = labels.reshape((labels.shape[0], 1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.10f}')
                    running_loss = 0.0
        # CONTINUE HERE, MAKE PREDS FILL UP SIM MATRIX
        print("predicting")
        #lower_triangle_indices = np.tril_indices(self.N, -1) # -1 gives lower triangle without diagonal (0 includes diagonal)
        num_preds = self.query_size * 6
        edges, objects = self.qs.select_batch("uncert", "pairs", num_preds)
        ind1, ind2 = edges[:, 0], edges[:, 1]

        #self.saved_ind1, self.saved_ind2 = self.select_similarities_all2()
        #self.num_feedback = saved_fb
        #indices = self.random.choice(ind_not_queried, num_preds)
        #ind1, ind2 = lower_triangle_indices[0][indices], lower_triangle_indices[1][indices] 
        dat1 = self.X[ind1].reshape(num_preds, 1, input_dim)
        dat2 = self.X[ind2].reshape(num_preds, 1, input_dim)
        dat_all = torch.Tensor(np.concatenate((dat1, dat2), axis=1))
        dat_all = torch.utils.data.TensorDataset(dat_all)
        test_loader = torch.utils.data.DataLoader(dat_all, shuffle=False, batch_size=1024)
        preds = []
        #self.saved_queries = self.similarity_matrix[self.saved_ind1, self.saved_ind2]
        for i, data in enumerate(test_loader, 0):
            input = data[0].to(self.device)
            pred = self.net(input)
            pred = nn.Sigmoid()(pred)
            preds.extend(pred[:, 0].tolist())
        print("NUM PREDS: ", num_preds)
        countt = 0
        #self.saved_ind11 = []
        #self.saved_ind22 = []
        #self.saved_queries = []
        for i1, i2, pred in zip(ind1, ind2, preds):
            prob = [1-pred, pred]
            entropy = scipy_entropy(prob)
            if entropy > 0.05:
                continue
            #self.edges_predicted[i1, i2] = True
            #self.saved_queries.append(self.similarity_matrix[i1, i2])
            #self.saved_ind11.append(i1)
            #self.saved_ind22.append(i2)
            countt += 1
            #pred = (pred - 0.5) * 2
            if pred >= 0.5:
                pred = 0.25
            else:
                pred = -0.25
            self.update_similarity(i1, i2, custom_query=pred, update_freq=False)
        #self.saved_queries = self.similarity_matrix[self.saved_ind11, self.saved_ind22]
        print("COUNTT: ", countt)
            #self.similarity_matrix[i1, i2] = pred

    def infer_similarities2(self): 
        num_inferred = 0
        confidence_limit = 1
        for i in range(0, self.N):
            current_indices_pos = []
            current_indices_neg = []
            for j in range(0, self.N):
                if i == j:
                    continue
                #if self.similarity_matrix[i, j] > 0.5:
                if self.feedback_freq[i, j] > confidence_limit:
                    if self.pairwise_similarities[i, j] > 0:
                        current_indices_pos.append(j)
                    if self.pairwise_similarities[i, j] < 0:
                        current_indices_neg.append(j)
            for k in itertools.permutations(current_indices_pos, 2): 
                if self.feedback_freq[k] <= confidence_limit:
                    self.update_similarity(k[0], k[1], custom_query=1)
                    num_inferred += 1
            for pos_ind in current_indices_pos:
                for neg_ind in current_indices_neg:
                    if self.feedback_freq[pos_ind, neg_ind] <= confidence_limit:
                        self.update_similarity(pos_ind, neg_ind, custom_query=-1)
                        num_inferred += 1
        return num_inferred

    def infer_similarities(self):
        for k1 in range(0, self.num_clusters):
            for k2 in range(0, k1 + 1):
                c1 = self.clustering[k1]
                c2 = self.clustering[k2]

                pairwise_sims = self.parwise_similarities[np.ix_(c1, c2)]
                pairwise_counts = self.feedback_freq[np.ix_(c1, c2)]

                # If the clusters are the same, only consider the lower triangular part of the matrix, excluding the diagonal
                if k1 == k2:
                    mask = np.tril(np.ones_like(pairwise_sims, dtype=bool), k=-1)
                    pairwise_sims = np.where(mask, pairwise_sims, np.nan)
                    pairwise_counts = np.where(mask, pairwise_counts, np.nan)

                sims_with_counts_gt_1 = pairwise_sims[pairwise_counts > 1]

                if np.isnan(sims_with_counts_gt_1).all():  # If all elements are NaN, skip this iteration
                    continue

                mean_sim = np.nanmean(sims_with_counts_gt_1)

                # Identify the pairwise similarities with F <= 1
                sims_with_counts_le_1_idx = np.where(pairwise_counts <= 1)

                # Loop over every index in sims_with_counts_le_1_idx 
                for idx in zip(*sims_with_counts_le_1_idx):
                    idx_cluster1 = c1[idx[0]]
                    idx_cluster2 = c2[idx[1]]
                    self.pairwise_similarities[idx_cluster1, idx_cluster2] = mean_sim
                    self.pairwise_similarities[idx_cluster2, idx_cluster1] = mean_sim
                    #self.update_similarity(idx_cluster1, idx_cluster2, mean_sim)

    def violates_clustering(self, o1, o2): 
        return (not self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] >= 0) or \
                (self.in_same_cluster(o1, o2) and self.pairwise_similarities[o1, o2] < 0)

    def infer_similarities3(self): 
        num_inferred = 0
        confidence_limit = 1
        infer_count = np.zeros((self.N, self.N))
        inferred_values = np.zeros((self.N, self.N))
        for i in range(0, self.N):
            current_indices_pos = []
            current_indices_neg = []
            for j in range(0, self.N):
                if i == j:
                    continue
                #if self.similarity_matrix[i, j] > 0.5:
                if self.feedback_freq[i, j] > confidence_limit:
                    if self.pairwise_similarities[i, j] > 0:
                        current_indices_pos.append(j)
                    if self.pairwise_similarities[i, j] < 0:
                        current_indices_neg.append(j)
            for k in itertools.permutations(current_indices_pos, 2): 
                if self.feedback_freq[k] <= confidence_limit:
                    infer_count[k[0], k[1]] += 1
                    infer_count[k[1], k[0]] += 1
                    inferred_values[k[0], k[1]] += 1
                    inferred_values[k[1], k[0]] += 1

                    #self.update_similarity(k[0], k[1], custom_query=1)
                    #num_inferred += 1
            for pos_ind in current_indices_pos:
                for neg_ind in current_indices_neg:
                    if self.feedback_freq[pos_ind, neg_ind] <= confidence_limit:
                        #self.update_similarity(pos_ind, neg_ind, custom_query=-1)
                        #num_inferred += 1
                        infer_count[pos_ind, neg_ind] += 1
                        infer_count[neg_ind, pos_ind] += 1
                        inferred_values[pos_ind, neg_ind] -= 1
                        inferred_values[neg_ind, pos_ind] -= 1
        
        for i in range(0, self.N):
            for j in range(0, i):
                if infer_count[i, j] > 10:
                    if inferred_values[i, j] > 0:
                        self.pairwise_similarities[i, j] = 1
                        self.pairwise_similarities[j, i] = 1
                        self.pairwise_similarities[j, i] = 1
                    else:
                        self.pairwise_similarities[i, j] = -1
                        self.pairwise_similarities[j, i] = -1
                    self.feedback_freq[i, j] +=1
                    self.feedback_freq[j, i] +=1
        print("HERE: ", np.max((infer_count)))
                        

            
        return num_inferred


            